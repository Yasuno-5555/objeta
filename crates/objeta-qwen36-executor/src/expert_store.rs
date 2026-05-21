//! Expert Store — per-layer expert weight offloading with Metal buffer integration.
//!
//! For Qwen3.6-class MoE models, expert weights are MERGED per layer into
//! large tensors (e.g. gate_up_proj: [256*1024, 2048] BF16 ≈ 1GB).
//! Instead of per-expert caching, we offload ENTIRE LAYERS of expert weights
//! to SSD and load them on demand.
//!
//! Architecture:
//!   SSD (flat binary files, one per layer)
//!     → mmap → Metal buffer (zero-copy via ptr::copy to MTLBuffer)
//!       → MLX array wraps the Metal buffer
//!         → forward pass
//!
//! During inference:
//!   Layer L: compute using Metal buffer for layer L
//!            prefetch layer L+1 from SSD in background
//!   Layer L+1: Metal buffer already populated → no stall
//!
//! Memory model (Qwen3.6-35B-A3B, q4):
//!   Per layer expert weights: ~400MB (gate_up + down, all 256 experts)
//!   Active buffers: 2 layers × 400MB = 800MB (current + prefetch)
//!   Fits comfortably in 6.6GB free on M1 8GB

use std::collections::HashMap;
use std::fs::{self, File};
use std::io;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

// ── Error type ────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ExpertStoreError {
    Io(io::Error),
    LayerNotFound(u32),
    BufferNotLoaded(u32),
    AlreadyLoaded(u32),
}

impl From<io::Error> for ExpertStoreError {
    fn from(e: io::Error) -> Self {
        ExpertStoreError::Io(e)
    }
}

// ── ExpertStore ───────────────────────────────────────────────────────

/// Manages per-layer expert weight files on SSD with Metal buffer integration.
pub struct ExpertStore {
    /// Directory containing per-layer expert weight files.
    #[allow(dead_code)]
    ssd_dir: PathBuf,
    /// Number of layers.
    #[allow(dead_code)]
    n_layers: u32,
    /// Registered layer files: layer_idx → (file_path, size_bytes).
    layer_files: HashMap<u32, (PathBuf, u64)>,
    /// Currently loaded layers → Metal buffer pointers.
    loaded_buffers: HashMap<u32, MetalBuffer>,
    /// Layer currently being prefetched (None if idle).
    prefetching: Option<u32>,
    /// Prefetch thread handle.
    prefetch_handle: Option<thread::JoinHandle<()>>,
    /// Flag: stop prefetch worker.
    running: Arc<AtomicBool>,
}

/// A Metal buffer holding expert weights for one layer.
#[derive(Clone, Debug)]
pub struct MetalBuffer {
    pub ptr: u64,
    pub size_bytes: u64,
}

/// Transfer statistics for monitoring offloading performance.
#[derive(Clone, Debug, Default)]
pub struct TransferStats {
    pub layers_loaded: u64,
    pub bytes_transferred: u64,
    pub prefetches_completed: u64,
    pub prefetch_stalls: u64, // Times we had to wait for prefetch
    pub total_load_ms: u64,
}

impl ExpertStore {
    /// Get transfer statistics.
    pub fn stats(&self) -> TransferStats {
        TransferStats::default() // TODO: track actual stats
    }

    /// Swap double buffers: unload previous layer, ensure next layer is loaded.
    ///
    /// Typical inference loop:
    ///   layer 0: load(0, buf_a)
    ///   layer 1: prefetch_async(1, buf_b); compute layer 0 from buf_a
    ///            swap_buffers(0, 1, buf_a, buf_b) → unload 0, wait for 1
    ///   layer 2: prefetch_async(2, buf_a); compute layer 1 from buf_b
    ///            swap_buffers(1, 2, buf_b, buf_a) → unload 1, wait for 2
    pub fn swap_layer(
        &mut self,
        unload_idx: u32,
        next_idx: u32,
        next_buffer_ptr: u64,
    ) -> Result<MetalBuffer, ExpertStoreError> {
        // Unload previous layer to free Metal buffer memory
        self.unload_layer(unload_idx);

        // Wait for prefetch if still in progress
        self.prefetch_wait();

        // Load if not already loaded by prefetch
        if self.get_buffer(next_idx).is_err() {
            self.load_layer(next_idx, next_buffer_ptr)?;
        }

        self.get_buffer(next_idx).cloned()
    }
}

impl ExpertStore {
    /// Create a new expert store.
    pub fn new(ssd_dir: &str, n_layers: u32) -> Self {
        fs::create_dir_all(ssd_dir).ok();
        Self {
            ssd_dir: PathBuf::from(ssd_dir),
            n_layers,
            layer_files: HashMap::new(),
            loaded_buffers: HashMap::new(),
            prefetching: None,
            prefetch_handle: None,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Register a layer's expert weight file on SSD.
    ///
    /// The file should be a flat binary containing all expert weights
    /// for this layer (gate_up_proj + down_proj concatenated).
    /// File format: [gate_up_proj (n_experts * 2 * ffn * hidden * sizeof(dtype))]
    ///               [down_proj (n_experts * hidden * ffn * sizeof(dtype))]
    pub fn register_layer(&mut self, layer_idx: u32, filepath: &str, size_bytes: u64) {
        self.layer_files
            .insert(layer_idx, (PathBuf::from(filepath), size_bytes));
    }

    /// Load a layer's expert weights from SSD into a Metal buffer.
    ///
    /// Uses mmap for efficient SSD read, then copies to the provided Metal buffer.
    /// Returns the Metal buffer pointer + size.
    pub fn load_layer(
        &mut self,
        layer_idx: u32,
        metal_buffer_ptr: u64,
    ) -> Result<MetalBuffer, ExpertStoreError> {
        let (filepath, size_bytes) = self
            .layer_files
            .get(&layer_idx)
            .ok_or(ExpertStoreError::LayerNotFound(layer_idx))?
            .clone();

        // Memory-map the file (zero-copy SSD access via OS page cache)
        let file = File::open(&filepath)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Copy from mmap to Metal buffer (unified memory, single copy)
        unsafe {
            ptr::copy_nonoverlapping(
                mmap.as_ptr(),
                metal_buffer_ptr as *mut u8,
                size_bytes as usize,
            );
        }

        let buffer = MetalBuffer {
            ptr: metal_buffer_ptr,
            size_bytes,
        };

        self.loaded_buffers.insert(layer_idx, buffer.clone());
        Ok(buffer)
    }

    /// Load a layer directly from a memory-mapped file (no Metal buffer copy yet).
    /// Returns a pointer to the mmap'd region.
    pub fn mmap_layer(&self, layer_idx: u32) -> Result<(*const u8, u64), ExpertStoreError> {
        let (filepath, _size_bytes) = self
            .layer_files
            .get(&layer_idx)
            .ok_or(ExpertStoreError::LayerNotFound(layer_idx))?
            .clone();

        let file = File::open(&filepath)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Leak the mmap to keep it alive — caller is responsible for cleanup
        let ptr = mmap.as_ptr();
        let size = mmap.len() as u64;
        std::mem::forget(mmap);

        Ok((ptr, size))
    }

    /// Get a loaded layer's Metal buffer.
    pub fn get_buffer(&self, layer_idx: u32) -> Result<&MetalBuffer, ExpertStoreError> {
        self.loaded_buffers
            .get(&layer_idx)
            .ok_or(ExpertStoreError::BufferNotLoaded(layer_idx))
    }

    /// Unload a layer to free Metal buffer memory.
    pub fn unload_layer(&mut self, layer_idx: u32) {
        self.loaded_buffers.remove(&layer_idx);
    }

    /// Start background prefetch of a layer.
    ///
    /// The prefetch worker loads the layer's expert weights from SSD
    /// into the Metal buffer while the GPU is computing the current layer.
    pub fn prefetch_async(
        &mut self,
        layer_idx: u32,
        metal_buffer_ptr: u64,
    ) -> Result<(), ExpertStoreError> {
        // Check if layer file exists
        if !self.layer_files.contains_key(&layer_idx) {
            return Err(ExpertStoreError::LayerNotFound(layer_idx));
        }

        // Check if already loaded
        if self.loaded_buffers.contains_key(&layer_idx) {
            return Ok(());
        }

        let (filepath, size_bytes) = self.layer_files.get(&layer_idx).unwrap().clone();
        self.prefetching = Some(layer_idx);
        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();

        self.prefetch_handle = Some(thread::spawn(move || {
            // mmap and copy to Metal buffer
            if let Ok(file) = File::open(&filepath) {
                if let Ok(mmap) = unsafe { memmap2::Mmap::map(&file) } {
                    unsafe {
                        ptr::copy_nonoverlapping(
                            mmap.as_ptr(),
                            metal_buffer_ptr as *mut u8,
                            size_bytes as usize,
                        );
                    }
                }
            }
            running.store(false, Ordering::SeqCst);
        }));

        Ok(())
    }

    /// Check if prefetch is complete.
    pub fn prefetch_done(&self) -> bool {
        !self.running.load(Ordering::SeqCst)
    }

    /// Wait for prefetch to complete.
    pub fn prefetch_wait(&mut self) {
        if let Some(handle) = self.prefetch_handle.take() {
            let _ = handle.join();
            self.prefetching = None;
        }
    }

    /// Get the number of registered layers.
    pub fn n_registered(&self) -> u32 {
        self.layer_files.len() as u32
    }

    /// Get the number of currently loaded layers.
    pub fn n_loaded(&self) -> u32 {
        self.loaded_buffers.len() as u32
    }

    /// Get total SSD storage used by all registered layers.
    pub fn total_ssd_bytes(&self) -> u64 {
        self.layer_files.values().map(|(_, size)| size).sum()
    }
}

impl Drop for ExpertStore {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.prefetch_handle.take() {
            let _ = handle.join();
        }
    }
}

// ── C API ─────────────────────────────────────────────────────────────

/// Create a new ExpertStore. Returns opaque pointer.
#[no_mangle]
pub extern "C" fn expert_store_create(
    ssd_dir: *const std::os::raw::c_char,
    n_layers: u32,
) -> *mut ExpertStore {
    let dir = unsafe {
        std::ffi::CStr::from_ptr(ssd_dir)
            .to_string_lossy()
            .into_owned()
    };
    let store = Box::new(ExpertStore::new(&dir, n_layers));
    Box::into_raw(store)
}

/// Register a layer file on SSD.
#[no_mangle]
pub extern "C" fn expert_store_register_layer(
    store: *mut ExpertStore,
    layer_idx: u32,
    filepath: *const std::os::raw::c_char,
    size_bytes: u64,
) -> i32 {
    let store = unsafe { &mut *store };
    let path = unsafe {
        std::ffi::CStr::from_ptr(filepath)
            .to_string_lossy()
            .into_owned()
    };
    store.register_layer(layer_idx, &path, size_bytes);
    0
}

/// Load a layer into a Metal buffer. Returns 0 on success.
#[no_mangle]
pub extern "C" fn expert_store_load_layer(
    store: *mut ExpertStore,
    layer_idx: u32,
    metal_buffer_ptr: u64,
) -> i32 {
    let store = unsafe { &mut *store };
    match store.load_layer(layer_idx, metal_buffer_ptr) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Get a loaded buffer's pointer. Returns 0 if not loaded.
#[no_mangle]
pub extern "C" fn expert_store_get_buffer_ptr(store: *const ExpertStore, layer_idx: u32) -> u64 {
    let store = unsafe { &*store };
    store.get_buffer(layer_idx).map(|b| b.ptr).unwrap_or(0)
}

/// Get a loaded buffer's size. Returns 0 if not loaded.
#[no_mangle]
pub extern "C" fn expert_store_get_buffer_size(store: *const ExpertStore, layer_idx: u32) -> u64 {
    let store = unsafe { &*store };
    store
        .get_buffer(layer_idx)
        .map(|b| b.size_bytes)
        .unwrap_or(0)
}

/// Start async prefetch of a layer.
#[no_mangle]
pub extern "C" fn expert_store_prefetch_async(
    store: *mut ExpertStore,
    layer_idx: u32,
    metal_buffer_ptr: u64,
) -> i32 {
    let store = unsafe { &mut *store };
    match store.prefetch_async(layer_idx, metal_buffer_ptr) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Check if async prefetch is complete. Returns 1 if done, 0 if in progress.
#[no_mangle]
pub extern "C" fn expert_store_prefetch_done(store: *const ExpertStore) -> i32 {
    let store = unsafe { &*store };
    if store.prefetch_done() {
        1
    } else {
        0
    }
}

/// Wait for async prefetch to complete.
#[no_mangle]
pub extern "C" fn expert_store_prefetch_wait(store: *mut ExpertStore) {
    let store = unsafe { &mut *store };
    store.prefetch_wait();
}

/// Unload a layer to free Metal buffer memory.
#[no_mangle]
pub extern "C" fn expert_store_unload_layer(store: *mut ExpertStore, layer_idx: u32) {
    let store = unsafe { &mut *store };
    store.unload_layer(layer_idx);
}

/// Destroy the expert store and free all resources.
#[no_mangle]
pub extern "C" fn expert_store_destroy(store: *mut ExpertStore) {
    if !store.is_null() {
        let _ = unsafe { Box::from_raw(store) };
    }
}

/// Double-buffer swap: unload old layer, ensure new layer is loaded.
/// Returns the buffer pointer for the new layer, or 0 on error.
#[no_mangle]
pub extern "C" fn expert_store_swap_layer(
    store: *mut ExpertStore,
    unload_idx: u32,
    next_idx: u32,
    next_buffer_ptr: u64,
) -> u64 {
    let store = unsafe { &mut *store };
    match store.swap_layer(unload_idx, next_idx, next_buffer_ptr) {
        Ok(buf) => buf.ptr,
        Err(_) => 0,
    }
}

/// Get transfer statistics as JSON string. Caller must free with expert_store_free_stats.
#[no_mangle]
pub extern "C" fn expert_store_get_stats_json(
    store: *const ExpertStore,
) -> *mut std::os::raw::c_char {
    let store = unsafe { &*store };
    let stats = store.stats();
    let json = format!(
        r#"{{"layers_loaded":{},"bytes_transferred":{},"prefetches_completed":{},"prefetch_stalls":{},"total_load_ms":{}}}"#,
        stats.layers_loaded,
        stats.bytes_transferred,
        stats.prefetches_completed,
        stats.prefetch_stalls,
        stats.total_load_ms
    );
    let c_str = std::ffi::CString::new(json).unwrap();
    c_str.into_raw()
}

/// Free stats JSON string.
#[no_mangle]
pub extern "C" fn expert_store_free_stats(ptr: *mut std::os::raw::c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = std::ffi::CString::from_raw(ptr);
        }
    }
}

/// Get total SSD storage used.
#[no_mangle]
pub extern "C" fn expert_store_total_ssd_bytes(store: *const ExpertStore) -> u64 {
    let store = unsafe { &*store };
    store.total_ssd_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_basic_store() {
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_string_lossy();

        // Create a fake layer file
        let file_path = dir.path().join("layer_0.bin");
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let mut file = File::create(&file_path).unwrap();
        file.write_all(&data).unwrap();

        let mut store = ExpertStore::new(&dir_path, 1);
        store.register_layer(0, &file_path.to_string_lossy(), data.len() as u64);

        assert_eq!(store.n_registered(), 1);
        assert_eq!(store.total_ssd_bytes(), data.len() as u64);

        // Test mmap
        let (ptr, size) = store.mmap_layer(0).unwrap();
        assert_eq!(size, data.len() as u64);
        let slice = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        assert_eq!(slice[0], data[0]);
        assert_eq!(slice[4095], data[4095]);

        dir.close().ok();
    }
}
