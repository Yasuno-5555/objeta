//! KV Arena — fixed-size Metal buffer manager.
//!
//! Eliminates MLX graph accumulation from KV cache operations by using
//! direct CPU memcpy into pre-allocated Metal buffers.
//!
//! On Apple Silicon (M1–M4), Metal buffers reside in unified memory,
//! so CPU-side writes are directly visible to the GPU after MLX's
//! next `synchronize()` call. No graph nodes are created.
//!
//! Architecture:
//!   Python (MLX) allocates KV buffers as mx.zeros(..., dtype=mx.float16)
//!     → extracts MTLBuffer pointer
//!     → registers with Rust KVArena
//!   Rust KVArena writes K/V directly into the MTLBuffer (ptr::copy)
//!   Python reads via MLX array views wrapping buffer slices

use std::ptr;

// ── Constants ─────────────────────────────────────────────────────

/// f16 element size in bytes.
const F16_BYTES: u32 = 2;

// ── KVArena ───────────────────────────────────────────────────────

#[derive(Debug)]
pub struct KVArena {
    /// Number of transformer layers.
    n_layers: u32,
    /// Number of KV heads per layer.
    n_kv_heads: u32,
    /// Head dimension.
    head_dim: u32,
    /// Maximum sequence length (static allocation).
    max_seq_len: u32,
    /// MLX array stride: elements per position per head (= head_dim).
    head_stride: u32,
    /// MLX array stride: elements between heads at same position (= max_seq_len * head_dim).
    head_offset: u64,
    /// Total elements per layer buffer.
    elements_per_layer: u64,
    /// Total bytes per layer buffer (f16).
    #[allow(dead_code)]
    bytes_per_layer: u64,
    /// K buffer GPU pointers per layer (0 = unregistered).
    k_ptrs: Vec<u64>,
    /// V buffer GPU pointers per layer (0 = unregistered).
    v_ptrs: Vec<u64>,
    /// Current sequence length.
    seq_len: u32,
}

impl KVArena {
    pub fn new(
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
    ) -> Self {
        // MLX row-major layout: shape (n_kv_heads, max_seq_len, head_dim)
        // Element [h, p, d] at offset: h * max_seq_len * head_dim + p * head_dim + d
        let head_stride = head_dim;
        let head_offset = max_seq_len as u64 * head_dim as u64;
        let elements_per_layer = n_kv_heads as u64 * max_seq_len as u64 * head_dim as u64;
        let bytes_per_layer = elements_per_layer * F16_BYTES as u64;
        Self {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            head_stride,
            head_offset,
            elements_per_layer,
            bytes_per_layer,
            k_ptrs: vec![0u64; n_layers as usize],
            v_ptrs: vec![0u64; n_layers as usize],
            seq_len: 0,
        }
    }

    // ── Buffer registration ──────────────────────────────────────

    /// Register a K buffer pointer for a layer.
    /// `ptr` is the GPU device pointer (CPU-accessible on Apple Silicon).
    pub fn register_k(&mut self, layer_idx: u32, ptr: u64) {
        if (layer_idx as usize) < self.k_ptrs.len() {
            self.k_ptrs[layer_idx as usize] = ptr;
        }
    }

    /// Register a V buffer pointer for a layer.
    pub fn register_v(&mut self, layer_idx: u32, ptr: u64) {
        if (layer_idx as usize) < self.v_ptrs.len() {
            self.v_ptrs[layer_idx as usize] = ptr;
        }
    }

    /// Register both K and V buffers for a layer at once.
    pub fn register_layer(&mut self, layer_idx: u32, k_ptr: u64, v_ptr: u64) {
        self.register_k(layer_idx, k_ptr);
        self.register_v(layer_idx, v_ptr);
    }

    /// True if all layer buffers are registered.
    pub fn is_ready(&self) -> bool {
        self.k_ptrs.iter().all(|&p| p != 0)
            && self.v_ptrs.iter().all(|&p| p != 0)
    }

    // ── Write ────────────────────────────────────────────────────

    /// Write K and V into the arena at `position`.
    ///
    /// MLX buffer layout: shape (n_kv_heads, max_seq_len, head_dim) in row-major.
    /// Element [h, p, d] at offset: h * max_seq_len * head_dim + p * head_dim + d.
    /// k_data/v_data: contiguous f16, n_kv_heads * head_dim elements,
    ///                interleaved as [head0_d0..d63, head1_d0..d63, ...].
    pub fn write_kv(
        &self,
        layer_idx: u32,
        position: u32,
        k_data: *const u16,
        v_data: *const u16,
    ) {
        if position >= self.max_seq_len {
            return;
        }
        let k_ptr = self.k_ptrs[layer_idx as usize] as *mut u16;
        let v_ptr = self.v_ptrs[layer_idx as usize] as *mut u16;
        if k_ptr.is_null() || v_ptr.is_null() {
            return;
        }
        let hd = self.head_stride as usize;
        let hop = self.head_offset as usize;
        let pos_base = position as usize * hd;
        // Per-head copy: each head's data at this position is non-contiguous
        for h in 0..self.n_kv_heads as usize {
            let dst_off = h * hop + pos_base;
            let src_off = h * hd;
            unsafe {
                ptr::copy_nonoverlapping(k_data.add(src_off), k_ptr.add(dst_off), hd);
                ptr::copy_nonoverlapping(v_data.add(src_off), v_ptr.add(dst_off), hd);
            }
        }
    }

    /// Write only K at `position`.
    pub fn write_k(&self, layer_idx: u32, position: u32, k_data: *const u16) {
        if position >= self.max_seq_len {
            return;
        }
        let k_ptr = self.k_ptrs[layer_idx as usize] as *mut u16;
        if k_ptr.is_null() {
            return;
        }
        let hd = self.head_stride as usize;
        let hop = self.head_offset as usize;
        let pos_base = position as usize * hd;
        for h in 0..self.n_kv_heads as usize {
            unsafe {
                ptr::copy_nonoverlapping(
                    k_data.add(h * hd), k_ptr.add(h * hop + pos_base), hd);
            }
        }
    }

    /// Write only V at `position`.
    pub fn write_v(&self, layer_idx: u32, position: u32, v_data: *const u16) {
        if position >= self.max_seq_len {
            return;
        }
        let v_ptr = self.v_ptrs[layer_idx as usize] as *mut u16;
        if v_ptr.is_null() {
            return;
        }
        let hd = self.head_stride as usize;
        let hop = self.head_offset as usize;
        let pos_base = position as usize * hd;
        for h in 0..self.n_kv_heads as usize {
            unsafe {
                ptr::copy_nonoverlapping(
                    v_data.add(h * hd), v_ptr.add(h * hop + pos_base), hd);
            }
        }
    }

    /// Write K/V for ALL layers at the same position.
    pub fn write_all_layers(
        &self,
        position: u32,
        all_k: *const u16,
        all_v: *const u16,
    ) {
        let hd = self.head_stride as usize;
        let hop = self.head_offset as usize;
        let heads = self.n_kv_heads as usize;
        let pos_base = position as usize * hd;
        for layer in 0..self.n_layers as usize {
            let k_ptr = self.k_ptrs[layer] as *mut u16;
            let v_ptr = self.v_ptrs[layer] as *mut u16;
            if k_ptr.is_null() || v_ptr.is_null() {
                continue;
            }
            let layer_src_k = unsafe { all_k.add(layer * heads * hd) };
            let layer_src_v = unsafe { all_v.add(layer * heads * hd) };
            for h in 0..heads {
                unsafe {
                    ptr::copy_nonoverlapping(
                        layer_src_k.add(h * hd), k_ptr.add(h * hop + pos_base), hd);
                    ptr::copy_nonoverlapping(
                        layer_src_v.add(h * hd), v_ptr.add(h * hop + pos_base), hd);
                }
            }
        }
    }

    // ── Read (pointer-based — caller wraps as MLX array) ─────────
    // NOTE: get_k_slice / get_v_slice return non-contiguous pointers
    // in MLX row-major layout. Prefer reading via the original MLX arrays
    // (K_cache[:, :seq_len, :]) which correctly handles non-contiguous views.

    /// Get a raw pointer to the K slice [start, end).
    /// WARNING: In MLX layout, position slices are non-contiguous across heads.
    /// Prefer reading via MLX array views instead.
    #[allow(dead_code)]
    pub fn get_k_slice(&self, layer_idx: u32, start: u32, end: u32) -> (*const u16, u64) {
        let ptr = self.k_ptrs[layer_idx as usize];
        if ptr == 0 || start >= end {
            return (std::ptr::null(), 0);
        }
        // Returns pointer to head 0 at `start` — only valid for that head
        let offset = start as u64 * self.head_dim as u64;
        let n_elems = (end - start) as u64 * self.head_dim as u64;
        unsafe { ((ptr as *const u16).add(offset as usize), n_elems) }
    }

    /// Get a raw pointer to the V slice [start, end).
    /// WARNING: Same non-contiguous caveat as get_k_slice.
    #[allow(dead_code)]
    pub fn get_v_slice(&self, layer_idx: u32, start: u32, end: u32) -> (*const u16, u64) {
        let ptr = self.v_ptrs[layer_idx as usize];
        if ptr == 0 || start >= end {
            return (std::ptr::null(), 0);
        }
        let offset = start as u64 * self.head_dim as u64;
        let n_elems = (end - start) as u64 * self.head_dim as u64;
        unsafe { ((ptr as *const u16).add(offset as usize), n_elems) }
    }

    /// Get a full K buffer pointer (for MLX array wrapping).
    pub fn get_k_ptr(&self, layer_idx: u32) -> (u64, u64) {
        let ptr = self.k_ptrs[layer_idx as usize];
        if ptr == 0 {
            return (0, 0);
        }
        (ptr, self.elements_per_layer)
    }

    /// Get a full V buffer pointer.
    pub fn get_v_ptr(&self, layer_idx: u32) -> (u64, u64) {
        let ptr = self.v_ptrs[layer_idx as usize];
        if ptr == 0 {
            return (0, 0);
        }
        (ptr, self.elements_per_layer)
    }

    // ── Copy between arenas ──────────────────────────────────────

    /// Copy KV cache from another arena for a single layer.
    /// Copies `up_to_position` positions, handling MLX non-contiguous layout.
    pub fn copy_from(
        &self,
        src: &KVArena,
        layer_idx: u32,
        up_to_position: u32,
    ) {
        if up_to_position == 0 || up_to_position > self.max_seq_len {
            return;
        }
        let hd = self.head_dim as usize;
        let hop = self.head_offset as usize;
        let n_pos = up_to_position as usize;
        let dst_k = self.k_ptrs[layer_idx as usize] as *mut u16;
        let dst_v = self.v_ptrs[layer_idx as usize] as *mut u16;
        let src_k = src.k_ptrs[layer_idx as usize] as *const u16;
        let src_v = src.v_ptrs[layer_idx as usize] as *const u16;
        if dst_k.is_null() || src_k.is_null() || dst_v.is_null() || src_v.is_null() {
            return;
        }
        // Per-head copy: each head's active region is contiguous in memory
        for h in 0..self.n_kv_heads as usize {
            let head_base = h * hop;
            unsafe {
                ptr::copy_nonoverlapping(
                    src_k.add(head_base), dst_k.add(head_base), n_pos * hd);
                ptr::copy_nonoverlapping(
                    src_v.add(head_base), dst_v.add(head_base), n_pos * hd);
            }
        }
    }

    /// Copy KV from another arena for ALL layers.
    pub fn copy_all_from(
        &self,
        src: &KVArena,
        up_to_position: u32,
    ) {
        for layer in 0..self.n_layers {
            self.copy_from(src, layer, up_to_position);
        }
    }

    // ── Zero / reset ─────────────────────────────────────────────

    /// Zero the active region (up to seq_len) of all buffers.
    /// Per-head zeroing to handle MLX non-contiguous layout.
    pub fn zero_active(&self) {
        let hd = self.head_dim as usize;
        let hop = self.head_offset as usize;
        let n_pos = self.seq_len as usize;
        for layer in 0..self.n_layers as usize {
            let k_ptr = self.k_ptrs[layer] as *mut u16;
            let v_ptr = self.v_ptrs[layer] as *mut u16;
            if k_ptr.is_null() || v_ptr.is_null() {
                continue;
            }
            for h in 0..self.n_kv_heads as usize {
                let head_base = h * hop;
                unsafe {
                    ptr::write_bytes(k_ptr.add(head_base), 0, n_pos * hd);
                    ptr::write_bytes(v_ptr.add(head_base), 0, n_pos * hd);
                }
            }
        }
    }

    /// Zero the entire buffer (full capacity).
    /// Entire buffer is one contiguous allocation — single memset per buffer.
    pub fn zero_all(&self) {
        for layer in 0..self.n_layers as usize {
            let n_elems = self.elements_per_layer as usize;
            let k_ptr = self.k_ptrs[layer] as *mut u16;
            let v_ptr = self.v_ptrs[layer] as *mut u16;
            if !k_ptr.is_null() {
                unsafe { ptr::write_bytes(k_ptr, 0, n_elems); }
            }
            if !v_ptr.is_null() {
                unsafe { ptr::write_bytes(v_ptr, 0, n_elems); }
            }
        }
    }

    // ── Accessors ────────────────────────────────────────────────

    #[inline]
    pub fn n_layers(&self) -> u32 { self.n_layers }

    #[inline]
    pub fn n_kv_heads(&self) -> u32 { self.n_kv_heads }

    #[inline]
    pub fn head_dim(&self) -> u32 { self.head_dim }

    #[inline]
    pub fn max_seq_len(&self) -> u32 { self.max_seq_len }

    #[inline]
    pub fn head_stride(&self) -> u32 { self.head_stride }

    #[inline]
    pub fn seq_len(&self) -> u32 { self.seq_len }

    pub fn set_seq_len(&mut self, seq_len: u32) {
        self.seq_len = seq_len.min(self.max_seq_len);
    }
}

// ══════════════════════════════════════════════════════════════════
// C API (ctypes FFI)
// ══════════════════════════════════════════════════════════════════

use std::ffi::c_int;
use std::os::raw::c_void;

pub type LKOKVArena = KVArena;

/// Create a new KV Arena. Returns opaque pointer.
/// Parameters must match the model config.
#[no_mangle]
pub extern "C" fn lko_kv_arena_create(
    n_layers: c_int,
    n_kv_heads: c_int,
    head_dim: c_int,
    max_seq_len: c_int,
) -> *mut LKOKVArena {
    Box::into_raw(Box::new(KVArena::new(
        n_layers as u32,
        n_kv_heads as u32,
        head_dim as u32,
        max_seq_len as u32,
    )))
}

/// Destroy a KV Arena.
#[no_mangle]
pub extern "C" fn lko_kv_arena_destroy(arena: *mut LKOKVArena) {
    if !arena.is_null() {
        unsafe { drop(Box::from_raw(arena)); }
    }
}

/// Register a K buffer pointer for a layer.
#[no_mangle]
pub extern "C" fn lko_kv_arena_register_k(
    arena: *mut LKOKVArena,
    layer_idx: c_int,
    ptr: u64,
) {
    if arena.is_null() { return; }
    unsafe { (*arena).register_k(layer_idx as u32, ptr); }
}

/// Register a V buffer pointer for a layer.
#[no_mangle]
pub extern "C" fn lko_kv_arena_register_v(
    arena: *mut LKOKVArena,
    layer_idx: c_int,
    ptr: u64,
) {
    if arena.is_null() { return; }
    unsafe { (*arena).register_v(layer_idx as u32, ptr); }
}

/// Check if all buffers are registered.
#[no_mangle]
pub extern "C" fn lko_kv_arena_is_ready(arena: *mut LKOKVArena) -> c_int {
    if arena.is_null() { return 0; }
    unsafe { (*arena).is_ready() as c_int }
}

/// Write K/V at position for a single layer.
/// k_data: f16 buffer of n_kv_heads * head_dim elements.
/// v_data: same.
#[no_mangle]
pub extern "C" fn lko_kv_arena_write(
    arena: *mut LKOKVArena,
    layer_idx: c_int,
    position: c_int,
    k_data: *const c_void,
    v_data: *const c_void,
) {
    if arena.is_null() { return; }
    unsafe {
        (*arena).write_kv(
            layer_idx as u32,
            position as u32,
            k_data as *const u16,
            v_data as *const u16,
        );
    }
}

/// Get pointer to K slice [start, end).
/// out_ptr: receives the GPU buffer pointer at the slice start.
/// out_len: receives the number of f16 elements.
#[no_mangle]
pub extern "C" fn lko_kv_arena_get_k_slice(
    arena: *mut LKOKVArena,
    layer_idx: c_int,
    start: c_int,
    end: c_int,
    out_ptr: *mut u64,
    out_len: *mut u64,
) {
    if arena.is_null() { return; }
    let (ptr, len) = unsafe { (*arena).get_k_slice(layer_idx as u32, start as u32, end as u32) };
    if !out_ptr.is_null() {
        unsafe { *out_ptr = ptr as u64; }
    }
    if !out_len.is_null() {
        unsafe { *out_len = len; }
    }
}

/// Get pointer to V slice [start, end).
#[no_mangle]
pub extern "C" fn lko_kv_arena_get_v_slice(
    arena: *mut LKOKVArena,
    layer_idx: c_int,
    start: c_int,
    end: c_int,
    out_ptr: *mut u64,
    out_len: *mut u64,
) {
    if arena.is_null() { return; }
    let (ptr, len) = unsafe { (*arena).get_v_slice(layer_idx as u32, start as u32, end as u32) };
    if !out_ptr.is_null() {
        unsafe { *out_ptr = ptr as u64; }
    }
    if !out_len.is_null() {
        unsafe { *out_len = len; }
    }
}

/// Get full K buffer info for a layer.
/// out_ptr: GPU buffer pointer.
/// out_len: total f16 elements in the buffer.
#[no_mangle]
pub extern "C" fn lko_kv_arena_get_k_ptr(
    arena: *mut LKOKVArena,
    layer_idx: c_int,
    out_ptr: *mut u64,
    out_len: *mut u64,
) {
    if arena.is_null() { return; }
    let (ptr, len) = unsafe { (*arena).get_k_ptr(layer_idx as u32) };
    if !out_ptr.is_null() { unsafe { *out_ptr = ptr; } }
    if !out_len.is_null() { unsafe { *out_len = len; } }
}

/// Get full V buffer info for a layer.
#[no_mangle]
pub extern "C" fn lko_kv_arena_get_v_ptr(
    arena: *mut LKOKVArena,
    layer_idx: c_int,
    out_ptr: *mut u64,
    out_len: *mut u64,
) {
    if arena.is_null() { return; }
    let (ptr, len) = unsafe { (*arena).get_v_ptr(layer_idx as u32) };
    if !out_ptr.is_null() { unsafe { *out_ptr = ptr; } }
    if !out_len.is_null() { unsafe { *out_len = len; } }
}

/// Copy KV from `src_arena` into `dst_arena` for `layer_idx`,
/// copying `up_to_position` positions.
#[no_mangle]
pub extern "C" fn lko_kv_arena_copy_layer(
    dst_arena: *mut LKOKVArena,
    src_arena: *mut LKOKVArena,
    layer_idx: c_int,
    up_to_position: c_int,
) {
    if dst_arena.is_null() || src_arena.is_null() { return; }
    let dst = unsafe { &*dst_arena };
    let src = unsafe { &*src_arena };
    dst.copy_from(src, layer_idx as u32, up_to_position as u32);
}

/// Copy all layers from src to dst.
#[no_mangle]
pub extern "C" fn lko_kv_arena_copy_all(
    dst_arena: *mut LKOKVArena,
    src_arena: *mut LKOKVArena,
    up_to_position: c_int,
) {
    if dst_arena.is_null() || src_arena.is_null() { return; }
    let dst = unsafe { &*dst_arena };
    let src = unsafe { &*src_arena };
    dst.copy_all_from(src, up_to_position as u32);
}

/// Set sequence length.
#[no_mangle]
pub extern "C" fn lko_kv_arena_set_seq_len(
    arena: *mut LKOKVArena,
    seq_len: c_int,
) {
    if arena.is_null() { return; }
    unsafe { (*arena).set_seq_len(seq_len as u32); }
}

/// Get sequence length.
#[no_mangle]
pub extern "C" fn lko_kv_arena_get_seq_len(arena: *mut LKOKVArena) -> c_int {
    if arena.is_null() { return 0; }
    unsafe { (*arena).seq_len() as c_int }
}

/// Zero the active region (up to seq_len) of all buffers.
#[no_mangle]
pub extern "C" fn lko_kv_arena_zero_active(arena: *mut LKOKVArena) {
    if arena.is_null() { return; }
    unsafe { (*arena).zero_active(); }
}

/// Zero all buffers entirely.
#[no_mangle]
pub extern "C" fn lko_kv_arena_zero_all(arena: *mut LKOKVArena) {
    if arena.is_null() { return; }
    unsafe { (*arena).zero_all(); }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_arena() -> KVArena {
        // TinyLlama-like: 22 layers, 4 kv_heads, 64 head_dim, 256 max_seq
        KVArena::new(2, 4, 64, 256)
    }

    fn make_buffer(n_elems: usize) -> Vec<u16> {
        vec![0u16; n_elems]
    }

    #[test]
    fn test_create() {
        let arena = KVArena::new(22, 4, 64, 2048);
        assert_eq!(arena.n_layers(), 22);
        assert_eq!(arena.n_kv_heads(), 4);
        assert_eq!(arena.head_dim(), 64);
        assert_eq!(arena.max_seq_len(), 2048);
        assert_eq!(arena.head_stride(), 64); // head_dim
        assert_eq!(arena.seq_len(), 0);
    }

    #[test]
    fn test_register_and_write() {
        let n_layers = 2u32;
        let (n_kv, max_s, hd) = (4usize, 256usize, 64usize);
        let total = n_kv * max_s * hd;
        let mut arena = KVArena::new(n_layers, n_kv as u32, hd as u32, max_s as u32);

        let mut k0 = make_buffer(total);
        let mut v0 = make_buffer(total);
        let mut k1 = make_buffer(total);
        let mut v1 = make_buffer(total);

        arena.register_layer(0, k0.as_mut_ptr() as u64, v0.as_mut_ptr() as u64);
        arena.register_layer(1, k1.as_mut_ptr() as u64, v1.as_mut_ptr() as u64);
        assert!(arena.is_ready());

        // Create test K/V data: n_kv * hd = 256 elements
        let k_data: Vec<u16> = (0..(n_kv * hd) as u16).collect();
        let v_data: Vec<u16> = (100..100 + (n_kv * hd) as u16).collect();

        // Write at position 5
        arena.write_kv(0, 5, k_data.as_ptr(), v_data.as_ptr());

        // Verify: per-head data at correct MLX offsets
        // Element [h, p, d] at offset: h * max_s * hd + p * hd + d
        let max_s = 256usize;
        let hd = 64usize;
        for h in 0..4 {
            let base = h * max_s * hd + 5 * hd;
            for i in 0..hd {
                assert_eq!(k0[base + i], k_data[h * hd + i], "K mismatch h={} i={}", h, i);
                assert_eq!(v0[base + i], v_data[h * hd + i], "V mismatch h={} i={}", h, i);
            }
        }

        // Position 0 should still be zero
        assert_eq!(k0[0], 0);
        assert_eq!(v0[0], 0);
    }

    #[test]
    fn test_write_out_of_bounds() {
        let (n_kv, max_s, hd) = (4usize, 100usize, 64usize);
        let total = n_kv * max_s * hd;
        let mut arena = KVArena::new(1, n_kv as u32, hd as u32, max_s as u32);
        let mut k = make_buffer(total);
        let mut v = make_buffer(total);
        arena.register_layer(0, k.as_mut_ptr() as u64, v.as_mut_ptr() as u64);

        let k_data = make_buffer(256);
        let v_data = make_buffer(256);

        // Write at max position — should be fine
        arena.write_kv(0, 99, k_data.as_ptr(), v_data.as_ptr());

        // Write beyond max — should silently skip
        arena.write_kv(0, 100, k_data.as_ptr(), v_data.as_ptr());
        arena.write_kv(0, 200, k_data.as_ptr(), v_data.as_ptr());
    }

    #[test]
    fn test_write_and_read_slice() {
        let (n_kv, max_s, hd) = (4usize, 256usize, 64usize);
        let total = n_kv * max_s * hd;
        let mut arena = KVArena::new(1, n_kv as u32, hd as u32, max_s as u32);
        let mut k = make_buffer(total);
        let mut v = make_buffer(total);
        arena.register_layer(0, k.as_mut_ptr() as u64, v.as_mut_ptr() as u64);

        // Write test pattern at positions 0..5: all heads filled with pos value
        for pos in 0..5u32 {
            let k_pat: Vec<u16> = vec![pos as u16; n_kv * hd];
            let v_pat: Vec<u16> = vec![(pos + 100) as u16; n_kv * hd];
            arena.write_kv(0, pos, k_pat.as_ptr(), v_pat.as_ptr());
        }

        // get_k_slice returns pointer to head 0 at `start`
        // Head 0's positions are contiguous (offset = pos * head_dim)
        let (k_ptr, k_len) = arena.get_k_slice(0, 1, 4);
        assert_eq!(k_len, 3 * 64); // 3 positions × 64 head_dim (head 0 only)
        let k_slice = unsafe { std::slice::from_raw_parts(k_ptr, k_len as usize) };
        assert_eq!(k_slice[0], 1);
        assert_eq!(k_slice[64], 2);
        assert_eq!(k_slice[128], 3);
    }

    #[test]
    fn test_copy_between_arenas() {
        let n_layers = 2u32;
        let (n_kv, max_s, hd) = (4usize, 256usize, 64usize);
        let total = n_kv * max_s * hd;
        let per_pos = n_kv * hd;
        let mut arena_a = KVArena::new(n_layers, n_kv as u32, hd as u32, max_s as u32);
        let mut arena_b = KVArena::new(n_layers, n_kv as u32, hd as u32, max_s as u32);

        let mut ka0 = make_buffer(total);
        let mut va0 = make_buffer(total);
        let mut ka1 = make_buffer(total);
        let mut va1 = make_buffer(total);
        let mut kb0 = make_buffer(total);
        let mut vb0 = make_buffer(total);
        let mut kb1 = make_buffer(total);
        let mut vb1 = make_buffer(total);

        arena_a.register_layer(0, ka0.as_mut_ptr() as u64, va0.as_mut_ptr() as u64);
        arena_a.register_layer(1, ka1.as_mut_ptr() as u64, va1.as_mut_ptr() as u64);
        arena_b.register_layer(0, kb0.as_mut_ptr() as u64, vb0.as_mut_ptr() as u64);
        arena_b.register_layer(1, kb1.as_mut_ptr() as u64, vb1.as_mut_ptr() as u64);

        for pos in 0..3u32 {
            let k_pat: Vec<u16> = vec![pos as u16 + 42; per_pos];
            let v_pat: Vec<u16> = vec![pos as u16 + 142; per_pos];
            arena_a.write_kv(0, pos, k_pat.as_ptr(), v_pat.as_ptr());
            arena_a.write_kv(1, pos, k_pat.as_ptr(), v_pat.as_ptr());
        }

        arena_b.copy_all_from(&arena_a, 3);

        // Verify: per-head comparison at each position
        let a_k: [&[u16]; 2] = [&ka0, &ka1];
        let b_k: [&[u16]; 2] = [&kb0, &kb1];
        for layer in 0..2usize {
            for h in 0..n_kv {
                let h_base = h * max_s * hd;
                for pos in 0..3usize {
                    let off = h_base + pos * hd;
                    assert_eq!(&a_k[layer][off..off + hd],
                               &b_k[layer][off..off + hd],
                               "K mismatch layer={} pos={} h={}", layer, pos, h);
                }
            }
        }
    }

    #[test]
    fn test_zero() {
        let (n_kv, max_s, hd) = (4usize, 256usize, 64usize);
        let total = n_kv * max_s * hd;
        let mut arena = KVArena::new(1, n_kv as u32, hd as u32, max_s as u32);
        let mut k = make_buffer(total);

        // Fill with non-zero
        for i in 0..k.len() {
            k[i] = 0xFFFF;
        }

        arena.register_layer(0, k.as_mut_ptr() as u64, k.as_mut_ptr() as u64);
        arena.set_seq_len(10);
        arena.zero_active();

        // First 10 positions across all heads should be zero
        for h in 0..n_kv {
            let h_base = h * max_s * hd;
            for i in 0..10 * hd {
                assert_eq!(k[h_base + i], 0, "k[h={}][i={}] should be 0", h, i);
            }
        }
        // Position 10, head 0 should still be 0xFFFF (beyond zeroed region)
        assert_eq!(k[10 * hd], 0xFFFF);
    }

    #[test]
    fn test_unregistered_skip() {
        // Write to unregistered layer should not crash
        let arena = KVArena::new(2, 4, 64, 256);
        let k_data = make_buffer(256);
        let v_data = make_buffer(256);
        arena.write_kv(0, 0, k_data.as_ptr(), v_data.as_ptr());
        // No panic
    }

    #[test]
    fn test_c_api_flow() {
        let (n_kv, max_s, hd) = (4usize, 256usize, 64usize);
        let total = n_kv * max_s * hd;
        let arena = lko_kv_arena_create(2, n_kv as i32, hd as i32, max_s as i32);
        assert!(!arena.is_null());

        let mut k = make_buffer(total);
        let mut v = make_buffer(total);

        lko_kv_arena_register_k(arena, 0, k.as_mut_ptr() as u64);
        lko_kv_arena_register_v(arena, 0, v.as_mut_ptr() as u64);
        lko_kv_arena_register_k(arena, 1, k.as_mut_ptr() as u64);
        lko_kv_arena_register_v(arena, 1, v.as_mut_ptr() as u64);

        assert_eq!(lko_kv_arena_is_ready(arena), 1);

        let data = make_buffer(n_kv * hd);
        lko_kv_arena_write(arena, 0, 3, data.as_ptr() as *const c_void, data.as_ptr() as *const c_void);

        lko_kv_arena_set_seq_len(arena, 4);
        assert_eq!(lko_kv_arena_get_seq_len(arena), 4);

        let mut out_ptr: u64 = 0;
        let mut out_len: u64 = 0;
        lko_kv_arena_get_k_slice(arena, 0, 0, 4, &mut out_ptr, &mut out_len);
        assert!(out_ptr != 0);
        assert_eq!(out_len, 4 * 64); // 4 positions × head_dim (head 0 only)

        lko_kv_arena_destroy(arena);
    }
}
