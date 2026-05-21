#![recursion_limit = "512"]

//! objeta Qwen3.6 executor — Rust SIMD + Metal GPU dispatch.

pub mod attention;
pub mod expert_cache;
pub mod metal_dispatch;
pub mod os_telemetry;
pub mod qwen36_forward;
pub mod runtime_pack;
pub mod runtime_governor;
pub mod runtime_profile;
pub mod runtime_tuner;
pub mod strategy;
pub mod moe_stats;
pub mod runner_governor;
pub mod runner_residency;
pub mod qwen36_ffi;

pub mod expert_store;
pub mod kv_arena;
pub mod moe_dispatch;
mod quantize;
pub mod speculative;

pub use expert_store::ExpertStore;
pub use kv_arena::KVArena;

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_char;

// ── Error handling ──────────────────────────────────────────────

#[derive(Debug)]
pub enum Error {
    BufferNotFound(String),
    InvalidConfig,
}

// ── Weight buffer registry ──────────────────────────────────────

/// A registered weight buffer — pointer + size on the GPU.
#[derive(Clone, Debug)]
pub struct WeightBuffer {
    pub ptr: u64, // GPU device pointer
    pub size_bytes: u64,
}

// ── Layer config (from model config) ─────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LayerConfig {
    pub hidden_dim: i32,
    pub ffn_dim: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub norm_eps: f32,
}

// ── Executor ─────────────────────────────────────────────────────

pub struct Executor {
    /// Registered weight buffers: name → (ptr, size)
    buffers: HashMap<String, WeightBuffer>,
    /// Layer config cache
    config: Option<LayerConfig>,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            config: None,
        }
    }

    /// Register a weight buffer (pointer from MLX array).
    pub fn register_buffer(&mut self, name: &str, ptr: u64, size_bytes: u64) {
        self.buffers
            .insert(name.to_string(), WeightBuffer { ptr, size_bytes });
    }

    /// Set layer config.
    pub fn set_config(&mut self, config: LayerConfig) {
        self.config = Some(config);
    }

    /// Get a registered buffer by name.
    pub fn get_buffer(&self, name: &str) -> Result<WeightBuffer, Error> {
        self.buffers
            .get(name)
            .cloned()
            .ok_or_else(|| Error::BufferNotFound(format!("Buffer '{}' not registered", name)))
    }
}

// ══════════════════════════════════════════════════════════════════
// C API (consumed by Python via ctypes)
// ══════════════════════════════════════════════════════════════════

pub type LKOExecutor = Executor;

/// Create a new executor instance. Returns opaque pointer.
#[no_mangle]
pub extern "C" fn lko_executor_create() -> *mut LKOExecutor {
    Box::into_raw(Box::new(Executor::new()))
}

/// Destroy an executor instance.
#[no_mangle]
pub extern "C" fn lko_executor_destroy(exec: *mut LKOExecutor) {
    if !exec.is_null() {
        unsafe {
            drop(Box::from_raw(exec));
        }
    }
}

/// Register a weight buffer.
/// `name`: null-terminated string.
/// `ptr`: GPU device pointer (from MLX).
/// `size_bytes`: buffer size.
#[no_mangle]
pub extern "C" fn lko_executor_register_buffer(
    exec: *mut LKOExecutor,
    name: *const c_char,
    ptr: u64,
    size_bytes: u64,
) -> i32 {
    if exec.is_null() || name.is_null() {
        return -1;
    }
    let exec = unsafe { &mut *exec };
    let name_str = unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("");
    exec.register_buffer(name_str, ptr, size_bytes);
    0
}

/// Set layer configuration.
#[no_mangle]
pub extern "C" fn lko_executor_set_config(
    exec: *mut LKOExecutor,
    config: *const LayerConfig,
) -> i32 {
    if exec.is_null() || config.is_null() {
        return -1;
    }
    let exec = unsafe { &mut *exec };
    exec.set_config(unsafe { *config });
    0
}

// ── Test ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_destroy() {
        let exec = Executor::new();
        assert!(exec.buffers.is_empty());
    }

    #[test]
    fn test_register_buffer() {
        let mut exec = Executor::new();
        exec.register_buffer("test", 0xDEADBEEF, 4096);
        let buf = exec.get_buffer("test").unwrap();
        assert_eq!(buf.ptr, 0xDEADBEEF);
        assert_eq!(buf.size_bytes, 4096);
    }

    #[test]
    fn test_c_api() {
        let exec = lko_executor_create();
        assert!(!exec.is_null());
        lko_executor_destroy(exec);
    }
}
