//! Metal GPU dispatch via ObjC wrapper (metal_wrapper.m).
//!
//! The C wrapper handles all ObjC/Metal API calls.
//! Rust calls simple C functions via FFI.

use std::ffi::c_void;
use std::path::Path;

type MetalGpu = *mut c_void;

extern "C" {
    fn metal_init(metallib_path: *const i8) -> MetalGpu;
    fn metal_destroy(gpu: MetalGpu);
    fn metal_expert_gemv(
        gpu: MetalGpu,
        q4_data: *const u8, q4_len: u64,
        x: *const f32, k: u64,
        y: *mut f32, m: u64,
        num_blocks: u64,
    ) -> i32;
    fn metal_fp16_gemv(
        gpu: MetalGpu,
        w: *const u16, w_len: u64,
        x: *const f32, k: u64,
        y: *mut f32, m: u64,
    ) -> i32;
}

pub struct MetalExpertGpu {
    handle: MetalGpu,
}

// Metal is thread-safe on Apple GPUs
unsafe impl Send for MetalExpertGpu {}
unsafe impl Sync for MetalExpertGpu {}

impl MetalExpertGpu {
    pub fn new(metallib_path: &Path) -> Option<Self> {
        let path_cstr = std::ffi::CString::new(metallib_path.to_str()?).ok()?;
        let handle = unsafe { metal_init(path_cstr.as_ptr()) };
        if handle.is_null() { None } else { Some(MetalExpertGpu { handle }) }
    }

    pub fn dispatch_fp16_gemv(&self, w: &[u16], x: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        unsafe {
            metal_fp16_gemv(self.handle, w.as_ptr(), (w.len() * 2) as u64, x.as_ptr(), k as u64, y.as_mut_ptr(), m as u64);
        }
        y
    }

    pub fn dispatch_expert(
        &self, q4_data: &[u8], x: &[f32],
        m: usize, k: usize, num_blocks: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        unsafe {
            metal_expert_gemv(
                self.handle,
                q4_data.as_ptr(), q4_data.len() as u64,
                x.as_ptr(), k as u64,
                y.as_mut_ptr(), m as u64,
                num_blocks as u64,
            );
        }
        y
    }
}

impl Drop for MetalExpertGpu {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { metal_destroy(self.handle); }
        }
    }
}

// ── Global instance for C API ─────────────────────────────────────────────

use std::sync::Mutex;
static METAL_GPU: Mutex<Option<MetalExpertGpu>> = Mutex::new(None);

#[no_mangle]
pub extern "C" fn lko_metal_init(metallib_path: *const i8) -> i32 {
    use std::ffi::CStr;
    let path = unsafe { CStr::from_ptr(metallib_path) }.to_string_lossy();
    let mut gpu = METAL_GPU.lock().unwrap();
    if gpu.is_some() { return 1; }
    match MetalExpertGpu::new(std::path::Path::new(path.as_ref())) {
        Some(instance) => { *gpu = Some(instance); 1 }
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn lko_metal_expert_gemv(
    q4_data: *const u8, q4_len: i32,
    x: *const f32, k: i32,
    y: *mut f32, m: i32,
    num_blocks: i32,
) -> i32 {
    let gpu_guard = METAL_GPU.lock().unwrap();
    let gpu = match gpu_guard.as_ref() {
        Some(g) => g,
        None => return -1,
    };
    let q4 = unsafe { std::slice::from_raw_parts(q4_data, q4_len as usize) };
    let inp = unsafe { std::slice::from_raw_parts(x, k as usize) };
    let result = gpu.dispatch_expert(q4, inp, m as usize, k as usize, num_blocks as usize);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), y, m as usize); }
    m
}

extern "C" {
    fn metal_multi_expert_gemv(
        gpu: MetalGpu,
        all_q4: *const u8, q4_len: u64,
        expert_offsets: *const u32, n_offsets: u64,
        x: *const f32, k: u64,
        y: *mut f32,
        output_offsets: *const u32,
        n_experts: u32,
        total_output_elems: u64,
    ) -> i32;
    fn metal_fused_gqa(
        gpu: MetalGpu,
        w_qkv: *const f32, w_qkv_bytes: u64,
        h: *const f32, rope_cos: *const f32, rope_sin: *const f32,
        pos: u32, seq_len: u32, max_seq: u32,
        k_cache: *mut f32, v_cache: *mut f32, kv_bytes: u64,
        attn_out: *mut f32,
    ) -> i32;
}

impl MetalExpertGpu {
    pub fn dispatch_multi_expert(
        &self,
        all_q4: &[u8],
        expert_offsets: &[u32],  // [M, K, n_blocks, q4_off] × n_experts
        x: &[f32],
        output_offsets: &[u32],  // per-expert output start
        n_experts: u32,
        total_elems: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; total_elems];
        unsafe {
            metal_multi_expert_gemv(
                self.handle,
                all_q4.as_ptr(), all_q4.len() as u64,
                expert_offsets.as_ptr(), expert_offsets.len() as u64,
                x.as_ptr(), x.len() as u64,
                y.as_mut_ptr(),
                output_offsets.as_ptr(),
                n_experts,
                total_elems as u64,
            );
        }
        y
    }

    pub fn dispatch_fused_gqa(&self,
        w_qkv: &[f32], h: &[f32],
        rope_cos: &[f32], rope_sin: &[f32],
        pos: u32, seq_len: u32, max_seq: u32,
        k_cache: &mut [f32], v_cache: &mut [f32],
    ) -> Vec<f32> {
        let mut attn_out = vec![0.0f32; 4096];
        unsafe {
            metal_fused_gqa(self.handle,
                w_qkv.as_ptr(), (w_qkv.len() * 4) as u64,
                h.as_ptr(), rope_cos.as_ptr(), rope_sin.as_ptr(),
                pos, seq_len, max_seq,
                k_cache.as_mut_ptr(), v_cache.as_mut_ptr(), (k_cache.len() * 4) as u64,
                attn_out.as_mut_ptr(),
            );
        }
        attn_out
    }
}

#[no_mangle]
pub extern "C" fn lko_metal_fp16_gemv(
    w: *const u16, w_bytes: i32,
    x: *const f32, k: i32,
    y: *mut f32, m: i32,
) -> i32 {
    let gpu_guard = METAL_GPU.lock().unwrap();
    let gpu = match gpu_guard.as_ref() {
        Some(g) => g,
        None => return -1,
    };
    let result = gpu.dispatch_fp16_gemv(
        unsafe { std::slice::from_raw_parts(w, (w_bytes / 2) as usize) },
        unsafe { std::slice::from_raw_parts(x, k as usize) },
        m as usize, k as usize,
    );
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), y, m as usize); }
    m
}

#[no_mangle]
pub extern "C" fn lko_metal_fused_gqa(
    w_qkv: *const f32, w_qkv_bytes: i32,
    h: *const f32,
    rope_cos: *const f32, rope_sin: *const f32,
    pos: i32, seq_len: i32, max_seq: i32,
    k_cache: *mut f32, v_cache: *mut f32, kv_bytes: i32,
    attn_out: *mut f32,
) -> i32 {
    let gpu_guard = METAL_GPU.lock().unwrap();
    let gpu = match gpu_guard.as_ref() {
        Some(g) => g,
        None => return -1,
    };
    let w = unsafe { std::slice::from_raw_parts(w_qkv, (w_qkv_bytes/4) as usize) };
    let h_slice = unsafe { std::slice::from_raw_parts(h, 2048) };
    let cos = unsafe { std::slice::from_raw_parts(rope_cos, (max_seq * 128) as usize) };
    let sin = unsafe { std::slice::from_raw_parts(rope_sin, (max_seq * 128) as usize) };
    let kc = unsafe { std::slice::from_raw_parts_mut(k_cache, (kv_bytes/4) as usize) };
    let vc = unsafe { std::slice::from_raw_parts_mut(v_cache, (kv_bytes/4) as usize) };

    let result = gpu.dispatch_fused_gqa(
        w, h_slice, cos, sin,
        pos as u32, seq_len as u32, max_seq as u32,
        kc, vc,
    );
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), attn_out, 4096); }
    4096
}

#[no_mangle]
pub extern "C" fn lko_metal_multi_expert(
    all_q4: *const u8, q4_len: i32,
    expert_offsets: *const u32, n_offsets: i32,
    x: *const f32, k: i32,
    y: *mut f32,
    output_offsets: *const u32,
    n_experts: i32,
    total_elems: i32,
) -> i32 {
    let gpu_guard = METAL_GPU.lock().unwrap();
    let gpu = match gpu_guard.as_ref() {
        Some(g) => g,
        None => return -1,
    };
    let q4 = unsafe { std::slice::from_raw_parts(all_q4, q4_len as usize) };
    let off = unsafe { std::slice::from_raw_parts(expert_offsets, n_offsets as usize) };
    let inp = unsafe { std::slice::from_raw_parts(x, k as usize) };
    let out_off = unsafe { std::slice::from_raw_parts(output_offsets, n_experts as usize) };
    let result = gpu.dispatch_multi_expert(
        q4, off, inp, out_off,
        n_experts as u32, total_elems as usize,
    );
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), y, total_elems as usize); }
    total_elems
}
