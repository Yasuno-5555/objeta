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
        q4_data: *const u8,
        q4_len: u64,
        x: *const f32,
        k: u64,
        y: *mut f32,
        m: u64,
        num_blocks: u64,
    ) -> i32;
    fn metal_fp16_gemv(
        gpu: MetalGpu,
        w: *const u16,
        w_len: u64,
        x: *const f32,
        k: u64,
        y: *mut f32,
        m: u64,
    ) -> i32;
    // Fused GQA
    fn metal_gqa_init(
        gpu: MetalGpu,
        rope_cos: *const f32,
        rope_sin: *const f32,
        max_seq: u32,
    ) -> i32;
    fn metal_gqa_load_weights(
        gpu: MetalGpu,
        layer_idx: u32,
        w_qkv: *const u16,
        w_qkv_bytes: u64,
        w_o: *const u16,
        w_o_bytes: u64,
        q_norm: *const f32,
        q_norm_bytes: u64,
        k_norm: *const f32,
        k_norm_bytes: u64,
    ) -> i32;
    fn metal_fused_gqa(
        gpu: MetalGpu,
        layer_idx: u32,
        h: *const f32,
        pos: u32,
        seq_len: u32,
        max_seq: u32,
        k_cache: *mut f32,
        v_cache: *mut f32,
        kv_bytes: u64,
        attn_out: *mut f32,
    ) -> i32;
    fn metal_gqa_oproj(
        gpu: MetalGpu,
        layer_idx: u32,
        w_o: *const u16,
        w_o_bytes: u64,
        attn_out: *const f32,
        output: *mut f32,
        m: u32,
        k: u32,
    ) -> i32;
    fn metal_multi_expert_gemv(
        gpu: MetalGpu,
        all_q4: *const u8,
        q4_len: u64,
        expert_offsets: *const u32,
        n_offsets: u64,
        x: *const f32,
        k: u64,
        y: *mut f32,
        output_offsets: *const u32,
        n_experts: u32,
        total_output_elems: u64,
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
        if handle.is_null() {
            None
        } else {
            Some(MetalExpertGpu { handle })
        }
    }

    pub fn dispatch_fp16_gemv(&self, w: &[u16], x: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        unsafe {
            metal_fp16_gemv(
                self.handle,
                w.as_ptr(),
                (w.len() * 2) as u64,
                x.as_ptr(),
                k as u64,
                y.as_mut_ptr(),
                m as u64,
            );
        }
        y
    }

    pub fn dispatch_expert(
        &self,
        q4_data: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
        num_blocks: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        unsafe {
            metal_expert_gemv(
                self.handle,
                q4_data.as_ptr(),
                q4_data.len() as u64,
                x.as_ptr(),
                k as u64,
                y.as_mut_ptr(),
                m as u64,
                num_blocks as u64,
            );
        }
        y
    }

    pub fn dispatch_multi_expert(
        &self,
        all_q4: &[u8],
        expert_offsets: &[u32],
        x: &[f32],
        output_offsets: &[u32],
        n_experts: u32,
        total_elems: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; total_elems];
        unsafe {
            metal_multi_expert_gemv(
                self.handle,
                all_q4.as_ptr(),
                all_q4.len() as u64,
                expert_offsets.as_ptr(),
                expert_offsets.len() as u64,
                x.as_ptr(),
                x.len() as u64,
                y.as_mut_ptr(),
                output_offsets.as_ptr(),
                n_experts,
                total_elems as u64,
            );
        }
        y
    }

    // ── Fused GQA ──────────────────────────────────────────────────────

    /// Initialize GQA persistent resources (RoPE tables). Call once.
    pub fn gqa_init(&self, rope_cos: &[f32], rope_sin: &[f32], max_seq: u32) -> i32 {
        unsafe { metal_gqa_init(self.handle, rope_cos.as_ptr(), rope_sin.as_ptr(), max_seq) }
    }

    /// Load per-layer GQA weights into persistent Metal buffers. Call before each GQA layer.
    pub fn gqa_load_weights(
        &self,
        layer_idx: usize,
        w_qkv: &[u16],
        w_o: &[u16],
        q_norm: &[f32],
        k_norm: &[f32],
    ) -> i32 {
        unsafe {
            metal_gqa_load_weights(
                self.handle,
                layer_idx as u32,
                w_qkv.as_ptr(),
                (w_qkv.len() * 2) as u64,
                w_o.as_ptr(),
                (w_o.len() * 2) as u64,
                q_norm.as_ptr(),
                (q_norm.len() * 4) as u64,
                k_norm.as_ptr(),
                (k_norm.len() * 4) as u64,
            )
        }
    }

    /// Dispatch fused GQA: QKV + RoPE + online softmax + V sum + Q-gate.
    /// Returns attn_out (4096 f32).
    pub fn dispatch_fused_gqa(
        &self,
        layer_idx: usize,
        h: &[f32],
        pos: u32,
        seq_len: u32,
        max_seq: u32,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
    ) -> Vec<f32> {
        let mut attn_out = vec![0.0f32; 4096];
        unsafe {
            metal_fused_gqa(
                self.handle,
                layer_idx as u32,
                h.as_ptr(),
                pos,
                seq_len,
                max_seq,
                k_cache.as_mut_ptr(),
                v_cache.as_mut_ptr(),
                (k_cache.len() * 4) as u64,
                attn_out.as_mut_ptr(),
            );
        }
        attn_out
    }

    /// Dispatch GQA O-proj: output = W_o @ attn_out.
    /// W_o is (m, k) = (2048, 4096) f16. attn_out is (4096,) f32.
    pub fn dispatch_gqa_oproj(
        &self,
        layer_idx: usize,
        w_o: &[u16],
        attn_out: &[f32],
        m: u32,
        k: u32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m as usize];
        unsafe {
            metal_gqa_oproj(
                self.handle,
                layer_idx as u32,
                w_o.as_ptr(),
                (w_o.len() * 2) as u64,
                attn_out.as_ptr(),
                output.as_mut_ptr(),
                m,
                k,
            );
        }
        output
    }
}

impl Drop for MetalExpertGpu {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                metal_destroy(self.handle);
            }
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
    if gpu.is_some() {
        return 1;
    }
    match MetalExpertGpu::new(std::path::Path::new(path.as_ref())) {
        Some(instance) => {
            *gpu = Some(instance);
            1
        }
        None => 0,
    }
}

// Convenience: access the global GPU instance.
pub fn with_gpu<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&MetalExpertGpu) -> R,
{
    let guard = METAL_GPU.lock().unwrap();
    guard.as_ref().map(f)
}

#[no_mangle]
pub extern "C" fn lko_metal_expert_gemv(
    q4_data: *const u8,
    q4_len: i32,
    x: *const f32,
    k: i32,
    y: *mut f32,
    m: i32,
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
    unsafe {
        std::ptr::copy_nonoverlapping(result.as_ptr(), y, m as usize);
    }
    m
}

#[no_mangle]
pub extern "C" fn lko_metal_fp16_gemv(
    w: *const u16,
    w_bytes: i32,
    x: *const f32,
    k: i32,
    y: *mut f32,
    m: i32,
) -> i32 {
    let gpu_guard = METAL_GPU.lock().unwrap();
    let gpu = match gpu_guard.as_ref() {
        Some(g) => g,
        None => return -1,
    };
    let result = gpu.dispatch_fp16_gemv(
        unsafe { std::slice::from_raw_parts(w, (w_bytes / 2) as usize) },
        unsafe { std::slice::from_raw_parts(x, k as usize) },
        m as usize,
        k as usize,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(result.as_ptr(), y, m as usize);
    }
    m
}

// ── Fused GQA C API ──────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn lko_metal_gqa_init(
    rope_cos: *const f32,
    rope_sin: *const f32,
    max_seq: i32,
) -> i32 {
    let gpu_guard = METAL_GPU.lock().unwrap();
    let gpu = match gpu_guard.as_ref() {
        Some(g) => g,
        None => return -1,
    };
    let cos = unsafe { std::slice::from_raw_parts(rope_cos, (max_seq * 32) as usize) };
    let sin = unsafe { std::slice::from_raw_parts(rope_sin, (max_seq * 32) as usize) };
    gpu.gqa_init(cos, sin, max_seq as u32)
}

#[no_mangle]
pub extern "C" fn lko_metal_gqa_load_weights(
    layer_idx: i32,
    w_qkv: *const u16,
    w_qkv_bytes: i32,
    w_o: *const u16,
    w_o_bytes: i32,
    q_norm: *const f32,
    q_norm_len: i32,
    k_norm: *const f32,
    k_norm_len: i32,
) -> i32 {
    let gpu_guard = METAL_GPU.lock().unwrap();
    let gpu = match gpu_guard.as_ref() {
        Some(g) => g,
        None => return -1,
    };
    let qkv = if w_qkv.is_null() || w_qkv_bytes <= 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(w_qkv, (w_qkv_bytes / 2) as usize) }
    };
    let o = if w_o.is_null() || w_o_bytes <= 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(w_o, (w_o_bytes / 2) as usize) }
    };
    let qn = if q_norm.is_null() || q_norm_len <= 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(q_norm, q_norm_len as usize) }
    };
    let kn = if k_norm.is_null() || k_norm_len <= 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(k_norm, k_norm_len as usize) }
    };
    gpu.gqa_load_weights(layer_idx as usize, qkv, o, qn, kn)
}

#[no_mangle]
pub extern "C" fn lko_metal_fused_gqa(
    layer_idx: i32,
    h: *const f32,
    pos: i32,
    seq_len: i32,
    max_seq: i32,
    k_cache: *mut f32,
    v_cache: *mut f32,
    kv_bytes: i32,
    attn_out: *mut f32,
) -> i32 {
    let gpu_guard = METAL_GPU.lock().unwrap();
    let gpu = match gpu_guard.as_ref() {
        Some(g) => g,
        None => return -1,
    };
    let h_slice = unsafe { std::slice::from_raw_parts(h, 2048) };
    let kc = unsafe { std::slice::from_raw_parts_mut(k_cache, (kv_bytes / 4) as usize) };
    let vc = unsafe { std::slice::from_raw_parts_mut(v_cache, (kv_bytes / 4) as usize) };
    let result = gpu.dispatch_fused_gqa(
        layer_idx as usize,
        h_slice,
        pos as u32,
        seq_len as u32,
        max_seq as u32,
        kc,
        vc,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(result.as_ptr(), attn_out, 4096);
    }
    4096
}

#[no_mangle]
pub extern "C" fn lko_metal_gqa_oproj(
    layer_idx: i32,
    w_o: *const u16,
    w_o_bytes: i32,
    attn_out: *const f32,
    output: *mut f32,
    m: i32,
    k: i32,
) -> i32 {
    let gpu_guard = METAL_GPU.lock().unwrap();
    let gpu = match gpu_guard.as_ref() {
        Some(g) => g,
        None => return -1,
    };
    let wo = unsafe { std::slice::from_raw_parts(w_o, (w_o_bytes / 2) as usize) };
    let attn = unsafe { std::slice::from_raw_parts(attn_out, k as usize) };
    let result = gpu.dispatch_gqa_oproj(layer_idx as usize, wo, attn, m as u32, k as u32);
    unsafe {
        std::ptr::copy_nonoverlapping(result.as_ptr(), output, m as usize);
    }
    m
}

#[no_mangle]
pub extern "C" fn lko_metal_multi_expert(
    all_q4: *const u8,
    q4_len: i32,
    expert_offsets: *const u32,
    n_offsets: i32,
    x: *const f32,
    k: i32,
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
        q4,
        off,
        inp,
        out_off,
        n_experts as u32,
        total_elems as usize,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(result.as_ptr(), y, total_elems as usize);
    }
    total_elems
}
