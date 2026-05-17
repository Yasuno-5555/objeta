//! Attention operations accelerated via Apple Accelerate BLAS.
//!
//! Uses cblas_sgemv for fp32 GEMV and manual fp16→fp32 conversion for weights.
//! All operations process single-token (decode) — batch ops not needed.

use std::os::raw::{c_float, c_int, c_char};

// ── Accelerate BLAS FFI ───────────────────────────────────────────────────

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemv(
        order: c_int,      // 101=RowMajor, 102=ColMajor
        trans: c_int,      // 111=NoTrans, 112=Trans
        m: c_int,          // rows of A
        n: c_int,          // cols of A
        alpha: *const c_float,
        a: *const c_float,
        lda: c_int,
        x: *const c_float,
        incx: c_int,
        beta: *const c_float,
        y: *mut c_float,
        incy: c_int,
    );
}

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;

/// y = alpha * W @ x + beta * y  where W is (M×K), x is (K,)
pub fn sgemv(W: &[f32], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; M];
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
            M as c_int, K as c_int,
            &alpha,
            W.as_ptr(), K as c_int,
            x.as_ptr(), 1,
            &beta,
            y.as_mut_ptr(), 1,
        );
    }
    y
}

/// y = W @ x where W is fp16 (M×K), x is fp32 (K,). Auto-converts W to fp32.
pub fn fp16_gemv(W_f16: &[u16], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    let W_f32: Vec<f32> = W_f16.iter().map(|&h| f16_to_f32(h)).collect();
    sgemv(&W_f32, x, M, K)
}

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as u32;
    if exp == 0 {
        if mantissa == 0 { f32::from_bits(sign) }
        else { (mantissa as f32) * 2f32.powi(1 - 15 - 10) * if sign == 0 { 1.0 } else { -1.0 } }
    } else if exp == 31 {
        if mantissa == 0 { f32::from_bits(sign | 0x7f80_0000) }
        else { f32::NAN }
    } else {
        f32::from_bits(sign | (((exp + 127 - 15) as u32) << 23) | (mantissa << 13))
    }
}

// ── RMSNorm ───────────────────────────────────────────────────────────────

pub fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms = (mean_sq + 1e-6).sqrt();
    x.iter().zip(weight.iter()).map(|(&v, &w)| (v / rms) * w).collect()
}

// ── Softmax ───────────────────────────────────────────────────────────────

pub fn softmax_inplace(x: &mut [f32], dim: usize) {
    for chunk in x.chunks_mut(dim) {
        let max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in chunk.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in chunk.iter_mut() { *v /= sum.max(1e-12); }
    }
}

// ── SiLU ──────────────────────────────────────────────────────────────────

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

// ── GQA Attention ─────────────────────────────────────────────────────────
// Q: (n_heads, head_dim), K_cache: (n_kv, seq_len, head_dim), V_cache: same
// Returns: (n_heads * head_dim,) output

pub fn gqa_attention(
    q: &[f32], k_cache: &[f32], v_cache: &[f32],
    n_heads: usize, n_kv: usize, head_dim: usize, seq_len: usize,
) -> Vec<f32> {
    let n_rep = n_heads / n_kv;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / n_rep;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];

        // Compute attention scores
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let mut dot = 0.0f32;
            let k_t = &k_cache[(kv_h * seq_len + t) * head_dim..(kv_h * seq_len + t + 1) * head_dim];
            for d in 0..head_dim { dot += q_h[d] * k_t[d]; }
            scores[t] = dot * scale;
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - max_s).exp(); sum += *s; }
        for s in scores.iter_mut() { *s /= sum.max(1e-12); }

        // Weighted sum
        let out_h = &mut output[h * head_dim..(h + 1) * head_dim];
        for t in 0..seq_len {
            let v_t = &v_cache[(kv_h * seq_len + t) * head_dim..(kv_h * seq_len + t + 1) * head_dim];
            let a = scores[t];
            for d in 0..head_dim { out_h[d] += a * v_t[d]; }
        }
    }
    output
}

// ── C API ─────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn lko_sgemv(
    w: *const c_float, m: c_int, k: c_int,
    x: *const c_float,
    y: *mut c_float,
) -> c_int {
    let W = unsafe { std::slice::from_raw_parts(w, (m * k) as usize) };
    let X = unsafe { std::slice::from_raw_parts(x, k as usize) };
    let result = sgemv(W, X, m as usize, k as usize);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), y, m as usize); }
    m
}

#[no_mangle]
pub extern "C" fn lko_fp16_gemv(
    w: *const u16, m: c_int, k: c_int,
    x: *const c_float,
    y: *mut c_float,
) -> c_int {
    let W = unsafe { std::slice::from_raw_parts(w, (m * k) as usize) };
    let X = unsafe { std::slice::from_raw_parts(x, k as usize) };
    let result = fp16_gemv(W, X, m as usize, k as usize);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), y, m as usize); }
    m
}

#[no_mangle]
pub extern "C" fn lko_rms_norm(
    x: *mut c_float, weight: *const c_float, n: c_int,
) -> c_int {
    let X = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let W = unsafe { std::slice::from_raw_parts(weight, n as usize) };
    let result = rms_norm(X, W);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), x, n as usize); }
    n
}

#[no_mangle]
pub extern "C" fn lko_gqa_attention(
    q: *const c_float, k_cache: *const c_float, v_cache: *const c_float,
    n_heads: c_int, n_kv: c_int, head_dim: c_int, seq_len: c_int,
    output: *mut c_float,
) -> c_int {
    let q_slice = unsafe { std::slice::from_raw_parts(q, (n_heads * head_dim) as usize) };
    let k_slice = unsafe { std::slice::from_raw_parts(k_cache, (n_kv * seq_len * head_dim) as usize) };
    let v_slice = unsafe { std::slice::from_raw_parts(v_cache, (n_kv * seq_len * head_dim) as usize) };
    let result = gqa_attention(q_slice, k_slice, v_slice,
        n_heads as usize, n_kv as usize, head_dim as usize, seq_len as usize);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), output, result.len()); }
    (n_heads * head_dim)
}

#[no_mangle]
pub extern "C" fn lko_silu(x: *const c_float, y: *mut c_float, n: c_int) -> c_int {
    let X = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let result = silu(X);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), y, n as usize); }
    n
}
