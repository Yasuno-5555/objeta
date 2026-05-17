//! Qwen3.6 full forward pass in Rust with NEON SIMD + rayon.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use rayon::prelude::*;

pub const HDIM: usize = 2048;
const HEAD_DIM: usize = 256;
const N_KV: usize = 2;
const N_Q_ATTN: usize = 16; // Q heads for full GQA

// ── Metal fused GQA FFI ──────────────────────────────────────────────────

extern "C" {
    // Init Metal GQA resources (RoPE tables, once)
    fn lko_metal_gqa_init(rope_cos: *const f32, rope_sin: *const f32, max_seq: i32) -> i32;
    // Load per-layer GQA weights into persistent Metal buffers
    fn lko_metal_gqa_load_weights(w_qkv: *const u16, w_qkv_bytes: i32, w_o: *const u16, w_o_bytes: i32) -> i32;
    // Dispatch fused GQA (QKV + RoPE + attention + Q-gate) — returns attn_out (4096 f32)
    fn lko_metal_fused_gqa(
        h: *const f32, pos: i32, seq_len: i32, max_seq: i32,
        k_cache: *mut f32, v_cache: *mut f32, kv_bytes: i32,
        attn_out: *mut f32,
    ) -> i32;
    // Dispatch GQA O-proj: output = W_o @ attn_out (f16 weights, f32 attn_out → f32 output)
    fn lko_metal_gqa_oproj(
        w_o: *const u16, w_o_bytes: i32,
        attn_out: *const f32,
        output: *mut f32, m: i32, k: i32,
    ) -> i32;
}

/// Try Metal fused GQA. Returns Some(output) on success, None if Metal unavailable.
fn gqa_metal_try(
    w_qkv: &[u16], w_o: &[u16], h: &[f32],
    pos: u32, seq_len: u32, max_seq: u32,
    k_cache: &mut [f32], v_cache: &mut [f32],
    first_call: &mut bool,
) -> Option<Vec<f32>> {
    let ok = unsafe {
        lko_metal_gqa_load_weights(
            w_qkv.as_ptr(), (w_qkv.len() * 2) as i32,
            w_o.as_ptr(), (w_o.len() * 2) as i32,
        ) == 0
    };
    if !ok { if *first_call { eprintln!("[objeta] Metal GQA: load_weights failed"); *first_call = false; } return None; }
    let mut attn_out = vec![0.0f32; 4096];
    let ok = unsafe {
        lko_metal_fused_gqa(
            h.as_ptr(), pos as i32, seq_len as i32, max_seq as i32,
            k_cache.as_mut_ptr(), v_cache.as_mut_ptr(), (k_cache.len() * 4) as i32,
            attn_out.as_mut_ptr(),
        ) == 4096
    };
    if !ok { if *first_call { eprintln!("[objeta] Metal GQA: fused_gqa failed (kernel missing?)"); *first_call = false; } return None; }
    let mut output = vec![0.0f32; HDIM];
    let ok = unsafe {
        lko_metal_gqa_oproj(
            w_o.as_ptr(), (w_o.len() * 2) as i32,
            attn_out.as_ptr(), output.as_mut_ptr(), HDIM as i32, 4096i32,
        ) == HDIM as i32
    };
    if !ok { if *first_call { eprintln!("[objeta] Metal GQA: oproj failed"); *first_call = false; } return None; }
    Some(output)
}

fn rope_cache(max_seq: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0f32; max_seq * hd / 2];
    let mut sin = vec![0.0f32; max_seq * hd / 2];
    for pos in 0..max_seq {
        for i in 0..hd/2 {
            let theta = 1.0 / 10000.0f32.powf(2.0 * i as f32 / hd as f32);
            cos[pos * hd/2 + i] = (pos as f32 * theta).cos();
            sin[pos * hd/2 + i] = (pos as f32 * theta).sin();
        }
    }
    (cos, sin)
}

// ── GEMV f32 (NEON + rayon) ──────────────────────────────────────────────
///
/// GEMV writing into a pre-allocated buffer (zero allocation).
pub fn fill_gemv_f32(y: &mut [f32], W: &[f32], x: &[f32], M: usize, K: usize) {
    assert!(y.len() >= M);
    if M < 128 {
        for i in 0..M {
            let row = &W[i * K..(i + 1) * K];
            y[i] = dot_f32(row, x);
        }
    } else {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let row = &W[i * K..(i + 1) * K];
            *yi = dot_f32(row, x);
        });
    }
}

pub fn gemv_f32(W: &[f32], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; M];
    if M < 128 {
        for i in 0..M {
            let row = &W[i * K..(i + 1) * K];
            y[i] = dot_f32(row, x);
        }
    } else {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let row = &W[i * K..(i + 1) * K];
            *yi = dot_f32(row, x);
        });
    }
    y
}

/// Direct f16 GEMV using manual f16→f32 + NEON FMA — zero intermediate allocation.
/// Reads f16 weights without full f32 conversion buffer. 2x less memory BW.
pub fn gemv_f16_direct(W: &[u16], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; M];
    if M < 64 {
        for i in 0..M {
            let row = &W[i * K..(i + 1) * K];
            y[i] = dot_f16_manual(row, x);
        }
    } else {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let row = &W[i * K..(i + 1) * K];
            *yi = dot_f16_manual(row, x);
        });
    }
    y
}

/// Manual f16→f32 conversion + NEON FMA dot product. No unstable intrinsics.
#[inline]
fn dot_f16_manual(a: &[u16], b: &[f32]) -> f32 {
    let n = a.len();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut sum = vdupq_n_f32(0.0);
        let mut j = 0usize;
        // Process 8 f16 at a time: unpack to f32 then FMA
        while j + 8 <= n {
            // Load 8 f32 inputs
            let x0 = vld1q_f32(b.as_ptr().add(j));
            let x1 = vld1q_f32(b.as_ptr().add(j + 4));
            // Convert 8 f16→f32 manually
            let w0 = f16_to_f32(a[j]);
            let w1 = f16_to_f32(a[j+1]);
            let w2 = f16_to_f32(a[j+2]);
            let w3 = f16_to_f32(a[j+3]);
            let w4 = f16_to_f32(a[j+4]);
            let w5 = f16_to_f32(a[j+5]);
            let w6 = f16_to_f32(a[j+6]);
            let w7 = f16_to_f32(a[j+7]);
            sum = vfmaq_f32(sum, vld1q_f32([w0,w1,w2,w3].as_ptr()), x0);
            sum = vfmaq_f32(sum, vld1q_f32([w4,w5,w6,w7].as_ptr()), x1);
            j += 8;
        }
        while j + 4 <= n {
            let w0 = f16_to_f32(a[j]);
            let w1 = f16_to_f32(a[j+1]);
            let w2 = f16_to_f32(a[j+2]);
            let w3 = f16_to_f32(a[j+3]);
            let x0 = vld1q_f32(b.as_ptr().add(j));
            sum = vfmaq_f32(sum, vld1q_f32([w0,w1,w2,w3].as_ptr()), x0);
            j += 4;
        }
        let mut s = vaddvq_f32(sum);
        while j < n { s += f16_to_f32(a[j]) * b[j]; j += 1; }
        return s;
    }
    #[cfg(not(target_arch = "aarch64"))]
    { a.iter().zip(b.iter()).map(|(h, x)| f16_to_f32(*h) * x).sum() }
}

// ── GEMV f16 (f16→f32 conversion + NEON GEMV) ───────────────────────────

pub fn gemv_f16_buf(W: &[u16], x: &[f32], M: usize, K: usize, buf: &mut Vec<f32>) -> Vec<f32> {
    buf.clear();
    buf.reserve(M * K);
    buf.extend(W.iter().map(|&h| f16_to_f32(h)));
    gemv_f32(buf, x, M, K)
}

pub fn gemv_f16(W: &[u16], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    let w_f32: Vec<f32> = W.iter().map(|&h| f16_to_f32(h)).collect();
    gemv_f32(&w_f32, x, M, K)
}

/// Optimized dot product using NEON where available.
#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let mut j = 0usize;
        while j + 16 <= n {
            let a0 = vld1q_f32(a.as_ptr().add(j));
            let b0 = vld1q_f32(b.as_ptr().add(j));
            let a1 = vld1q_f32(a.as_ptr().add(j + 4));
            let b1 = vld1q_f32(b.as_ptr().add(j + 4));
            let a2 = vld1q_f32(a.as_ptr().add(j + 8));
            let b2 = vld1q_f32(b.as_ptr().add(j + 8));
            let a3 = vld1q_f32(a.as_ptr().add(j + 12));
            let b3 = vld1q_f32(b.as_ptr().add(j + 12));
            sum = vfmaq_f32(sum, a0, b0);
            sum = vfmaq_f32(sum, a1, b1);
            sum = vfmaq_f32(sum, a2, b2);
            sum = vfmaq_f32(sum, a3, b3);
            j += 16;
        }
        while j + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(j));
            let bv = vld1q_f32(b.as_ptr().add(j));
            sum = vfmaq_f32(sum, av, bv);
            j += 4;
        }
        let mut s = vaddvq_f32(sum);
        for r in j..n { s += a[r] * b[r]; }
        return s;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b.iter()).map(|(x,y)| x*y).sum()
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = (h >> 10) & 0x1f;
    let m = h as u32 & 0x3ff;
    if e == 0 {
        if m == 0 { f32::from_bits(s) }
        else { (m as f32) * 2f32.powi(-24) * if s == 0 { 1.0 } else { -1.0 } }
    } else if e == 31 {
        if m == 0 { f32::from_bits(s | 0x7f80_0000) } else { f32::NAN }
    } else {
        f32::from_bits(s | (((e + 112) as u32) << 23) | (m << 13))
    }
}

// ── RMSNorm ───────────────────────────────────────────────────────────────

pub fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let n = x.len();
    let msq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (msq + 1e-6).sqrt();
    x.iter().zip(weight.iter()).map(|(&v, &w)| v * inv * w).collect()
}

// ── C API exports for Python ──────────────────────────────────────────────

/// FP32 GEMV: y = W @ x. W is (M, K) row-major f32.
#[no_mangle]
pub extern "C" fn lko_q36_f32_gemv(
    w: *const f32, m: i32, k: i32,
    x: *const f32, y: *mut f32,
) -> i32 {
    let W = unsafe { std::slice::from_raw_parts(w, (m * k) as usize) };
    let X = unsafe { std::slice::from_raw_parts(x, k as usize) };
    let r = gemv_f32(W, X, m as usize, k as usize);
    unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), y, m as usize) };
    m
}

/// FP16 GEMV: y = W @ x. W is (M, K) row-major f16.
#[no_mangle]
pub extern "C" fn lko_q36_f16_gemv(
    w: *const u16, m: i32, k: i32,
    x: *const f32, y: *mut f32,
) -> i32 {
    let W = unsafe { std::slice::from_raw_parts(w, (m * k) as usize) };
    let X = unsafe { std::slice::from_raw_parts(x, k as usize) };
    let r = gemv_f16(W, X, m as usize, k as usize);
    unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), y, m as usize) };
    m
}

/// RMSNorm: x = RMSNorm(x, weight), in-place.
#[no_mangle]
pub extern "C" fn lko_q36_rms_norm(
    x: *mut f32, weight: *const f32, n: i32,
) -> i32 {
    let X = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let W = unsafe { std::slice::from_raw_parts(weight, n as usize) };
    let r = rms_norm(X, W);
    unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), x, n as usize) };
    n
}

// ── Softmax (in-place, per row of dim) ───────────────────────────────────

pub fn softmax_inplace(x: &mut [f32], dim: usize) {
    for chunk in x.chunks_mut(dim) {
        let max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in chunk.iter_mut() { *v = (*v - max).exp(); sum += *v; }
        for v in chunk.iter_mut() { *v /= sum.max(1e-12); }
    }
}

#[no_mangle]
pub extern "C" fn lko_q36_softmax(x: *mut f32, rows: i32, dim: i32) -> i32 {
    let n = (rows * dim) as usize;
    let X = unsafe { std::slice::from_raw_parts_mut(x, n) };
    softmax_inplace(X, dim as usize);
    n as i32
}

// ── SiLU (in-place) ──────────────────────────────────────────────────────

pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() { *v = *v / (1.0 + (-*v).exp()); }
}

#[no_mangle]
pub extern "C" fn lko_q36_silu(x: *mut f32, n: i32) -> i32 {
    let X = unsafe { std::slice::from_raw_parts_mut(x, n as usize) };
    silu_inplace(X);
    n
}

// ── L2 Normalize (per row of dim) ────────────────────────────────────────

pub fn l2_norm_rows(x: &mut [f32], rows: usize, dim: usize) {
    let inv_scale = 1.0 / (dim as f32).sqrt();
    for row in x.chunks_mut(dim) {
        let sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sq + 1e-6).sqrt() * inv_scale;
        for v in row.iter_mut() { *v *= inv; }
    }
}

#[no_mangle]
pub extern "C" fn lko_q36_l2_norm(x: *mut f32, rows: i32, dim: i32, with_scale: i32) -> i32 {
    let X = unsafe { std::slice::from_raw_parts_mut(x, (rows * dim) as usize) };
    let scale = 1.0 / (dim as f32).sqrt();
    for row in X.chunks_mut(dim as usize) {
        let sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sq + 1e-6).sqrt();
        let s = if with_scale != 0 { inv * scale } else { inv };
        for v in row.iter_mut() { *v *= s; }
    }
    rows * dim
}

// ── DeltaNet State Update ────────────────────────────────────────────────
// S: (n_heads, K, V) = (32, 128, 128) stored flat f32
// k: (n_heads, K) f32
// q: (n_heads, K) f32
// v: (n_heads, V) f32
// beta: (n_heads,) f32
// exp_g: (n_heads,) f32
// output: (n_heads, V) f32

pub fn delta_state_update(
    S: &mut [f32], k: &[f32], q: &[f32], v: &[f32],
    beta: &[f32], exp_g: &[f32],
    n_heads: usize, kv: usize, vd: usize,
    output: &mut [f32],
) {
    let head_size = kv * vd;
    // Parallelize across heads — each head is independent
    S.par_chunks_mut(head_size)
        .zip(k.par_chunks(kv))
        .zip(q.par_chunks(kv))
        .zip(v.par_chunks(vd))
        .zip(output.par_chunks_mut(vd))
        .zip(beta.par_iter().zip(exp_g.par_iter()))
        .for_each(|(((((S_h, k_h), q_h), v_h), out_h), (&beta_h, &g_h))| {
            // S *= exp(g)
            for s in S_h.iter_mut() { *s *= g_h; }

            // kv_mem = S^T @ k
            let mut kv_mem = vec![0.0f32; vd];
            for j in 0..vd {
                let mut s = 0.0;
                for i in 0..kv {
                    s += S_h[i * vd + j] * k_h[i];
                }
                kv_mem[j] = s;
            }

            // S += k ⊗ delta
            for j in 0..vd {
                let delta_val = (v_h[j] - kv_mem[j]) * beta_h;
                for i in 0..kv {
                    S_h[i * vd + j] += k_h[i] * delta_val;
                }
            }

            // output = S^T @ q
            for j in 0..vd {
                let mut s = 0.0;
                for i in 0..kv {
                    s += S_h[i * vd + j] * q_h[i];
                }
                out_h[j] = s;
            }
        });
}

#[no_mangle]
pub extern "C" fn lko_q36_delta_update(
    S: *mut f32, k: *const f32, q: *const f32, v: *const f32,
    beta: *const f32, exp_g: *const f32,
    n_heads: i32, kv_dim: i32, v_dim: i32,
    output: *mut f32,
) -> i32 {
    let nh = n_heads as usize; let kd = kv_dim as usize; let vd = v_dim as usize;
    let S_mut = unsafe { std::slice::from_raw_parts_mut(S, nh * kd * vd) };
    let k_slice = unsafe { std::slice::from_raw_parts(k, nh * kd) };
    let q_slice = unsafe { std::slice::from_raw_parts(q, nh * kd) };
    let v_slice = unsafe { std::slice::from_raw_parts(v, nh * vd) };
    let beta_slice = unsafe { std::slice::from_raw_parts(beta, nh) };
    let g_slice = unsafe { std::slice::from_raw_parts(exp_g, nh) };
    let out = unsafe { std::slice::from_raw_parts_mut(output, nh * vd) };

    delta_state_update(S_mut, k_slice, q_slice, v_slice, beta_slice, g_slice, nh, kd, vd, out);
    (nh * vd) as i32
}

// ── RMSNormGated ─────────────────────────────────────────────────────────
// output: (n_heads, V) — apply RMSNorm per head, then gate with z * sigmoid(z)

pub fn rms_norm_gated(
    output: &[f32], z: &[f32], w_norm: &[f32],
    n_heads: usize, v_dim: usize,
) -> Vec<f32> {
    let mut gated = vec![0.0f32; n_heads * v_dim];
    for h in 0..n_heads {
        let o = &output[h * v_dim..(h + 1) * v_dim];
        let zr = &z[h * v_dim..(h + 1) * v_dim];
        let go = &mut gated[h * v_dim..(h + 1) * v_dim];

        let sq: f32 = o.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sq / v_dim as f32 + 1e-6).sqrt();

        for d in 0..v_dim {
            let on = o[d] * inv_rms * w_norm[d];
            go[d] = on * zr[d] / (1.0 + (-zr[d]).exp());
        }
    }
    gated
}

#[no_mangle]
pub extern "C" fn lko_q36_rms_norm_gated(
    output: *const f32, z: *const f32, w_norm: *const f32,
    n_heads: i32, v_dim: i32,
    gated_out: *mut f32,
) -> i32 {
    let nh = n_heads as usize; let vd = v_dim as usize;
    let o = unsafe { std::slice::from_raw_parts(output, nh * vd) };
    let zs = unsafe { std::slice::from_raw_parts(z, nh * vd) };
    let wn = unsafe { std::slice::from_raw_parts(w_norm, vd) };
    let r = rms_norm_gated(o, zs, wn, nh, vd);
    unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), gated_out, r.len()) };
    r.len() as i32
}

// ── Sigmoid (element-wise) ───────────────────────────────────────────────

pub fn sigmoid_inplace(x: &mut [f32]) {
    for v in x.iter_mut() { *v = 1.0 / (1.0 + (-*v).exp()); }
}

#[no_mangle]
pub extern "C" fn lko_q36_sigmoid(x: *mut f32, n: i32) -> i32 {
    let X = unsafe { std::slice::from_raw_parts_mut(x, n as usize) };
    sigmoid_inplace(X);
    n
}

// ── GQA Attention (fused: QKV gemv + RoPE + softmax + weighted sum + output proj) ──

/// Rayon-parallel f16 GEMV: converts f16→f32 on the fly, zero intermediate buffer.
fn gemv_f16_par(W: &[u16], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    (0..M).into_par_iter().map(|row| {
        let r = &W[row*K..(row+1)*K];
        r.iter().zip(x.iter()).map(|(&h, &xv)| f16_to_f32(h) * xv).sum()
    }).collect()
}

/// Full GQA attention in Rust. All buffers pre-allocated.
/// Returns via `output` (n_heads * head_dim f32).
pub fn gqa_attention_fused(
    // Weights (f32, pre-loaded)
    w_qkv: &[u16], w_o: &[u16],
    // Input
    h: &[f32],
    // KV cache
    k_cache: &mut [f32], v_cache: &mut [f32],
    // RoPE
    rope_cos: &[f32], rope_sin: &[f32],
    // Dimensions
    n_heads: usize, n_kv: usize, head_dim: usize,
    pos: usize, seq_len: usize, max_seq: usize,
    // Output
    output: &mut [f32],
    // Scratch buffers
    qkv_buf: &mut [f32], q_buf: &mut [f32], k_buf: &mut [f32], v_buf: &mut [f32],
    attn_out: &mut [f32],
    scores: &mut [f32], attn_w: &mut [f32],
) {
    let q_sz = n_heads * head_dim * 2;  // 4096 (Q) + 4096 (Q-gate) = 8192
    let k_sz = n_kv * head_dim;  // 512
    let v_sz = n_kv * head_dim;  // 512
    let total = q_sz + k_sz + v_sz;  // 9216
    let K = h.len();

    // QKV projection
    qkv_buf[..total].copy_from_slice(&gemv_f16(w_qkv, h, total, K));

    // Split: q(4352), k(512), v(512)
    let n_q = n_heads * head_dim; // 4096
    q_buf[..n_q].copy_from_slice(&qkv_buf[..n_q]);
    let q_gate: Vec<f32> = qkv_buf[n_q..q_sz].iter().map(|&v| 1.0/(1.0+(-v).exp())).collect();
    k_buf[..k_sz].copy_from_slice(&qkv_buf[q_sz..q_sz + k_sz]);
    v_buf[..v_sz].copy_from_slice(&qkv_buf[q_sz + k_sz..total]);

    // RoPE
    let half = head_dim / 2;
    let c = &rope_cos[pos * half..(pos + 1) * half];
    let s = &rope_sin[pos * half..(pos + 1) * half];
    for h in 0..n_heads {
        for i in 0..half {
            let qe = q_buf[h*head_dim+i]; let qo = q_buf[h*head_dim+half+i];
            q_buf[h*head_dim+i] = qe*c[i] - qo*s[i];
            q_buf[h*head_dim+half+i] = qe*s[i] + qo*c[i];
        }
    }
    for h in 0..n_kv {
        for i in 0..half {
            let ke = k_buf[h*head_dim+i]; let ko = k_buf[h*head_dim+half+i];
            k_buf[h*head_dim+i] = ke*c[i] - ko*s[i];
            k_buf[h*head_dim+half+i] = ke*s[i] + ko*c[i];
        }
    }

    // Write KV cache
    for h in 0..n_kv {
        let k_off = h * max_seq * head_dim + pos * head_dim;
        let v_off = h * max_seq * head_dim + pos * head_dim;
        for d in 0..head_dim {
            k_cache[k_off + d] = k_buf[h * head_dim + d];
            v_cache[v_off + d] = v_buf[h * head_dim + d];
        }
    }

    // Attention: for each head, compute scores, softmax, weighted sum
    let n_rep = n_heads / n_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_heads {
        let kv_h = h / n_rep;
        let qh = &q_buf[h * head_dim..(h + 1) * head_dim];
        let oh = &mut attn_out[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);

        // Scores with NEON dot_f32
        let mut max_s = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let kt = &k_cache[(kv_h * max_seq + t) * head_dim..(kv_h * max_seq + t) * head_dim + head_dim];
            scores[t] = dot_f32(qh, kt) * scale;
            if scores[t] > max_s { max_s = scores[t]; }
        }

        // Softmax
        let mut sum = 0.0;
        for t in 0..seq_len { attn_w[t] = (scores[t] - max_s).exp(); sum += attn_w[t]; }
        for t in 0..seq_len { attn_w[t] /= sum.max(1e-12); }

        // Weighted sum
        for t in 0..seq_len {
            let vt = &v_cache[(kv_h * max_seq + t) * head_dim..(kv_h * max_seq + t) * head_dim + head_dim];
            let a = attn_w[t];
            for d in 0..head_dim { oh[d] += a * vt[d]; }
        }

        // Apply gate: q_gate (4096,) element-wise × attn_out (4096,)
        let g_off = h * head_dim;
        for d in 0..head_dim { oh[d] *= q_gate[g_off + d]; }
    }

    // Output projection
    let ao = gemv_f16(w_o, attn_out, HDIM, n_heads * head_dim);
    output[..HDIM].copy_from_slice(&ao);
}

// ── Fused DeltaNet Layer (1 C call = entire DeltaNet forward) ────────────

/// Runs complete DeltaNet forward in Rust.
/// Returns attention output (HDIM f32).
/// Pre-converts f16 weights to f32 once to avoid per-row allocation overhead.
pub fn delta_net_fused(
    w_qkv: &[u16], w_z: &[u16], w_b: &[f32], w_a: &[f32],
    w_out: &[u16], w_conv: &[f32], w_norm: &[f32],
    dt_bias: &[f32], a_log: &[f32],
    h: &[f32],
    conv_state: &mut [f32], conv_ptr: &mut usize,
    S_state: &mut [f32],
    ao_out: &mut [f32],
    scratch_f32: &mut Vec<f32>,
) {
    let mixed_qkv = gemv_f16_buf(w_qkv, h, 8192, HDIM, scratch_f32);
    let z = gemv_f16_buf(w_z, h, 4096, HDIM, scratch_f32);
    let b = gemv_f32(w_b, h, 32, HDIM);
    let a_vec = gemv_f32(w_a, h, 32, HDIM);

    // Conv1d ring buffer
    let ptr = *conv_ptr;
    for c in 0..8192 { conv_state[c * 4 + ptr] = mixed_qkv[c]; }
    let new_ptr = (ptr + 1) % 4;
    *conv_ptr = new_ptr;
    let order = [(ptr + 1) % 4, (ptr + 2) % 4, (ptr + 3) % 4, ptr];  // weight[3]=newest

    let mut qkv_conv = vec![0.0f32; 8192];
    for c in 0..8192 {
        let mut s = 0.0;
        for t in 0..4 {
            s += w_conv[c * 4 + t] * conv_state[c * 4 + order[t]];
        }
        qkv_conv[c] = s / (1.0 + (-s).exp()); // SiLU
    }

    let q: Vec<f32> = qkv_conv[..2048].to_vec();
    let k: Vec<f32> = qkv_conv[2048..4096].to_vec();
    let v: Vec<f32> = qkv_conv[4096..].to_vec();

    let mut q_rep = vec![0.0f32; 32 * 128];
    let mut k_rep = vec![0.0f32; 32 * 128];
    let v_rs = v.clone();
    let z_rs = z.clone();

    for h in 0..32 {
        let sh = h / 2;
        for d in 0..128 {
            q_rep[h*128+d] = q[sh*128+d];
            k_rep[h*128+d] = k[sh*128+d];
        }
    }

    let beta: Vec<f32> = b.iter().map(|&x| 1.0/(1.0+(-x).exp())).collect();
    let exp_g: Vec<f32> = (0..32).map(|i| {
        let sp = (a_vec[i] + dt_bias[i]).max(-10.0).exp().ln_1p();
        (-a_log[i].exp() * sp).exp()
    }).collect();

    // L2 norm q,k
    for h in 0..32 {
        let mut qn = 0.0f32; let mut kn = 0.0f32;
        for d in 0..128 { qn += q_rep[h*128+d].powi(2); kn += k_rep[h*128+d].powi(2); }
        let qs = 1.0 / (qn.sqrt() + 1e-6) / (128.0f32).sqrt();
        let ks = 1.0 / (kn.sqrt() + 1e-6);
        for d in 0..128 { q_rep[h*128+d] *= qs; k_rep[h*128+d] *= ks; }
    }

    // Delta state update
    let mut output = vec![0.0f32; 32 * 128];
    delta_state_update(S_state, &k_rep, &q_rep, &v_rs, &beta, &exp_g, 32, 128, 128, &mut output);

    // RMSNormGated
    let gated = rms_norm_gated(&output, &z_rs, w_norm, 32, 128);

    // Output projection
    let ao = gemv_f16_buf(w_out, &gated, HDIM, 4096, scratch_f32);
    ao_out.copy_from_slice(&ao);
}

/// One fused layer forward call.
/// Returns attention output in `ao_out` (HDIM f32).
/// Updates kv_cache or delta_state in place.
#[no_mangle]
pub extern "C" fn lko_q36_fused_layer(
    // Weights: packed f32 arrays (all weights for one layer concatenated)
    w_ptr: *const f32, w_sizes: *const i32,  // [size of each weight matrix] × N_MATS
    n_mats: i32,
    // Input
    h: *const f32,
    // State
    conv_state: *mut f32, conv_ptr: *mut i32,
    S_state: *mut f32,
    k_cache: *mut f32, v_cache: *mut f32,
    rope_cos: *const f32, rope_sin: *const f32,
    // Dimensions
    pos: i32, seq_len: i32, max_seq: i32,
    layer_type: i32,  // 0=DeltaNet, 1=FullGQA, 2=NoAttn
    // Output
    ao_out: *mut f32,
) -> i32 {
    // Stub — fused API not yet used. Use individual C calls.
    -1
}

// ═══════════════════════════════════════════════════════════════════════════
// LKO Scheduler: Phase-aware execution policy
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq)]
enum AttnPolicy {
    Full,       // Full GQA or DeltaNet forward
    Collapse,   // Koopman identity (skip, J≈I)
    Skip,       // No attention at all
}

#[derive(Clone, Copy, PartialEq)]
enum MoEPolicy {
    Full,       // Dequantize + GEMV all routed experts
    Adaptive,   // Entropy-conditioned top-k pruning
    Skip,       // Skip MoE entirely
}

#[derive(Clone, Copy)]
struct LayerPolicy {
    attn: AttnPolicy,
    moe: MoEPolicy,
    precision_bits: u8,  // target precision for weights (3-16)
    is_steering: bool,   // is this a GQA course-correction layer?
}

/// Build static policy table for Qwen3.6-35B-A3B based on LKO phase measurements.
/// GQA at layers 3,7,11,15,19,23,27,31,35,39 (every 4th).
/// UNFOLD: L0-L2 (sacred), ISOMETRIC: L3-L35, DIVERGENT: L36-L39.
fn build_policy_table(fusion_ratio: f64, moe_on_deltanet: bool) -> [LayerPolicy; 40] {
    let stride = (1.0 / fusion_ratio.max(0.01)).round() as usize;
    let mut delta_count: usize = 0;
    let mut table = [LayerPolicy {
        attn: AttnPolicy::Full, moe: MoEPolicy::Full, precision_bits: 16, is_steering: false
    }; 40];

    for l in 0..40 {
        let is_gqa = l % 4 == 3;
        let phase = if l < 3 { "unfold" } else if l > 35 { "divergent" } else { "isometric" };

        let (attn, moe, prec, steering) = if is_gqa {
            delta_count = 0;
            // GQA: course correction — always full, high precision
            (AttnPolicy::Full, MoEPolicy::Adaptive, 16, true)
        } else {
            delta_count += 1;
            let compute = delta_count % stride.max(1) == 0;
            let attn = if compute { AttnPolicy::Full } else { AttnPolicy::Collapse };
            let moe = if moe_on_deltanet { MoEPolicy::Adaptive } else { MoEPolicy::Skip };
            let prec = match phase {
                "unfold" => 16,
                "divergent" => 8,
                _ => 4, // ISOMETRIC: low precision (q4)
            };
            (attn, moe, prec, false)
        };

        table[l] = LayerPolicy { attn, moe, precision_bits: prec, is_steering: steering };
    }
    table
}

// ═══════════════════════════════════════════════════════════════════════════
// Full Executor: owns weights, KV caches, DeltaNet states, 40-layer loop
// ═══════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Per-layer weight set. All weights pre-converted to f32.
struct LayerWeights {
    // Large weights: f16 (half memory). Total per layer: ~72MB vs ~145MB for f32.
    w_qkv: Vec<u16>, w_o: Vec<u16>, w_z: Vec<u16>,
    se_gate: Vec<u16>, se_up: Vec<u16>, se_down: Vec<u16>,
    // Small weights: f32 (negligible memory)
    w_b: Vec<f32>, w_a: Vec<f32>, w_conv: Vec<f32>, w_norm: Vec<f32>,
    dt_bias: Vec<f32>, a_log: Vec<f32>, se_gate_w: Vec<f32>,
    input_norm: Vec<f32>, post_norm: Vec<f32>,
    is_gqa: bool, has_attn: bool,
    qkv_M: usize, qkv_K: usize, o_M: usize, o_K: usize,
}

pub struct Qwen36Runner {
    embed: memmap2::Mmap,   // mmap'd embed_tokens.bin (2GB, zero-copy)
    final_norm: Vec<f32>,
    layers: Vec<LayerWeights>,
    // KV caches
    kv_k: Vec<Vec<f32>>,    // per layer: (n_kv × max_seq × head_dim)
    kv_v: Vec<Vec<f32>>,
    // DeltaNet states
    conv_states: Vec<Vec<f32>>,  // per layer: (8192 × 4)
    conv_ptrs: Vec<usize>,
    S_states: Vec<Vec<f32>>,     // per layer: (32 × 128 × 128)
    // RoPE
    rope_cos: Vec<f32>, rope_sin: Vec<f32>,
    // MoE: pre-loaded routers + cached mmaps
    routers: Vec<Vec<f32>>,
    gu_mmaps: Vec<memmap2::Mmap>,
    down_mmaps: Vec<memmap2::Mmap>,
    // Scratch buffers (reused per forward pass)
    scratch_qkv: Vec<f32>, scratch_q: Vec<f32>, scratch_k: Vec<f32>, scratch_v: Vec<f32>,
    scratch_attn_out: Vec<f32>,
    scratch_scores: Vec<f32>, scratch_attn: Vec<f32>,
    scratch_f32: Vec<f32>, // reusable f16→f32 conversion buffer
    max_seq: usize,
    /// DeltaNet fusion: fraction of DeltaNet layers to compute (1.0=all, 0.33=1 per GQA block)
    pub fusion_ratio: f64,
    /// Skip MoE+shared expert on non-GQA (DeltaNet) layers
    pub moe_on_deltanet: bool,
    /// Scheduler: phase-aware execution policy per layer
    policy_table: [LayerPolicy; 40],
    /// Whether Metal fused GQA is available (tested at init)
    pub metal_gqa_ok: bool,
    metal_gqa_first_fail: bool,
    /// Expert residency cache: (layer, eid) → (gate_f32, up_f32, down_f32)
    expert_cache: std::collections::HashMap<(usize, usize), (Vec<f32>, Vec<f32>, Vec<f32>)>,
    expert_cache_order: Vec<(usize, usize)>, // LRU order, front = most recent
    expert_cache_max: usize,
    /// Expert cache status: number of experts cached per layer (0 = not built)
    pub expert_cache_size: usize,
    /// Per-layer expert frequency data collected during warmup
    expert_freq_ready: bool,
    /// Pre-allocated scratch buffers for MoE GEMV (reused, zero allocation)
    moe_gate_buf: Vec<f32>,      // 512
    moe_up_buf: Vec<f32>,        // 512
    moe_hidden_buf: Vec<f32>,    // 512
    moe_down_buf: Vec<f32>,      // 2048
}

impl Qwen36Runner {
    pub fn new(bin_dir: &Path, max_seq: usize) -> Option<Self> {
        // mmap embed to save 2GB RAM
        let embed_path = bin_dir.join("embed_tokens.bin");
        let embed_file = std::fs::File::open(&embed_path).ok()?;
        let embed = unsafe { memmap2::Mmap::map(&embed_file).ok()? };
        let n_vocab = embed.len() / (HDIM * 4); // f32 = 4 bytes

        let norm_bytes = std::fs::read(bin_dir.join("final_norm.bin")).ok()?;
        let final_norm: Vec<f32> = norm_bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();

        // Load all 40 layers
        let mut layers = Vec::with_capacity(40);
        for l in 0..40 {
            layers.push(load_layer_weights(bin_dir, l)?);
        }

        // Build scheduler policy table
        let policy_table = build_policy_table(0.33, false);

        // Apply strategy.json if present (family-aware precision)
        if let Some(strategy) = crate::strategy::load_strategy(bin_dir) {
            let ec = &strategy.executor_config;
            for l in 0..40 {
                let lw = &mut layers[l];
                let ffn_b = ec.ffn_bits.get(l).copied().unwrap_or(4);
                let qo_b = ec.attn_qo_bits.get(l).copied().unwrap_or(16);
                let kv_b = ec.attn_kv_bits.get(l).copied().unwrap_or(16);

                if ffn_b < 16 {
                    lw.se_gate = crate::strategy::requantize_f16(&lw.se_gate, ffn_b);
                    lw.se_up = crate::strategy::requantize_f16(&lw.se_up, ffn_b);
                    lw.se_down = crate::strategy::requantize_f16(&lw.se_down, ffn_b);
                }
                // Attention Q projection (part of w_qkv for GQA, separate for DeltaNet)
                // For GQA layers: Q is first N_Q_ATTN*HEAD_DIM elements of w_qkv
                // For DeltaNet: separate q_proj
                if qo_b < 16 {
                    if lw.is_gqa {
                        // GQA: w_qkv contains Q+K+V concatenated. Q is first part.
                        let q_size = 16 * HEAD_DIM; // N_Q_ATTN * HEAD_DIM
                        let q_part: Vec<u16> = lw.w_qkv[..q_size].to_vec();
                        let q_quant = crate::strategy::requantize_f16(&q_part, qo_b);
                        lw.w_qkv[..q_size].copy_from_slice(&q_quant);
                        // O projection
                        lw.w_o = crate::strategy::requantize_f16(&lw.w_o, qo_b);
                    }
                    // DeltaNet layers: Q/O handled separately
                }
                if kv_b < 16 && lw.is_gqa {
                    // GQA: K and V are after Q in w_qkv
                    let q_size = 16 * HEAD_DIM;
                    let kv_size = 2 * HEAD_DIM + 2 * HEAD_DIM; // K(2*256) + V(2*256)
                    let kv_part: Vec<u16> = lw.w_qkv[q_size..q_size + kv_size].to_vec();
                    let kv_quant = crate::strategy::requantize_f16(&kv_part, kv_b);
                    lw.w_qkv[q_size..q_size + kv_size].copy_from_slice(&kv_quant);
                }
            }
        }

        // KV caches (only for full GQA layers, l % 4 == 3)
        let kv_size = 2 * max_seq * HEAD_DIM; // n_kv=2, hd=256
        let kv_k: Vec<_> = (0..40).map(|l| {
            if l % 4 == 3 { vec![0.0f32; kv_size] } else { Vec::new() }
        }).collect();
        let kv_v: Vec<_> = (0..40).map(|l| {
            if l % 4 == 3 { vec![0.0f32; kv_size] } else { Vec::new() }
        }).collect();

        // DeltaNet states (for linear attention layers)
        let conv_states: Vec<_> = (0..40).map(|l| {
            if l % 4 != 3 && layers[l].has_attn { vec![0.0f32; 8192 * 4] } else { Vec::new() }
        }).collect();
        let conv_ptrs = vec![0usize; 40];
        let S_states: Vec<_> = (0..40).map(|l| {
            if l % 4 != 3 && layers[l].has_attn { vec![0.0f32; 32 * 128 * 128] } else { Vec::new() }
        }).collect();

        let (rope_cos, rope_sin) = rope_cache(max_seq, HEAD_DIM);

        // Init Metal GQA persistent resources (RoPE tables, once)
        let metal_gqa_ok = unsafe {
            let r = lko_metal_gqa_init(rope_cos.as_ptr(), rope_sin.as_ptr(), max_seq as i32);
            r == 0
        };
        if metal_gqa_ok {
            eprintln!("[objeta] Metal GQA: OK");
        } else {
            eprintln!("[objeta] Metal GQA: unavailable, using CPU fallback");
        }

        // Pre-load routers + mmap MoE weights
        let mut routers = Vec::with_capacity(40);
        let mut gu_mmaps = Vec::with_capacity(40);
        let mut down_mmaps = Vec::with_capacity(40);
        for l in 0..40 {
            let rpath = bin_dir.join(format!("layer_{}_router.bin", l));
            let rbytes = std::fs::read(&rpath).unwrap_or_default();
            let r: Vec<f32> = rbytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
            routers.push(r);

            let gu_f = std::fs::File::open(bin_dir.join(format!("layer_{}_gate_up.bin", l))).unwrap();
            let d_f = std::fs::File::open(bin_dir.join(format!("layer_{}_down.bin", l))).unwrap();
            gu_mmaps.push(unsafe { memmap2::Mmap::map(&gu_f).unwrap() });
            down_mmaps.push(unsafe { memmap2::Mmap::map(&d_f).unwrap() });
        }

        // Init per-layer expert caches
        unsafe { crate::moe_dispatch::lko_moe_init_caches(40); }
        // Init per-layer frequency trackers
        unsafe { crate::moe_dispatch::lko_moe_init_freq_tracker(40); }

        Some(Qwen36Runner {
            embed, final_norm, layers,
            kv_k, kv_v, conv_states, conv_ptrs, S_states,
            rope_cos, rope_sin,
            routers, gu_mmaps, down_mmaps,
            policy_table,
            scratch_qkv: vec![0.0f32; 9216],
    scratch_q: vec![0.0f32; 16*256],
            scratch_k: vec![0.0f32; 2*256], scratch_v: vec![0.0f32; 2*256],
            scratch_attn_out: vec![0.0f32; 16*256],
    scratch_scores: vec![0.0f32; max_seq], scratch_attn: vec![0.0f32; max_seq],
            scratch_f32: Vec::with_capacity(20_000_000), // ~80MB, for largest f16→f32 GEMV
            max_seq,
            fusion_ratio: 0.33,
            moe_on_deltanet: false,  // matches initial policy_table
            metal_gqa_ok, metal_gqa_first_fail: true,
            expert_cache: std::collections::HashMap::new(),
            expert_cache_order: Vec::new(),
            expert_cache_max: 50, // ~630MB, small — OS page cache handles q4→f32
            expert_cache_size: 0,
            expert_freq_ready: false,
            moe_gate_buf: vec![0.0f32; 512],
            moe_up_buf: vec![0.0f32; 512],
            moe_hidden_buf: vec![0.0f32; 512],
            moe_down_buf: vec![0.0f32; HDIM],
        })
    }

    /// Forward pass WITH timing breakdown. Returns (h, [deltanet_ms, gqa_ms, shared_ms, moe_ms]).
    pub fn forward_timed(&mut self, token_id: usize, pos: usize, seq_len: usize) -> (Vec<f32>, [f64; 5]) {
        use std::time::Instant;
        let mut h = {
    let ptr = unsafe { self.embed.as_ptr().add(token_id * HDIM * 4) as *const f32 };
    (0..HDIM).map(|i| unsafe { *ptr.add(i) }).collect::<Vec<f32>>()
};
        let mut t_delta = 0.0f64;
        let mut t_gqa = 0.0f64;
        let mut t_shared = 0.0f64;
        let mut t_moe = 0.0f64;
        let mut t_norm = 0.0f64;

        let stride = (1.0 / self.fusion_ratio.max(0.01)).round() as usize;
        let mut delta_count = 0usize;
        let mut deltas_skipped = 0u32;

        for l in 0..40 {
            let policy = &self.policy_table[l];
            let lw = &self.layers[l];

            // ── Norm (always, cheap) ──
            let t0 = Instant::now();
            if !lw.input_norm.is_empty() {
                h = rms_norm(&h, &lw.input_norm);
            }
            t_norm += t0.elapsed().as_secs_f64();

            // ── Attention (policy-driven) ──
            let ao = match policy.attn {
                AttnPolicy::Full => {
                    if policy.is_steering {
                        // GQA steering layer: Metal fused kernel
                        let t0 = Instant::now();
                        let ao = if self.metal_gqa_ok {
                            if let Some(ao) = gqa_metal_try(
                                &lw.w_qkv, &lw.w_o, &h,
                                pos as u32, seq_len as u32, self.max_seq as u32,
                                &mut self.kv_k[l], &mut self.kv_v[l],
                                &mut self.metal_gqa_first_fail,
                            ) { ao } else {
                                let mut ao = vec![0.0f32; HDIM];
                                gqa_attention_fused(
                                    &lw.w_qkv, &lw.w_o, &h,
                                    &mut self.kv_k[l], &mut self.kv_v[l],
                                    &self.rope_cos, &self.rope_sin,
                                    16, 2, HEAD_DIM, pos, seq_len, self.max_seq,
                                    &mut ao,
                                    &mut self.scratch_qkv, &mut self.scratch_q,
                                    &mut self.scratch_k, &mut self.scratch_v,
                                    &mut self.scratch_attn_out,
                                    &mut self.scratch_scores, &mut self.scratch_attn,
                                );
                                ao
                            }
                        } else {
                            let mut ao = vec![0.0f32; HDIM];
                            gqa_attention_fused(
                                &lw.w_qkv, &lw.w_o, &h,
                                &mut self.kv_k[l], &mut self.kv_v[l],
                                &self.rope_cos, &self.rope_sin,
                                16, 2, HEAD_DIM, pos, seq_len, self.max_seq,
                                &mut ao,
                                &mut self.scratch_qkv, &mut self.scratch_q,
                                &mut self.scratch_k, &mut self.scratch_v,
                                &mut self.scratch_attn_out,
                                &mut self.scratch_scores, &mut self.scratch_attn,
                            );
                            ao
                        };
                        t_gqa += t0.elapsed().as_secs_f64();
                        ao
                    } else {
                        // DeltaNet transport layer
                        let t0 = Instant::now();
                        let mut ao = vec![0.0f32; HDIM];
                        delta_net_fused(
                            &lw.w_qkv, &lw.w_z, &lw.w_b, &lw.w_a,
                            &lw.w_o, &lw.w_conv, &lw.w_norm,
                            &lw.dt_bias, &lw.a_log,
                            &h,
                            &mut self.conv_states[l], &mut self.conv_ptrs[l],
                            &mut self.S_states[l],
                            &mut ao,
                            &mut self.scratch_f32,
                        );
                        t_delta += t0.elapsed().as_secs_f64();
                        ao
                    }
                }
                AttnPolicy::Collapse => {
                    // Koopman collapse: J≈I, identity skip
                    vec![0.0f32; HDIM]
                }
                AttnPolicy::Skip => {
                    vec![0.0f32; HDIM]
                }
            };

            for i in 0..HDIM { h[i] += ao[i]; }

            // ── Post-attention norm ──
            let t0 = Instant::now();
            if !lw.post_norm.is_empty() {
                h = rms_norm(&h, &lw.post_norm);
            }
            t_norm += t0.elapsed().as_secs_f64();

            // ── Shared expert (policy-driven) ──
            let t0 = Instant::now();
            if policy.moe != MoEPolicy::Skip && !lw.se_gate.is_empty() {
                let gate = gemv_f16(&lw.se_gate, &h, 512, HDIM);
                let up = gemv_f16(&lw.se_up, &h, 512, HDIM);
                let mut hidden = gate.clone();
                for i in 0..512 { hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i]; }
                let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
                let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h)).exp());
                for i in 0..HDIM { h[i] += se_out[i] * se_gate; }
            }
            t_shared += t0.elapsed().as_secs_f64();

            // ── MoE dispatch (policy-driven) ──
            let t0 = Instant::now();
            if policy.moe != MoEPolicy::Skip {
                let moe_out = self.call_moe(&h, l);
                for i in 0..HDIM { h[i] += moe_out[i]; }
            }
            t_moe += t0.elapsed().as_secs_f64();
        }

        let n_full: usize = self.policy_table.iter().filter(|p| p.attn == AttnPolicy::Full).count();
        let n_collapse: usize = self.policy_table.iter().filter(|p| p.attn == AttnPolicy::Collapse).count();
        eprintln!("TIMING: delta={:.0}ms gqa={:.0}ms shared={:.0}ms moe={:.0}ms | scheduler: full={} collapse={} fusion={:.2}",
            t_delta*1000.0, t_gqa*1000.0, t_shared*1000.0, t_moe*1000.0,
            n_full, n_collapse, self.fusion_ratio);

        (h, [t_delta, t_gqa, t_shared, t_moe, t_norm])
    }

    /// Full 40-layer forward pass. Returns hidden state (HDIM f32).
    pub fn forward(&mut self, token_id: usize, pos: usize, seq_len: usize) -> Vec<f32> {
        let mut h = {
    let ptr = unsafe { self.embed.as_ptr().add(token_id * HDIM * 4) as *const f32 };
    (0..HDIM).map(|i| unsafe { *ptr.add(i) }).collect::<Vec<f32>>()
};

        for l in 0..40 {
            let policy = self.policy_table[l];
            let lw = &self.layers[l];

            if !lw.input_norm.is_empty() {
                h = rms_norm(&h, &lw.input_norm);
            }

            let ao = match policy.attn {
                AttnPolicy::Full => {
                    if policy.is_steering {
                        if self.metal_gqa_ok {
                            if let Some(ao) = gqa_metal_try(
                                &lw.w_qkv, &lw.w_o, &h,
                                pos as u32, seq_len as u32, self.max_seq as u32,
                                &mut self.kv_k[l], &mut self.kv_v[l],
                                &mut self.metal_gqa_first_fail,
                            ) { ao } else {
                                let mut ao = vec![0.0f32; HDIM];
                                gqa_attention_fused(
                                    &lw.w_qkv, &lw.w_o, &h,
                                    &mut self.kv_k[l], &mut self.kv_v[l],
                                    &self.rope_cos, &self.rope_sin,
                                    16, 2, HEAD_DIM, pos, seq_len, self.max_seq,
                                    &mut ao,
                                    &mut self.scratch_qkv, &mut self.scratch_q,
                                    &mut self.scratch_k, &mut self.scratch_v,
                                    &mut self.scratch_attn_out,
                                    &mut self.scratch_scores, &mut self.scratch_attn,
                                );
                                ao
                            }
                        } else {
                            let mut ao = vec![0.0f32; HDIM];
                            gqa_attention_fused(
                                &lw.w_qkv, &lw.w_o, &h,
                                &mut self.kv_k[l], &mut self.kv_v[l],
                                &self.rope_cos, &self.rope_sin,
                                16, 2, HEAD_DIM, pos, seq_len, self.max_seq,
                                &mut ao,
                                &mut self.scratch_qkv, &mut self.scratch_q,
                                &mut self.scratch_k, &mut self.scratch_v,
                                &mut self.scratch_attn_out,
                                &mut self.scratch_scores, &mut self.scratch_attn,
                            );
                            ao
                        }
                    } else {
                        let mut ao = vec![0.0f32; HDIM];
                        delta_net_fused(
                            &lw.w_qkv, &lw.w_z, &lw.w_b, &lw.w_a,
                            &lw.w_o, &lw.w_conv, &lw.w_norm,
                            &lw.dt_bias, &lw.a_log,
                            &h,
                            &mut self.conv_states[l], &mut self.conv_ptrs[l],
                            &mut self.S_states[l],
                            &mut ao,
                            &mut self.scratch_f32,
                        );
                        ao
                    }
                }
                AttnPolicy::Collapse | AttnPolicy::Skip => {
                    vec![0.0f32; HDIM]
                }
            };

            for i in 0..HDIM { h[i] += ao[i]; }

            if !lw.post_norm.is_empty() {
                h = rms_norm(&h, &lw.post_norm);
            }

            if policy.moe != MoEPolicy::Skip && !lw.se_gate.is_empty() {
                let gate = gemv_f16(&lw.se_gate, &h, 512, HDIM);
                let up = gemv_f16(&lw.se_up, &h, 512, HDIM);
                let mut hidden = gate.clone();
                for i in 0..512 { hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i]; }
                let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
                let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h)).exp());
                for i in 0..HDIM { h[i] += se_out[i] * se_gate; }
            }

            if policy.moe != MoEPolicy::Skip {
                let moe_out = self.call_moe(&h, l);
                for i in 0..HDIM { h[i] += moe_out[i]; }
            }
        }

        h
    }

    fn call_moe(&mut self, h: &[f32], l: usize) -> Vec<f32> {
        // Adaptive top-k
        let (eidx, ew, _n_selected) = crate::moe_dispatch::router_topk_adaptive(
            &self.routers[l], h, 8,
        );

        let mut out = vec![0.0f32; HDIM];

        // ── Phase 1: gather expert IDs, separate cached from uncached ──
        let n = eidx.len();
        let mut uncached_ids: Vec<usize> = Vec::with_capacity(n);
        let mut uncached_rws: Vec<f32> = Vec::with_capacity(n);

        for (&eid, &rw) in eidx.iter().zip(ew.iter()) {
            let eid = eid as usize;
            let key = (l, eid);

            // Check hashmap cache first
            if let Some(pos) = self.expert_cache_order.iter().position(|k| *k == key) {
                self.expert_cache_order.remove(pos);
                self.expert_cache_order.insert(0, key);
                let (gate, up, down) = &self.expert_cache[&key];
                // Cached: compute with pre-allocated scratch (fast, no I/O)
                let gate_out = &mut self.moe_gate_buf;
                fill_gemv_f32(gate_out, gate, h, 512, HDIM);
                let up_out = &mut self.moe_up_buf;
                fill_gemv_f32(up_out, up, h, 512, HDIM);
                let hidden = &mut self.moe_hidden_buf;
                for i in 0..512 { hidden[i] = gate_out[i] / (1.0 + (-gate_out[i]).exp()) * up_out[i]; }
                let down_out = &mut self.moe_down_buf;
                fill_gemv_f32(down_out, down, hidden, HDIM, 512);
                for i in 0..HDIM { out[i] += down_out[i] * rw; }
            } else {
                uncached_ids.push(eid);
                uncached_rws.push(rw);
            }
        }

        if uncached_ids.is_empty() { return out; }

        // ── Phase 2: parallel mmap read + dequant + GEMV for uncached ──
        // Raw pointer through usize for Send+Sync (mmap is read-only, concurrent reads safe)
        let gu_addr = self.gu_mmaps[l].as_ptr() as usize;
        let d_addr = self.down_mmaps[l].as_ptr() as usize;

        let results: Vec<(Vec<f32>, (usize, usize), (Vec<f32>, Vec<f32>, Vec<f32>))> =
            uncached_ids.par_iter().zip(uncached_rws.par_iter()).map(|(&eid, &rw)| {
                let gu_off = eid * 1_310_720;
                let d_off = eid * 655_360;
                let gu_ptr = unsafe { (gu_addr as *const u8).add(gu_off) };
                let d_ptr = unsafe { (d_addr as *const u8).add(d_off) };
                let (gate, up, down) = crate::moe_dispatch::dequantize_expert_f32(
                    gu_ptr, 1_310_720, d_ptr, 655_360,
                );

                // GEMV with local scratch (allocation: ~8KB, vs SSD read: ~15ms)
                let mut gate_out = vec![0.0f32; 512];
                fill_gemv_f32(&mut gate_out, &gate, h, 512, HDIM);
                let mut up_out = vec![0.0f32; 512];
                fill_gemv_f32(&mut up_out, &up, h, 512, HDIM);
                let mut hidden = vec![0.0f32; 512];
                for i in 0..512 { hidden[i] = gate_out[i] / (1.0 + (-gate_out[i]).exp()) * up_out[i]; }
                let mut down_out = vec![0.0f32; HDIM];
                fill_gemv_f32(&mut down_out, &down, &hidden, HDIM, 512);
                for v in &mut down_out { *v *= rw; }

                (down_out, (l, eid), (gate, up, down))
            }).collect();

        // ── Phase 3: sum outputs + update cache ──
        for (output, key, entry) in results {
            for i in 0..HDIM { out[i] += output[i]; }
            self.expert_cache.insert(key, entry);
            self.expert_cache_order.insert(0, key);
        }
        while self.expert_cache_order.len() > 50 {
            if let Some(old_key) = self.expert_cache_order.pop() {
                self.expert_cache.remove(&old_key);
            }
        }

        out
    }

    /// Warmup: actually dequantize + GEMV the routed experts to fault q4 pages into OS cache.
    /// Accesses ALL bytes of each expert's q4 data, forcing the kernel to cache them in RAM.
    pub fn warmup(&mut self, n_tokens: usize) {
        if self.expert_freq_ready { return; }
        eprintln!("[objeta] Warming OS page cache by dequantizing real experts ({n_tokens} tokens)...");
        let vocab = self.embed.len() / (HDIM * 4);
        let step = vocab.max(1) / n_tokens.max(1);
        for t in 0..n_tokens {
            let tid = (t * step).min(vocab - 1);
            let h = self.embed_token(tid);
            for l in 0..40 {
                let run_moe = self.layers[l].is_gqa || self.moe_on_deltanet;
                if !run_moe { continue; }
                // Actually dequantize + GEMV all routed experts (reads ALL q4 pages into OS cache)
                let _ = self.call_moe(&h, l);
            }
        }
        // Clear hashmap cache after warmup (these were random tokens, not our prompt)
        self.expert_cache.clear();
        self.expert_cache_order.clear();
        self.expert_freq_ready = true;
        eprintln!("[objeta] OS page cache warmed ({} experts faulted in).", n_tokens * 10 * 8);
    }

    /// Get token embedding from mmap (zero-copy, returns Vec for convenience).
    fn embed_token(&self, token_id: usize) -> Vec<f32> {
        let ptr = unsafe { self.embed.as_ptr().add(token_id * HDIM * 4) as *const f32 };
        (0..HDIM).map(|i| unsafe { *ptr.add(i) }).collect()
    }

    /// Build per-layer expert caches from warmup frequency data.
    /// Only caches layers where MoE actually runs (GQA layers if moe_on_deltanet=false).
    pub fn build_expert_caches(&mut self, cache_size: usize) {
        if !self.expert_freq_ready {
            eprintln!("[objeta] Warning: building caches without warmup — hit rate will be low");
        }
        let total_layers: usize = (0..40).filter(|&l| self.layers[l].is_gqa || self.moe_on_deltanet).count();
        let mem_per_expert_mb = (512.0 * 2048.0 * 3.0 * 4.0) / (1024.0 * 1024.0); // ~12.6MB f32
        let est_mb = total_layers as f64 * cache_size as f64 * mem_per_expert_mb;
        eprintln!("[objeta] Building expert caches ({cache_size} experts/layer × {total_layers} layers, ~{:.0}MB)...", est_mb);

        let mut total = 0i32;
        for l in 0..40 {
            let run_moe = self.layers[l].is_gqa || self.moe_on_deltanet;
            if !run_moe { continue; }
            let n = unsafe {
                crate::moe_dispatch::lko_moe_build_cache(
                    l as i32,
                    self.gu_mmaps[l].as_ptr(), self.gu_mmaps[l].len() as i32,
                    self.down_mmaps[l].as_ptr(), self.down_mmaps[l].len() as i32,
                    cache_size as i32,
                )
            };
            total += n;
        }
        self.expert_cache_size = cache_size;
        eprintln!("[objeta] Cached {total} experts across {total_layers} active layers.");
    }
}

// ── Weight loading helpers ────────────────────────────────────────────────

/// Load f16 weights from mmap (kept as f16 to save memory).
fn load_f16_raw(data: &memmap2::Mmap, off: usize, nelem: usize) -> Vec<u16> {
    let ptr = unsafe { data.as_ptr().add(off) as *const u16 };
    let slice = unsafe { std::slice::from_raw_parts(ptr, nelem) };
    slice.to_vec()
}

fn load_f16_to_f32(data: &memmap2::Mmap, off: usize, nelem: usize) -> Vec<f32> {
    let ptr = unsafe { data.as_ptr().add(off) as *const u16 };
    let slice = unsafe { std::slice::from_raw_parts(ptr, nelem) };
    slice.iter().map(|&h| f16_to_f32(h)).collect()
}

fn load_layer_weights(bin_dir: &Path, l: usize) -> Option<LayerWeights> {
    let json_path = bin_dir.join(format!("layer_{}_attn_f16.json", l));
    let mut json_str = String::new();
    std::fs::File::open(&json_path).ok()?.read_to_string(&mut json_str).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&json_str).ok()?;

    let bin_path = bin_dir.join(format!("layer_{}_attn_f16.bin", l));
    let file = std::fs::File::open(&bin_path).ok()?;
    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };

    let get_f16 = |name: &str| -> Option<Vec<u16>> {
        let arr = meta.get(name)?.as_array()?;
        let off = arr[1].as_u64()? as usize;
        let nb = arr[2].as_u64()? as usize;
        Some(load_f16_raw(&mmap, off, nb / 2))
    };
    let get_f32 = |name: &str| -> Option<Vec<f32>> {
        let arr = meta.get(name)?.as_array()?;
        let off = arr[1].as_u64()? as usize;
        let nb = arr[2].as_u64()? as usize;
        Some(load_f16_to_f32(&mmap, off, nb / 2))
    };

    let is_gqa = l % 4 == 3;
    let has_attn = get_f16("linear_attn.in_proj_qkv.weight").is_some() || get_f16("self_attn.q_proj.weight").is_some();

    let (w_qkv, qkv_M, qkv_K) = if is_gqa {
        let qw = get_f16("self_attn.q_proj.weight")?;
        let kw = get_f16("self_attn.k_proj.weight")?;
        let vw = get_f16("self_attn.v_proj.weight")?;
        let qM = qw.len() / HDIM;
        let kM = kw.len() / HDIM;
        let vM = vw.len() / HDIM;
        let mut cat = Vec::with_capacity(qw.len() + kw.len() + vw.len());
        cat.extend_from_slice(&qw); cat.extend_from_slice(&kw); cat.extend_from_slice(&vw);
        (cat, qM + kM + vM, HDIM)
    } else if has_attn {
        let w = get_f16("linear_attn.in_proj_qkv.weight")?;
        let M = w.len() / HDIM;
        (w, M, HDIM)
    } else {
        (Vec::new(), 0, 0)
    };

    let (w_o, o_M, o_K) = if is_gqa {
        let w = get_f16("self_attn.o_proj.weight")?;
        let M = w.len() / (16 * 256);
        (w, M, 16 * 256)
    } else if has_attn {
        let w = get_f16("linear_attn.out_proj.weight")?;
        let M = HDIM;
        let K = w.len() / HDIM;
        (w, M, K)
    } else {
        (Vec::new(), 0, 0)
    };

    let w_z = if has_attn && !is_gqa { get_f16("linear_attn.in_proj_z.weight")? } else { Vec::new() };
    let w_b = if has_attn && !is_gqa { get_f32("linear_attn.in_proj_b.weight")? } else { Vec::new() };
    let w_a = if has_attn && !is_gqa { get_f32("linear_attn.in_proj_a.weight")? } else { Vec::new() };
    let w_conv = if has_attn && !is_gqa {
        let w = get_f32("linear_attn.conv1d.weight")?; // (8192, 1, 4) → reshape to (8192, 4)
        w
    } else { Vec::new() };
    let w_norm = if has_attn && !is_gqa { get_f32("linear_attn.norm.weight")? } else { Vec::new() };
    let dt_bias = if has_attn && !is_gqa { get_f32("linear_attn.dt_bias")? } else { Vec::new() };
    let a_log = if has_attn && !is_gqa { get_f32("linear_attn.A_log")? } else { Vec::new() };
    let se_gate = get_f16("mlp.shared_expert.gate_proj.weight").unwrap_or_default();
    let se_up = get_f16("mlp.shared_expert.up_proj.weight").unwrap_or_default();
    let se_down = get_f16("mlp.shared_expert.down_proj.weight").unwrap_or_default();
    let se_gate_w = get_f32("mlp.shared_expert_gate.weight").unwrap_or_default();
    let input_norm = get_f32("input_layernorm.weight").unwrap_or_default();
    let post_norm = get_f32("post_attention_layernorm.weight").unwrap_or_default();

    Some(LayerWeights {
        w_qkv, w_o, w_z, w_b, w_a, w_conv, w_norm, dt_bias, a_log,
        se_gate, se_up, se_down, se_gate_w,
        input_norm, post_norm,
        is_gqa, has_attn, qkv_M, qkv_K, o_M, o_K,
    })
}

// ── lm_head + top-k sampling (in Rust) ────────────────────────────────────

impl Qwen36Runner {
    /// Compute logits = embed @ hn, return top-k indices + values.
    /// Uses NEON+rayon for the massive matmul (248320 × 2048 = 509M FLOPs).
    pub fn lm_head_topk(&self, hn: &[f32], top_k: usize) -> (Vec<i32>, Vec<f32>) {
        let vocab = self.embed.len() / (HDIM * 4); // f32 = 4 bytes

        // Compute logits in parallel (embed is mmap'd, access via raw pointer is safe for read-only)
        let embed_data: &[f32] = unsafe {
            std::slice::from_raw_parts(self.embed.as_ptr() as *const f32, vocab * HDIM)
        };
        let logits: Vec<f32> = (0..vocab).into_par_iter().map(|v| {
            dot_f32(&embed_data[v * HDIM..(v + 1) * HDIM], hn)
        }).collect();

        // Top-k selection (partial sort)
        let mut indexed: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
        let k = top_k.min(indexed.len());
        indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        let indices: Vec<i32> = indexed.iter().map(|(i, _)| *i as i32).collect();
        let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
        (indices, values)
    }
}

// ── C API for full executor ──────────────────────────────────────────────

static mut RUNNER: Option<Qwen36Runner> = None;

#[no_mangle]
pub extern "C" fn lko_runner_init(bin_dir: *const i8, max_seq: i32) -> i32 {
    let path = unsafe { std::ffi::CStr::from_ptr(bin_dir) }.to_string_lossy();
    let runner = Qwen36Runner::new(Path::new(path.as_ref()), max_seq as usize);
    match runner {
        Some(r) => { unsafe { RUNNER = Some(r); } 1 }
        None => 0,
    }
}

/// Set DeltaNet fusion ratio: 1.0 = all layers (default), 0.33 = 1 per GQA block.
#[no_mangle]
pub extern "C" fn lko_runner_set_fusion_ratio(ratio: f64) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                let r = r;
                r.fusion_ratio = ratio.clamp(0.0, 1.0);
                r.policy_table = build_policy_table(r.fusion_ratio, r.moe_on_deltanet);
                1
            }
            None => 0,
        }
    }
}

/// Run warmup to collect expert routing frequencies.
#[no_mangle]
pub extern "C" fn lko_runner_warmup(n_tokens: i32) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    runner.warmup(n_tokens as usize);
    1
}

/// Build per-layer expert caches from warmup data.
#[no_mangle]
pub extern "C" fn lko_runner_build_caches(cache_size: i32) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    runner.build_expert_caches(cache_size as usize);
    runner.expert_cache_size as i32
}

/// Skip MoE dispatch + shared expert on non-GQA (DeltaNet) layers.
#[no_mangle]
pub extern "C" fn lko_runner_set_moe_on_deltanet(enabled: i32) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                let r = r;
                r.moe_on_deltanet = enabled != 0;
                r.policy_table = build_policy_table(r.fusion_ratio, r.moe_on_deltanet);
                1
            }
            None => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_forward(
    token_id: i32, pos: i32, seq_len: i32,
    h_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let h = runner.forward(token_id as usize, pos as usize, seq_len as usize);
    unsafe { std::ptr::copy_nonoverlapping(h.as_ptr(), h_out, HDIM); }
    HDIM as i32
}

/// Forward pass through only the first N layers. Returns hidden state after N layers.
#[no_mangle]
pub extern "C" fn lko_runner_forward_n(
    token_id: i32, pos: i32, seq_len: i32, n_layers: i32,
    h_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let mut h = {
        let ptr = unsafe { runner.embed.as_ptr().add(token_id as usize * HDIM * 4) as *const f32 };
        (0..HDIM).map(|i| unsafe { *ptr.add(i) }).collect::<Vec<f32>>()
    };
    let n = n_layers.min(40) as usize;
    for l in 0..n {
        let policy = runner.policy_table[l];
        let lw = &runner.layers[l];
        if !lw.input_norm.is_empty() {
            h = rms_norm(&h, &lw.input_norm);
        }
        let ao = match policy.attn {
            AttnPolicy::Full => {
                if policy.is_steering {
                    if runner.metal_gqa_ok {
                        if let Some(ao) = gqa_metal_try(
                            &lw.w_qkv, &lw.w_o, &h,
                            pos as u32, seq_len as u32, runner.max_seq as u32,
                            &mut runner.kv_k[l], &mut runner.kv_v[l],
                            &mut runner.metal_gqa_first_fail,
                        ) { ao } else {
                            let mut ao = vec![0.0f32; HDIM];
                            gqa_attention_fused(
                                &lw.w_qkv, &lw.w_o, &h,
                                &mut runner.kv_k[l], &mut runner.kv_v[l],
                                &runner.rope_cos, &runner.rope_sin,
                                16, 2, HEAD_DIM, pos as usize, seq_len as usize, runner.max_seq,
                                &mut ao, &mut runner.scratch_qkv, &mut runner.scratch_q,
                                &mut runner.scratch_k, &mut runner.scratch_v,
                                &mut runner.scratch_attn_out,
                                &mut runner.scratch_scores, &mut runner.scratch_attn,
                            );
                            ao
                        }
                    } else {
                        let mut ao = vec![0.0f32; HDIM];
                        gqa_attention_fused(
                            &lw.w_qkv, &lw.w_o, &h,
                            &mut runner.kv_k[l], &mut runner.kv_v[l],
                            &runner.rope_cos, &runner.rope_sin,
                            16, 2, HEAD_DIM, pos as usize, seq_len as usize, runner.max_seq,
                            &mut ao, &mut runner.scratch_qkv, &mut runner.scratch_q,
                            &mut runner.scratch_k, &mut runner.scratch_v,
                            &mut runner.scratch_attn_out,
                            &mut runner.scratch_scores, &mut runner.scratch_attn,
                        );
                        ao
                    }
                } else {
                    let mut ao = vec![0.0f32; HDIM];
                    delta_net_fused(
                        &lw.w_qkv, &lw.w_z, &lw.w_b, &lw.w_a,
                        &lw.w_o, &lw.w_conv, &lw.w_norm,
                        &lw.dt_bias, &lw.a_log,
                        &h,
                        &mut runner.conv_states[l], &mut runner.conv_ptrs[l],
                        &mut runner.S_states[l],
                        &mut ao,
                        &mut runner.scratch_f32,
                    );
                    ao
                }
            }
            AttnPolicy::Collapse | AttnPolicy::Skip => { vec![0.0f32; HDIM] }
        };
        for i in 0..HDIM { h[i] += ao[i]; }
        if !lw.post_norm.is_empty() {
            h = rms_norm(&h, &lw.post_norm);
        }
        if policy.moe != MoEPolicy::Skip && !lw.se_gate.is_empty() {
            let gate = gemv_f16(&lw.se_gate, &h, 512, HDIM);
            let up = gemv_f16(&lw.se_up, &h, 512, HDIM);
            let mut hidden = gate.clone();
            for i in 0..512 { hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i]; }
            let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
            let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h)).exp());
            for i in 0..HDIM { h[i] += se_out[i] * se_gate; }
        }
        if policy.moe != MoEPolicy::Skip {
            let moe_out = runner.call_moe(&h, l);
            for i in 0..HDIM { h[i] += moe_out[i]; }
        }
    }
    unsafe { std::ptr::copy_nonoverlapping(h.as_ptr(), h_out, HDIM); }
    HDIM as i32
}

/// Compute lm_head + top-k in Rust. Returns top_k indices/values via output buffers.
#[no_mangle]
pub extern "C" fn lko_runner_lm_head(
    hn: *const f32,
    top_k: i32,
    indices_out: *mut i32,
    values_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_ref() }.expect("runner not initialized");
    let h_slice = unsafe { std::slice::from_raw_parts(hn, HDIM) };
    let (indices, values) = runner.lm_head_topk(h_slice, top_k as usize);
    let k = indices.len().min(top_k as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(indices.as_ptr(), indices_out, k);
        std::ptr::copy_nonoverlapping(values.as_ptr(), values_out, k);
    }
    k as i32
}

/// Profiled forward pass. Returns timing breakdown in `timing_out` (5 f64: delta, gqa, shared, moe, norm).
#[no_mangle]
pub extern "C" fn lko_runner_forward_timed(
    token_id: i32, pos: i32, seq_len: i32,
    h_out: *mut f32,
    timing_out: *mut f64,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let (h, timing) = runner.forward_timed(token_id as usize, pos as usize, seq_len as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(h.as_ptr(), h_out, HDIM);
        std::ptr::copy_nonoverlapping(timing.as_ptr(), timing_out, 5);
    }
    HDIM as i32
}

/// Full generation step: forward(hidden) + RMSNorm + lm_head.
#[no_mangle]
pub extern "C" fn lko_runner_step(
    token_id: i32, pos: i32, seq_len: i32,
    hn_out: *mut f32,
    top_k: i32,
    indices_out: *mut i32,
    values_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let (h, timing) = runner.forward_timed(token_id as usize, pos as usize, seq_len as usize);

    // RMSNorm
    let hn = rms_norm(&h, &runner.final_norm);
    unsafe { std::ptr::copy_nonoverlapping(hn.as_ptr(), hn_out, HDIM); }

    // lm_head top-k
    let (indices, values) = runner.lm_head_topk(&hn, top_k as usize);
    let k = indices.len().min(top_k as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(indices.as_ptr(), indices_out, k);
        std::ptr::copy_nonoverlapping(values.as_ptr(), values_out, k);
    }
    k as i32
}
