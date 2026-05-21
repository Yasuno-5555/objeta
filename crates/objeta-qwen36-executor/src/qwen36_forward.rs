//! Qwen3.6 full forward pass in Rust with NEON SIMD + rayon.

use rayon::prelude::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub const HDIM: usize = 2048;
pub(crate) const HEAD_DIM: usize = 256;
pub(crate) const N_KV: usize = 2;
pub(crate) const N_Q_ATTN: usize = 16; // Q heads for full GQA

pub(crate) const EXPERT_GATE_UP_BYTES: u64 = 1_310_720;
pub(crate) const EXPERT_DOWN_BYTES: u64 = 655_360;
pub(crate) const EXPERT_TOTAL_BYTES: u64 = EXPERT_GATE_UP_BYTES + EXPERT_DOWN_BYTES;

use crate::moe_stats::{LayerTrace, StepTrace, MoELayerStats, MoEIoEvent, ForwardLayerStats};

// ── Metal fused GQA FFI ──────────────────────────────────────────────────

extern "C" {
    // Init Metal GQA resources (RoPE tables, once)
    fn lko_metal_gqa_init(rope_cos: *const f32, rope_sin: *const f32, max_seq: i32) -> i32;
    // Load per-layer GQA weights into persistent Metal buffers
    fn lko_metal_gqa_load_weights(
        layer_idx: i32,
        w_qkv: *const u16,
        w_qkv_bytes: i32,
        w_o: *const u16,
        w_o_bytes: i32,
        q_norm: *const f32,
        q_norm_len: i32,
        k_norm: *const f32,
        k_norm_len: i32,
    ) -> i32;
    // Dispatch fused GQA (QKV + RoPE + attention + Q-gate) — returns attn_out (4096 f32)
    fn lko_metal_fused_gqa(
        layer_idx: i32,
        h: *const f32,
        pos: i32,
        seq_len: i32,
        max_seq: i32,
        k_cache: *mut f32,
        v_cache: *mut f32,
        kv_bytes: i32,
        attn_out: *mut f32,
    ) -> i32;
    // Dispatch GQA O-proj: output = W_o @ attn_out (f16 weights, f32 attn_out → f32 output)
    fn lko_metal_gqa_oproj(
        layer_idx: i32,
        w_o: *const u16,
        w_o_bytes: i32,
        attn_out: *const f32,
        output: *mut f32,
        m: i32,
        k: i32,
    ) -> i32;
}

/// Try Metal fused GQA. Returns Some(output) on success, None if Metal unavailable.
pub(crate) fn gqa_metal_try(
    layer_idx: usize,
    w_qkv: &[u16],
    w_o: &[u16],
    h: &[f32],
    pos: u32,
    seq_len: u32,
    max_seq: u32,
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    first_call: &mut bool,
) -> Option<Vec<f32>> {
    let mut attn_out = vec![0.0f32; 4096];
    let ok = unsafe {
        lko_metal_fused_gqa(
            layer_idx as i32,
            h.as_ptr(),
            pos as i32,
            seq_len as i32,
            max_seq as i32,
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
            (k_cache.len() * 4) as i32,
            attn_out.as_mut_ptr(),
        ) == 4096
    };
    if !ok {
        if *first_call {
            eprintln!("[objeta] Metal GQA: fused_gqa failed (kernel missing?)");
            *first_call = false;
        }
        return None;
    }
    let mut output = vec![0.0f32; HDIM];
    let ok = unsafe {
        lko_metal_gqa_oproj(
            layer_idx as i32,
            std::ptr::null(),
            0, // pass null to use pre-loaded GPU weight buffer!
            attn_out.as_ptr(),
            output.as_mut_ptr(),
            HDIM as i32,
            4096i32,
        ) == HDIM as i32
    };
    if !ok {
        if *first_call {
            eprintln!("[objeta] Metal GQA: oproj failed");
            *first_call = false;
        }
        return None;
    }
    Some(output)
}

fn rope_cache(max_seq: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    // Qwen3.6-35B-A3B uses partial_rotary_factor = 0.25, rope_theta = 10_000_000
    let rotary_dim = (hd as f32 * 0.25) as usize;
    let half_rot = rotary_dim / 2;
    let mut cos = vec![0.0f32; max_seq * half_rot];
    let mut sin = vec![0.0f32; max_seq * half_rot];
    for pos in 0..max_seq {
        for i in 0..half_rot {
            let theta = 1.0 / 10000000.0f32.powf(2.0 * i as f32 / rotary_dim as f32);
            cos[pos * half_rot + i] = (pos as f32 * theta).cos();
            sin[pos * half_rot + i] = (pos as f32 * theta).sin();
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

extern "C" {
    fn cpu_fast_f16_gemv(w: *const u16, x: *const f32, y: *mut f32, m: usize, k: usize);
}

/// Direct f16 GEMV using manual f16→f32 + NEON FMA — zero intermediate allocation.
/// Reads f16 weights without full f32 conversion buffer. 2x less memory BW.
pub fn gemv_f16_direct(W: &[u16], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; M];
    if M < 64 {
        for i in 0..M {
            let row = &W[i * K..(i + 1) * K];
            unsafe {
                cpu_fast_f16_gemv(row.as_ptr(), x.as_ptr(), &mut y[i], 1, K);
            }
        }
    } else {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let row = &W[i * K..(i + 1) * K];
            unsafe {
                cpu_fast_f16_gemv(row.as_ptr(), x.as_ptr(), yi as *mut f32, 1, K);
            }
        });
    }
    y
}

// ── GEMV f16 (f16→f32 conversion + NEON GEMV) ───────────────────────────

pub fn gemv_f16_buf(W: &[u16], x: &[f32], M: usize, K: usize, _buf: &mut Vec<f32>) -> Vec<f32> {
    gemv_f16_direct(W, x, M, K)
}

pub fn gemv_f16(W: &[u16], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    gemv_f16_direct(W, x, M, K)
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
        for r in j..n {
            s += a[r] * b[r];
        }
        return s;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = (h >> 10) & 0x1f;
    let m = h as u32 & 0x3ff;
    if e == 0 {
        if m == 0 {
            f32::from_bits(s)
        } else {
            (m as f32) * 2f32.powi(-24) * if s == 0 { 1.0 } else { -1.0 }
        }
    } else if e == 31 {
        if m == 0 {
            f32::from_bits(s | 0x7f80_0000)
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits(s | (((e + 112) as u32) << 23) | (m << 13))
    }
}

// ── RMSNorm ───────────────────────────────────────────────────────────────

pub fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let n = x.len();
    let msq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (msq + 1e-6).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v * inv * w)
        .collect()
}

pub fn rms_norm_offset(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let n = x.len();
    let msq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (msq + 1e-6).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v * inv * (1.0 + w))
        .collect()
}



// ── Softmax (in-place, per row of dim) ───────────────────────────────────

pub fn softmax_inplace(x: &mut [f32], dim: usize) {
    for chunk in x.chunks_mut(dim) {
        let max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in chunk.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in chunk.iter_mut() {
            *v /= sum.max(1e-12);
        }
    }
}


// ── SiLU (in-place) ──────────────────────────────────────────────────────

pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}


// ── L2 Normalize (per row of dim) ────────────────────────────────────────

pub fn l2_norm_rows(x: &mut [f32], rows: usize, dim: usize) {
    let inv_scale = 1.0 / (dim as f32).sqrt();
    for row in x.chunks_mut(dim) {
        let sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sq + 1e-6).sqrt() * inv_scale;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
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
    S: &mut [f32],
    k: &[f32],
    q: &[f32],
    v: &[f32],
    beta: &[f32],
    exp_g: &[f32],
    n_heads: usize,
    kv: usize,
    vd: usize,
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
            for s in S_h.iter_mut() {
                *s *= g_h;
            }

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


// ── RMSNormGated ─────────────────────────────────────────────────────────
// output: (n_heads, V) — apply RMSNorm per head, then gate with z * sigmoid(z)

pub fn rms_norm_gated(
    output: &[f32],
    z: &[f32],
    w_norm: &[f32],
    n_heads: usize,
    v_dim: usize,
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


// ── Sigmoid (element-wise) ───────────────────────────────────────────────

pub fn sigmoid_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}


// ── GQA Attention (fused: QKV gemv + RoPE + softmax + weighted sum + output proj) ──

/// Rayon-parallel f16 GEMV: converts f16→f32 on the fly, zero intermediate buffer.
fn gemv_f16_par(W: &[u16], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    (0..M)
        .into_par_iter()
        .map(|row| {
            let r = &W[row * K..(row + 1) * K];
            r.iter()
                .zip(x.iter())
                .map(|(&h, &xv)| f16_to_f32(h) * xv)
                .sum()
        })
        .collect()
}

/// Full GQA attention in Rust. All buffers pre-allocated.
/// Returns via `output` (n_heads * head_dim f32).
pub fn gqa_attention_fused(
    // Weights (f32, pre-loaded)
    w_qkv: &[u16],
    w_o: &[u16],
    q_norm: &[f32],
    k_norm: &[f32],
    // Input
    h: &[f32],
    // KV cache
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    // RoPE
    rope_cos: &[f32],
    rope_sin: &[f32],
    // Dimensions
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    pos: usize,
    seq_len: usize,
    max_seq: usize,
    // Output
    output: &mut [f32],
    // Scratch buffers
    qkv_buf: &mut [f32],
    q_buf: &mut [f32],
    k_buf: &mut [f32],
    v_buf: &mut [f32],
    attn_out: &mut [f32],
    scores: &mut [f32],
    attn_w: &mut [f32],
) {
    let q_proj_sz = n_heads * head_dim * 2; // 8192 = query(256) + gate(256) per head
    let q_sz = n_heads * head_dim; // 4096
    let k_sz = n_kv * head_dim; // 512
    let v_sz = n_kv * head_dim; // 512
    let total = q_proj_sz + k_sz + v_sz; // 9216
    let K = h.len();

    // QKV projection
    qkv_buf[..total].copy_from_slice(&gemv_f16(w_qkv, h, total, K));

    if objeta_debug_enabled() && pos == seq_len - 1 {
        let h_norm_val: f32 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
        let qkv_norm: f32 = qkv_buf[..total].iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("  [RUST GQA DEBUG] h norm = {:.6}", h_norm_val);
        println!("  [RUST GQA DEBUG] qkv_buf norm = {:.6}", qkv_norm);
    }

    // Split q_proj into per-head query/gate chunks matching HF:
    // q_proj.view(..., 16, 512).chunk(2, dim=-1) -> query[16,256], gate[16,256]
    let mut q_gate = vec![0.0f32; q_sz];
    for h in 0..n_heads {
        let src = h * head_dim * 2;
        let dst = h * head_dim;
        q_buf[dst..dst + head_dim].copy_from_slice(&qkv_buf[src..src + head_dim]);
        for d in 0..head_dim {
            q_gate[dst + d] = 1.0 / (1.0 + (-qkv_buf[src + head_dim + d]).exp());
        }
    }
    k_buf[..k_sz].copy_from_slice(&qkv_buf[q_proj_sz..q_proj_sz + k_sz]);
    v_buf[..v_sz].copy_from_slice(&qkv_buf[q_proj_sz + k_sz..total]);

    if objeta_debug_enabled() && pos == seq_len - 1 {
        let q_proj_norm: f32 = q_buf[..q_sz].iter().map(|v| v * v).sum::<f32>().sqrt();
        let k_proj_norm: f32 = k_buf[..k_sz].iter().map(|v| v * v).sum::<f32>().sqrt();
        let v_proj_norm: f32 = v_buf[..v_sz].iter().map(|v| v * v).sum::<f32>().sqrt();
        println!(
            "  [RUST GQA DEBUG] q_proj norm = {:.6}, k_proj norm = {:.6}, v_proj norm = {:.6}",
            q_proj_norm, k_proj_norm, v_proj_norm
        );
    }

    // Per-head RMSNorm for Q/K, matching HF Qwen3MoeAttention (which uses 1.0 + weight offset).
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut sq = 0.0f32;
        for d in 0..head_dim {
            sq += q_buf[base + d] * q_buf[base + d];
        }
        let inv = 1.0 / (sq / head_dim as f32 + 1e-6).sqrt();
        for d in 0..head_dim {
            q_buf[base + d] = q_buf[base + d] * inv * (1.0 + q_norm[d]);
        }
    }
    for h in 0..n_kv {
        let base = h * head_dim;
        let mut sq = 0.0f32;
        for d in 0..head_dim {
            sq += k_buf[base + d] * k_buf[base + d];
        }
        let inv = 1.0 / (sq / head_dim as f32 + 1e-6).sqrt();
        for d in 0..head_dim {
            k_buf[base + d] = k_buf[base + d] * inv * (1.0 + k_norm[d]);
        }
    }

    if objeta_debug_enabled() && pos == seq_len - 1 {
        let q_norm_norm: f32 = q_buf[..q_sz].iter().map(|v| v * v).sum::<f32>().sqrt();
        let k_norm_norm: f32 = k_buf[..k_sz].iter().map(|v| v * v).sum::<f32>().sqrt();
        println!(
            "  [RUST GQA DEBUG] q_norm norm = {:.6}, k_norm norm = {:.6}",
            q_norm_norm, k_norm_norm
        );
    }

    // RoPE (partial_rotary_factor = 0.25)
    let rotary_dim = (head_dim as f32 * 0.25) as usize;
    let half_rot = rotary_dim / 2;
    let c = &rope_cos[pos * half_rot..(pos + 1) * half_rot];
    let s = &rope_sin[pos * half_rot..(pos + 1) * half_rot];
    for h in 0..n_heads {
        for i in 0..half_rot {
            let qe = q_buf[h * head_dim + i];
            let qo = q_buf[h * head_dim + half_rot + i];
            q_buf[h * head_dim + i] = qe * c[i] - qo * s[i];
            q_buf[h * head_dim + half_rot + i] = qe * s[i] + qo * c[i];
        }
    }
    for h in 0..n_kv {
        for i in 0..half_rot {
            let ke = k_buf[h * head_dim + i];
            let ko = k_buf[h * head_dim + half_rot + i];
            k_buf[h * head_dim + i] = ke * c[i] - ko * s[i];
            k_buf[h * head_dim + half_rot + i] = ke * s[i] + ko * c[i];
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
            let kt = &k_cache
                [(kv_h * max_seq + t) * head_dim..(kv_h * max_seq + t) * head_dim + head_dim];
            scores[t] = dot_f32(qh, kt) * scale;
            if scores[t] > max_s {
                max_s = scores[t];
            }
        }

        // Softmax
        let mut sum = 0.0;
        for t in 0..seq_len {
            attn_w[t] = (scores[t] - max_s).exp();
            sum += attn_w[t];
        }
        for t in 0..seq_len {
            attn_w[t] /= sum.max(1e-12);
        }

        // Weighted sum
        for t in 0..seq_len {
            let vt = &v_cache
                [(kv_h * max_seq + t) * head_dim..(kv_h * max_seq + t) * head_dim + head_dim];
            let a = attn_w[t];
            for d in 0..head_dim {
                oh[d] += a * vt[d];
            }
        }

        let g_off = h * head_dim;
        for d in 0..head_dim {
            oh[d] *= q_gate[g_off + d];
        }
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
    w_qkv: &[u16],
    w_z: &[u16],
    w_b: &[f32],
    w_a: &[f32],
    w_out: &[u16],
    w_conv: &[f32],
    w_norm: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    h: &[f32],
    conv_state: &mut [f32],
    conv_ptr: &mut usize,
    S_state: &mut [f32],
    ao_out: &mut [f32],
    scratch_f32: &mut Vec<f32>,
    layer_idx: usize,
    pos: usize,
) {
    let mixed_qkv = gemv_f16_buf(w_qkv, h, 8192, HDIM, scratch_f32);
    let z = gemv_f16_buf(w_z, h, 4096, HDIM, scratch_f32);
    let b = gemv_f32(w_b, h, 32, HDIM);
    let a_vec = gemv_f32(w_a, h, 32, HDIM);

    // Conv1d ring buffer
    let ptr = *conv_ptr;
    for c in 0..8192 {
        conv_state[c * 4 + ptr] = mixed_qkv[c];
    }
    let new_ptr = (ptr + 1) % 4;
    *conv_ptr = new_ptr;
    let order = [(ptr + 1) % 4, (ptr + 2) % 4, (ptr + 3) % 4, ptr]; // weight[3]=newest

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
            q_rep[h * 128 + d] = q[sh * 128 + d];
            k_rep[h * 128 + d] = k[sh * 128 + d];
        }
    }

    let beta: Vec<f32> = b.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let exp_g: Vec<f32> = (0..32)
        .map(|i| {
            let sp = (a_vec[i] + dt_bias[i]).max(-10.0).exp().ln_1p();
            (-a_log[i].exp() * sp).exp()
        })
        .collect();

    if objeta_debug_enabled() && layer_idx == 0 && pos == 0 {
        let mqkv_norm = mixed_qkv.iter().map(|v| v * v).sum::<f32>().sqrt();
        let z_norm = z.iter().map(|v| v * v).sum::<f32>().sqrt();
        let b_norm = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        let a_vec_norm = a_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        let qkv_a_norm = qkv_conv.iter().map(|v| v * v).sum::<f32>().sqrt();
        let beta_norm = beta.iter().map(|v| v * v).sum::<f32>().sqrt();
        let exp_g_norm = exp_g.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("[RUST DETAILED L0] mqkv norm: {:.6}", mqkv_norm);
        println!("[RUST DETAILED L0] z norm: {:.6}", z_norm);
        println!("[RUST DETAILED L0] b norm: {:.6}", b_norm);
        println!("[RUST DETAILED L0] a_vec norm: {:.6}", a_vec_norm);
        println!("[RUST DETAILED L0] qkv_a norm: {:.6}", qkv_a_norm);
        println!("[RUST DETAILED L0] beta norm: {:.6}", beta_norm);
        println!("[RUST DETAILED L0] exp(g) norm: {:.6}", exp_g_norm);
    }

    // L2 norm q,k
    for h in 0..32 {
        let mut qn = 0.0f32;
        let mut kn = 0.0f32;
        for d in 0..128 {
            qn += q_rep[h * 128 + d].powi(2);
            kn += k_rep[h * 128 + d].powi(2);
        }
        let qs = 1.0 / (qn.sqrt() + 1e-6) / (128.0f32).sqrt();
        let ks = 1.0 / (kn.sqrt() + 1e-6);
        for d in 0..128 {
            q_rep[h * 128 + d] *= qs;
            k_rep[h * 128 + d] *= ks;
        }
    }

    // Delta state update
    let mut output = vec![0.0f32; 32 * 128];
    delta_state_update(
        S_state,
        &k_rep,
        &q_rep,
        &v_rs,
        &beta,
        &exp_g,
        32,
        128,
        128,
        &mut output,
    );

    // RMSNormGated
    let gated = rms_norm_gated(&output, &z_rs, w_norm, 32, 128);

    // Output projection
    let ao = gemv_f16_buf(w_out, &gated, HDIM, 4096, scratch_f32);
    ao_out.copy_from_slice(&ao);

    if objeta_debug_enabled() && layer_idx == 0 && pos == 0 {
        let q_norm = q_rep.iter().map(|v| v * v).sum::<f32>().sqrt();
        let k_norm = k_rep.iter().map(|v| v * v).sum::<f32>().sqrt();
        let s_norm = S_state.iter().map(|v| v * v).sum::<f32>().sqrt();
        let out_norm = output.iter().map(|v| v * v).sum::<f32>().sqrt();
        let gated_norm = gated.iter().map(|v| v * v).sum::<f32>().sqrt();
        let ao_norm = ao.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("[RUST DETAILED L0] normalized q norm: {:.6}", q_norm);
        println!("[RUST DETAILED L0] normalized k norm: {:.6}", k_norm);
        println!("[RUST DETAILED L0] S_state norm: {:.6}", s_norm);
        println!("[RUST DETAILED L0] output norm: {:.6}", out_norm);
        println!("[RUST DETAILED L0] gated norm: {:.6}", gated_norm);
        println!("[RUST DETAILED L0] ao norm: {:.6}", ao_norm);
    }
}

/// One fused layer forward call.
/// Returns attention output in `ao_out` (HDIM f32).
/// Updates kv_cache or delta_state in place.

// ═══════════════════════════════════════════════════════════════════════════
// LKO Scheduler: Phase-aware execution policy
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum AttnPolicy {
    Full,     // Full GQA or DeltaNet forward
    Collapse, // Koopman identity (skip, J≈I)
    Skip,     // No attention at all
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum MoEPolicy {
    Full,     // Dequantize + GEMV all routed experts
    Adaptive, // Entropy-conditioned top-k pruning
    Skip,     // Skip MoE entirely
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerPolicy {
    pub(crate) attn: AttnPolicy,
    pub(crate) moe: MoEPolicy,
    pub(crate) precision_bits: u8, // target precision for weights (3-16)
    pub(crate) is_steering: bool,  // is this a GQA course-correction layer?
}

/// Build static policy table for Qwen3.6-35B-A3B based on LKO phase measurements.
/// GQA at layers 3,7,11,15,19,23,27,31,35,39 (every 4th).
/// UNFOLD: L0-L2 (sacred), ISOMETRIC: L3-L35, DIVERGENT: L36-L39.
pub(crate) fn build_policy_table(fusion_ratio: f64, moe_on_deltanet: bool) -> [LayerPolicy; 40] {
    let stride = (1.0 / fusion_ratio.max(0.01)).round() as usize;
    let mut delta_count: usize = 0;
    let mut table = [LayerPolicy {
        attn: AttnPolicy::Full,
        moe: MoEPolicy::Full,
        precision_bits: 16,
        is_steering: false,
    }; 40];

    for l in 0..40 {
        let is_gqa = l % 4 == 3;
        let phase = if l < 3 {
            "unfold"
        } else if l > 35 {
            "divergent"
        } else {
            "isometric"
        };

        let (attn, moe, prec, steering) = if is_gqa {
            delta_count = 0;
            // GQA: course correction — always full, high precision
            (AttnPolicy::Full, MoEPolicy::Adaptive, 16, true)
        } else {
            delta_count += 1;
            let compute = delta_count % stride.max(1) == 0;
            let attn = if compute {
                AttnPolicy::Full
            } else {
                AttnPolicy::Collapse
            };
            let moe = if moe_on_deltanet {
                MoEPolicy::Adaptive
            } else {
                MoEPolicy::Skip
            };
            let prec = match phase {
                "unfold" => 16,
                "divergent" => 8,
                _ => 4, // ISOMETRIC: low precision (q4)
            };
            (attn, moe, prec, false)
        };

        table[l] = LayerPolicy {
            attn,
            moe,
            precision_bits: prec,
            is_steering: steering,
        };
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
pub(crate) struct LayerWeights {
    // Large weights: f16 (half memory). Total per layer: ~72MB vs ~145MB for f32.
    pub(crate) w_qkv: Vec<u16>,
    pub(crate) w_o: Vec<u16>,
    pub(crate) w_z: Vec<u16>,
    pub(crate) se_gate: Vec<u16>,
    pub(crate) se_up: Vec<u16>,
    pub(crate) se_down: Vec<u16>,
    // Small weights: f32 (negligible memory)
    pub(crate) w_b: Vec<f32>,
    pub(crate) w_a: Vec<f32>,
    pub(crate) w_conv: Vec<f32>,
    pub(crate) w_norm: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) a_log: Vec<f32>,
    pub(crate) se_gate_w: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) input_norm: Vec<f32>,
    pub(crate) post_norm: Vec<f32>,
    pub(crate) is_gqa: bool,
    pub(crate) has_attn: bool,
    pub(crate) qkv_M: usize,
    pub(crate) qkv_K: usize,
    pub(crate) o_M: usize,
    pub(crate) o_K: usize,
}

use std::collections::BTreeSet;
use std::sync::Arc;

fn resident_cache_enabled(capacity: usize) -> bool {
    capacity > 0
}



fn objeta_timing_enabled() -> bool {
    std::env::var("OBJETA_TIMING")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
}

#[inline]
pub(crate) fn objeta_debug_enabled() -> bool {
    std::env::var("OBJETA_DEBUG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeConfigSource {
    Defaults,
    StrategyConfig,
    EnvDebugOverride,
    RuntimeProfile,
}

impl RuntimeConfigSource {
    pub(crate) fn rank(self) -> u8 {
        match self {
            Self::Defaults => 0,
            Self::StrategyConfig => 1,
            Self::EnvDebugOverride => 2,
            Self::RuntimeProfile => 3,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Defaults => "defaults",
            Self::StrategyConfig => "strategy_config",
            Self::EnvDebugOverride => "env_debug_override",
            Self::RuntimeProfile => "runtime_profile",
        }
    }
}

pub struct Qwen36Runner {
    pub(crate) embed: memmap2::Mmap,           // mmap'd embed_tokens.bin (2GB, zero-copy)
    pub(crate) lm_head: Option<memmap2::Mmap>, // mmap'd lm_head.bin when embeddings are untied
    pub(crate) final_norm: Vec<f32>,
    pub(crate) layers: Vec<LayerWeights>,
    // KV caches
    pub(crate) kv_k: Vec<Vec<f32>>, // per layer: (n_kv × max_seq × head_dim)
    pub(crate) kv_v: Vec<Vec<f32>>,
    // DeltaNet states
    pub(crate) conv_states: Vec<Vec<f32>>, // per layer: (8192 × 4)
    pub(crate) conv_ptrs: Vec<usize>,
    pub(crate) S_states: Vec<Vec<f32>>, // per layer: (32 × 128 × 128)
    // RoPE
    pub(crate) rope_cos: Vec<f32>,
    pub(crate) rope_sin: Vec<f32>,
    // MoE: pre-loaded routers + cached mmaps
    pub(crate) routers: Vec<Vec<f32>>,
    pub(crate) gu_mmaps: Vec<memmap2::Mmap>,
    pub(crate) down_mmaps: Vec<memmap2::Mmap>,
    // Scratch buffers (reused per forward pass)
    pub(crate) scratch_qkv: Vec<f32>,
    pub(crate) scratch_q: Vec<f32>,
    pub(crate) scratch_k: Vec<f32>,
    pub(crate) scratch_v: Vec<f32>,
    pub(crate) scratch_attn_out: Vec<f32>,
    pub(crate) scratch_scores: Vec<f32>,
    pub(crate) scratch_attn: Vec<f32>,
    pub(crate) scratch_f32: Vec<f32>, // reusable f16→f32 conversion buffer
    pub(crate) max_seq: usize,
    /// DeltaNet fusion: fraction of DeltaNet layers to compute (1.0=all, 0.33=1 per GQA block)
    pub fusion_ratio: f64,
    /// Skip MoE+shared expert on non-GQA (DeltaNet) layers
    pub moe_on_deltanet: bool,
    /// Whether routed MoE is globally enabled (used for isolation debugging)
    pub moe_enabled: bool,
    /// Scheduler: phase-aware execution policy per layer
    pub(crate) policy_table: [LayerPolicy; 40],
    /// Whether Metal fused GQA is available (tested at init)
    pub metal_gqa_ok: bool,
    pub(crate) metal_gqa_first_fail: bool,
    /// Expert residency manager for explicit memory limit and LRU/score eviction
    pub expert_residency_manager: crate::expert_cache::ExpertResidencyManager,
    /// Expert cache status: number of experts cached per layer (0 = not built)
    pub expert_cache_size: usize,
    /// Per-layer expert frequency data collected during warmup
    pub(crate) expert_freq_ready: bool,
    /// Pre-allocated scratch buffers for MoE GEMV (reused, zero allocation)
    pub(crate) moe_gate_buf: Vec<f32>, // 512
    pub(crate) moe_up_buf: Vec<f32>,     // 512
    pub(crate) moe_hidden_buf: Vec<f32>, // 512
    pub(crate) moe_down_buf: Vec<f32>,   // 2048
    pub(crate) fused_moe_scratch: crate::moe_dispatch::FusedMoeScratch,

    // Trace recording and replay fields
    pub record_trace_path: Option<String>,
    pub replay_traces: Option<Vec<StepTrace>>,
    pub current_step_trace: Option<StepTrace>,
    pub step_counter: usize,
    pub active_token_id: Option<usize>,
    pub decode_started: bool,
    pub decode_token_count: usize,
    pub current_governor_phase: crate::runtime_governor::GovernorPhase,
    pub last_repetition_kind: Option<crate::runtime_governor::RepetitionKind>,

    // Stats tracking fields
    pub moe_stats: Vec<MoELayerStats>,
    pub forward_stats: Vec<ForwardLayerStats>,
    pub lm_head_calls: u64,
    pub lm_head_wall_sec: f64,
    pub forward_calls: u64,
    pub forward_wall_sec: f64,
    pub moe_io_events: Vec<MoEIoEvent>,

    // Expert Policy config fields
    pub expert_policy: crate::strategy::ExpertPolicyConfig,
    pub moe_prune_mode: i32,
    pub moe_top_p: f32,
    pub moe_contrib_threshold: f32,
    pub min_experts: usize,
    pub max_experts: usize,
    pub moe_ema_output_norm: Vec<Vec<f32>>,
    // Debug overrides
    pub debug_force_attn_full: bool,
    pub debug_force_moe_skip: bool,
    pub use_fused_moe: bool,
    pub fused_down_mode: crate::moe_dispatch::FusedDownMode,
    pub last_decode_entropy: f32,
    pub last_predicted_token_id: i32,
    pub last_repetition_risk: bool,
    pub last_collapse_risk: bool,
    pub runtime_profile: crate::runtime_profile::RuntimeProfile,
    pub runtime_pack_status: crate::runtime_pack::RuntimePackLoadStatus,
    pub runtime_governor: crate::runtime_governor::RuntimeGovernor,
    pub runtime_config_source: RuntimeConfigSource,
    pub os_telemetry: crate::os_telemetry::MacOsMemoryTelemetryCollector,
}

impl Qwen36Runner {
    pub fn new(bin_dir: &Path, max_seq: usize) -> Option<Self> {
        // mmap embed to save 2GB RAM
        let embed_path = bin_dir.join("embed_tokens.bin");
        let embed_file = std::fs::File::open(&embed_path).ok()?;
        let embed = unsafe { memmap2::Mmap::map(&embed_file).ok()? };
        let _n_vocab = embed.len() / (HDIM * 4); // f32 = 4 bytes

        let lm_head = {
            let path = bin_dir.join("lm_head.bin");
            if path.exists() {
                let file = std::fs::File::open(&path).ok()?;
                eprintln!("[objeta] lm_head loaded from lm_head.bin");
                Some(unsafe { memmap2::Mmap::map(&file).ok()? })
            } else {
                eprintln!("[objeta] lm_head.bin missing, falling back to tied embed weights");
                None
            }
        };

        let norm_bytes = std::fs::read(bin_dir.join("final_norm.bin")).ok()?;
        let final_norm: Vec<f32> = norm_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Load all 40 layers
        let mut layers = Vec::with_capacity(40);
        for l in 0..40 {
            layers.push(load_layer_weights(bin_dir, l)?);
        }

        // Build scheduler policy table
        let policy_table = build_policy_table(0.33, false);

        // Apply strategy.json if present (family-aware precision).
        // For correctness debugging, allow disabling it without changing files.
        let disable_strategy = std::env::var("OBJETA_DISABLE_STRATEGY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !disable_strategy {
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
        } else {
            eprintln!("[objeta] strategy disabled via OBJETA_DISABLE_STRATEGY");
        }

        // KV caches (only for full GQA layers, l % 4 == 3)
        let kv_size = 2 * max_seq * HEAD_DIM; // n_kv=2, hd=256
        let kv_k: Vec<_> = (0..40)
            .map(|l| {
                if l % 4 == 3 {
                    vec![0.0f32; kv_size]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let kv_v: Vec<_> = (0..40)
            .map(|l| {
                if l % 4 == 3 {
                    vec![0.0f32; kv_size]
                } else {
                    Vec::new()
                }
            })
            .collect();

        // DeltaNet states (for linear attention layers)
        let conv_states: Vec<_> = (0..40)
            .map(|l| {
                if l % 4 != 3 && layers[l].has_attn {
                    vec![0.0f32; 8192 * 4]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let conv_ptrs = vec![0usize; 40];
        let S_states: Vec<_> = (0..40)
            .map(|l| {
                if l % 4 != 3 && layers[l].has_attn {
                    vec![0.0f32; 32 * 128 * 128]
                } else {
                    Vec::new()
                }
            })
            .collect();

        let (rope_cos, rope_sin) = rope_cache(max_seq, HEAD_DIM);

        // Init Metal GQA persistent resources (RoPE tables, once)
        let metal_gqa_ok = false;
        let _ = unsafe { lko_metal_gqa_init(rope_cos.as_ptr(), rope_sin.as_ptr(), max_seq as i32) };
        eprintln!("[objeta] Metal GQA: disabled pending kernel parity, using CPU fallback");

        // Pre-load routers + mmap MoE weights
        let mut routers = Vec::with_capacity(40);
        let mut gu_mmaps = Vec::with_capacity(40);
        let mut down_mmaps = Vec::with_capacity(40);
        for l in 0..40 {
            let rpath = bin_dir.join(format!("layer_{}_router.bin", l));
            let rbytes = std::fs::read(&rpath).unwrap_or_default();
            let r: Vec<f32> = rbytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            routers.push(r);

            let gu_f =
                std::fs::File::open(bin_dir.join(format!("layer_{}_gate_up.bin", l))).unwrap();
            let d_f = std::fs::File::open(bin_dir.join(format!("layer_{}_down.bin", l))).unwrap();
            gu_mmaps.push(unsafe { memmap2::Mmap::map(&gu_f).unwrap() });
            down_mmaps.push(unsafe { memmap2::Mmap::map(&d_f).unwrap() });
        }

        // Init per-layer expert caches
        unsafe {
            crate::moe_dispatch::lko_moe_init_caches(40);
        }
        // Init per-layer frequency trackers
        unsafe {
            crate::moe_dispatch::lko_moe_init_freq_tracker(40);
        }

        let mut runner = Qwen36Runner {
            embed,
            lm_head,
            final_norm,
            layers,
            kv_k,
            kv_v,
            conv_states,
            conv_ptrs,
            S_states,
            rope_cos,
            rope_sin,
            routers,
            gu_mmaps,
            down_mmaps,
            policy_table,
            scratch_qkv: vec![0.0f32; 9216],
            scratch_q: vec![0.0f32; 16 * 256],
            scratch_k: vec![0.0f32; 2 * 256],
            scratch_v: vec![0.0f32; 2 * 256],
            scratch_attn_out: vec![0.0f32; 16 * 256],
            scratch_scores: vec![0.0f32; max_seq],
            scratch_attn: vec![0.0f32; max_seq],
            scratch_f32: Vec::with_capacity(20_000_000), // ~80MB, for largest f16→f32 GEMV
            max_seq,
            fusion_ratio: 0.33,
            moe_on_deltanet: false, // matches initial policy_table
            moe_enabled: true,
            metal_gqa_ok,
            metal_gqa_first_fail: true,
            expert_residency_manager: crate::expert_cache::ExpertResidencyManager::new(0),
            expert_cache_size: 0,
            expert_freq_ready: false,
            moe_gate_buf: vec![0.0f32; 512],
            moe_up_buf: vec![0.0f32; 512],
            moe_hidden_buf: vec![0.0f32; 512],
            moe_down_buf: vec![0.0f32; HDIM],
            fused_moe_scratch: crate::moe_dispatch::FusedMoeScratch::default(),

            record_trace_path: None,
            replay_traces: None,
            current_step_trace: None,
            step_counter: 0,
            active_token_id: None,
            decode_started: false,
            decode_token_count: 0,
            current_governor_phase: crate::runtime_governor::GovernorPhase::Prefill,
            last_repetition_kind: None,

            moe_stats: vec![MoELayerStats::default(); 40],
            forward_stats: vec![ForwardLayerStats::default(); 40],
            lm_head_calls: 0,
            lm_head_wall_sec: 0.0,
            forward_calls: 0,
            forward_wall_sec: 0.0,
            moe_io_events: Vec::new(),

            expert_policy: crate::strategy::ExpertPolicyConfig::Exact,
            moe_prune_mode: 0,
            moe_top_p: 1.0,
            moe_contrib_threshold: 1.0,
            min_experts: 2,
            max_experts: 8,
            moe_ema_output_norm: vec![vec![1.0f32; 256]; 40],
            debug_force_attn_full: false,
            debug_force_moe_skip: false,
            use_fused_moe: std::env::var("OBJETA_USE_FUSED_MOE")
                .map(|v| v == "1")
                .unwrap_or(false),
            fused_down_mode: crate::moe_dispatch::FusedDownMode::from_env_default(),
            last_decode_entropy: 0.0,
            last_predicted_token_id: -1,
            last_repetition_risk: false,
            last_collapse_risk: false,
            runtime_profile: crate::runtime_profile::RuntimeProfile::default(),
            runtime_pack_status: crate::runtime_pack::RuntimePackLoadStatus::default(),
            runtime_governor: crate::runtime_governor::RuntimeGovernor::from_env(),
            runtime_config_source: RuntimeConfigSource::Defaults,
            os_telemetry: crate::os_telemetry::MacOsMemoryTelemetryCollector::default(),
        };

        if std::env::var_os("OBJETA_USE_FUSED_MOE").is_some()
            || std::env::var_os("OBJETA_FUSED_MOE_VARIANT").is_some()
            || std::env::var_os("OBJETA_RESIDENCY_GROUP_SIZE").is_some()
            || std::env::var_os("OBJETA_GROUP_PRERESOLVE_TOP_N").is_some()
            || std::env::var_os("OBJETA_GROUP_PRERESOLVE_MAX_BYTES").is_some()
        {
            runner.note_runtime_config_source(RuntimeConfigSource::EnvDebugOverride);
        }

        if let Ok(pack_path) = std::env::var("OBJETA_RUNTIME_PACK_PATH") {
            if !pack_path.trim().is_empty() {
                let _ = runner.load_runtime_pack(Path::new(pack_path.trim()));
            }
        }

        if let Ok(profile_path) = std::env::var("OBJETA_RUNTIME_PROFILE_PATH") {
            if !profile_path.trim().is_empty() {
                if let Ok(profile) =
                    crate::runtime_profile::load_runtime_profile(Path::new(profile_path.trim()))
                {
                    crate::runtime_profile::apply_runtime_profile(&mut runner, &profile);
                }
            }
        }

        Some(runner)
    }

    pub fn sync_legacy_policy_fields(&mut self) {
        match &self.expert_policy {
            crate::strategy::ExpertPolicyConfig::Exact => {
                self.moe_prune_mode = 0;
                self.moe_top_p = 1.0;
                self.moe_contrib_threshold = 1.0;
            }
            crate::strategy::ExpertPolicyConfig::LkoAware => {
                self.moe_prune_mode = 0;
                self.moe_top_p = 1.0;
                self.moe_contrib_threshold = 1.0;
                self.min_experts = 2;
                self.max_experts = 8;
            }
            crate::strategy::ExpertPolicyConfig::TopP {
                p,
                min_experts,
                max_experts,
            } => {
                self.moe_prune_mode = 0;
                self.moe_top_p = p.clamp(0.0, 1.0);
                self.moe_contrib_threshold = 1.0;
                self.min_experts = *min_experts;
                self.max_experts = *max_experts;
            }
            crate::strategy::ExpertPolicyConfig::Contribution {
                threshold,
                min_experts,
                max_experts,
                ..
            } => {
                self.moe_prune_mode = 1;
                self.moe_top_p = 1.0;
                self.moe_contrib_threshold = threshold.clamp(0.0, 1.0);
                self.min_experts = *min_experts;
                self.max_experts = *max_experts;
            }
            crate::strategy::ExpertPolicyConfig::AdaptiveEntropy {
                min_experts,
                max_experts,
                ..
            } => {
                self.moe_prune_mode = 0;
                self.moe_top_p = 1.0;
                self.moe_contrib_threshold = 1.0;
                self.min_experts = *min_experts;
                self.max_experts = *max_experts;
            }
        }
    }

    pub fn note_runtime_config_source(&mut self, source: RuntimeConfigSource) {
        if source.rank() >= self.runtime_config_source.rank() {
            self.runtime_config_source = source;
        }
    }


    pub fn effective_runtime_policy_kind(&self) -> &'static str {
        match &self.expert_policy {
            crate::strategy::ExpertPolicyConfig::Exact => "exact",
            crate::strategy::ExpertPolicyConfig::TopP { .. } => "top_p",
            crate::strategy::ExpertPolicyConfig::LkoAware => "lko_aware",
            crate::strategy::ExpertPolicyConfig::Contribution { .. } => "contribution",
            crate::strategy::ExpertPolicyConfig::AdaptiveEntropy { .. } => "adaptive_entropy",
        }
    }

    pub fn effective_moe_top_p(&self) -> f32 {
        match &self.expert_policy {
            crate::strategy::ExpertPolicyConfig::Exact => 1.0,
            crate::strategy::ExpertPolicyConfig::TopP { p, .. } => *p,
            crate::strategy::ExpertPolicyConfig::LkoAware => 1.0,
            crate::strategy::ExpertPolicyConfig::Contribution { .. } => 1.0,
            crate::strategy::ExpertPolicyConfig::AdaptiveEntropy { high_entropy_p, .. } => *high_entropy_p,
        }
    }

    pub fn effective_moe_min_experts(&self) -> usize {
        match &self.expert_policy {
            crate::strategy::ExpertPolicyConfig::Exact => 8,
            crate::strategy::ExpertPolicyConfig::TopP { min_experts, .. } => *min_experts,
            crate::strategy::ExpertPolicyConfig::LkoAware => self.min_experts.max(2),
            crate::strategy::ExpertPolicyConfig::Contribution { min_experts, .. } => *min_experts,
            crate::strategy::ExpertPolicyConfig::AdaptiveEntropy { min_experts, .. } => *min_experts,
        }
    }

    pub fn effective_moe_max_experts(&self) -> usize {
        match &self.expert_policy {
            crate::strategy::ExpertPolicyConfig::Exact => 8,
            crate::strategy::ExpertPolicyConfig::TopP { max_experts, .. } => *max_experts,
            crate::strategy::ExpertPolicyConfig::LkoAware => self.max_experts.max(self.min_experts.max(2)),
            crate::strategy::ExpertPolicyConfig::Contribution { max_experts, .. } => *max_experts,
            crate::strategy::ExpertPolicyConfig::AdaptiveEntropy { max_experts, .. } => *max_experts,
        }
    }



    pub fn final_norm_weights(&self) -> &[f32] {
        &self.final_norm
    }

    pub fn set_expert_policy(&mut self, policy: crate::strategy::ExpertPolicyConfig) {
        self.expert_policy = policy;
        self.sync_legacy_policy_fields();
    }

    /// Forward pass WITH timing breakdown. Returns (h, [deltanet_ms, gqa_ms, shared_ms, moe_ms]).
    pub fn forward_timed(
        &mut self,
        token_id: usize,
        pos: usize,
        seq_len: usize,
    ) -> (Vec<f32>, [f64; 5]) {
        use std::time::Instant;
        let forward_start = Instant::now();

        // Initialize trace step if recording
        if self.record_trace_path.is_some() {
            self.current_step_trace = Some(StepTrace {
                step: self.step_counter,
                token_id,
                entropy: 0.0,
                logits_topk_ids: Vec::new(),
                logits_topk_values: Vec::new(),
                layers: Vec::new(),
            });
        }

        let mut h = {
            let ptr = unsafe { self.embed.as_ptr().add(token_id * HDIM * 4) as *const f32 };
            (0..HDIM)
                .map(|i| unsafe { *ptr.add(i) })
                .collect::<Vec<f32>>()
        };
        let mut t_delta = 0.0f64;
        let mut t_gqa = 0.0f64;
        let mut t_shared = 0.0f64;
        let mut t_moe = 0.0f64;
        let mut t_norm = 0.0f64;

        // ── Group pre-resolve JIT setup ──
        let group_size = std::env::var("OBJETA_RESIDENCY_GROUP_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let top_n = std::env::var("OBJETA_GROUP_PRERESOLVE_TOP_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        self.expert_residency_manager.residency_group_size = group_size;

        if group_size > 1 {
            self.expert_residency_manager.group_preresolve_wall_ms = 0.0;
            self.expert_residency_manager.group_pinned_bytes = 0;
            self.expert_residency_manager.group_preloaded_expert_count = 0;
            self.expert_residency_manager.group_cache_miss_count = 0;
            self.expert_residency_manager
                .group_preresolve_skipped_by_budget = 0;
            self.expert_residency_manager
                .group_preresolve_requested_bytes = 0;
            self.expert_residency_manager.group_preresolve_loaded_bytes = 0;
            self.expert_residency_manager.group_preresolve_hit_rate = 0.0;
        }

        let stride = (1.0 / self.fusion_ratio.max(0.01)).round() as usize;
        let mut delta_count = 0usize;
        let mut deltas_skipped = 0u32;
        let mut n_moe_collapse = 0usize;

        for l in 0..40 {
            // ── JIT Group pre-resolve ──
            if group_size > 1
                && top_n > 0
                && !self.expert_residency_manager.is_bypass()
                && l % group_size == 0
            {
                let t0 = Instant::now();
                let expert_total_bytes = (crate::moe_dispatch::GU_EXPERT_BYTES
                    + crate::moe_dispatch::D_EXPERT_BYTES)
                    as usize;
                let predict_n = top_n;

                let group_start = l;
                let group_end = (group_start + group_size).min(40);
                let mut group_layers: Vec<(usize, Vec<usize>)> = Vec::new();
                for gl in group_start..group_end {
                    let top_ids = unsafe {
                        let trackers = crate::moe_dispatch::GLOBAL_FREQ.as_ref();
                        let mut ids = Vec::new();
                        if let Some(trackers) = trackers {
                            if gl < trackers.len() {
                                let freq = trackers[gl].read().unwrap();
                                let mut entries: Vec<(usize, u32)> =
                                    freq.iter().map(|(k, v)| (*k, *v)).collect();
                                entries.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
                                ids = entries.iter().take(predict_n).map(|(k, _)| *k).collect();
                            }
                        }
                        ids
                    };
                    group_layers.push((gl, top_ids));
                }
                let group_refs: Vec<(usize, &[usize])> = group_layers
                    .iter()
                    .map(|(gl, ids)| (*gl, ids.as_slice()))
                    .collect();
                let gu_slices: Vec<&[u8]> = (group_start..group_end)
                    .map(|gl| &self.gu_mmaps[gl][..])
                    .collect();
                let d_slices: Vec<&[u8]> = (group_start..group_end)
                    .map(|gl| &self.down_mmaps[gl][..])
                    .collect();

                self.expert_residency_manager.pre_resolve_group(
                    &group_refs,
                    &gu_slices,
                    &d_slices,
                    token_id,
                    expert_total_bytes,
                );
                let elapsed = t0.elapsed().as_secs_f64();
                self.expert_residency_manager.group_preresolve_wall_ms += elapsed * 1000.0;
            }

            let layer_start = Instant::now();
            let mut layer_t_delta = 0.0f64;
            let mut layer_t_gqa = 0.0f64;
            let mut layer_t_shared = 0.0f64;
            let mut layer_t_moe = 0.0f64;

            let policy = &self.policy_table[l];
            let lw = &self.layers[l];

            // Debug overrides
            let attn_policy = if self.debug_force_attn_full {
                AttnPolicy::Full
            } else {
                policy.attn
            };
            let moe_policy = if self.debug_force_moe_skip {
                MoEPolicy::Skip
            } else {
                policy.moe
            };

            // ── Norm (always, cheap) ──
            let t0 = Instant::now();
            let h_norm = if !lw.input_norm.is_empty() {
                rms_norm_offset(&h, &lw.input_norm)
            } else {
                h.clone()
            };
            t_norm += t0.elapsed().as_secs_f64();

            // ── Attention (policy-driven) ──
            let ao = match attn_policy {
                AttnPolicy::Full => {
                    if policy.is_steering {
                        // GQA steering layer: Metal fused kernel
                        let t0 = Instant::now();
                        let ao = if self.metal_gqa_ok {
                            if let Some(ao) = gqa_metal_try(
                                l,
                                &lw.w_qkv,
                                &lw.w_o,
                                &h_norm,
                                pos as u32,
                                seq_len as u32,
                                self.max_seq as u32,
                                &mut self.kv_k[l],
                                &mut self.kv_v[l],
                                &mut self.metal_gqa_first_fail,
                            ) {
                                ao
                            } else {
                                let mut ao = vec![0.0f32; HDIM];
                                gqa_attention_fused(
                                    &lw.w_qkv,
                                    &lw.w_o,
                                    &lw.q_norm,
                                    &lw.k_norm,
                                    &h_norm,
                                    &mut self.kv_k[l],
                                    &mut self.kv_v[l],
                                    &self.rope_cos,
                                    &self.rope_sin,
                                    16,
                                    2,
                                    HEAD_DIM,
                                    pos,
                                    seq_len,
                                    self.max_seq,
                                    &mut ao,
                                    &mut self.scratch_qkv,
                                    &mut self.scratch_q,
                                    &mut self.scratch_k,
                                    &mut self.scratch_v,
                                    &mut self.scratch_attn_out,
                                    &mut self.scratch_scores,
                                    &mut self.scratch_attn,
                                );
                                ao
                            }
                        } else {
                            let mut ao = vec![0.0f32; HDIM];
                            gqa_attention_fused(
                                &lw.w_qkv,
                                &lw.w_o,
                                &lw.q_norm,
                                &lw.k_norm,
                                &h_norm,
                                &mut self.kv_k[l],
                                &mut self.kv_v[l],
                                &self.rope_cos,
                                &self.rope_sin,
                                16,
                                2,
                                HEAD_DIM,
                                pos,
                                seq_len,
                                self.max_seq,
                                &mut ao,
                                &mut self.scratch_qkv,
                                &mut self.scratch_q,
                                &mut self.scratch_k,
                                &mut self.scratch_v,
                                &mut self.scratch_attn_out,
                                &mut self.scratch_scores,
                                &mut self.scratch_attn,
                            );
                            ao
                        };
                        let dur = t0.elapsed().as_secs_f64();
                        layer_t_gqa += dur;
                        t_gqa += dur;
                        ao
                    } else {
                        // DeltaNet transport layer
                        let t0 = Instant::now();
                        let mut ao = vec![0.0f32; HDIM];
                        delta_net_fused(
                            &lw.w_qkv,
                            &lw.w_z,
                            &lw.w_b,
                            &lw.w_a,
                            &lw.w_o,
                            &lw.w_conv,
                            &lw.w_norm,
                            &lw.dt_bias,
                            &lw.a_log,
                            &h_norm,
                            &mut self.conv_states[l],
                            &mut self.conv_ptrs[l],
                            &mut self.S_states[l],
                            &mut ao,
                            &mut self.scratch_f32,
                            l,
                            pos,
                        );
                        let dur = t0.elapsed().as_secs_f64();
                        layer_t_delta += dur;
                        t_delta += dur;
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

            let skip_moe = false;

            for i in 0..HDIM {
                h[i] += ao[i];
            }

            // ── Post-attention norm ──
            let t0 = Instant::now();
            let h_norm2 = if !lw.post_norm.is_empty() {
                rms_norm_offset(&h, &lw.post_norm)
            } else {
                h.clone()
            };
            t_norm += t0.elapsed().as_secs_f64();

            // Record trace norm for this layer
            if self.record_trace_path.is_some() {
                if let Some(ref mut step_trace) = self.current_step_trace {
                    let h_norm_val = {
                        let mut sum = 0.0f32;
                        for &x in &h_norm2 {
                            sum += x * x;
                        }
                        (sum / h_norm2.len() as f32).sqrt()
                    };
                    if let Some(lyr) = step_trace.layers.iter_mut().find(|lyr| lyr.layer == l) {
                        lyr.hidden_norm = h_norm_val;
                    } else {
                        step_trace.layers.push(LayerTrace {
                            layer: l,
                            hidden_norm: h_norm_val,
                            expert_ids: Vec::new(),
                            expert_weights: Vec::new(),
                        });
                    }
                }
            }

            // ── Shared expert (policy-driven) ──
            let t0 = Instant::now();
            if moe_policy != MoEPolicy::Skip && !skip_moe && !lw.se_gate.is_empty() {
                let gate = gemv_f16(&lw.se_gate, &h_norm2, 512, HDIM);
                let up = gemv_f16(&lw.se_up, &h_norm2, 512, HDIM);
                let mut hidden = gate.clone();
                for i in 0..512 {
                    hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i];
                }
                let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
                let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h_norm2)).exp());
                for i in 0..HDIM {
                    h[i] += se_out[i] * se_gate;
                }
            }
            let dur = t0.elapsed().as_secs_f64();
            layer_t_shared += dur;
            t_shared += dur;

            // ── MoE dispatch (policy-driven) ──
            let t0 = Instant::now();
            if self.moe_enabled && moe_policy != MoEPolicy::Skip && !skip_moe {
                let moe_out = self.call_moe(&h_norm2, l);
                for i in 0..HDIM {
                    h[i] += moe_out[i];
                }
            } else {
                n_moe_collapse += 1;
            }
            let dur = t0.elapsed().as_secs_f64();
            layer_t_moe += dur;
            t_moe += dur;

            // Update forward stats
            let layer_elapsed = layer_start.elapsed().as_secs_f64();
            self.forward_stats[l].calls += 1;
            self.forward_stats[l].total_layer_wall_sec += layer_elapsed;
            self.forward_stats[l].total_deltanet_wall_sec += layer_t_delta;
            self.forward_stats[l].total_gqa_wall_sec += layer_t_gqa;
            self.forward_stats[l].total_shared_wall_sec += layer_t_shared;
            self.forward_stats[l].total_moe_wall_sec += layer_t_moe;

            // ── JIT Group unpin ──
            if group_size > 1
                && top_n > 0
                && !self.expert_residency_manager.is_bypass()
                && ((l + 1) % group_size == 0 || l == 39)
            {
                self.expert_residency_manager.unpin_group();
            }
        }

        self.forward_calls += 1;
        self.forward_wall_sec += forward_start.elapsed().as_secs_f64();

        let n_full: usize = self
            .policy_table
            .iter()
            .filter(|p| p.attn == AttnPolicy::Full)
            .count();
        let n_collapse: usize = self
            .policy_table
            .iter()
            .filter(|p| p.attn == AttnPolicy::Collapse)
            .count();
        if objeta_timing_enabled() {
            eprintln!("TIMING: delta={:.0}ms gqa={:.0}ms shared={:.0}ms moe={:.0}ms | scheduler: full={} collapse={} moe_collapse={} fusion={:.2}",
                t_delta*1000.0, t_gqa*1000.0, t_shared*1000.0, t_moe*1000.0,
                n_full, n_collapse, n_moe_collapse, self.fusion_ratio);
        }

        (h, [t_delta, t_gqa, t_shared, t_moe, t_norm])
    }

    /// Full 40-layer forward pass. Returns hidden state (HDIM f32).
    pub fn forward(&mut self, token_id: usize, pos: usize, seq_len: usize) -> Vec<f32> {
        let mut h = {
            let ptr = unsafe { self.embed.as_ptr().add(token_id * HDIM * 4) as *const f32 };
            (0..HDIM)
                .map(|i| unsafe { *ptr.add(i) })
                .collect::<Vec<f32>>()
        };

        for l in 0..40 {
            let policy = self.policy_table[l];
            let lw = &self.layers[l];

            let h_norm = if !lw.input_norm.is_empty() {
                rms_norm_offset(&h, &lw.input_norm)
            } else {
                h.clone()
            };

            let ao = match policy.attn {
                AttnPolicy::Full => {
                    if policy.is_steering {
                        if self.metal_gqa_ok {
                            if let Some(ao) = gqa_metal_try(
                                l,
                                &lw.w_qkv,
                                &lw.w_o,
                                &h_norm,
                                pos as u32,
                                seq_len as u32,
                                self.max_seq as u32,
                                &mut self.kv_k[l],
                                &mut self.kv_v[l],
                                &mut self.metal_gqa_first_fail,
                            ) {
                                ao
                            } else {
                                let mut ao = vec![0.0f32; HDIM];
                                gqa_attention_fused(
                                    &lw.w_qkv,
                                    &lw.w_o,
                                    &lw.q_norm,
                                    &lw.k_norm,
                                    &h_norm,
                                    &mut self.kv_k[l],
                                    &mut self.kv_v[l],
                                    &self.rope_cos,
                                    &self.rope_sin,
                                    16,
                                    2,
                                    HEAD_DIM,
                                    pos,
                                    seq_len,
                                    self.max_seq,
                                    &mut ao,
                                    &mut self.scratch_qkv,
                                    &mut self.scratch_q,
                                    &mut self.scratch_k,
                                    &mut self.scratch_v,
                                    &mut self.scratch_attn_out,
                                    &mut self.scratch_scores,
                                    &mut self.scratch_attn,
                                );
                                ao
                            }
                        } else {
                            let mut ao = vec![0.0f32; HDIM];
                            gqa_attention_fused(
                                &lw.w_qkv,
                                &lw.w_o,
                                &lw.q_norm,
                                &lw.k_norm,
                                &h_norm,
                                &mut self.kv_k[l],
                                &mut self.kv_v[l],
                                &self.rope_cos,
                                &self.rope_sin,
                                16,
                                2,
                                HEAD_DIM,
                                pos,
                                seq_len,
                                self.max_seq,
                                &mut ao,
                                &mut self.scratch_qkv,
                                &mut self.scratch_q,
                                &mut self.scratch_k,
                                &mut self.scratch_v,
                                &mut self.scratch_attn_out,
                                &mut self.scratch_scores,
                                &mut self.scratch_attn,
                            );
                            ao
                        }
                    } else {
                        let mut ao = vec![0.0f32; HDIM];
                        delta_net_fused(
                            &lw.w_qkv,
                            &lw.w_z,
                            &lw.w_b,
                            &lw.w_a,
                            &lw.w_o,
                            &lw.w_conv,
                            &lw.w_norm,
                            &lw.dt_bias,
                            &lw.a_log,
                            &h_norm,
                            &mut self.conv_states[l],
                            &mut self.conv_ptrs[l],
                            &mut self.S_states[l],
                            &mut ao,
                            &mut self.scratch_f32,
                            l,
                            pos,
                        );
                        ao
                    }
                }
                AttnPolicy::Collapse | AttnPolicy::Skip => {
                    vec![0.0f32; HDIM]
                }
            };

            for i in 0..HDIM {
                h[i] += ao[i];
            }

            let h_norm2 = if !lw.post_norm.is_empty() {
                rms_norm_offset(&h, &lw.post_norm)
            } else {
                h.clone()
            };

            if policy.moe != MoEPolicy::Skip && !lw.se_gate.is_empty() {
                let gate = gemv_f16(&lw.se_gate, &h_norm2, 512, HDIM);
                let up = gemv_f16(&lw.se_up, &h_norm2, 512, HDIM);
                let mut hidden = gate.clone();
                for i in 0..512 {
                    hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i];
                }
                let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
                let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h_norm2)).exp());
                for i in 0..HDIM {
                    h[i] += se_out[i] * se_gate;
                }
            }

            if self.moe_enabled && policy.moe != MoEPolicy::Skip {
                let moe_out = self.call_moe(&h_norm2, l);
                for i in 0..HDIM {
                    h[i] += moe_out[i];
                }
            }
        }

        h
    }

    pub(crate) fn call_moe(&mut self, h: &[f32], l: usize) -> Vec<f32> {
        use std::time::Instant;
        let call_moe_start = Instant::now();
        let is_gqa_layer = self.layers[l].is_gqa;

        // Exact HF routing: always top-8 experts with renormalized softmax weights.
        let router_wall_start = Instant::now();
        let (mut eidx, mut ew) = crate::moe_dispatch::router_topk_cpu(&self.routers[l], h, 8);
        let router_wall_sec = router_wall_start.elapsed().as_secs_f64();
        self.moe_stats[l].total_router_wall_sec += router_wall_sec;
        self.moe_stats[l].last_router_top8_ids = eidx.clone();
        self.moe_stats[l].last_router_top8_weights = ew.clone();

        let candidate_build_start = Instant::now();
        self.moe_stats[l].last_candidate_ids = eidx.clone();
        self.moe_stats[l].last_candidate_weights = ew.clone();
        self.moe_stats[l].total_candidate_build_wall_sec +=
            candidate_build_start.elapsed().as_secs_f64();

        // Replay/Policy override
        let mut replayed = false;
        let policy_select_start = Instant::now();
        if let Some(ref traces) = self.replay_traces {
            if let Some(trace) = traces.iter().find(|t| t.step == self.step_counter) {
                if let Some(layer_trace) = trace.layers.iter().find(|lyr| lyr.layer == l) {
                    if !layer_trace.expert_ids.is_empty() {
                        eidx = layer_trace.expert_ids.clone();
                        ew = layer_trace.expert_weights.clone();
                        replayed = true;
                    }
                }
            }
        }

        if !replayed {
            match &self.expert_policy {
                crate::strategy::ExpertPolicyConfig::Exact => {}
                crate::strategy::ExpertPolicyConfig::LkoAware => {
                    if self.last_collapse_risk {
                        // Transition/high-entropy token: preserve full top-8.
                    } else if !is_gqa_layer
                        && !self.last_repetition_risk
                        && self.last_decode_entropy <= 1.0
                    {
                        // Shared-only on stable DeltaNet-only layers.
                        eidx.clear();
                        ew.clear();
                    } else {
                        let (target_p, min_experts) = if is_gqa_layer {
                            (0.90f32, 4usize)
                        } else if self.last_decode_entropy <= 1.35 {
                            (0.80f32, 2usize)
                        } else {
                            (0.85f32, 2usize)
                        };
                        let items: Vec<crate::moe_dispatch::SelectedExpert> = eidx
                            .iter()
                            .copied()
                            .zip(ew.iter().copied())
                            .map(|(expert_id, routing_weight)| {
                                crate::moe_dispatch::SelectedExpert {
                                    expert_id,
                                    routing_weight,
                                }
                            })
                            .collect();
                        let (pruned, _, _, _, _) = crate::moe_dispatch::prune_selected_top_p(
                            items,
                            target_p,
                            min_experts,
                            8,
                        );
                        eidx = pruned.iter().map(|e| e.expert_id).collect();
                        ew = pruned.iter().map(|e| e.routing_weight).collect();
                    }
                }
                crate::strategy::ExpertPolicyConfig::TopP {
                    p,
                    min_experts,
                    max_experts,
                } => {
                    let mut items: Vec<(usize, f32)> =
                        eidx.iter().copied().zip(ew.iter().copied()).collect();
                    items
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    if l == 0 && self.step_counter == 1 {
                        println!(
                            "[DEBUG TopP] l={} step={} p={} min={} max={} items={:?}",
                            l, self.step_counter, p, min_experts, max_experts, items
                        );
                    }

                    let mut kept_idx = Vec::new();
                    let mut kept_w = Vec::new();
                    let mut cum_w = 0.0f32;
                    for (i, (id, w)) in items.into_iter().enumerate() {
                        kept_idx.push(id);
                        kept_w.push(w);
                        cum_w += w;
                        if cum_w >= *p && (i + 1) >= *min_experts {
                            break;
                        }
                        if (i + 1) >= *max_experts {
                            break;
                        }
                    }
                    if l == 0 && self.step_counter == 1 {
                        println!("[DEBUG TopP] kept={}", kept_idx.len());
                    }
                    let sum: f32 = kept_w.iter().sum::<f32>().max(1e-12);
                    for w in &mut kept_w {
                        *w /= sum;
                    }
                    eidx = kept_idx;
                    ew = kept_w;
                }
                crate::strategy::ExpertPolicyConfig::Contribution {
                    threshold,
                    min_experts,
                    max_experts,
                    ema_beta,
                } => {
                    let mut items: Vec<(usize, f32, f32)> = eidx
                        .iter()
                        .copied()
                        .zip(ew.iter().copied())
                        .map(|(id, w)| {
                            let ema = self.moe_ema_output_norm[l][id];
                            (id, w, w * ema)
                        })
                        .collect();
                    items
                        .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

                    let total_score: f32 = items.iter().map(|x| x.2).sum::<f32>().max(1e-12);
                    let mut kept_idx = Vec::new();
                    let mut kept_w = Vec::new();
                    let mut cum_score = 0.0f32;
                    for (i, (id, w, score)) in items.into_iter().enumerate() {
                        kept_idx.push(id);
                        kept_w.push(w);
                        cum_score += score;
                        if (cum_score / total_score) >= *threshold && (i + 1) >= *min_experts {
                            break;
                        }
                        if (i + 1) >= *max_experts {
                            break;
                        }
                    }
                    let sum: f32 = kept_w.iter().sum::<f32>().max(1e-12);
                    for w in &mut kept_w {
                        *w /= sum;
                    }
                    eidx = kept_idx;
                    ew = kept_w;

                    // Update EMA
                    for id in 0..256 {
                        self.moe_ema_output_norm[l][id] *= *ema_beta;
                    }
                    for (&id, &w) in eidx.iter().zip(ew.iter()) {
                        self.moe_ema_output_norm[l][id] += (1.0 - *ema_beta) * w;
                    }
                }
                crate::strategy::ExpertPolicyConfig::AdaptiveEntropy {
                    low_entropy_p,
                    mid_entropy_p,
                    high_entropy_p,
                    low_entropy_threshold,
                    mid_entropy_threshold,
                    min_experts,
                    max_experts,
                    ..
                } => {
                    let entropy: f32 = -ew
                        .iter()
                        .map(|&w| if w > 1e-10 { w * w.ln() } else { 0.0 })
                        .sum::<f32>();
                    let target_p = if entropy < *low_entropy_threshold {
                        *low_entropy_p
                    } else if entropy < *mid_entropy_threshold {
                        *mid_entropy_p
                    } else {
                        *high_entropy_p
                    };
                    let mut items: Vec<(usize, f32)> =
                        eidx.iter().copied().zip(ew.iter().copied()).collect();
                    items
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut kept_idx = Vec::new();
                    let mut kept_w = Vec::new();
                    let mut cum_w = 0.0f32;
                    for (i, (id, w)) in items.into_iter().enumerate() {
                        kept_idx.push(id);
                        kept_w.push(w);
                        cum_w += w;
                        if cum_w >= target_p && (i + 1) >= *min_experts {
                            break;
                        }
                        if (i + 1) >= *max_experts {
                            break;
                        }
                    }
                    let sum: f32 = kept_w.iter().sum::<f32>().max(1e-12);
                    for w in &mut kept_w {
                        *w /= sum;
                    }
                    eidx = kept_idx;
                    ew = kept_w;
                }
            }
        }
        self.moe_stats[l].total_policy_select_wall_sec +=
            policy_select_start.elapsed().as_secs_f64();
        self.moe_stats[l].last_selected_ids = eidx.clone();
        self.moe_stats[l].last_selected_weights = ew.clone();
        self.moe_stats[l].last_selected_count = eidx.len();
        self.moe_stats[l].last_selected_renormalized = !matches!(
            self.expert_policy,
            crate::strategy::ExpertPolicyConfig::Exact
                | crate::strategy::ExpertPolicyConfig::LkoAware
        ) && !replayed;
        self.moe_stats[l].last_attention_type = if is_gqa_layer {
            "gqa".to_string()
        } else {
            "deltanet".to_string()
        };
        self.moe_stats[l].last_dispatch_ids = eidx.clone();
        self.moe_stats[l].last_dispatch_weights = ew.clone();

        // Record trace for this layer
        if self.record_trace_path.is_some() {
            if let Some(ref mut step_trace) = self.current_step_trace {
                let h_norm_val = {
                    let mut sum = 0.0f32;
                    for &x in h {
                        sum += x * x;
                    }
                    (sum / h.len() as f32).sqrt()
                };
                let mut found = false;
                for lyr in &mut step_trace.layers {
                    if lyr.layer == l {
                        lyr.hidden_norm = h_norm_val;
                        lyr.expert_ids = eidx.iter().map(|&id| id as usize).collect();
                        lyr.expert_weights = ew.clone();
                        found = true;
                        break;
                    }
                }
                if !found {
                    step_trace.layers.push(LayerTrace {
                        layer: l,
                        hidden_norm: h_norm_val,
                        expert_ids: eidx.iter().map(|&id| id as usize).collect(),
                        expert_weights: ew.clone(),
                    });
                }
            }
        }

        let res =
            self.execute_selected_moe(h, l, &eidx, &ew, self.use_fused_moe, self.fused_down_mode);
        self.moe_stats[l].total_call_moe_wall_sec += call_moe_start.elapsed().as_secs_f64();
        res.output
    }

    fn compute_dequantized_expert_scratch_bytes(&self) -> usize {
        let ffn_dim = 512;
        let gate_up_f32_len = 2 * ffn_dim * HDIM;
        let down_f32_len = HDIM * ffn_dim;
        (gate_up_f32_len + down_f32_len) * std::mem::size_of::<f32>()
    }

    pub(crate) fn execute_selected_moe(
        &mut self,
        h: &[f32],
        l: usize,
        eidx: &[usize],
        ew: &[f32],
        use_fused: bool,
        fused_down_mode: crate::moe_dispatch::FusedDownMode,
    ) -> crate::moe_dispatch::MoeExecutionResult {
        use crate::moe_dispatch::{
            call_moe_pipeline, MoeExecutionBackend, MoeExecutionResult, SelectedExpert,
        };

        let token_id = self
            .current_step_trace
            .as_ref()
            .map(|trace| trace.token_id)
            .unwrap_or(0);

        let selected: Vec<SelectedExpert> = eidx
            .iter()
            .zip(ew.iter())
            .map(|(&id, &w)| SelectedExpert {
                expert_id: id,
                routing_weight: w,
            })
            .collect();

        let backend = if use_fused {
            MoeExecutionBackend::FusedQ4
        } else {
            MoeExecutionBackend::LegacyDequantF32
        };

        let res = call_moe_pipeline(
            l,
            h,
            None,
            Some(selected),
            backend,
            fused_down_mode,
            Some(&mut self.expert_residency_manager),
            Some((&self.gu_mmaps[l], &self.down_mmaps[l])),
            &mut self.fused_moe_scratch,
            token_id,
        );

        // Accumulate statistics
        self.moe_stats[l].calls += 1;
        self.moe_stats[l].total_executed_experts += res.telemetry.selected_expert_count as u64;
        self.moe_stats[l].total_executed_mass += res.telemetry.routing_mass_kept as f64;
        self.moe_stats[l].total_dropped_mass += res.telemetry.routing_mass_dropped as f64;
        self.moe_stats[l].total_executed_mass_pre_renorm +=
            res.telemetry.routing_mass_kept_pre_renorm as f64;
        self.moe_stats[l].total_dropped_mass_pre_renorm +=
            res.telemetry.routing_mass_dropped_pre_renorm as f64;
        self.moe_stats[l].total_routing_mass_sum_after_renorm +=
            res.telemetry.routing_mass_sum_after_renorm as f64;
        self.moe_stats[l].total_warm_hit_count += res.telemetry.warm_hit_count;
        self.moe_stats[l].total_cold_hit_count += res.telemetry.cold_hit_count;
        self.moe_stats[l].total_bytes_read += res.telemetry.actual_expert_bytes_loaded;
        self.moe_stats[l].total_logical_bytes_requested +=
            res.telemetry.logical_expert_bytes_requested;
        self.moe_stats[l].total_actual_bytes_loaded += res.telemetry.actual_expert_bytes_loaded;
        self.moe_stats[l].total_resident_cache_bytes_reused +=
            res.telemetry.resident_cache_bytes_reused;
        self.moe_stats[l].total_resident_cache_hit_count += res.telemetry.warm_hit_count;
        self.moe_stats[l].total_resident_cache_miss_count += res.telemetry.cold_hit_count;
        self.moe_stats[l].total_direct_cold_load_count += res.telemetry.direct_cold_load_count;
        self.moe_stats[l].total_dequantized_scratch_bytes +=
            res.telemetry.dequantized_scratch_bytes;

        self.moe_stats[l].total_compute_sec += res.telemetry.routed_exec_wall;
        self.moe_stats[l].total_exec_wall_sec += res.telemetry.routed_exec_wall;
        self.moe_stats[l].total_routed_exec_wall_sec += res.telemetry.routed_exec_wall;
        self.moe_stats[l].total_cache_lookup_wall_sec += res.telemetry.cache_lookup_wall;
        self.moe_stats[l].total_candidate_build_wall_sec += res.telemetry.candidate_build_wall;
        self.moe_stats[l].total_router_wall_sec += res.telemetry.router_wall;
        self.moe_stats[l].total_policy_select_wall_sec += res.telemetry.policy_select_wall;
        self.moe_stats[l].total_stats_wall_sec += res.telemetry.stats_wall;
        self.moe_stats[l].total_cache_key_build_wall_sec += res.telemetry.cache_key_build_wall;
        self.moe_stats[l].total_cache_hit_lookup_wall_sec += res.telemetry.cache_hit_lookup_wall;
        self.moe_stats[l].total_cache_miss_load_wall_sec += res.telemetry.cache_miss_load_wall;
        self.moe_stats[l].total_cache_eviction_wall_sec += res.telemetry.cache_eviction_wall;
        self.moe_stats[l].total_cache_insert_wall_sec += res.telemetry.cache_insert_wall;
        self.moe_stats[l].total_cache_page_clone_wall_sec += res.telemetry.cache_page_clone_wall;

        self.moe_stats[l].total_fused_gate_up_sec += res.telemetry.fused_gate_up_sec;
        self.moe_stats[l].total_fused_swiglu_sec += res.telemetry.fused_swiglu_sec;
        self.moe_stats[l].total_fused_down_accum_sec += res.telemetry.fused_down_accum_sec;
        self.moe_stats[l].total_fused_alloc_sec += res.telemetry.fused_alloc_sec;
        self.moe_stats[l].total_fused_stats_sec += res.telemetry.fused_stats_sec;

        if use_fused {
            self.moe_stats[l].total_load_sec += res.telemetry.fused_gate_up_sec;
            self.moe_stats[l].total_gemv_sec += res.telemetry.fused_gate_up_sec;
            self.moe_stats[l].total_accumulate_sec += res.telemetry.fused_down_accum_sec;
        }

        self.moe_stats[l].last_selected_ids = res.telemetry.selected_expert_ids.clone();
        self.moe_stats[l].last_selected_weights = res.telemetry.selected_expert_weights.clone();
        self.moe_stats[l].last_selected_count = res.telemetry.selected_expert_count;
        self.moe_stats[l].last_selected_renormalized = res.telemetry.selected_renormalized;
        self.moe_stats[l].last_dispatch_ids = res.telemetry.selected_expert_ids.clone();
        self.moe_stats[l].last_dispatch_weights = res.telemetry.selected_expert_weights.clone();
        for &eid in &res.telemetry.selected_expert_ids {
            self.moe_stats[l].unique_expert_ids.insert(eid);
        }

        self.moe_io_events.push(MoEIoEvent {
            step: self.step_counter,
            token_id,
            layer_id: l,
            selected_experts: res.telemetry.selected_expert_ids.clone(),
            selected_weights: res.telemetry.selected_expert_weights.clone(),
            logical_bytes: res.telemetry.logical_expert_bytes_requested,
            actual_loaded_bytes: res.telemetry.actual_expert_bytes_loaded,
            resident_hits: res.telemetry.warm_hit_count,
            cold_loads: res.telemetry.cold_hit_count,
            resident_bytes: res.telemetry.resident_cache_resident_bytes,
            pinned_resident_bytes: res.telemetry.pinned_resident_bytes,
            token_window_peak_resident_bytes: res.telemetry.token_window_peak_resident_bytes,
            eviction_count: res.telemetry.eviction_count,
            eviction_count_during_token: res.telemetry.eviction_count_during_token,
            eviction_count_at_token_end: res.telemetry.eviction_count_at_token_end,
            dequantized_scratch_bytes: res.telemetry.dequantized_scratch_bytes,

            logical_expert_bytes_requested: res.telemetry.logical_expert_bytes_requested,
            actual_expert_bytes_loaded: res.telemetry.actual_expert_bytes_loaded,
            resident_cache_bytes_reused: res.telemetry.resident_cache_bytes_reused,
            resident_cache_resident_bytes: res.telemetry.resident_cache_resident_bytes,
            routing_mass_kept: res.telemetry.routing_mass_kept,
            routing_mass_dropped: res.telemetry.routing_mass_dropped,
            routing_mass_kept_pre_renorm: res.telemetry.routing_mass_kept_pre_renorm,
            routing_mass_dropped_pre_renorm: res.telemetry.routing_mass_dropped_pre_renorm,
            routing_mass_sum_after_renorm: res.telemetry.routing_mass_sum_after_renorm,
        });

        res
    }

    /// Warmup: actually dequantize + GEMV the routed experts to fault q4 pages into OS cache.
    /// Accesses ALL bytes of each expert's q4 data, forcing the kernel to cache them in RAM.
    pub fn warmup(&mut self, n_tokens: usize) {
        if self.expert_freq_ready {
            return;
        }
        eprintln!(
            "[objeta] Warming OS page cache by dequantizing real experts ({n_tokens} tokens)..."
        );
        let vocab = self.embed.len() / (HDIM * 4);
        let step = vocab.max(1) / n_tokens.max(1);
        for t in 0..n_tokens {
            let tid = (t * step).min(vocab - 1);
            let h = self.embed_token(tid);
            for l in 0..40 {
                let run_moe = self.layers[l].is_gqa || self.moe_on_deltanet;
                if !run_moe {
                    continue;
                }
                // Actually dequantize + GEMV all routed experts (reads ALL q4 pages into OS cache)
                let _ = self.call_moe(&h, l);
            }
        }
        // Clear hashmap cache after warmup (these were random tokens, not our prompt)
        self.expert_residency_manager.resident.clear();
        self.expert_residency_manager.metadata.clear();
        self.expert_freq_ready = true;
        eprintln!(
            "[objeta] OS page cache warmed ({} experts faulted in).",
            n_tokens * 10 * 8
        );
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
        self.expert_residency_manager.resident.clear();
        self.expert_residency_manager.metadata.clear();
        self.expert_cache_size = cache_size;
        self.expert_residency_manager.capacity_bytes = cache_size as u64 * EXPERT_TOTAL_BYTES;
        if cache_size == 0 {
            eprintln!("[objeta] Resident expert cache disabled (capacity=0); all experts use direct/cold path.");
            return;
        }
        let total_layers: usize = (0..40)
            .filter(|&l| self.layers[l].is_gqa || self.moe_on_deltanet)
            .count();
        let mem_per_expert_mb = (EXPERT_TOTAL_BYTES as f64) / (1024.0 * 1024.0); // ~1.97MB Q4 compressed
        let est_mb = total_layers as f64 * cache_size as f64 * mem_per_expert_mb;
        eprintln!("[objeta] Building expert caches ({cache_size} experts/layer × {total_layers} layers, ~{:.1}MB Q4)...", est_mb);

        let mut total = 0i32;
        for l in 0..40 {
            let run_moe = self.layers[l].is_gqa || self.moe_on_deltanet;
            if !run_moe {
                continue;
            }
            let top_ids = crate::moe_dispatch::get_top_frequent_expert_ids(l, cache_size);
            for &eid in &top_ids {
                let key = crate::expert_cache::ExpertPageKey {
                    layer_id: l,
                    expert_id: eid,
                    precision: 4,
                };
                let gu_off = eid * EXPERT_GATE_UP_BYTES as usize;
                let d_off = eid * EXPERT_DOWN_BYTES as usize;
                let gu_slice =
                    Arc::from(&self.gu_mmaps[l][gu_off..gu_off + EXPERT_GATE_UP_BYTES as usize]);
                let d_slice =
                    Arc::from(&self.down_mmaps[l][d_off..d_off + EXPERT_DOWN_BYTES as usize]);
                self.expert_residency_manager.resident.insert(
                    key,
                    crate::expert_cache::ExpertPage {
                        gate_up_bytes: gu_slice,
                        down_bytes: d_slice,
                    },
                );
                self.expert_residency_manager.metadata.insert(
                    key,
                    crate::expert_cache::ExpertPageMeta {
                        bytes: EXPERT_TOTAL_BYTES as usize,
                        last_used_token: 0,
                        use_count: 0,
                        ema_gate: 0.0,
                        load_count: 1,
                    },
                );
                total += 1;
            }
        }
        eprintln!("[objeta] Cached {total} experts across {total_layers} active layers.");
    }
}

#[cfg(test)]
mod tests {
    use crate::expert_cache::{ExpertPage, ExpertPageKey, ExpertResidencyManager};
    use std::sync::Arc;

    #[test]
    fn resident_cache_disabled_bypasses_insert() {
        let mut manager = ExpertResidencyManager::new(0);
        let bytes = 1000;
        let page = manager.ensure_resident(0, 0, 4, bytes, 1, 0.5, || ExpertPage {
            gate_up_bytes: Arc::new([1, 2, 3]),
            down_bytes: Arc::new([4, 5, 6]),
        });
        assert_eq!(manager.resident.len(), 0);
        assert_eq!(page.gate_up_bytes.as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn resident_cache_enabled_evicts_to_capacity() {
        // Capacity fits exactly 2 experts of size 1000 bytes
        let mut manager = ExpertResidencyManager::new(2000);

        let load_1 = || ExpertPage {
            gate_up_bytes: Arc::new([1]),
            down_bytes: Arc::new([1]),
        };
        let load_2 = || ExpertPage {
            gate_up_bytes: Arc::new([2]),
            down_bytes: Arc::new([2]),
        };
        let load_3 = || ExpertPage {
            gate_up_bytes: Arc::new([3]),
            down_bytes: Arc::new([3]),
        };

        // Insert first
        manager.ensure_resident(0, 1, 4, 1000, 1, 0.5, load_1);
        assert_eq!(manager.resident.len(), 1);

        // Insert second
        manager.ensure_resident(0, 2, 4, 1000, 2, 0.5, load_2);
        assert_eq!(manager.resident.len(), 2);

        // Insert third - triggers eviction.
        // Recency score for 1 is 1 * 1.0 = 1.0. For 2 is 2.0.
        // So expert 1 has lower score and should be evicted.
        manager.ensure_resident(0, 3, 4, 1000, 3, 0.5, load_3);
        assert_eq!(manager.resident.len(), 2);

        let key1 = ExpertPageKey {
            layer_id: 0,
            expert_id: 1,
            precision: 4,
        };
        let key2 = ExpertPageKey {
            layer_id: 0,
            expert_id: 2,
            precision: 4,
        };
        let key3 = ExpertPageKey {
            layer_id: 0,
            expert_id: 3,
            precision: 4,
        };

        assert!(!manager.resident.contains_key(&key1));
        assert!(manager.resident.contains_key(&key2));
        assert!(manager.resident.contains_key(&key3));
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
    std::fs::File::open(&json_path)
        .ok()?
        .read_to_string(&mut json_str)
        .ok()?;
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
    let has_attn = get_f16("linear_attn.in_proj_qkv.weight").is_some()
        || get_f16("self_attn.q_proj.weight").is_some();

    let (w_qkv, qkv_M, qkv_K) = if is_gqa {
        let qw = get_f16("self_attn.q_proj.weight")?;
        let kw = get_f16("self_attn.k_proj.weight")?;
        let vw = get_f16("self_attn.v_proj.weight")?;
        let qM = qw.len() / HDIM;
        let kM = kw.len() / HDIM;
        let vM = vw.len() / HDIM;
        let mut cat = Vec::with_capacity(qw.len() + kw.len() + vw.len());
        cat.extend_from_slice(&qw);
        cat.extend_from_slice(&kw);
        cat.extend_from_slice(&vw);
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

    let w_z = if has_attn && !is_gqa {
        get_f16("linear_attn.in_proj_z.weight")?
    } else {
        Vec::new()
    };
    let w_b = if has_attn && !is_gqa {
        get_f32("linear_attn.in_proj_b.weight")?
    } else {
        Vec::new()
    };
    let w_a = if has_attn && !is_gqa {
        get_f32("linear_attn.in_proj_a.weight")?
    } else {
        Vec::new()
    };
    let w_conv = if has_attn && !is_gqa {
        let w = get_f32("linear_attn.conv1d.weight")?; // (8192, 1, 4) → reshape to (8192, 4)
        w
    } else {
        Vec::new()
    };
    let w_norm = if has_attn && !is_gqa {
        get_f32("linear_attn.norm.weight")?
    } else {
        Vec::new()
    };
    let dt_bias = if has_attn && !is_gqa {
        get_f32("linear_attn.dt_bias")?
    } else {
        Vec::new()
    };
    let a_log = if has_attn && !is_gqa {
        get_f32("linear_attn.A_log")?
    } else {
        Vec::new()
    };
    let se_gate = get_f16("mlp.shared_expert.gate_proj.weight").unwrap_or_default();
    let se_up = get_f16("mlp.shared_expert.up_proj.weight").unwrap_or_default();
    let se_down = get_f16("mlp.shared_expert.down_proj.weight").unwrap_or_default();
    let se_gate_w = get_f32("mlp.shared_expert_gate.weight").unwrap_or_default();
    let q_norm = get_f32("self_attn.q_norm.weight").unwrap_or_default();
    let k_norm = get_f32("self_attn.k_norm.weight").unwrap_or_default();
    let input_norm = get_f32("input_layernorm.weight").unwrap_or_default();
    let post_norm = get_f32("post_attention_layernorm.weight").unwrap_or_default();

    Some(LayerWeights {
        w_qkv,
        w_o,
        w_z,
        w_b,
        w_a,
        w_conv,
        w_norm,
        dt_bias,
        a_log,
        se_gate,
        se_up,
        se_down,
        se_gate_w,
        q_norm,
        k_norm,
        input_norm,
        post_norm,
        is_gqa,
        has_attn,
        qkv_M,
        qkv_K,
        o_M,
        o_K,
    })
}

// ── lm_head + top-k sampling (in Rust) ────────────────────────────────────

impl Qwen36Runner {
    #[inline]
    fn classify_repetition_kind(token_id: i32) -> crate::runtime_governor::RepetitionKind {
        match token_id {
            248045 => crate::runtime_governor::RepetitionKind::StructuralMarkerLoop,
            248044 | 248046 => crate::runtime_governor::RepetitionKind::SpecialTokenLoop,
            _ => crate::runtime_governor::RepetitionKind::SemanticRepetition,
        }
    }

    /// Compute logits = embed @ hn, return top-k indices + values.
    /// Uses NEON+rayon for the massive matmul (248320 × 2048 = 509M FLOPs).
    pub fn lm_head_topk(&mut self, hn: &[f32], top_k: usize) -> (Vec<i32>, Vec<f32>) {
        let t0 = std::time::Instant::now();
        let lm_head_mmap = self.lm_head.as_ref().unwrap_or(&self.embed);
        let vocab = lm_head_mmap.len() / (HDIM * 4); // f32 = 4 bytes

        // Compute logits in parallel (embed is mmap'd, access via raw pointer is safe for read-only)
        let lm_head_data: &[f32] = unsafe {
            std::slice::from_raw_parts(lm_head_mmap.as_ptr() as *const f32, vocab * HDIM)
        };
        let logits: Vec<f32> = (0..vocab)
            .into_par_iter()
            .map(|v| dot_f32(&lm_head_data[v * HDIM..(v + 1) * HDIM], hn))
            .collect();

        // Top-k selection (partial sort)
        let mut indexed: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
        let k = top_k.min(indexed.len());
        indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        let indices: Vec<i32> = indexed.iter().map(|(i, _)| *i as i32).collect();
        let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
        if let Some((top1_id, _)) = indexed
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let top1 = *top1_id as i32;
            self.last_repetition_risk = self.last_predicted_token_id == top1;
            self.last_repetition_kind = if self.last_repetition_risk {
                Some(Self::classify_repetition_kind(top1))
            } else {
                None
            };
            self.last_predicted_token_id = top1;
        }
        self.lm_head_calls += 1;
        self.lm_head_wall_sec += t0.elapsed().as_secs_f64();
        (indices, values)
    }

    /// Compute logits = embed @ hn, return top-k indices + values + logit entropy.
    /// Shannon entropy is computed on the full vocab distribution in a single pass.
    pub fn lm_head_topk_with_entropy(
        &mut self,
        hn: &[f32],
        top_k: usize,
    ) -> (Vec<i32>, Vec<f32>, f32) {
        let t0 = std::time::Instant::now();
        let lm_head_mmap = self.lm_head.as_ref().unwrap_or(&self.embed);
        let vocab = lm_head_mmap.len() / (HDIM * 4); // f32 = 4 bytes

        let lm_head_data: &[f32] = unsafe {
            std::slice::from_raw_parts(lm_head_mmap.as_ptr() as *const f32, vocab * HDIM)
        };
        let logits: Vec<f32> = (0..vocab)
            .into_par_iter()
            .map(|v| dot_f32(&lm_head_data[v * HDIM..(v + 1) * HDIM], hn))
            .collect();

        // Calculate max logit
        let max_logit = logits
            .par_iter()
            .cloned()
            .reduce(|| f32::NEG_INFINITY, f32::max);

        // Sum exponent and calculate entropy
        let (sum_exp, sum_exp_log_exp) = logits
            .par_iter()
            .map(|&x| {
                let e = (x - max_logit).exp();
                (e, e * (x - max_logit))
            })
            .reduce(|| (0.0f32, 0.0f32), |a, b| (a.0 + b.0, a.1 + b.1));

        let entropy = if sum_exp > 1e-10 {
            sum_exp.ln() - (sum_exp_log_exp / sum_exp)
        } else {
            0.0f32
        };

        // Top-k selection (partial sort)
        let mut indexed: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
        let k = top_k.min(indexed.len());
        indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        let indices: Vec<i32> = indexed.iter().map(|(i, _)| *i as i32).collect();
        let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
        if let Some((top1_id, _)) = indexed
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let top1 = *top1_id as i32;
            self.last_repetition_risk = self.last_predicted_token_id == top1;
            self.last_repetition_kind = if self.last_repetition_risk {
                Some(Self::classify_repetition_kind(top1))
            } else {
                None
            };
            self.last_predicted_token_id = top1;
        }
        self.last_decode_entropy = entropy;
        self.last_collapse_risk = entropy <= 0.01;
        self.lm_head_calls += 1;
        self.lm_head_wall_sec += t0.elapsed().as_secs_f64();

        // Record trace endpoints if trace path is open
        if self.record_trace_path.is_some() {
            if let Some(ref mut step_trace) = self.current_step_trace {
                step_trace.entropy = entropy;
                step_trace.logits_topk_ids = indices.clone();
                step_trace.logits_topk_values = values.clone();
            }
        }

        (indices, values, entropy)
    }

    pub fn finish_step(&mut self) {
        if let Some(token_id) = self.active_token_id {
            let step = self.step_counter;
            let token_events: Vec<&MoEIoEvent> = self
                .moe_io_events
                .iter()
                .filter(|event| event.step == step && event.token_id == token_id)
                .collect();
            let count = token_events.len() as f32;
            let avg_selected_experts = if count > 0.0 {
                token_events
                    .iter()
                    .map(|event| event.selected_experts.len() as f32)
                    .sum::<f32>()
                    / count
            } else {
                0.0
            };
            let avg_routing_mass_kept = if count > 0.0 {
                token_events
                    .iter()
                    .map(|event| event.routing_mass_kept)
                    .sum::<f32>()
                    / count
            } else {
                0.0
            };
            let avg_routing_mass_dropped = if count > 0.0 {
                token_events
                    .iter()
                    .map(|event| event.routing_mass_dropped)
                    .sum::<f32>()
                    / count
            } else {
                0.0
            };
            let avg_routing_mass_kept_pre_renorm = if count > 0.0 {
                token_events
                    .iter()
                    .map(|event| event.routing_mass_kept_pre_renorm)
                    .sum::<f32>()
                    / count
            } else {
                0.0
            };
            let avg_routing_mass_dropped_pre_renorm = if count > 0.0 {
                token_events
                    .iter()
                    .map(|event| event.routing_mass_dropped_pre_renorm)
                    .sum::<f32>()
                    / count
            } else {
                0.0
            };
            let avg_routing_mass_sum_after_renorm = if count > 0.0 {
                token_events
                    .iter()
                    .map(|event| event.routing_mass_sum_after_renorm)
                    .sum::<f32>()
                    / count
            } else {
                0.0
            };
            self.collect_runtime_observation(
                step,
                token_id,
                avg_selected_experts,
                avg_routing_mass_kept,
                avg_routing_mass_dropped,
                avg_routing_mass_kept_pre_renorm,
                avg_routing_mass_dropped_pre_renorm,
                avg_routing_mass_sum_after_renorm,
            );
        }
        self.expert_residency_manager.end_token_residency();
        if let Some(path) = &self.record_trace_path {
            if let Some(ref trace) = self.current_step_trace {
                if let Ok(json_line) = serde_json::to_string(trace) {
                    use std::io::Write;
                    // Append json line to the record file
                    if let Ok(mut file) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                    {
                        let _ = writeln!(file, "{}", json_line);
                    }
                }
            }
        }
        self.current_step_trace = None;
        self.active_token_id = None;
        self.step_counter += 1;
    }

}
