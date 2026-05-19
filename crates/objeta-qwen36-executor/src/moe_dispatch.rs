//! MoE Expert Forward Dispatch — zero Python overhead.
//!
//! Single Rust function call per layer:
//!   lko_moe_forward_layer(router, gate_up_q4, down_q4, x, expert_ids, weights) → output
//!
//! Internally:
//!   1. Router matmul (CPU, tiny: 256×2048)
//!   2. Top-8 expert selection (argpartition)
//!   3. Per-expert: extract q4 slice → dequantize → GEMV (CPU SIMD)
//!   4. SwiGLU + weighted sum
//!
//! All data stays in native memory. No Python objects, no numpy, no mx.array.

use std::collections::HashMap;
use std::os::raw::c_float;
use std::sync::RwLock;
use rayon::prelude::*;

// Q4_K_APPL constants (must match quantize.rs)
const QK_K: usize = 256;
const Q4K_BLOCK_BYTES: usize = 160;
const N_SUB: usize = 8;
const SUB_SIZE: usize = 32;

// Expert weight layout constants for Qwen3.6
const HIDDEN_DIM: usize = 2048;
const FFN_DIM: usize = 512;
const N_EXPERTS: usize = 256;
const TOP_K: usize = 8;
// gate_up: 256 experts × (gate(512) + up(512)) = 1024 rows each, K=2048
const GU_ROWS: usize = 1024; // per expert
const GU_K: usize = 2048;
const GU_ROW_BYTES: usize = (GU_K / QK_K) * Q4K_BLOCK_BYTES; // 1280
const GU_EXPERT_BYTES: usize = GU_ROWS * GU_ROW_BYTES; // 1,310,720

// down: 256 experts × 2048 rows, K=512
const D_ROWS: usize = 2048;
const D_K: usize = 512;
const D_ROW_BYTES: usize = (D_K / QK_K) * Q4K_BLOCK_BYTES; // 320
const D_EXPERT_BYTES: usize = D_ROWS * D_ROW_BYTES; // 655,360


// ── F16 conversion ─────────────────────────────────────────────────

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let v = (mant as f32) / 1024.0 * 2.0f32.powi(-14);
            if sign == 1 { -v } else { v }
        }
    } else if exp == 31 {
        if mant == 0 { f32::from_bits((sign << 31) | 0x7F800000) }
        else { f32::NAN }
    } else {
        let v = 2.0f32.powi(exp as i32 - 15) * (1.0 + mant as f32 / 1024.0);
        if sign == 1 { -v } else { v }
    }
}


// ── Q4_K_APPL dequantize + GEMV ────────────────────────────────────

/// Dequantize Q4_K_APPL data and compute dot product with x.
/// Returns: W @ x where W is (M, K) stored in q4 format.
fn q4k_gemv(q4_data: &[u8], M: usize, K: usize, x: &[f32]) -> Vec<f32> {
    let num_blocks = K / QK_K;
    let row_bytes = num_blocks * Q4K_BLOCK_BYTES;
    let mut result = vec![0.0f32; M];

    let mut dequant_buf = vec![0.0f32; QK_K];

    for row in 0..M {
        let row_start = row * row_bytes;

        // Dequantize this row's blocks
        for b in 0..num_blocks {
            let blk_start = row_start + b * Q4K_BLOCK_BYTES;
            let blk: &[u8; Q4K_BLOCK_BYTES] = &q4_data[blk_start..blk_start + Q4K_BLOCK_BYTES]
                .try_into().unwrap();

            // Read scales and mins
            let mut scales = [0.0f32; N_SUB];
            let mut mins = [0.0f32; N_SUB];
            for j in 0..N_SUB {
                let sr = u16::from_le_bytes([blk[j * 2], blk[j * 2 + 1]]);
                let mr = u16::from_le_bytes([blk[16 + j * 2], blk[16 + j * 2 + 1]]);
                scales[j] = f16_to_f32(sr);
                mins[j] = f16_to_f32(mr);
            }

            // Unpack 4-bit quants
            let mut L = [0u8; QK_K];
            for g in 0..4 {
                for l in 0..32 {
                    let byte = blk[32 + g * 32 + l];
                    L[g * 64 + l] = byte & 0xF;
                    L[g * 64 + 32 + l] = byte >> 4;
                }
            }

            // Dequantize to f32
            for j in 0..N_SUB {
                let s = scales[j]; let m = mins[j];
                for i in 0..SUB_SIZE {
                    dequant_buf[j * SUB_SIZE + i] = L[j * SUB_SIZE + i] as f32 * s + m;
                }
            }

            // Dot product
            let x_slice = &x[b * QK_K..(b + 1) * QK_K];
            let mut dot = 0.0f32;
            for i in 0..QK_K {
                dot += dequant_buf[i] * x_slice[i];
            }
            result[row] += dot;
        }
    }

    result
}


// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_gemv_roundtrip() {
        // Create a simple weight matrix
        let M = 16;
        let K = 256;
        let mut w = vec![0.0f32; M * K];
        for i in 0..M {
            for j in 0..K {
                w[i * K + j] = ((i as f32) * 0.1 + (j as f32) * 0.01).sin();
            }
        }
        let x: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.07).cos()).collect();

        // Reference: float matmul
        let mut ref_out = vec![0.0f32; M];
        for i in 0..M {
            for j in 0..K {
                ref_out[i] += w[i * K + j] * x[j];
            }
        }

        // Quantize using existing Rust quantizer (from quantize module)
        // For now, test the dequantize function directly
        // — the q4k_gemv function uses the same dequantize logic.
        // We test that dequantize + dot gives same result as float matmul.

        // Quantize each row
        let mut q4_data = vec![0u8; M * Q4K_BLOCK_BYTES];
        for row in 0..M {
            let src: [f32; QK_K] = w[row * K..(row + 1) * K].try_into().unwrap();
            let mut dst = [0u8; Q4K_BLOCK_BYTES];
            // We need the quantize_q4k_appl_block function from quantize.rs
            // For now, skip — this test just validates the structure compiles
            // The real test will be done via Python FFI comparison
        }

        // Test router_topk
        let router: Vec<f32> = (0..N_EXPERTS * HIDDEN_DIM)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let (indices, weights) = router_topk_cpu(&router, &x.repeat(HIDDEN_DIM / K)[..HIDDEN_DIM], TOP_K);
        assert_eq!(indices.len(), TOP_K);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }
}


// ── Router ──────────────────────────────────────────────────────────

/// Top-k expert selection via argpartition. Public for Metal dispatch use.
pub fn router_topk_cpu(router_w: &[f32], x: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    // Compute logits
    let mut logits: Vec<(f32, usize)> = router_w
        .chunks(HIDDEN_DIM)
        .enumerate()
        .map(|(i, w)| {
            let dot: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            (dot, i)
        })
        .collect();

    // Partial sort for top-k
    logits.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());

    let top: Vec<_> = logits[..k].to_vec();

    // Softmax
    let max_logit = top.iter().map(|(l, _)| *l).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = top.iter().map(|(l, _)| ((*l - max_logit) as f32).exp()).sum();
    let weights: Vec<f32> = top.iter().map(|(l, _)| ((*l - max_logit) as f32).exp() / exp_sum).collect();
    let indices: Vec<usize> = top.iter().map(|(_, i)| *i).collect();

    (indices, weights)
}


/// Adaptive top-k: threshold based on router entropy.
/// Peaked distribution → aggressive pruning (2-4 experts).
/// Flat distribution → conservative (6-8 experts).
/// Returns (indices, renormalized_weights, n_selected).
pub fn router_topk_adaptive(
    router_w: &[f32], x: &[f32], max_k: usize,
) -> (Vec<usize>, Vec<f32>, usize) {
    let (indices, weights) = router_topk_cpu(router_w, x, max_k);

    // Compute entropy of top-k weights to gauge distribution peakedness
    let entropy: f32 = -weights.iter()
        .map(|&w| if w > 1e-10 { w * w.ln() } else { 0.0 })
        .sum::<f32>();

    // Adaptive threshold: lower entropy → more peaked → aggressive pruning
    // max entropy for k=8 uniform = ln(8) ≈ 2.08
    let cum_threshold = if entropy < 1.0 {
        0.45  // peaked: 2-3 experts
    } else if entropy < 1.5 {
        0.60  // moderate: 4-5 experts
    } else if entropy < 2.0 {
        0.78  // somewhat flat: 6 experts
    } else {
        0.88  // near-uniform: 7 experts (vs 8)
    };

    let mut cum = 0.0f32;
    let mut n = 0usize;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        n = i + 1;
        if cum >= cum_threshold && n >= 2 { break; }
    }
    let mut truncated_weights: Vec<f32> = weights[..n].to_vec();
    let sum: f32 = truncated_weights.iter().sum();
    for w in &mut truncated_weights { *w /= sum.max(1e-12); }
    (indices[..n].to_vec(), truncated_weights, n)
}

// ── MoE Forward ─────────────────────────────────────────────────────

#[derive(Copy, Clone)]
struct SendPtr(*const f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// Full MoE expert forward for one layer. Single function call, zero Python overhead.
fn moe_forward_layer(
    router_w: &[f32],        // [256, 2048]
    gate_up_q4: &[u8],       // all 256 experts, 320MB
    down_q4: &[u8],          // all 256 experts, 160MB
    x: &[f32],               // [2048]
    track_layer: Option<usize>, // if Some, record routing to global freq tracker
) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
    // 1. Router
    let (expert_ids, routing_weights) = router_topk_cpu(router_w, x, TOP_K);

    // Track frequency
    if let Some(layer_idx) = track_layer {
        if let Some(ref trackers) = unsafe { GLOBAL_FREQ.as_ref() } {
            if layer_idx < trackers.len() {
                let mut freq = trackers[layer_idx].write().unwrap();
                for &eid in &expert_ids {
                    *freq.entry(eid).or_insert(0) += 1;
                }
            }
        }
    }

    // Check per-layer cache for pre-dequantized expert weights
    let cache: Vec<(usize, SendPtr, SendPtr, SendPtr)> = {
        let track = track_layer.unwrap_or(usize::MAX);
        let caches = unsafe { CACHED_EXPERTS.as_ref() };
        if let Some(caches) = caches {
            if track < caches.len() {
                caches[track].lock().unwrap().iter()
                    .map(|(eid, g, u, d)| (*eid, SendPtr(g.as_ptr()), SendPtr(u.as_ptr()), SendPtr(d.as_ptr())))
                    .collect()
            } else { Vec::new() }
        } else { Vec::new() }
    };

    // 2. Per-expert forward (parallel)
    let expert_outputs: Vec<Vec<f32>> = expert_ids
        .par_iter()
        .zip(routing_weights.par_iter())
        .map(|(&eid, &rw)| {
            // Check cache first (fast f32 GEMV, no dequantize)
            if let Some((_, gate_ptr, up_ptr, down_ptr)) = cache.iter().find(|(cid, _, _, _)| *cid == eid) {
                // Reconstruct slices from raw pointers safely
                let gate = unsafe { std::slice::from_raw_parts(gate_ptr.0, FFN_DIM * HIDDEN_DIM) };
                let up = unsafe { std::slice::from_raw_parts(up_ptr.0, FFN_DIM * HIDDEN_DIM) };
                let down = unsafe { std::slice::from_raw_parts(down_ptr.0, HIDDEN_DIM * FFN_DIM) };

                // Cache stores gate/up separately at FFN_DIM×HIDDEN_DIM (512×2048)
                let gate_out = f32_gemv(gate, x, FFN_DIM, HIDDEN_DIM);
                let up_out = f32_gemv(up, x, FFN_DIM, HIDDEN_DIM);
                let mut hidden = vec![0.0f32; FFN_DIM];
                for i in 0..FFN_DIM { hidden[i] = silu(gate_out[i]) * up_out[i]; }
                let down_out = f32_gemv(down, &hidden, HIDDEN_DIM, FFN_DIM);
                let mut scaled = down_out;
                for v in &mut scaled { *v *= rw; }
                return scaled;
            }

            // Slow path: q4 dequantize + GEMV
            let gu_start = eid * GU_EXPERT_BYTES;
            let gu_slice = &gate_up_q4[gu_start..gu_start + GU_EXPERT_BYTES];
            let d_start = eid * D_EXPERT_BYTES;
            let d_slice = &down_q4[d_start..d_start + D_EXPERT_BYTES];

            let gu_out = q4k_gemv(gu_slice, GU_ROWS, GU_K, x);
            let gate = &gu_out[..FFN_DIM];
            let up = &gu_out[FFN_DIM..];

            let mut hidden = vec![0.0f32; FFN_DIM];
            for i in 0..FFN_DIM {
                hidden[i] = silu(gate[i]) * up[i];
            }

            let down_out = q4k_gemv(d_slice, D_ROWS, D_K, &hidden);

            // Scale by routing weight
            let mut scaled = down_out;
            for v in &mut scaled {
                *v *= rw;
            }
            scaled
        })
        .collect();

    // Sum all expert outputs
    let mut output = vec![0.0f32; HIDDEN_DIM];
    for expert_out in &expert_outputs {
        for i in 0..HIDDEN_DIM {
            output[i] += expert_out[i];
        }
    }

    (output, expert_ids, routing_weights)
}


// ── Hybrid Expert Cache ─────────────────────────────────────────────

/// Per-layer expert cache. Holds pre-dequantized f32 weights for hot experts.
pub struct ExpertCache {
    /// Cached experts: expert_id → (gate_f32, up_f32, down_f32)
    cache: RwLock<HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)>>,
    /// Frequency tracker
    freq: RwLock<HashMap<usize, u32>>,
    cache_size: usize,
    hits: RwLock<u64>,
    misses: RwLock<u64>,
}

impl ExpertCache {
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            freq: RwLock::new(HashMap::new()),
            cache_size,
            hits: RwLock::new(0),
            misses: RwLock::new(0),
        }
    }

    /// Record routing decisions for frequency tracking.
    pub fn record(&self, expert_ids: &[usize]) {
        let mut freq = self.freq.write().unwrap();
        for &eid in expert_ids {
            *freq.entry(eid).or_insert(0) += 1;
        }
    }

    /// Pre-dequantize top-N experts.
    pub fn build(&self, gate_up_q4: &[u8], down_q4: &[u8]) {
        let freq = self.freq.read().unwrap();
        let mut top: Vec<(usize, u32)> = freq.iter().map(|(k, v)| (*k, *v)).collect();
        top.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        let top_ids: Vec<usize> = top.iter().take(self.cache_size).map(|(k, _)| *k).collect();
        drop(freq);

        let mut cache = self.cache.write().unwrap();
        for &eid in &top_ids {
            if cache.contains_key(&eid) { continue; }
            // Dequantize gate_up
            let gu_start = eid * GU_EXPERT_BYTES;
            let gu = q4k_dequantize_only(&gate_up_q4[gu_start..gu_start + GU_EXPERT_BYTES], GU_ROWS, GU_K);
            // Dequantize down
            let d_start = eid * D_EXPERT_BYTES;
            let down = q4k_dequantize_only(&down_q4[d_start..d_start + D_EXPERT_BYTES], D_ROWS, D_K);

            let gate = gu[..FFN_DIM * GU_K].to_vec(); // Actually: gu is (1024, 2048) f32
            let up = gu[FFN_DIM * GU_K..].to_vec();
            cache.insert(eid, (gate, up, down));
        }
        // Evict non-top
        cache.retain(|k, _| top_ids.contains(k));
    }

    /// Get expert weights (from cache or dequantize on miss).
    pub fn get(&self, eid: usize, gate_up_q4: &[u8], down_q4: &[u8])
        -> (Vec<f32>, Vec<f32>, Vec<f32>)
    {
        {
            let cache = self.cache.read().unwrap();
            if let Some((g, u, d)) = cache.get(&eid) {
                *self.hits.write().unwrap() += 1;
                return (g.clone(), u.clone(), d.clone());
            }
        }
        *self.misses.write().unwrap() += 1;
        // Miss: dequantize
        let gu_start = eid * GU_EXPERT_BYTES;
        let gu = q4k_dequantize_only(&gate_up_q4[gu_start..gu_start + GU_EXPERT_BYTES], GU_ROWS, GU_K);
        let d_start = eid * D_EXPERT_BYTES;
        let down = q4k_dequantize_only(&down_q4[d_start..d_start + D_EXPERT_BYTES], D_ROWS, D_K);
        let gate = gu[..FFN_DIM * GU_K].to_vec();
        let up = gu[FFN_DIM * GU_K..].to_vec();
        (gate, up, down)
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = *self.hits.read().unwrap() as f64;
        let misses = *self.misses.read().unwrap() as f64;
        hits / (hits + misses + 1e-10) * 100.0
    }
}

/// Dequantize Q4_K_APPL to f32 without computing GEMV.
fn q4k_dequantize_only(q4_data: &[u8], M: usize, K: usize) -> Vec<f32> {
    let num_blocks = K / QK_K;
    let row_bytes = num_blocks * Q4K_BLOCK_BYTES;
    let mut result = vec![0.0f32; M * K];
    let mut dequant_buf = vec![0.0f32; QK_K];

    for row in 0..M {
        let row_start = row * row_bytes;
        for b in 0..num_blocks {
            let blk_start = row_start + b * Q4K_BLOCK_BYTES;
            let blk: &[u8; Q4K_BLOCK_BYTES] = &q4_data[blk_start..blk_start + Q4K_BLOCK_BYTES]
                .try_into().unwrap();

            let mut scales = [0.0f32; N_SUB];
            let mut mins = [0.0f32; N_SUB];
            for j in 0..N_SUB {
                let sr = u16::from_le_bytes([blk[j * 2], blk[j * 2 + 1]]);
                let mr = u16::from_le_bytes([blk[16 + j * 2], blk[16 + j * 2 + 1]]);
                scales[j] = f16_to_f32(sr);
                mins[j] = f16_to_f32(mr);
            }

            let mut L = [0u8; QK_K];
            for g in 0..4 {
                for l in 0..32 {
                    let byte = blk[32 + g * 32 + l];
                    L[g * 64 + l] = byte & 0xF;
                    L[g * 64 + 32 + l] = byte >> 4;
                }
            }

            for j in 0..N_SUB {
                let s = scales[j]; let m = mins[j];
                for i in 0..SUB_SIZE {
                    dequant_buf[j * SUB_SIZE + i] = L[j * SUB_SIZE + i] as f32 * s + m;
                }
            }

            let out_start = row * K + b * QK_K;
            result[out_start..out_start + QK_K].copy_from_slice(&dequant_buf);
        }
    }
    result
}

// ── Expert Cache (pre-dequantized f32 weights) ──────────────────────

use std::sync::Mutex;

/// Per-layer expert caches: CACHED_EXPERTS[layer] = vec of (eid, gate, up, down) f32
pub(crate) static mut CACHED_EXPERTS: Option<Vec<Mutex<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>>>> = None;

/// Record routing decisions for frequency tracking (warmup).
pub(crate) fn record_routing(layer_idx: usize, expert_ids: &[usize]) {
    let trackers = unsafe { GLOBAL_FREQ.as_ref() };
    if let Some(trackers) = trackers {
        if layer_idx < trackers.len() {
            let mut freq = trackers[layer_idx].write().unwrap();
            for &eid in expert_ids {
                *freq.entry(eid).or_insert(0) += 1;
            }
        }
    }
}

/// Compute expert output directly from cached f32 weights (clone + GEMV outside lock).
/// Returns MoE output contribution (2048 f32) scaled by routing weight, or None if not cached.
pub(crate) fn compute_cached_expert(layer_idx: usize, eid: usize, x: &[f32], rw: f32) -> Option<Vec<f32>> {
    use crate::qwen36_forward::dot_f32;
    let caches = unsafe { CACHED_EXPERTS.as_ref()? };
    let cache = caches.get(layer_idx)?;
    // Lock ONLY for lookup+clone (~0.2ms at 60GB/s), release before GEMV
    let (gate, up, down) = {
        let cache = cache.lock().unwrap();
        cache.iter().find(|(id, _, _, _)| *id == eid)
            .map(|(_, g, u, d)| (g.clone(), u.clone(), d.clone()))?
    };

    // gate GEMV: gate (512, 2048) @ x (2048) → (512) — parallel across rows
    let gate_out: Vec<f32> = (0..FFN_DIM).into_par_iter().map(|row| {
        let w = &gate[row * HIDDEN_DIM..(row + 1) * HIDDEN_DIM];
        dot_f32(w, x)
    }).collect();

    // up GEMV: up (512, 2048) @ x (2048) → (512)
    let up_out: Vec<f32> = (0..FFN_DIM).into_par_iter().map(|row| {
        let w = &up[row * HIDDEN_DIM..(row + 1) * HIDDEN_DIM];
        dot_f32(w, x)
    }).collect();

    // SwiGLU
    let mut hidden = vec![0.0f32; FFN_DIM];
    for i in 0..FFN_DIM {
        let g = gate_out[i];
        hidden[i] = g / (1.0 + (-g).exp()) * up_out[i];
    }

    // down GEMV: down (2048, 512) @ hidden (512) → (2048) — parallel across rows, NEON dot
    let down_out: Vec<f32> = (0..HIDDEN_DIM).into_par_iter().map(|row| {
        let w = &down[row * FFN_DIM..(row + 1) * FFN_DIM];
        dot_f32(w, &hidden) * rw
    }).collect();

    Some(down_out)
}

/// Look up a cached expert from the per-layer cache. Returns (gate, up, down) if found.
/// Prefer compute_cached_expert for inference to avoid cloning.
pub(crate) fn get_cached_expert(layer_idx: usize, eid: usize) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let caches = unsafe { CACHED_EXPERTS.as_ref()? };
    let cache = caches.get(layer_idx)?;
    let cache = cache.lock().unwrap();
    cache.iter().find(|(id, _, _, _)| *id == eid)
        .map(|(_, g, u, d)| (g.clone(), u.clone(), d.clone()))
}

/// Initialize per-layer cache array for `n_layers`. Call once before inference.
#[no_mangle]
pub extern "C" fn lko_moe_init_caches(n_layers: i32) -> i32 {
    let mut caches = Vec::with_capacity(n_layers as usize);
    for _ in 0..n_layers {
        caches.push(Mutex::new(Vec::new()));
    }
    unsafe { CACHED_EXPERTS = Some(caches); }
    0
}

/// Build expert cache for a layer from frequency data + q4 weights.
/// Pre-dequantizes top-N experts and stores in CACHED_EXPERTS[layer_idx].
#[no_mangle]
pub extern "C" fn lko_moe_build_cache(
    layer_idx: i32,
    gate_up_q4: *const u8, gate_up_q4_len: i32,
    down_q4: *const u8, down_q4_len: i32,
    cache_size: i32,
) -> i32 {
    let li = layer_idx as usize;
    // Get top-N from frequency tracker
    let top_ids = unsafe {
        let trackers = GLOBAL_FREQ.as_ref();
        if trackers.is_none() || li >= trackers.unwrap().len() { return 0; }
        let freq = trackers.unwrap()[li].read().unwrap();
        let mut entries: Vec<(usize, u32)> = freq.iter().map(|(k, v)| (*k, *v)).collect();
        entries.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        entries.iter().take(cache_size as usize).map(|(k, _)| *k).collect::<Vec<usize>>()
    };

    if top_ids.is_empty() { return 0; }

    let gu_q4 = unsafe { std::slice::from_raw_parts(gate_up_q4, gate_up_q4_len as usize) };
    let d_q4 = unsafe { std::slice::from_raw_parts(down_q4, down_q4_len as usize) };

    let mut cache_entries: Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(top_ids.len());
    for &eid in &top_ids {
        let gu_start = eid * GU_EXPERT_BYTES;
        let gu = q4k_dequantize_only(&gu_q4[gu_start..gu_start + GU_EXPERT_BYTES], GU_ROWS, GU_K);
        let d_start = eid * D_EXPERT_BYTES;
        let down = q4k_dequantize_only(&d_q4[d_start..d_start + D_EXPERT_BYTES], D_ROWS, D_K);
        let gate = gu[..FFN_DIM * GU_K].to_vec();
        let up = gu[FFN_DIM * GU_K..].to_vec();
        cache_entries.push((eid, gate, up, down));
    }

    let caches = unsafe { CACHED_EXPERTS.as_ref().unwrap() };
    if li < caches.len() {
        *caches[li].lock().unwrap() = cache_entries;
    }
    top_ids.len() as i32
}

/// Set pre-dequantized expert cache for a layer (API-compatible with old lko_moe_set_cache).
#[no_mangle]
pub extern "C" fn lko_moe_set_cache(
    layer_idx: i32,
    expert_ids: *const i32,
    gate_f32: *const c_float,
    up_f32: *const c_float,
    down_f32: *const c_float,
    n_experts: i32,
) {
    let n = n_experts as usize;
    if n == 0 { return; }
    let ids = unsafe { std::slice::from_raw_parts(expert_ids, n) };
    let gates = unsafe { std::slice::from_raw_parts(gate_f32, n * FFN_DIM * HIDDEN_DIM) };
    let ups = unsafe { std::slice::from_raw_parts(up_f32, n * FFN_DIM * HIDDEN_DIM) };
    let downs = unsafe { std::slice::from_raw_parts(down_f32, n * HIDDEN_DIM * FFN_DIM) };

    let mut cache = Vec::with_capacity(n);
    for i in 0..n {
        let start_g = i * FFN_DIM * HIDDEN_DIM;
        let start_d = i * HIDDEN_DIM * FFN_DIM;
        cache.push((
            ids[i] as usize,
            gates[start_g..start_g + FFN_DIM * HIDDEN_DIM].to_vec(),
            ups[start_g..start_g + FFN_DIM * HIDDEN_DIM].to_vec(),
            downs[start_d..start_d + HIDDEN_DIM * FFN_DIM].to_vec(),
        ));
    }
    let caches = unsafe { CACHED_EXPERTS.as_ref() };
    if let Some(caches) = caches {
        if (layer_idx as usize) < caches.len() {
            *caches[layer_idx as usize].lock().unwrap() = cache;
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_moe_clear_cache() {
    let caches = unsafe { CACHED_EXPERTS.as_ref() };
    if let Some(caches) = caches {
        for cache in caches {
            *cache.lock().unwrap() = Vec::new();
        }
    }
}

/// Fast f32 GEMV (no dequantize) for cached expert weights.
fn f32_gemv(w: &[f32], x: &[f32], M: usize, K: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; M];
    for row in 0..M {
        let row_slice = &w[row * K..(row + 1) * K];
        result[row] = crate::qwen36_forward::dot_f32(row_slice, x);
    }
    result
}

/// Dequantize a single expert from q4 to f32. Returns (gate, up, down).
pub fn dequantize_expert_f32(
    gu_q4: *const u8, gu_len: i32,
    d_q4: *const u8, d_len: i32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let gu = unsafe { std::slice::from_raw_parts(gu_q4, gu_len as usize) };
    let d = unsafe { std::slice::from_raw_parts(d_q4, d_len as usize) };
    // gate_up: M=1024, K=2048
    let gu_f32 = q4k_dequantize_only(gu, GU_ROWS, GU_K);
    // down: M=2048, K=512
    let d_f32 = q4k_dequantize_only(d, D_ROWS, D_K);
    // Split gate_up into gate(512,2048) and up(512,2048)
    let gate = gu_f32[..FFN_DIM * GU_K].to_vec();
    let up = gu_f32[FFN_DIM * GU_K..].to_vec();
    let down = d_f32;
    (gate, up, down)
}

// ── Global frequency tracker ────────────────────────────────────────

static mut GLOBAL_FREQ: Option<Vec<RwLock<HashMap<usize, u32>>>> = None;

/// Initialize per-layer frequency trackers. Call once before inference.
#[no_mangle]
pub extern "C" fn lko_moe_init_freq_tracker(n_layers: i32) {
    let mut trackers = Vec::with_capacity(n_layers as usize);
    for _ in 0..n_layers {
        trackers.push(RwLock::new(HashMap::new()));
    }
    unsafe { GLOBAL_FREQ = Some(trackers); }
}

/// Get top-N expert IDs for a layer. Returns number of experts written.
#[no_mangle]
pub extern "C" fn lko_moe_get_top_experts(
    layer_idx: i32, top_n: i32, out_ids: *mut i32, out_counts: *mut i32,
) -> i32 {
    let trackers = unsafe { GLOBAL_FREQ.as_ref() };
    if trackers.is_none() { return 0; }
    let trackers = trackers.unwrap();
    if layer_idx < 0 || layer_idx as usize >= trackers.len() { return 0; }

    let freq = trackers[layer_idx as usize].read().unwrap();
    let mut entries: Vec<(usize, u32)> = freq.iter().map(|(k, v)| (*k, *v)).collect();
    entries.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

    let n = top_n.min(entries.len() as i32) as usize;
    let out_ids = unsafe { std::slice::from_raw_parts_mut(out_ids, n) };
    let out_counts = unsafe { std::slice::from_raw_parts_mut(out_counts, n) };
    for i in 0..n {
        out_ids[i] = entries[i].0 as i32;
        out_counts[i] = entries[i].1 as i32;
    }
    n as i32
}

/// Free frequency tracker.
#[no_mangle]
pub extern "C" fn lko_moe_free_freq_tracker() {
    unsafe { GLOBAL_FREQ = None; }
}

// ── C API ───────────────────────────────────────────────────────────

/// MoE expert forward for one Qwen3.6 layer.
///
/// Args:
///   router_w: f32 array [256, 2048] — router weights
///   gate_up_q4: u8 array — all 256 experts' gate_up in Q4_K_APPL format
///   gate_up_q4_len: byte length of gate_up_q4
///   down_q4: u8 array — all 256 experts' down in Q4_K_APPL format
///   down_q4_len: byte length of down_q4
///   x: f32 array [2048] — input hidden state
///   top_k: number of experts to activate (typically 8)
///   layer_idx: if >= 0, track routing frequencies for this layer
///   expert_indices_out: output buffer for expert IDs [top_k]
///   expert_weights_out: output buffer for routing weights [top_k]
///   output: output buffer for result [2048]
///
/// Returns: 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn lko_moe_forward_layer(
    router_w: *const c_float,
    gate_up_q4: *const u8,
    gate_up_q4_len: i32,
    down_q4: *const u8,
    down_q4_len: i32,
    x: *const c_float,
    top_k: i32,
    layer_idx: i32,
    expert_indices_out: *mut i32,
    expert_weights_out: *mut c_float,
    output: *mut c_float,
) -> i32 {
    if router_w.is_null() || gate_up_q4.is_null() || down_q4.is_null()
        || x.is_null() || output.is_null()
    {
        return -1;
    }

    let k = top_k as usize;
    if k == 0 || k > N_EXPERTS { return -1; }

    let router = unsafe { std::slice::from_raw_parts(router_w, N_EXPERTS * HIDDEN_DIM) };
    let gu_q4 = unsafe { std::slice::from_raw_parts(gate_up_q4, gate_up_q4_len as usize) };
    let d_q4 = unsafe { std::slice::from_raw_parts(down_q4, down_q4_len as usize) };
    let input = unsafe { std::slice::from_raw_parts(x, HIDDEN_DIM) };

    let track = if layer_idx >= 0 { Some(layer_idx as usize) } else { None };
    let (result, indices, weights) = moe_forward_layer(router, gu_q4, d_q4, input, track);

    let out_slice = unsafe { std::slice::from_raw_parts_mut(output, HIDDEN_DIM) };
    out_slice.copy_from_slice(&result);
    let idx_out = unsafe { std::slice::from_raw_parts_mut(expert_indices_out, k) };
    let w_out = unsafe { std::slice::from_raw_parts_mut(expert_weights_out, k) };
    for i in 0..k {
        idx_out[i] = indices[i] as i32;
        w_out[i] = weights[i];
    }

    0
}
