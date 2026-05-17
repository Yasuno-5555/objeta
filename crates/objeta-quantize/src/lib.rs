//! objeta-quantize — Phase-adaptive quantization with Lyapunov-aware bit allocation.
//!
//! Core algorithm:
//!   1. Compute per-layer sensitivity = Lyapunov estimate × zone multiplier × inversion factor
//!   2. Allocate bits proportional to log(sensitivity) with budget constraint
//!   3. Output per-layer format assignments (q2, q3, q4, q5, q8, fp16)
//!
//! This replaces uniform quantization (all q4) with trajectory-aware precision:
//!   - UNFOLD (L2): high precision — basin compiler, J≠I
//!   - ISOMETRIC (L3-L13): aggressive quantization — λ≈0, safe
//!   - DIVERGENT (L14-L21): medium precision — λ>0 but J≈I
//!   - SYNC (L0-L1): medium precision — short, anti-damped

use objeta_core::{LayerZone, PhaseProfile, StabilityMap};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tracing::info;

// ── Quantization Plan ───────────────────────────────────────────────────────

/// Complete per-layer quantization plan, serializable to JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationPlan {
    pub model_name: String,
    pub phase: String,
    pub family: String,
    pub n_layers: usize,
    pub hidden_dim: usize,
    /// Average bits per weight (across all layers)
    pub average_bits: f64,
    /// Total compressed size (bytes, weights only)
    pub total_bytes: u64,
    /// Original fp16 size for comparison
    pub fp16_bytes: u64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Per-layer quantization entries
    pub layers: Vec<LayerQuantization>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQuantization {
    pub layer_idx: usize,
    pub zone: String,
    /// Allocated bits per weight
    pub bits: u8,
    /// Quantization format tag
    pub format: String,
    /// Lyapunov estimate
    pub lyapunov: f64,
    /// Sensitivity score (normalized, 0-1)
    pub sensitivity: f64,
    /// Justification for this allocation
    pub reason: String,
    /// Block bytes for this format
    pub block_bytes: usize,
}

// ── Bit Budget ──────────────────────────────────────────────────────────────

/// Bit allocation budget configuration.
#[derive(Debug, Clone)]
pub struct BitBudget {
    /// Target average bits per weight across all layers
    pub target_avg_bits: f64,
    /// Available bit widths (sorted)
    pub available_bits: Vec<u8>,
    /// Minimum bits for any layer
    pub min_bits: u8,
    /// Maximum bits for any layer
    pub max_bits: u8,
}

impl Default for BitBudget {
    fn default() -> Self {
        Self {
            target_avg_bits: 4.0, // match uniform q4
            available_bits: vec![2, 3, 4, 5, 8, 16],
            min_bits: 2,
            max_bits: 16,
        }
    }
}

// ── Sensitivity Computation ─────────────────────────────────────────────────

/// Zone amplification multipliers.
///
/// Based on LKO empirical data:
///   - UNFOLD: J≠I, σ_max≈48, 2.7× dominant per-layer sensitivity
///   - DIVERGENT: λ>0, perturbation amplification 2.6×
///   - SYNC: anti-damped (D<0), short but structurally important
///   - ISOMETRIC: λ≈0, J≈I, CFL α>5 — maximally safe
const ZONE_MULTIPLIER: &[(LayerZone, f64)] = &[
    (LayerZone::Unfold, 2.7),
    (LayerZone::Divergent, 1.8),
    (LayerZone::Sync, 1.0),
    (LayerZone::IsometricLocal, 0.4),
    (LayerZone::IsometricGlobal, 0.5),
];

/// Inversion zone bonus multiplier. Layers where cos(Δ_l, Δ_{l+1}) < -0.01
/// get extra sensitivity because sign flips amplify errors directionally.
const INVERSION_MULTIPLIER: f64 = 1.5;

/// Lyapunov floor — minimum assumed amplification to avoid zero sensitivity.
const LYAPUNOV_FLOOR: f64 = 0.5;

/// Compute per-layer sensitivity scores from a stability map.
///
/// sensitivity_l = lyapunov_l × zone_multiplier × [inversion_multiplier]
///
/// Lyapunov is floored at 0.5 to avoid zero sensitivity in anti-damped zones.
fn compute_sensitivity(stability: &StabilityMap) -> Vec<f64> {
    stability
        .per_layer
        .iter()
        .map(|ls| {
            let lyap = ls.lyapunov.max(LYAPUNOV_FLOOR);
            let zone_mult = ZONE_MULTIPLIER
                .iter()
                .find(|(z, _)| *z == ls.zone)
                .map(|(_, m)| *m)
                .unwrap_or(1.0);
            let inv_mult = if ls.inversion_active {
                INVERSION_MULTIPLIER
            } else {
                1.0
            };
            lyap * zone_mult * inv_mult
        })
        .collect()
}

// ── Bit Allocation ──────────────────────────────────────────────────────────

/// Allocate bits to layers proportional to log(sensitivity).
///
/// The objective is to minimize:
///   Σ_l sensitivity_l × reconstruction_error(bits_l)
///
/// where reconstruction_error(bits) ∝ 2^(-bits) (standard quantization theory).
///
/// Solving the constrained optimization gives:
///   bits_l ∝ log(sensitivity_l) / log(2)
///
/// We then snap to the nearest available bit width and adjust to meet budget.
fn allocate_bits(
    sensitivity: &[f64],
    budget: &BitBudget,
) -> Vec<u8> {
    let n = sensitivity.len();
    let available = &budget.available_bits;

    // log(sensitivity) → continuous bit allocation
    let log_sens: Vec<f64> = sensitivity.iter().map(|s| s.max(1e-10).ln()).collect();
    let mean_log = log_sens.iter().sum::<f64>() / n as f64;

    // Map log sensitivity to bits: bits = target_avg + (log(s) - mean_log) / log(2)
    let cont_bits: Vec<f64> = log_sens
        .iter()
        .map(|ls| {
            let raw = budget.target_avg_bits + (ls - mean_log) / std::f64::consts::LN_2;
            raw.clamp(budget.min_bits as f64, budget.max_bits as f64)
        })
        .collect();

    // Snap to nearest available bit width
    let mut bits: Vec<u8> = cont_bits
        .iter()
        .map(|&cb| {
            available
                .iter()
                .min_by(|&a, &b| {
                    ((*a as f64) - cb)
                        .abs()
                        .partial_cmp(&((*b as f64) - cb).abs())
                        .unwrap()
                })
                .copied()
                .unwrap_or(4)
        })
        .collect();

    // Adjust to meet budget: if average is off, bump highest-sensitivity layers up
    // or lowest-sensitivity layers down
    let avg: f64 = bits.iter().map(|&b| b as f64).sum::<f64>() / n as f64;
    if (avg - budget.target_avg_bits).abs() > 0.1 {
        if avg > budget.target_avg_bits {
            // Too many bits: reduce lowest-sensitivity layers
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| sensitivity[a].partial_cmp(&sensitivity[b]).unwrap());
            let mut excess = ((avg - budget.target_avg_bits) * n as f64).round() as i32;
            for &idx in &indices {
                if excess <= 0 {
                    break;
                }
                let cur = bits[idx];
                if let Some(&lower) = available.iter().filter(|&&b| b < cur).max() {
                    bits[idx] = lower;
                    excess -= (cur - lower) as i32;
                }
            }
        } else {
            // Too few bits: increase highest-sensitivity layers
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| sensitivity[b].partial_cmp(&sensitivity[a]).unwrap());
            let mut deficit = ((budget.target_avg_bits - avg) * n as f64).round() as i32;
            for &idx in &indices {
                if deficit <= 0 {
                    break;
                }
                let cur = bits[idx];
                if let Some(&higher) = available.iter().filter(|&&b| b > cur).min() {
                    bits[idx] = higher;
                    deficit -= (higher - cur) as i32;
                }
            }
        }
    }

    bits
}

// ── Format Mapping ──────────────────────────────────────────────────────────

fn format_for_bits(bits: u8) -> (&'static str, usize) {
    match bits {
        2 => ("q2_k_appl", 96),
        3 => ("q3_k_appl", 128),
        4 => ("q4_k_appl", 160),
        5 => ("q5_k_appl", 192),
        8 => ("q8_k_appl", 288),
        16 => ("fp16", 512),
        _ => ("q4_k_appl", 160),
    }
}

fn reason_for_layer(
    idx: usize,
    zone: LayerZone,
    bits: u8,
    sensitivity: f64,
    lyapunov: f64,
    inversion: bool,
) -> String {
    let zone_name = format!("{:?}", zone);
    let mut parts = vec![format!("zone={}", zone_name)];

    if idx == 2 {
        parts.push("basin_compiler=J≠I".into());
    }
    if lyapunov > 1.5 {
        parts.push(format!("lyapunov={:.1}", lyapunov));
    }
    if inversion {
        parts.push("inversion_active".into());
    }
    if bits <= 2 {
        parts.push("safe_zone=λ≈0".into());
    }
    if bits >= 8 {
        parts.push("stability_critical".into());
    }

    parts.join(", ")
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Generate a complete quantization plan from a phase profile.
///
/// Uses Lyapunov-weighted sensitivity to allocate bits per layer.
pub fn generate_plan(profile: &PhaseProfile) -> QuantizationPlan {
    generate_plan_with_budget(profile, &BitBudget::default())
}

/// Generate a quantization plan with a custom bit budget.
pub fn generate_plan_with_budget(profile: &PhaseProfile, budget: &BitBudget) -> QuantizationPlan {
    info!(
        "Generating quantization plan: {} layers, target_avg={:.1} bits",
        profile.n_layers,
        budget.target_avg_bits
    );

    let stability = objeta_routing::generate_stability_map(profile);
    let sensitivity = compute_sensitivity(&stability);
    let bits = allocate_bits(&sensitivity, budget);

    // Normalize sensitivity for display
    let max_sens = sensitivity
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(1.0);

    let layers: Vec<LayerQuantization> = (0..profile.n_layers)
        .map(|l| {
            let ls = &stability.per_layer[l];
            let (format_name, block_bytes) = format_for_bits(bits[l]);
            let sens_norm = if max_sens > 1e-10 {
                sensitivity[l] / max_sens
            } else {
                0.0
            };

            LayerQuantization {
                layer_idx: l,
                zone: format!("{:?}", ls.zone),
                bits: bits[l],
                format: format_name.to_string(),
                lyapunov: ls.lyapunov,
                sensitivity: sens_norm,
                reason: reason_for_layer(
                    l,
                    ls.zone,
                    bits[l],
                    sens_norm,
                    ls.lyapunov,
                    ls.inversion_active,
                ),
                block_bytes,
            }
        })
        .collect();

    // Compute aggregate statistics
    let avg_bits = bits.iter().map(|&b| b as f64).sum::<f64>() / bits.len() as f64;
    let fp16_bytes = (profile.hidden_dim * profile.ffn_dim * profile.n_layers * 2) as u64;
    let total_bytes: u64 = layers
        .iter()
        .map(|lq| {
            // Approximate: rows * blocks_per_row * block_bytes
            let rows = profile.ffn_dim;
            let cols = profile.hidden_dim;
            let nblocks = (cols + 255) / 256; // QK_K = 256
            (rows * nblocks * lq.block_bytes) as u64
        })
        .sum();

    let compression_ratio = if total_bytes > 0 {
        fp16_bytes as f64 / total_bytes as f64
    } else {
        1.0
    };

    info!(
        "Plan: avg={:.1} bits, total={:.1}MB, fp16={:.1}MB, ratio={:.1}x",
        avg_bits,
        total_bytes as f64 / 1_000_000.0,
        fp16_bytes as f64 / 1_000_000.0,
        compression_ratio
    );

    QuantizationPlan {
        model_name: profile.model_name.clone(),
        phase: format!("{:?}", profile.phase),
        family: format!("{:?}", profile.family),
        n_layers: profile.n_layers,
        hidden_dim: profile.hidden_dim,
        average_bits: avg_bits,
        total_bytes,
        fp16_bytes,
        compression_ratio,
        layers,
    }
}

/// Compute memory savings vs. uniform quantization.
pub fn compute_savings(plan: &QuantizationPlan) -> SavingsReport {
    let uniform_q4_bytes = plan.layers.iter().map(|lq| {
        let rows = plan.hidden_dim; // approximate
        let nblocks = (plan.hidden_dim + 255) / 256;
        (rows * nblocks * 160) as u64 // q4 = 160 bytes/block
    }).sum::<u64>();

    SavingsReport {
        uniform_q4_bytes,
        phase_adaptive_bytes: plan.total_bytes,
        savings_bytes: uniform_q4_bytes.saturating_sub(plan.total_bytes),
        savings_percent: if uniform_q4_bytes > 0 {
            (uniform_q4_bytes - plan.total_bytes) as f64 / uniform_q4_bytes as f64 * 100.0
        } else {
            0.0
        },
        high_precision_layers: plan.layers.iter().filter(|lq| lq.bits >= 8).count(),
        low_precision_layers: plan.layers.iter().filter(|lq| lq.bits <= 2).count(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavingsReport {
    pub uniform_q4_bytes: u64,
    pub phase_adaptive_bytes: u64,
    pub savings_bytes: u64,
    pub savings_percent: f64,
    pub high_precision_layers: usize,
    pub low_precision_layers: usize,
}

impl std::fmt::Display for SavingsReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "  Uniform q4:     {:>8.1} MB",
            self.uniform_q4_bytes as f64 / 1_000_000.0
        )?;
        writeln!(
            f,
            "  Phase-adaptive: {:>8.1} MB",
            self.phase_adaptive_bytes as f64 / 1_000_000.0
        )?;
        writeln!(
            f,
            "  Savings:        {:>8.1} MB ({:.1}%)",
            self.savings_bytes as f64 / 1_000_000.0,
            self.savings_percent
        )?;
        writeln!(
            f,
            "  High-precision layers (≥8bit): {}",
            self.high_precision_layers
        )?;
        writeln!(
            f,
            "  Low-precision layers  (≤2bit): {}",
            self.low_precision_layers
        )
    }
}

// ── Attention Backbone Plan — LKO Phase C findings ─────────────────────────

/// Per-component quantization assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentQuantization {
    pub layer_idx: usize,
    pub zone: String,
    /// FFN weights (gate_proj, up_proj, down_proj)
    pub ffn_bits: u8,
    /// Attention Q/O projection (query, output)
    pub attn_qo_bits: u8,
    /// Attention K/V projection (key, value)
    pub attn_kv_bits: u8,
    pub ffn_format: String,
    pub attn_qo_format: String,
    pub attn_kv_format: String,
}

/// Complete per-component quantization plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPlan {
    pub model_name: String,
    pub phase: String,
    pub family: String,
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub average_bits: f64,
    pub total_bytes: u64,
    pub fp16_bytes: u64,
    pub compression_ratio: f64,
    /// Weight fractions for averaging
    pub ffn_weight_fraction: f64,
    pub attn_qo_weight_fraction: f64,
    pub attn_kv_weight_fraction: f64,
    pub layers: Vec<ComponentQuantization>,
}

/// Generate an attention backbone quantization plan.
///
/// Based on Phase C experimental findings:
///   - Attention Q/O = q5 (transport routing — mandatory)
///   - Attention K/V = q4 (memory storage — moderate)
///   - FFN = q3.5 (local modulation — can degrade)
///
/// This is NOT layer-wise allocation. It is component-wise allocation
/// based on the discovery that attention determines trajectory transport
/// capacity while FFN can be aggressively compressed.
pub fn generate_attention_backbone_plan(profile: &PhaseProfile) -> ComponentPlan {
    generate_attention_backbone_with_params(profile, 5, 4, 3.5)
}

/// Attention backbone with custom precision levels.
pub fn generate_attention_backbone_with_params(
    profile: &PhaseProfile,
    attn_qo_bits: u8,
    attn_kv_bits: u8,
    ffn_bits: f64,
) -> ComponentPlan {
    let stability = objeta_routing::generate_stability_map(profile);
    let n = profile.n_layers;

    // Per-component format lookup (floor to get aggressive quantization)
    let ffn_bits_u8 = ffn_bits.floor() as u8;
    let (ffn_fmt, ffn_block_bytes) = format_for_bits(ffn_bits_u8);
    let (qo_fmt, qo_block_bytes) = format_for_bits(attn_qo_bits);
    let (kv_fmt, kv_block_bytes) = format_for_bits(attn_kv_bits);

    let layers: Vec<ComponentQuantization> = (0..n)
        .map(|l| {
            let ls = &stability.per_layer[l];
            ComponentQuantization {
                layer_idx: l,
                zone: format!("{:?}", ls.zone),
                ffn_bits: ffn_bits_u8,
                attn_qo_bits,
                attn_kv_bits,
                ffn_format: ffn_fmt.to_string(),
                attn_qo_format: qo_fmt.to_string(),
                attn_kv_format: kv_fmt.to_string(),
            }
        })
        .collect();

    // Weight fractions (approximate for TinyLlama/Qwen3.6)
    // FFN: gate + up + down ≈ 3 * ffn_dim * hidden
    // Attn QO: Q + O ≈ 2 * hidden * hidden (plus head overhead)
    // Attn KV: K + V ≈ 2 * hidden * kv_dim ≈ smaller in GQA
    let ffn_w = 3.0 * profile.ffn_dim as f64 * profile.hidden_dim as f64;
    let attn_qo_w = 2.0 * profile.hidden_dim as f64 * profile.hidden_dim as f64;
    let attn_kv_w = 2.0 * profile.hidden_dim as f64
        * (profile.n_kv_heads * profile.head_dim) as f64;
    let total_w = ffn_w + attn_qo_w + attn_kv_w;

    let ffn_frac = ffn_w / total_w;
    let qo_frac = attn_qo_w / total_w;
    let kv_frac = attn_kv_w / total_w;

    let avg_bits = ffn_bits * ffn_frac
        + attn_qo_bits as f64 * qo_frac
        + attn_kv_bits as f64 * kv_frac;

    // Estimate total bytes (FFN + Attn) — per-layer, per-matrix
    let blocks_per_row = (profile.hidden_dim + 255) / 256;
    // FFN: gate(ffn×hidden) + up(ffn×hidden) + down(hidden×ffn)
    let ffn_bytes = (2 * profile.ffn_dim * blocks_per_row * ffn_block_bytes
                     + profile.hidden_dim * ((profile.ffn_dim + 255) / 256) * ffn_block_bytes) * n;
    // Attn QO: Q_proj(hidden×hidden) + O_proj(hidden×hidden)
    let qo_bytes = 2 * profile.hidden_dim * blocks_per_row * qo_block_bytes * n;
    // Attn KV: K_proj(kv_heads*head_dim×hidden) + V_proj(kv_heads*head_dim×hidden)
    let kv_rows = profile.n_kv_heads * profile.head_dim;
    let kv_bytes = 2 * kv_rows * blocks_per_row * kv_block_bytes * n;

    let total_bytes = (ffn_bytes + qo_bytes + kv_bytes) as u64;
    let fp16_bytes = (total_w * 2.0 * n as f64) as u64;
    let compression_ratio = if total_bytes > 0 { fp16_bytes as f64 / total_bytes as f64 } else { 1.0 };

    info!(
        "Attention Backbone plan: avg={:.2} bits (FFN={} QO={} KV={}), ratio={:.1}x",
        avg_bits, ffn_fmt, qo_fmt, kv_fmt, compression_ratio
    );

    ComponentPlan {
        model_name: profile.model_name.clone(),
        phase: format!("{:?}", profile.phase),
        family: format!("{:?}", profile.family),
        n_layers: n,
        hidden_dim: profile.hidden_dim,
        average_bits: avg_bits,
        total_bytes,
        fp16_bytes,
        compression_ratio,
        ffn_weight_fraction: ffn_frac,
        attn_qo_weight_fraction: qo_frac,
        attn_kv_weight_fraction: kv_frac,
        layers,
    }
}

// ── Reference Static Allocation (from LKO data) ────────────────────────────

/// Generate a quantization plan using static LKO-derived rules.
///
/// This is the simpler, empirically-validated approach:
///   L0-L1:    q4 (SYNC — anti-damped but short)
///   L2:       fp16 (UNFOLD — basin compiler, mandatory)
///   L3-L13:   q2 (ISOMETRIC — λ≈0, maximally safe)
///   L14-L20:  q5 (DIVERGENT — λ>0, conservative)
///   L21:      q4 (output — needs reasonable fidelity)
pub fn generate_static_plan(profile: &PhaseProfile) -> QuantizationPlan {
    let n = profile.n_layers;
    let diverge_start = (0.7 * n as f64).ceil() as usize;
    let isometric_end = diverge_start.saturating_sub(1);

    let mut bits = vec![4u8; n]; // default q4

    for l in 0..n {
        bits[l] = if l <= 1 {
            4 // SYNC: q4
        } else if l == 2 {
            16 // UNFOLD: fp16 (mandatory)
        } else if l <= isometric_end {
            2 // ISOMETRIC: q2 (ultra-aggressive)
        } else if l < n - 1 {
            5 // DIVERGENT: q5 (conservative)
        } else {
            4 // Last layer: q4
        };
    }

    build_plan_from_bits(profile, &bits)
}

fn build_plan_from_bits(profile: &PhaseProfile, bits: &[u8]) -> QuantizationPlan {
    let stability = objeta_routing::generate_stability_map(profile);
    let sensitivity = compute_sensitivity(&stability);
    let max_sens = sensitivity
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(1.0);

    let layers: Vec<LayerQuantization> = (0..profile.n_layers)
        .map(|l| {
            let ls = &stability.per_layer[l];
            let (format_name, block_bytes) = format_for_bits(bits[l]);
            let sens_norm = if max_sens > 1e-10 {
                sensitivity[l] / max_sens
            } else {
                0.0
            };

            LayerQuantization {
                layer_idx: l,
                zone: format!("{:?}", ls.zone),
                bits: bits[l],
                format: format_name.to_string(),
                lyapunov: ls.lyapunov,
                sensitivity: sens_norm,
                reason: reason_for_layer(
                    l,
                    ls.zone,
                    bits[l],
                    sens_norm,
                    ls.lyapunov,
                    ls.inversion_active,
                ),
                block_bytes,
            }
        })
        .collect();

    let avg_bits = bits.iter().map(|&b| b as f64).sum::<f64>() / bits.len() as f64;
    let fp16_bytes = (profile.hidden_dim * profile.ffn_dim * profile.n_layers * 2) as u64;
    let total_bytes: u64 = layers
        .iter()
        .map(|lq| {
            let rows = profile.ffn_dim;
            let nblocks = (profile.hidden_dim + 255) / 256;
            (rows * nblocks * lq.block_bytes) as u64
        })
        .sum();
    let compression_ratio = if total_bytes > 0 {
        fp16_bytes as f64 / total_bytes as f64
    } else {
        1.0
    };

    QuantizationPlan {
        model_name: profile.model_name.clone(),
        phase: format!("{:?}", profile.phase),
        family: format!("{:?}", profile.family),
        n_layers: profile.n_layers,
        hidden_dim: profile.hidden_dim,
        average_bits: avg_bits,
        total_bytes,
        fp16_bytes,
        compression_ratio,
        layers,
    }
}

// ── Family-Aware Runtime Strategy ──────────────────────────────────────────

use objeta_core::{
    ComponentPrecision, ComputePolicy, ExecutorConfig, Family, Phase, RuntimeStrategy,
    SensitivityDominance,
};

/// Generate a family-aware runtime strategy from phase profile.
///
/// Cross-family validated rules (2026-05-17):
///
///   Family A — Residual Transport (TinyLlama, Llama):
///     Attention determines transport capacity (8.8x asymmetry).
///     Strategy: Attn q5+, FFN q3.5. Preserve KV/attention precision.
///
///   Family B Phase 1 — Aligned Field (Qwen2.5-0.5B):
///     FFN coherence is critical (0.1x asymmetry — inverted).
///     Strategy: FFN q5+, Attn q4. Preserve FFN precision.
///
///   Family B Phase 3 — Mixed Field (Qwen3.6-35B):
///     GQA/Full-attn layers are steering backbone. Delta/linear can degrade.
///     Strategy: GQA q5+, DeltaNet q4, FFN q3-q4.
///     Steering layers (every 4th) get highest precision.
pub fn generate_runtime_strategy(profile: &PhaseProfile) -> RuntimeStrategy {
    let dominance = classify_dominance(profile);
    let confidence = compute_confidence(profile);
    let steering_layers = detect_steering_layers(profile);
    let comp_precision = component_precision_for(profile, dominance);
    let layer_policies = per_layer_policies(profile, dominance, &steering_layers);
    let description = strategy_description(dominance);
    let executor_config = build_executor_config(profile, &comp_precision, &steering_layers);

    RuntimeStrategy {
        model_name: profile.model_name.clone(),
        family: profile.family,
        phase: profile.phase,
        dominance,
        confidence,
        layer_policies,
        component_precision: comp_precision,
        steering_layers,
        description,
        executor_config,
    }
}

fn compute_confidence(profile: &PhaseProfile) -> f64 {
    // Confidence based on signal clarity of the phase metrics
    let valid_intra: Vec<f64> = profile.layers.iter().filter_map(|l| l.intra_cos).collect();
    if valid_intra.is_empty() {
        return 0.5; // no signal → low confidence
    }

    let mean_intra = valid_intra.iter().sum::<f64>() / valid_intra.len() as f64;
    let intra_std = {
        let m = mean_intra;
        (valid_intra.iter().map(|v| (v - m).powi(2)).sum::<f64>() / valid_intra.len() as f64).sqrt()
    };

    let mean_eff = profile.layers.iter().map(|l| l.effective_rank).sum::<f64>()
        / profile.n_layers.max(1) as f64;

    // Confidence rules based on experimental data:
    // - Phase 1 (aligned): high intra_cos, low std, low eff_rank → high confidence
    // - Phase 2 (split): moderate intra, moderate std → moderate confidence
    // - Phase 3 (mixed): high std → high confidence
    match profile.family {
        Family::ResidualTransport => {
            // Family A: confidence from intra_cos being close to 0 (independent attn/ffn)
            let separation = (mean_intra - 0.0).abs();
            if separation < 0.3 { 0.92 } else if separation < 0.5 { 0.75 } else { 0.55 }
        }
        Family::SphericalSteering => {
            if mean_intra > 0.95 && mean_eff < 5.0 {
                0.95 // clear Phase 1 signal
            } else if intra_std > 0.3 {
                0.91 // clear Phase 3 signal (mixed field)
            } else {
                0.65 // ambiguous — could be transitional
            }
        }
    }
}

fn build_executor_config(
    profile: &PhaseProfile,
    precision: &ComponentPrecision,
    steering_layers: &[usize],
) -> ExecutorConfig {
    let n = profile.n_layers;
    let ffn_bits: Vec<u8> = (0..n)
        .map(|l| if steering_layers.contains(&l) && l > 2 {
            // Steering layers get one level higher FFN
            (precision.ffn_bits + 1).min(8)
        } else {
            precision.ffn_bits
        })
        .collect();

    let attn_qo_bits: Vec<u8> = (0..n)
        .map(|l| if steering_layers.contains(&l) {
            8u8 // steering layers: q8 minimum
        } else {
            precision.attn_qo_bits
        })
        .collect();

    let attn_kv_bits: Vec<u8> = vec![precision.attn_kv_bits; n];

    let (fusion_ratio, moe_on_deltanet) = match profile.family {
        Family::SphericalSteering if profile.phase == Phase::MixedField => (0.33, false),
        Family::SphericalSteering => (0.5, true),
        Family::ResidualTransport => (1.0, true),
    };

    // Performance estimates based on M1 8GB benchmark data (2026-05-17)
    let avg = precision.average_bits;
    let base_tok_s = 0.21;
    let fusion_speedup = if fusion_ratio < 0.4 { 4.0 } else if fusion_ratio < 0.7 { 2.5 } else { 1.0 };
    let moe_speedup = if !moe_on_deltanet { 2.5 } else { 1.0 };
    let estimated_tok_per_sec = base_tok_s * fusion_speedup * moe_speedup;
    let estimated_vram_gb = profile.hidden_dim as f64 * profile.ffn_dim as f64 * n as f64
        * avg / 8.0 / 1e9 * 3.5;
    let estimated_ppl_delta = if avg >= 5.0 { 0.1 } else if avg >= 4.0 { 0.8 } else { 2.5 };

    ExecutorConfig {
        ffn_bits,
        attn_qo_bits,
        attn_kv_bits,
        fusion_ratio,
        moe_on_deltanet,
        estimated_tok_per_sec,
        estimated_vram_gb,
        estimated_ppl_delta,
    }
}

fn classify_dominance(profile: &PhaseProfile) -> SensitivityDominance {
    match (profile.family, profile.phase) {
        // Family A: Residual Transport → attention is the transport bottleneck
        (Family::ResidualTransport, _) => SensitivityDominance::AttentionBandwidth,

        // Family B: depends on phase
        (Family::SphericalSteering, Phase::Collapse1D) => {
            // Phase 1: aligned field → FFN coherence dominates (Qwen2.5 pattern)
            SensitivityDominance::FfnCoherence
        }
        (Family::SphericalSteering, Phase::MixedField) => {
            // Phase 3: mixed field → GQA steering backbone (Qwen3.6 pattern)
            SensitivityDominance::SteeringBackbone
        }
        (Family::SphericalSteering, Phase::Split2D) => {
            // Phase 2 is Family A territory, but SphericalSteering + Split2D is unusual
            // Default to attention bandwidth
            SensitivityDominance::AttentionBandwidth
        }
    }
}

fn detect_steering_layers(profile: &PhaseProfile) -> Vec<usize> {
    // Steering layers: layers with large relative_steering or inversion activity
    // For Qwen3.6: every 4th layer is GQA (steering backbone)
    // For generic models: layers where ||Δ|| >> mean
    if profile.n_layers == 40 && profile.phase == Phase::MixedField {
        // Qwen3.6 pattern: GQA layers at L3, L7, L11, L15, L19, L23, L27, L31, L35, L39
        return (0..40).filter(|l| l % 4 == 3).collect();
    }

    // Generic detection: layers with high relative steering or in inversion zone
    let mean_rel = profile.layers.iter()
        .filter_map(|l| l.relative_steering)
        .sum::<f64>() / profile.n_layers.max(1) as f64;

    profile.layers.iter()
        .filter(|l| {
            let high_steering = l.relative_steering.unwrap_or(0.0) > mean_rel * 1.5;
            let in_inversion = l.steering_cos.unwrap_or(1.0) < -0.01;
            high_steering || in_inversion
        })
        .map(|l| l.layer_idx)
        .collect()
}

fn component_precision_for(
    profile: &PhaseProfile,
    dominance: SensitivityDominance,
) -> ComponentPrecision {
    match dominance {
        SensitivityDominance::AttentionBandwidth => {
            // Family A: Attn determines transport. FFN can degrade.
            // TinyLlama data: Attn5+FFN3 = 14.4 PPL, Attn3+FFN5 = 127.6
            let (ffn_b, qo_b, kv_b) = (3u8, 5u8, 4u8);
            let avg = compute_avg_bits(profile, ffn_b, qo_b, kv_b);
            ComponentPrecision {
                attn_qo_bits: qo_b, attn_kv_bits: kv_b, ffn_bits: ffn_b,
                average_bits: avg, compression_ratio: 16.0 / avg,
            }
        }
        SensitivityDominance::FfnCoherence => {
            // Family B Phase 1: FFN coherence is critical. Attn can degrade.
            // Qwen2.5 data: Attn3+FFN5 = 26.7 PPL, Attn5+FFN3 = 253.3
            let (ffn_b, qo_b, kv_b) = (5u8, 4u8, 4u8);
            let avg = compute_avg_bits(profile, ffn_b, qo_b, kv_b);
            ComponentPrecision {
                attn_qo_bits: qo_b, attn_kv_bits: kv_b, ffn_bits: ffn_b,
                average_bits: avg, compression_ratio: 16.0 / avg,
            }
        }
        SensitivityDominance::SteeringBackbone => {
            // Family B Phase 3: Steering layers (GQA) need high precision.
            // DeltaNet and FFN can degrade.
            // Qwen3.6 probe: GQA q5+ essential, DeltaNet q4 OK, FFN q3-q4
            let (ffn_b, qo_b, kv_b) = (3u8, 5u8, 4u8);
            let avg = compute_avg_bits(profile, ffn_b, qo_b, kv_b);
            ComponentPrecision {
                attn_qo_bits: qo_b, attn_kv_bits: kv_b, ffn_bits: ffn_b,
                average_bits: avg, compression_ratio: 16.0 / avg,
            }
        }
    }
}

fn compute_avg_bits(profile: &PhaseProfile, ffn_b: u8, qo_b: u8, kv_b: u8) -> f64 {
    let ffn_w = 3.0 * profile.ffn_dim as f64 * profile.hidden_dim as f64;
    let attn_qo_w = 2.0 * profile.hidden_dim as f64 * profile.hidden_dim as f64;
    let attn_kv_w = 2.0 * profile.n_kv_heads as f64 * profile.head_dim as f64
        * profile.hidden_dim as f64;
    let total_w = ffn_w + attn_qo_w + attn_kv_w;
    ffn_b as f64 * ffn_w / total_w
        + qo_b as f64 * attn_qo_w / total_w
        + kv_b as f64 * attn_kv_w / total_w
}

fn per_layer_policies(
    profile: &PhaseProfile,
    dominance: SensitivityDominance,
    steering_layers: &[usize],
) -> Vec<ComputePolicy> {
    (0..profile.n_layers)
        .map(|l| {
            let is_steering = steering_layers.contains(&l);
            let zone = profile.layers[l].zone;

            match (dominance, is_steering, zone) {
                // Steering layers always get full precision
                (_, true, _) => ComputePolicy::FullPrecision,

                // Family A: UNFOLD (L2) = basin compiler
                (SensitivityDominance::AttentionBandwidth, _, Some(objeta_core::LayerZone::Unfold)) => {
                    ComputePolicy::FullPrecision
                }

                // Default per-zone policies
                (_, _, Some(objeta_core::LayerZone::Sync)) => ComputePolicy::StandardQuantize,
                (_, _, Some(objeta_core::LayerZone::Unfold)) => ComputePolicy::FullPrecision,
                (_, _, Some(objeta_core::LayerZone::IsometricLocal)) => ComputePolicy::AggressiveQuantize,
                (_, _, Some(objeta_core::LayerZone::IsometricGlobal)) => ComputePolicy::AggressiveQuantize,
                (_, _, Some(objeta_core::LayerZone::Divergent)) => ComputePolicy::StandardQuantize,
                (_, _, None) => ComputePolicy::StandardQuantize,
            }
        })
        .collect()
}

fn strategy_description(dominance: SensitivityDominance) -> String {
    match dominance {
        SensitivityDominance::AttentionBandwidth =>
            "Family A — Residual Transport: Preserve attention precision (transport bottleneck). \
             FFN can be aggressively quantized. Attn q5+, FFN q3+.".into(),
        SensitivityDominance::FfnCoherence =>
            "Family B Phase 1 — Aligned Field: Preserve FFN precision (coherence bottleneck). \
             Attention can be quantized. FFN q5+, Attn q4+.".into(),
        SensitivityDominance::SteeringBackbone =>
            "Family B Phase 3 — Mixed Field: Preserve steering layer precision (GQA backbone). \
             DeltaNet/linear layers and FFN can be quantized. GQA q5+, Delta q4, FFN q3+.".into(),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use objeta_core::{
        Family, LayerProfile, LayerStability, LayerZone, Phase, PhaseProfile, StabilityMap,
    };

    fn make_tinyllama_profile() -> PhaseProfile {
        let n = 22;
        let mut layers = Vec::new();
        for l in 0..n {
            let zone = if l <= 1 {
                LayerZone::Sync
            } else if l == 2 {
                LayerZone::Unfold
            } else if l <= 13 {
                LayerZone::IsometricLocal
            } else {
                LayerZone::Divergent
            };
            let lyap = match zone {
                LayerZone::Sync => 0.8,
                LayerZone::Unfold => 3.5,
                LayerZone::IsometricLocal => 1.0,
                LayerZone::IsometricGlobal => 1.0,
                LayerZone::Divergent => 2.1,
            };
            layers.push(LayerProfile {
                layer_idx: l,
                steering_cos: None,
                intra_cos: None,
                effective_rank: 8.0,
                residual_cos: None,
                hidden_norm: None,
                relative_steering: Some(lyap * 0.01),
                position_gradient: None,
                non_normality: None,
                zone: Some(zone),
                lyapunov_estimate: Some(lyap),
            });
        }
        PhaseProfile {
            model_name: "TinyLlama-1.1B".into(),
            n_layers: n,
            hidden_dim: 2048,
            ffn_dim: 5632,
            n_heads: 32,
            n_kv_heads: 32,
            head_dim: 64,
            vocab_size: 32000,
            phase: Phase::Split2D,
            family: Family::ResidualTransport,
            layers,
            inversion_layers: vec![7, 8, 9, 10, 11, 12, 13, 14],
            inversion_onset: Some(7),
            realignment_onset: Some(15),
            refresh_layers: vec![3, 8],
            coupling_strength: 0.1,
            ffn_compression_ratio: 0.1,
            zone_policies: vec![],
        }
    }

    #[test]
    fn test_static_plan_unfold_is_fp16() {
        let profile = make_tinyllama_profile();
        let plan = generate_static_plan(&profile);
        let l2 = &plan.layers[2];
        assert_eq!(l2.bits, 16, "L2 (UNFOLD) must be fp16, got {} bits", l2.bits);
    }

    #[test]
    fn test_static_plan_isometric_is_low() {
        let profile = make_tinyllama_profile();
        let plan = generate_static_plan(&profile);
        let l10 = &plan.layers[10];
        assert!(l10.bits <= 2, "ISOMETRIC L10 should be q2, got {} bits", l10.bits);
    }

    #[test]
    fn test_static_plan_divergent_gt_isometric() {
        let profile = make_tinyllama_profile();
        let plan = generate_static_plan(&profile);
        let iso_bit = plan.layers[10].bits;
        let div_bit = plan.layers[18].bits;
        assert!(div_bit > iso_bit, "DIVERGENT ({}bit) should require more bits than ISOMETRIC ({}bit)", div_bit, iso_bit);
    }

    #[test]
    fn test_lyapunov_aware_allocation() {
        let profile = make_tinyllama_profile();
        let budget = BitBudget {
            target_avg_bits: 4.0,
            ..Default::default()
        };
        let plan = generate_plan_with_budget(&profile, &budget);
        // L2 should get highest bits
        let max_bits = plan.layers.iter().map(|lq| lq.bits).max().unwrap();
        assert_eq!(plan.layers[2].bits, max_bits, "L2 should get highest precision");
        // Average should be close to target
        assert!((plan.average_bits - budget.target_avg_bits).abs() < 1.5,
            "avg bits {:.1} should be near target {:.1}", plan.average_bits, budget.target_avg_bits);
    }
}
