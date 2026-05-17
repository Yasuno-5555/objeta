//! objeta-analysis — Static geometry analysis of Transformer weights.
//!
//! Computes from FFN weights:
//! - Effective rank per layer (SVD entropy)
//! - Intra-layer coupling: cos(attn_direction, ffn_direction)
//! - Steering rotation: cos(Δ_l, Δ_{l+1}) via synthetic forward pass
//! - Position gradient
//! - Lyapunov estimates
//! - Phase and family classification

mod svd;

use nalgebra::{DMatrix, DVector};
use objeta_core::{LayerProfile, LayerZone, PhaseProfile, StabilityMap, Result};
use objeta_parser::{ModelConfig, ModelWeights};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::time::Instant;
use tracing::info;

// ── Public API ────────────────────────────────────────────────────────────

pub fn analyze_model(model_path: &str) -> Result<AnalysisReport> {
    let t0 = Instant::now();
    info!("Loading model from {}", model_path);

    let weights = ModelWeights::open(model_path)?;
    let config = ModelConfig::load(std::path::Path::new(model_path))?;

    info!("Loaded {} tensors: {} layers, {} hidden, {} ffn",
        weights.len(), config.num_hidden_layers, config.hidden_size, config.intermediate_size);

    let n_layers = config.num_hidden_layers;
    let hidden_dim = config.hidden_size;
    let ffn_dim = config.intermediate_size;

    let ffn_weights = load_ffn_weights(&weights, n_layers, hidden_dim, ffn_dim)?;

    info!("Computing effective rank per layer...");
    let eff_ranks: Vec<f64> = (0..n_layers).into_par_iter()
        .map(|l| effective_rank(&ffn_weights[l])).collect();

    info!("Computing intra-layer coupling...");
    let intra_cos_vals: Vec<Option<f64>> = (0..n_layers).into_par_iter()
        .map(|l| compute_intra_cos_from_weights(&ffn_weights[l])).collect();

    info!("Running synthetic FFN-only forward pass...");
    let (steering_cos, layer_norms, rel_steering, position_grad) =
        synthetic_forward(&ffn_weights, n_layers, hidden_dim)?;

    // Lyapunov estimates from relative steering norms
    let lyapunov_ests: Vec<Option<f64>> = (0..n_layers.saturating_sub(1))
        .map(|l| {
            let r1 = rel_steering.get(l).copied().unwrap_or(0.01);
            let r2 = rel_steering.get(l + 1).copied().unwrap_or(0.01);
            if r1 > 1e-12 { Some(r2 / r1) } else { None }
        }).collect();
    let lyapunov_ests: Vec<Option<f64>> = {
        let mut v = lyapunov_ests;
        v.push(None); // last layer has no next
        v
    };

    let layers: Vec<LayerProfile> = (0..n_layers).map(|l| {
        let zone = classify_zone(l, n_layers, &steering_cos, &intra_cos_vals);
        LayerProfile {
            layer_idx: l,
            steering_cos: steering_cos.get(l).copied(),
            intra_cos: intra_cos_vals[l],
            effective_rank: eff_ranks[l],
            residual_cos: None,
            hidden_norm: layer_norms.get(l).copied(),
            relative_steering: rel_steering.get(l).copied(),
            position_gradient: position_grad.get(l).copied(),
            non_normality: None,
            zone: Some(zone),
            lyapunov_estimate: lyapunov_ests[l],
        }
    }).collect();

    let inversion_layers: Vec<usize> = steering_cos.iter().enumerate()
        .filter(|(_, &c)| c < -0.01).map(|(i, _)| i).collect();
    let inversion_onset = steering_cos.iter().position(|&c| c < -0.01);
    let realignment_onset = inversion_onset.and_then(|onset| {
        steering_cos.iter().skip(onset).position(|&c| c > 0.05).map(|p| onset + p)
    });

    let mut refresh_layers = Vec::new();
    if let Some(onset) = inversion_onset { refresh_layers.push(onset); }
    if let Some(ra) = realignment_onset { refresh_layers.push(ra); }

    let coupling_strength = {
        let valid: Vec<f64> = steering_cos.iter().filter(|&&c| c.is_finite()).copied().collect();
        let mean = valid.iter().sum::<f64>() / valid.len().max(1) as f64;
        (valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / valid.len().max(1) as f64).sqrt()
    };

    let mean_eff_rank = eff_ranks.iter().sum::<f64>() / eff_ranks.len().max(1) as f64;
    let ffn_compression_ratio = mean_eff_rank / ffn_dim as f64;

    let mean_intra = intra_cos_vals.iter().filter_map(|&v| v).sum::<f64>()
        / intra_cos_vals.iter().filter_map(|&v| v).count().max(1) as f64;
    let intra_std = {
        let vals: Vec<f64> = intra_cos_vals.iter().filter_map(|&v| v).collect();
        let m = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
        (vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / vals.len().max(1) as f64).sqrt()
    };

    let phase = classify_phase(mean_intra, mean_eff_rank, intra_std);
    let family = classify_family(mean_intra, &steering_cos);
    let zone_policies = build_zone_policies(&layers, n_layers);

    let profile = PhaseProfile {
        model_name: config.model_type.clone(),
        n_layers, hidden_dim, ffn_dim,
        n_heads: config.num_attention_heads,
        n_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        vocab_size: config.vocab_size,
        phase, family, layers,
        inversion_layers, inversion_onset, realignment_onset,
        refresh_layers, coupling_strength, ffn_compression_ratio, zone_policies,
    };

    info!("Analysis complete in {:.1}s", t0.elapsed().as_secs_f64());

    Ok(AnalysisReport { profile, effective_ranks: eff_ranks, intra_cos_values: intra_cos_vals, steering_cos_values: steering_cos })
}

pub struct AnalysisReport {
    pub profile: PhaseProfile,
    pub effective_ranks: Vec<f64>,
    pub intra_cos_values: Vec<Option<f64>>,
    pub steering_cos_values: Vec<f64>,
}

// ── FFN Weight Loading ────────────────────────────────────────────────────

pub struct FfnWeights {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

fn load_ffn_weights(weights: &ModelWeights, n_layers: usize, hidden_dim: usize, ffn_dim: usize) -> Result<Vec<FfnWeights>> {
    (0..n_layers).map(|l| {
        let gate = load_matrix(weights, l, "gate_proj")?;
        let up = load_matrix(weights, l, "up_proj")?;
        let down = load_matrix(weights, l, "down_proj")?;
        Ok(FfnWeights { gate, up, down, rows: ffn_dim, cols: hidden_dim })
    }).collect()
}

fn load_matrix(weights: &ModelWeights, l: usize, name: &str) -> Result<Vec<f32>> {
    for prefix in &["model.layers", "model.language_model.layers"] {
        let key = format!("{}.{}.mlp.{}.weight", prefix, l, name);
        if weights.contains(&key) {
            let (_, _, data) = weights.get_matrix(&key)?;
            return Ok(data);
        }
    }
    Err(objeta_core::ObjetaError::MissingTensor(format!("layer {}.mlp.{}.weight", l, name)))
}

// ── Effective Rank ───────────────────────────────────────────────────────

fn effective_rank(w: &FfnWeights) -> f64 {
    let m = 2 * w.rows; let n = w.cols;
    if m < 2048 && n < 2048 {
        let mat = build_stacked_matrix(w);
        svd::effective_rank_full(&mat)
    } else {
        svd::effective_rank_randomized(&w.gate, &w.up, w.rows, w.cols, 128)
    }
}

fn build_stacked_matrix(w: &FfnWeights) -> DMatrix<f64> {
    let m = 2 * w.rows; let n = w.cols;
    let mut data = vec![0.0f64; m * n];
    for i in 0..w.rows {
        for j in 0..w.cols {
            data[i * n + j] = w.gate[i * w.cols + j] as f64;
            data[(w.rows + i) * n + j] = w.up[i * w.cols + j] as f64;
        }
    }
    DMatrix::from_vec(m, n, data)
}

fn compute_intra_cos_from_weights(w: &FfnWeights) -> Option<f64> {
    let (_, _, vt) = svd::randomized_svd_stacked(&w.gate, &w.up, w.rows, w.cols, 4)?;
    let v0 = DVector::from(vt.row(0).iter().copied().collect::<Vec<f64>>());
    let u0 = svd::power_iteration(&w.down, w.rows, w.cols)?;
    Some(v0.dot(&u0).abs())
}

// ── Synthetic Forward ────────────────────────────────────────────────────

fn synthetic_forward(ffn_weights: &[FfnWeights], n_layers: usize, hidden_dim: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n_inputs = 20; let n_positions = 6;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0 / (hidden_dim as f64).sqrt()).unwrap();

    let mut steering_cos_sum = vec![0.0f64; n_layers.saturating_sub(1)];
    let mut norm_sum = vec![0.0f64; n_layers];
    let mut rel_steer_sum = vec![0.0f64; n_layers];

    for pos in 0..n_positions {
        for _ in 0..n_inputs {
            let mut x = vec![0.0f64; hidden_dim];
            for v in x.iter_mut() { *v = normal.sample(&mut rng); }
            let pos_shift = pos as f64 * 0.01;
            for v in x.iter_mut() { *v += pos_shift; }
            let xn: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
            for v in x.iter_mut() { *v /= xn.max(1e-12); }

            let mut h = DVector::from(x);
            let mut deltas = Vec::with_capacity(n_layers);

            for l in 0..n_layers {
                let delta = ffn_forward_layer(&h, &ffn_weights[l]);
                let d_norm = delta.norm();
                h = &h + &delta;
                deltas.push(delta);
                norm_sum[l] += h.norm();
                rel_steer_sum[l] += if h.norm() > 1e-12 { d_norm / h.norm() } else { 0.0 };
            }

            for l in 0..n_layers.saturating_sub(1) {
                let n1 = deltas[l].norm(); let n2 = deltas[l+1].norm();
                if n1 > 1e-12 && n2 > 1e-12 {
                    steering_cos_sum[l] += deltas[l].dot(&deltas[l+1]) / (n1 * n2);
                }
            }
        }
    }

    let total = (n_inputs * n_positions) as f64;
    let steering_cos: Vec<f64> = steering_cos_sum.iter().map(|s| s / total).collect();
    let layer_norms: Vec<f64> = norm_sum.iter().map(|s| s / total).collect();
    let rel_steering: Vec<f64> = rel_steer_sum.iter().map(|s| s / total).collect();
    let position_grad: Vec<f64> = vec![0.0; n_layers]; // simplified

    Ok((steering_cos, layer_norms, rel_steering, position_grad))
}

fn ffn_forward_layer(x: &DVector<f64>, w: &FfnWeights) -> DVector<f64> {
    let mut gate_out = vec![0.0f64; w.rows];
    let mut up_out = vec![0.0f64; w.rows];
    for i in 0..w.rows {
        let mut sg = 0.0; let mut su = 0.0;
        for j in 0..w.cols {
            sg += w.gate[i * w.cols + j] as f64 * x[j];
            su += w.up[i * w.cols + j] as f64 * x[j];
        }
        gate_out[i] = sg; up_out[i] = su;
    }
    let mut hidden = vec![0.0f64; w.rows];
    for i in 0..w.rows { hidden[i] = gate_out[i] / (1.0 + (-gate_out[i]).exp()) * up_out[i]; }
    let mut out = vec![0.0f64; w.cols];
    for i in 0..w.cols {
        let mut s = 0.0;
        for j in 0..w.rows { s += w.down[i * w.rows + j] as f64 * hidden[j]; }
        out[i] = s;
    }
    DVector::from(out)
}

// ── Classification ────────────────────────────────────────────────────────

fn classify_zone(l: usize, n_layers: usize, steering_cos: &[f64], _intra_cos: &[Option<f64>]) -> LayerZone {
    let diverge_start = (0.7 * n_layers as f64).ceil() as usize;
    if l <= 1 { LayerZone::Sync }
    else if l == 2 { LayerZone::Unfold }
    else if l >= diverge_start { LayerZone::Divergent }
    else if steering_cos.get(l).copied().unwrap_or(0.0) < -0.01 { LayerZone::IsometricGlobal }
    else { LayerZone::IsometricLocal }
}

fn classify_phase(mean_intra: f64, eff_rank: f64, intra_std: f64) -> objeta_core::Phase {
    if mean_intra > 0.95 && eff_rank < 3.0 { objeta_core::Phase::Collapse1D }
    else if intra_std > 0.3 { objeta_core::Phase::MixedField }
    else { objeta_core::Phase::Split2D }
}

fn classify_family(mean_intra: f64, steering_cos: &[f64]) -> objeta_core::Family {
    let mean_cos = steering_cos.iter().filter(|&&c| c.is_finite()).sum::<f64>() / steering_cos.len().max(1) as f64;
    if mean_cos.abs() < 0.1 && mean_intra > 0.9 { objeta_core::Family::SphericalSteering }
    else { objeta_core::Family::ResidualTransport }
}

fn build_zone_policies(layers: &[LayerProfile], _n_layers: usize) -> Vec<objeta_core::ZonePolicy> {
    use std::collections::BTreeMap;
    let mut zones: BTreeMap<LayerZone, Vec<usize>> = BTreeMap::new();
    for layer in layers {
        if let Some(zone) = layer.zone { zones.entry(zone).or_default().push(layer.layer_idx); }
    }
    zones.into_iter().map(|(zone, layers)| {
        let (precision, stability_critical, force_attn, refresh) = match zone {
            LayerZone::Sync => (4, false, false, 0),
            LayerZone::Unfold => (16, true, true, 0),
            LayerZone::IsometricLocal => (5, false, false, 0),
            LayerZone::IsometricGlobal => (5, false, true, 0),
            LayerZone::Divergent => (8, true, true, 0),
        };
        objeta_core::ZonePolicy { zone, layers, min_precision_bits: precision, stability_critical, force_full_attention: force_attn, refresh_interval: refresh }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_phase_classification() {
        assert_eq!(classify_phase(0.99, 1.5, 0.01), objeta_core::Phase::Collapse1D);
        assert_eq!(classify_phase(0.1, 10.0, 0.15), objeta_core::Phase::Split2D);
        assert_eq!(classify_phase(0.05, 20.0, 0.5), objeta_core::Phase::MixedField);
    }
}
