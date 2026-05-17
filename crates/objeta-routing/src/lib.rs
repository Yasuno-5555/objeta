//! objeta-routing — Phase-aware precision routing for stability orchestration.
//!
//! Generates per-layer precision schedules based on phase profiles.
//! Used by both Path A (MoE compiler) and Path B (adaptive runtime).

use objeta_core::{LayerZone, PhaseProfile, StabilityMap, LayerStability, ComputePolicy};

/// Generate a stability map from a phase profile.
///
/// This is the core output of objeta analyze → usable by any runtime
/// that wants phase-aware precision control.
pub fn generate_stability_map(profile: &PhaseProfile) -> StabilityMap {
    let n_layers = profile.n_layers;
    let per_layer: Vec<LayerStability> = (0..n_layers)
        .map(|l| {
            let layer = &profile.layers[l];
            let zone = layer.zone.unwrap_or(LayerZone::Sync);

            // Lyapunov estimate from relative steering magnitude
            let lyapunov = layer.lyapunov_estimate.unwrap_or(
                layer.relative_steering.unwrap_or(0.01)
            );

            // Precision assignment based on phase zone
            let (precision_bits, full_attention, is_refresh) = match zone {
                LayerZone::Sync => (4, false, false),           // q3-q4: short, anti-damped
                LayerZone::Unfold => (16, true, false),         // fp16: J≠I, mandatory
                LayerZone::IsometricLocal => (5, false, false), // q4-q5: λ≈0, safe
                LayerZone::IsometricGlobal => (5, true, false), // q4-q5: need attention for modulation
                LayerZone::Divergent => (8, true, false),       // q8: λ>0, conservative
            };

            // Refresh points: L3 (Type I) and L8 (Type II)
            let is_refresh_point = l == 3 || l == 8;

            // Inversion active if layer is in the inversion zone
            let inversion_active = profile.inversion_layers.contains(&l);

            LayerStability {
                layer_idx: l,
                zone,
                lyapunov,
                precision_bits,
                full_attention,
                is_refresh_point,
                inversion_active,
            }
        })
        .collect();

    StabilityMap {
        model_name: profile.model_name.clone(),
        n_layers,
        per_layer,
    }
}

/// Generate compute policies for each layer.
pub fn generate_policies(profile: &PhaseProfile) -> Vec<ComputePolicy> {
    let map = generate_stability_map(profile);
    map.per_layer.iter().map(|ls| {
        match ls.precision_bits {
            0..=3 => ComputePolicy::AggressiveQuantize,
            4..=5 => ComputePolicy::StandardQuantize,
            6..=8 => ComputePolicy::ConservativeQuantize,
            _ => ComputePolicy::FullPrecision,
        }
    }).collect()
}
