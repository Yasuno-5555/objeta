//! objeta-phase — Phase detection and classification for Transformer models.
//!
//! Determines the dynamical regime of the model from per-layer geometry metrics.
//! The classification is based on:
//! - intra_cos (coupling between attention and FFN output directions)
//! - effective_rank (dimensionality of the FFN weight manifold)
//! - steering_cos variance (heterogeneity of the layer-to-layer flow)

use objeta_core::{Family, LayerProfile, Phase};

/// Detect the global phase from an array of layer profiles.
///
/// Algorithm:
/// 1. Compute mean intra_cos across all layers
/// 2. Compute effective rank averaged across layers
/// 3. Compute std dev of intra_cos (measures field heterogeneity)
/// 4. Apply decision boundaries:
///    - Phase 1: mean_intra > 0.95 AND global_eff_rank < 3
///    - Phase 2: mean_intra ≈ 0 AND intra_std < 0.3
///    - Phase 3: intra_std > 0.3 (sign-mixed field)
pub fn detect_phase(layers: &[LayerProfile]) -> Phase {
    let valid_intra: Vec<f64> = layers.iter().filter_map(|l| l.intra_cos).collect();
    if valid_intra.is_empty() {
        // Without intra_cos data, fall back to effective rank
        return detect_phase_from_rank(layers);
    }

    let mean_intra = valid_intra.iter().sum::<f64>() / valid_intra.len() as f64;
    let intra_std = {
        let m = mean_intra;
        (valid_intra.iter().map(|v| (v - m).powi(2)).sum::<f64>() / valid_intra.len() as f64)
            .sqrt()
    };

    let mean_eff_rank = layers.iter().map(|l| l.effective_rank).sum::<f64>()
        / layers.len().max(1) as f64;

    classify(mean_intra, mean_eff_rank, intra_std)
}

fn classify(mean_intra: f64, global_eff_rank: f64, intra_std: f64) -> Phase {
    // Phase 1: aligned field — all layers point in same direction
    if mean_intra > 0.95 && global_eff_rank < 3.0 {
        return Phase::Collapse1D;
    }

    // Phase 3: sign-mixed field — layers have both positive and negative intra_cos
    if intra_std > 0.3 {
        return Phase::MixedField;
    }

    // Phase 2: orthogonal field — attention and FFN are independent
    Phase::Split2D
}

/// Fallback phase detection from effective rank only.
fn detect_phase_from_rank(layers: &[LayerProfile]) -> Phase {
    let mean_eff_rank = layers.iter().map(|l| l.effective_rank).sum::<f64>()
        / layers.len().max(1) as f64;

    if mean_eff_rank < 3.0 {
        Phase::Collapse1D
    } else if mean_eff_rank < 15.0 {
        Phase::Split2D
    } else {
        Phase::MixedField
    }
}

/// Detect the transport family.
///
/// Family A (Residual Transport): Dense Transformers, h_{l+1} ≈ h_l.
///   Characterized by moderate-to-high residual cos and moderate intra_cos.
///
/// Family B (Spherical Steering): MoE hybrids, h ⟂ Δ.
///   Characterized by near-zero residual cos and high intra_cos (aligned).
pub fn detect_family(layers: &[LayerProfile]) -> Family {
    let valid_intra: Vec<f64> = layers.iter().filter_map(|l| l.intra_cos).collect();
    let mean_intra = if valid_intra.is_empty() {
        0.0
    } else {
        valid_intra.iter().sum::<f64>() / valid_intra.len() as f64
    };

    let mean_steering = layers
        .iter()
        .filter_map(|l| l.steering_cos)
        .sum::<f64>()
        / layers.len().max(1) as f64;

    // Family B: near-orthogonal steering (mean cos ≈ 0) AND high intra coupling
    if mean_steering.abs() < 0.1 && mean_intra > 0.9 {
        Family::SphericalSteering
    } else {
        Family::ResidualTransport
    }
}

/// Find inversion zone boundaries.
///
/// Inversion zone: consecutive layers where cos(Δ_l, Δ_{l+1}) < -0.01.
/// This indicates the model is rotating its coordinate system between layers.
pub fn find_inversion_zone(
    steering_cos: &[f64],
) -> (Vec<usize>, Option<usize>, Option<usize>) {
    let inversion_layers: Vec<usize> = steering_cos
        .iter()
        .enumerate()
        .filter(|(_, &c)| c < -0.01)
        .map(|(i, _)| i)
        .collect();

    let onset = steering_cos.iter().position(|&c| c < -0.01);

    let realignment = onset.and_then(|o| {
        steering_cos.iter().skip(o).position(|&c| c > 0.05).map(|p| o + p)
    });

    (inversion_layers, onset, realignment)
}

/// Identify refresh layers — layers where attention is most critical
/// for maintaining routing diversity.
///
/// These are the boundaries between dynamical zones: where the steering
/// direction changes sign (inversion onset) and where it realigns.
pub fn find_refresh_layers(
    inversion_onset: Option<usize>,
    realignment_onset: Option<usize>,
) -> Vec<usize> {
    let mut refresh = Vec::new();
    if let Some(onset) = inversion_onset {
        refresh.push(onset);
    }
    if let Some(ra) = realignment_onset {
        if !refresh.contains(&ra) {
            refresh.push(ra);
        }
    }
    refresh
}

#[cfg(test)]
mod tests {
    use super::*;
    use objeta_core::{LayerProfile, LayerZone};

    fn make_layer(idx: usize, intra: f64, eff_rank: f64, steering: f64) -> LayerProfile {
        LayerProfile {
            layer_idx: idx,
            steering_cos: Some(steering),
            intra_cos: Some(intra),
            effective_rank: eff_rank,
            residual_cos: None,
            hidden_norm: None,
            relative_steering: None,
            position_gradient: None,
            non_normality: None,
            zone: None,
        }
    }

    #[test]
    fn test_detect_phase_1_collapse() {
        let layers: Vec<_> = (0..22)
            .map(|i| make_layer(i, 0.999, 1.5, 0.05))
            .collect();
        assert_eq!(detect_phase(&layers), Phase::Collapse1D);
    }

    #[test]
    fn test_detect_phase_2_split() {
        let layers: Vec<_> = (0..22)
            .map(|i| make_layer(i, 0.05, 10.0, 0.03))
            .collect();
        assert_eq!(detect_phase(&layers), Phase::Split2D);
    }

    #[test]
    fn test_detect_phase_3_mixed() {
        // Mixed signs: some positive, some negative intra_cos
        let mut layers = Vec::new();
        for i in 0..22 {
            let intra = if i % 3 == 0 { -0.3 } else { 0.4 };
            layers.push(make_layer(i, intra, 25.0, 0.05));
        }
        assert_eq!(detect_phase(&layers), Phase::MixedField);
    }

    #[test]
    fn test_inversion_zone() {
        let cos_vals = vec![0.1, 0.05, -0.02, -0.05, -0.03, 0.01, 0.08, 0.15];
        let (inv, onset, realign) = find_inversion_zone(&cos_vals);
        assert_eq!(inv, vec![2, 3, 4]);
        assert_eq!(onset, Some(2));
        assert_eq!(realign, Some(6));
    }

    #[test]
    fn test_no_inversion() {
        let cos_vals = vec![0.1, 0.05, 0.02, 0.03, 0.01];
        let (inv, onset, realign) = find_inversion_zone(&cos_vals);
        assert!(inv.is_empty());
        assert_eq!(onset, None);
        assert_eq!(realign, None);
    }
}
