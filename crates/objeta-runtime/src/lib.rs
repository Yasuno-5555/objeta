//! objeta-runtime — Stability-aware runtime policy generation.
//!
//! Generates per-layer compute policies and refresh schedules
//! from phase profiles. Output consumed by adaptive runtimes.

use objeta_core::{LayerZone, PhaseProfile, StabilityMap, LayerStability};
use tracing::info;

/// Generate a complete stability map from a phase profile.
pub fn compile_stability_map(profile: &PhaseProfile) -> StabilityMap {
    info!(
        "Generating stability map: {} layers, phase={:?}, family={:?}",
        profile.n_layers, profile.phase, profile.family
    );

    objeta_routing::generate_stability_map(profile)
}

/// Generate per-layer precision configuration as JSON-serializable struct.
pub fn generate_precision_config(profile: &PhaseProfile) -> PrecisionConfig {
    let map = compile_stability_map(profile);
    let entries: Vec<PrecisionEntry> = map.per_layer.iter().map(|ls| {
        PrecisionEntry {
            layer: ls.layer_idx,
            zone: format!("{:?}", ls.zone),
            precision_bits: ls.precision_bits,
            full_attention: ls.full_attention,
            is_refresh: ls.is_refresh_point,
            inversion: ls.inversion_active,
            lyapunov: ls.lyapunov,
        }
    }).collect();

    PrecisionConfig {
        model: profile.model_name.clone(),
        phase: format!("{:?}", profile.phase),
        family: format!("{:?}", profile.family),
        n_layers: profile.n_layers,
        layers: entries,
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PrecisionConfig {
    pub model: String,
    pub phase: String,
    pub family: String,
    pub n_layers: usize,
    pub layers: Vec<PrecisionEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PrecisionEntry {
    pub layer: usize,
    pub zone: String,
    pub precision_bits: u8,
    pub full_attention: bool,
    pub is_refresh: bool,
    pub inversion: bool,
    pub lyapunov: f64,
}
