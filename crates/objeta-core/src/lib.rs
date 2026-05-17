//! objeta-core — Shared types for Transformer stability orchestration.
//!
//! ## Final theory (2026-05-16)
//!
//! Transformer = stiff dense dynamical system.
//!   - observable geometry is low-dimensional
//!   - generative dynamics is high-dimensional (full-rank operator)
//!   - compression target: compute QUALITY, not compute QUANTITY
//!
//! objeta is a stability orchestrator, not an operator decomposer.

use serde::{Deserialize, Serialize};

// ── Phase & Family ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    #[serde(rename = "collapse_1d")]
    Collapse1D,
    #[serde(rename = "split_2d")]
    Split2D,
    #[serde(rename = "mixed_field")]
    MixedField,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Family {
    #[serde(rename = "residual_transport")]
    ResidualTransport,
    #[serde(rename = "spherical_steering")]
    SphericalSteering,
}

// ── Zones ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LayerZone {
    Sync,
    Unfold,
    IsometricLocal,
    IsometricGlobal,
    Divergent,
}

// ── Per-Layer Profile ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProfile {
    pub layer_idx: usize,
    pub steering_cos: Option<f64>,
    pub intra_cos: Option<f64>,
    pub effective_rank: f64,
    pub residual_cos: Option<f64>,
    pub hidden_norm: Option<f64>,
    pub relative_steering: Option<f64>,
    pub position_gradient: Option<f64>,
    pub non_normality: Option<f64>,
    pub zone: Option<LayerZone>,
    /// Local Lyapunov estimate: ||Δ_{l+1}|| / ||Δ_l|| — >1 indicates divergence
    pub lyapunov_estimate: Option<f64>,
}

// ── Phase Profile ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseProfile {
    pub model_name: String,
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub phase: Phase,
    pub family: Family,
    pub layers: Vec<LayerProfile>,
    pub inversion_layers: Vec<usize>,
    pub inversion_onset: Option<usize>,
    pub realignment_onset: Option<usize>,
    pub refresh_layers: Vec<usize>,
    pub coupling_strength: f64,
    pub ffn_compression_ratio: f64,
    pub zone_policies: Vec<ZonePolicy>,
}

// ── Zone Policy ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZonePolicy {
    pub zone: LayerZone,
    pub layers: Vec<usize>,
    /// Recommended precision in bits (2-16)
    pub min_precision_bits: u8,
    /// Whether this zone is stability-critical (error amplification)
    pub stability_critical: bool,
    /// Whether to force full attention in this zone
    pub force_full_attention: bool,
    /// Refresh interval (layers between forced full compute)
    pub refresh_interval: usize,
}

// ── Stability Map ─────────────────────────────────────────────────────────

/// Per-layer stability classification for phase-aware quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMap {
    pub model_name: String,
    pub n_layers: usize,
    pub per_layer: Vec<LayerStability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStability {
    pub layer_idx: usize,
    pub zone: LayerZone,
    /// Local Lyapunov estimate
    pub lyapunov: f64,
    /// Recommended precision bits
    pub precision_bits: u8,
    /// Whether full attention is required
    pub full_attention: bool,
    /// Whether this is a refresh point
    pub is_refresh_point: bool,
    /// Inversion active (cos(Δ_l, Δ_{l+1}) < 0)
    pub inversion_active: bool,
}

// ── Compute Policy ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputePolicy {
    /// Aggressive quantization (q3-q4)
    AggressiveQuantize,
    /// Standard quantization (q4-q5)
    StandardQuantize,
    /// Conservative quantization (q8)
    ConservativeQuantize,
    /// Full precision (fp16)
    FullPrecision,
    /// Skip this layer (identity — only for λ≈0 zones)
    Skip,
}

// ── Runtime Strategy (cross-family validated) ─────────────────────────────

/// The dominant sensitivity mechanism determining what the runtime must preserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensitivityDominance {
    /// Attention determines transport capacity. FFN can degrade.
    /// Family A: Residual Transport (TinyLlama, Llama)
    AttentionBandwidth,
    /// FFN coherence is critical. Attention can degrade.
    /// Family B Phase 1: Aligned Field (Qwen2.5-0.5B)
    FfnCoherence,
    /// GQA/Full-attention layers are steering backbone. Delta/linear can degrade.
    /// Family B Phase 3: Mixed Field (Qwen3.6-35B)
    SteeringBackbone,
}

/// Per-family runtime strategy generated by the compiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStrategy {
    pub model_name: String,
    pub family: Family,
    pub phase: Phase,
    pub dominance: SensitivityDominance,
    /// Confidence score (0-1) for the family classification
    pub confidence: f64,
    /// Per-layer compute policies
    pub layer_policies: Vec<ComputePolicy>,
    /// Component-level precision recommendations
    pub component_precision: ComponentPrecision,
    /// Which layers are steering-critical (major course corrections)
    pub steering_layers: Vec<usize>,
    /// Strategy description
    pub description: String,
    /// Per-component format tags (ready for executor consumption)
    pub executor_config: ExecutorConfig,
}

/// Executor-ready quantization config — maps directly to weight loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// Per-layer FFN precision bits
    pub ffn_bits: Vec<u8>,
    /// Per-layer Attention Q/O precision bits
    pub attn_qo_bits: Vec<u8>,
    /// Per-layer Attention K/V precision bits
    pub attn_kv_bits: Vec<u8>,
    /// DeltaNet fusion ratio: 1.0=all, 0.33=1 per GQA block
    pub fusion_ratio: f64,
    /// Skip MoE on non-GQA (DeltaNet) layers
    pub moe_on_deltanet: bool,
    /// Expected tok/s on reference hardware (M1 8GB)
    pub estimated_tok_per_sec: f64,
    /// Expected VRAM usage in GB
    pub estimated_vram_gb: f64,
    /// Expected perplexity degradation vs fp16
    pub estimated_ppl_delta: f64,
}

/// Per-component precision assignment for the entire model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPrecision {
    /// Attention Q/O projection bits (transport routing)
    pub attn_qo_bits: u8,
    /// Attention K/V projection bits (memory storage)
    pub attn_kv_bits: u8,
    /// FFN bits (local field modulation)
    pub ffn_bits: u8,
    /// Effective average bits per weight
    pub average_bits: f64,
    /// Estimated compression ratio vs fp16
    pub compression_ratio: f64,
}

// ── Error ─────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ObjetaError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Analysis error: {0}")]
    Analysis(String),
    #[error("Unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("Missing tensor: {0}")]
    MissingTensor(String),
}

pub type Result<T> = std::result::Result<T, ObjetaError>;
