//! objeta-os — LKO Reflexive Runtime OS.
//!
//! LLM inference is not static computation.
//! It is adaptive dynamical resource allocation.
//!
//! The OS provides:
//! 1. **Observation** — entropy, steering, routing, attention divergence
//! 2. **Classification** — token class (STABLE/STEERING/TRANSITION/REPETITIVE)
//! 3. **Allocation** — precision budget, attention mode, MoE mode, layer policy
//! 4. **Stabilization** — collapse detection and recovery
//!
//! Core principle:
//!   observe → classify → allocate → execute

use objeta_core::{Family, LayerZone};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// Token Classes
// ═══════════════════════════════════════════════════════════════════════════════

/// Token compute budget class for heterogeneous execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenClass {
    /// Same token repeating — near-zero compute needed
    Repetitive,
    /// Low entropy, low steering — aggressive skip safe
    Stable,
    /// Normal conditions — phase-dependent policy
    Default,
    /// High steering magnitude — full attention needed
    Steering,
    /// Entropy + steering spike — full precision mandatory
    Transition,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Execution Modes
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttnMode {
    /// Full attention — all heads, all KV
    Full,
    /// Reduced KV context window
    Reduced,
    /// Reuse cached attention weights (Frozen-QK)
    Cached,
    /// No attention at all (identity skip)
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoeMode {
    /// Normal top-k routing
    Full,
    /// Entropy-based adaptive k
    Adaptive,
    /// Reduced expert count
    Sparse,
    /// Bypass MoE entirely
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// Full precision (fp16)
    Fp16,
    /// Conservative quantization (q8)
    Q8,
    /// Standard quantization (q5)
    Q5,
    /// Aggressive quantization (q4)
    Q4,
    /// Ultra-aggressive (q3)
    Q3,
}

impl PrecisionMode {
    pub fn bits(&self) -> u8 {
        match self {
            PrecisionMode::Fp16 => 16,
            PrecisionMode::Q8 => 8,
            PrecisionMode::Q5 => 5,
            PrecisionMode::Q4 => 4,
            PrecisionMode::Q3 => 3,
        }
    }

    pub fn from_bits(bits: u8) -> Self {
        match bits {
            0..=3 => PrecisionMode::Q3,
            4 => PrecisionMode::Q4,
            5 => PrecisionMode::Q5,
            6..=8 => PrecisionMode::Q8,
            _ => PrecisionMode::Fp16,
        }
    }

    pub fn downgrade(&self) -> Self {
        match self {
            PrecisionMode::Fp16 => PrecisionMode::Q8,
            PrecisionMode::Q8 => PrecisionMode::Q5,
            PrecisionMode::Q5 => PrecisionMode::Q4,
            PrecisionMode::Q4 => PrecisionMode::Q3,
            PrecisionMode::Q3 => PrecisionMode::Q3,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Layer Policy
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-layer execution policy — compiled at init, constant per layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPolicy {
    pub layer_idx: usize,
    pub attn_mode: AttnMode,
    pub moe_mode: MoeMode,
    pub precision: PrecisionMode,
    pub phase: LayerZone,
    /// Never skip this layer (UNFOLD, output)
    pub is_sacred: bool,
    /// Course-correction layer (GQA, DIVERGENT)
    pub is_steering: bool,
    /// Whether this layer requires recomputation (not cacheable)
    pub recompute: bool,
}

impl LayerPolicy {
    pub fn sacred(layer_idx: usize, phase: LayerZone) -> Self {
        Self {
            layer_idx,
            attn_mode: AttnMode::Full,
            moe_mode: MoeMode::Full,
            precision: PrecisionMode::Fp16,
            phase,
            is_sacred: true,
            is_steering: false,
            recompute: true,
        }
    }

    pub fn transport(layer_idx: usize, phase: LayerZone) -> Self {
        Self {
            layer_idx,
            attn_mode: AttnMode::Cached,
            moe_mode: MoeMode::Adaptive,
            precision: PrecisionMode::Q4,
            phase,
            is_sacred: false,
            is_steering: false,
            recompute: false,
        }
    }

    pub fn steering(layer_idx: usize, phase: LayerZone) -> Self {
        Self {
            layer_idx,
            attn_mode: AttnMode::Full,
            moe_mode: MoeMode::Full,
            precision: PrecisionMode::Q8,
            phase,
            is_sacred: false,
            is_steering: true,
            recompute: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Observation
// ═══════════════════════════════════════════════════════════════════════════════

/// Runtime observation signals — measured every token.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Observation {
    /// Softmax entropy: 0=peaked (certain), 1=uniform (uncertain)
    pub entropy: f64,
    /// Steering magnitude: 1 - cos(h_t, h_{t-1})
    pub steering: f64,
    /// MoE routing entropy (if applicable)
    pub routing_entropy: Option<f64>,
    /// Attention map change: 1 - cos(A_t, A_{t-1})
    pub attention_divergence: Option<f64>,
    /// Current token index in sequence
    pub token_index: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Logit of the top-1 token
    pub top1_logit: f64,
    /// Whether the predicted token repeats the previous
    pub is_repeat: bool,
}

impl Observation {
    /// Classify token from observation signals.
    pub fn classify(&self) -> TokenClass {
        // Repetition is the strongest signal
        if self.is_repeat {
            return TokenClass::Repetitive;
        }

        // Transition: entropy spike + steering spike
        if self.entropy > 0.2 && self.steering > 0.6 {
            return TokenClass::Transition;
        }

        // Steering: high trajectory change
        if self.steering > 0.5 {
            return TokenClass::Steering;
        }

        // Stable: low entropy, low steering
        if self.entropy < 0.05 && self.steering < 0.4 {
            return TokenClass::Stable;
        }

        TokenClass::Default
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Observation Pipeline — runtime signal measurement
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default)]
pub struct ObservationPipeline {
    pub prev_hidden: Option<Vec<f64>>,
    pub prev_attn_weights: std::collections::HashMap<usize, Vec<Vec<f64>>>,
}

impl ObservationPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.prev_hidden = None;
        self.prev_attn_weights.clear();
    }

    /// Compute entropy, top-1 logit, and repeat flag from logits.
    ///
    /// Returns (entropy, top1_logit, top1_token_id).
    pub fn observe_logits(&self, logits: &[f64]) -> (f64, f64, usize) {
        if logits.is_empty() {
            return (0.0, 0.0, 0);
        }

        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let stable: Vec<f64> = logits.iter().map(|&x| x - max_logit).collect();

        let mut probs: Vec<f64> = stable.iter().map(|&x| x.exp()).collect();
        let sum_probs: f64 = probs.iter().sum();

        if sum_probs > 0.0 {
            for p in &mut probs {
                *p /= sum_probs;
            }
        }

        let max_ent = (logits.len() as f64).ln();
        let mut shannon = 0.0;
        for &p in &probs {
            if p > 0.0 {
                shannon -= p * p.ln();
            }
        }

        let entropy = if max_ent > 0.0 { shannon / max_ent } else { 0.0 };

        let mut top1 = 0;
        let mut top1_logit = logits[0];
        for (i, &val) in logits.iter().enumerate() {
            if val > top1_logit {
                top1_logit = val;
                top1 = i;
            }
        }

        (entropy, top1_logit, top1)
    }

    /// Compute steering magnitude: 1 - cos(h_t, h_{t-1}).
    ///
    /// Returns steering in [0, 2]. 0 = identical, 2 = opposite.
    pub fn observe_hidden(&mut self, hidden: &[f64]) -> f64 {
        if hidden.is_empty() {
            return 0.0;
        }

        if let Some(prev) = &self.prev_hidden {
            let h_norm = (hidden.iter().map(|&x| x * x).sum::<f64>()).sqrt();
            let prev_norm = (prev.iter().map(|&x| x * x).sum::<f64>()).sqrt();
            let dot_prod = hidden.iter().zip(prev.iter()).map(|(&x, &y)| x * y).sum::<f64>();
            let cos = dot_prod / (h_norm * prev_norm + 1e-12);
            self.prev_hidden = Some(hidden.to_vec());
            1.0 - cos
        } else {
            self.prev_hidden = Some(hidden.to_vec());
            0.0
        }
    }

    /// Compute attention divergence from previous step.
    ///
    /// attn_weights shape: (n_heads, seq_len).
    /// Returns mean(1 - cos(A_head_t, A_head_{t-1})) across heads.
    pub fn observe_attention(&mut self, layer_idx: usize, attn_weights: &[Vec<f64>]) -> Option<f64> {
        if attn_weights.is_empty() {
            return None;
        }

        if let Some(prev) = self.prev_attn_weights.get(&layer_idx) {
            if prev.len() != attn_weights.len() || prev[0].len() != attn_weights[0].len() {
                self.prev_attn_weights.insert(layer_idx, attn_weights.to_vec());
                return None;
            }

            let n_heads = attn_weights.len();
            let mut divergences = Vec::with_capacity(n_heads);

            for h in 0..n_heads {
                let head_curr = &attn_weights[h];
                let head_prev = &prev[h];

                let norm_curr = (head_curr.iter().map(|&x| x * x).sum::<f64>()).sqrt();
                let norm_prev = (head_prev.iter().map(|&x| x * x).sum::<f64>()).sqrt();
                let dot_prod = head_curr.iter().zip(head_prev.iter()).map(|(&x, &y)| x * y).sum::<f64>();

                let cos = dot_prod / (norm_curr * norm_prev + 1e-12);
                divergences.push(1.0 - cos);
            }

            self.prev_attn_weights.insert(layer_idx, attn_weights.to_vec());
            let mean_div = divergences.iter().sum::<f64>() / n_heads as f64;
            Some(mean_div)
        } else {
            self.prev_attn_weights.insert(layer_idx, attn_weights.to_vec());
            None
        }
    }
}

// ── Standalone convenience functions ──

/// Normalized Shannon entropy from logits.
pub fn compute_entropy(logits: &[f64]) -> f64 {
    if logits.is_empty() {
        return 0.0;
    }

    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let stable: Vec<f64> = logits.iter().map(|&x| x - max_logit).collect();

    let mut probs: Vec<f64> = stable.iter().map(|&x| x.exp()).collect();
    let sum_probs: f64 = probs.iter().sum();

    if sum_probs > 0.0 {
        for p in &mut probs {
            *p /= sum_probs;
        }
    }

    let max_ent = (logits.len() as f64).ln();
    let mut shannon = 0.0;
    for &p in &probs {
        if p > 0.0 {
            shannon -= p * p.ln();
        }
    }

    if max_ent > 0.0 { shannon / max_ent } else { 0.0 }
}

/// 1 - cos(h_curr, h_prev).
pub fn compute_steering(h_curr: &[f64], h_prev: &[f64]) -> f64 {
    if h_curr.is_empty() || h_prev.is_empty() {
        return 0.0;
    }

    let norm_curr = (h_curr.iter().map(|&x| x * x).sum::<f64>()).sqrt();
    let norm_prev = (h_prev.iter().map(|&x| x * x).sum::<f64>()).sqrt();
    let dot_prod = h_curr.iter().zip(h_prev.iter()).map(|(&x, &y)| x * y).sum::<f64>();

    let cos = dot_prod / (norm_curr * norm_prev + 1e-12);
    1.0 - cos
}

/// Mean per-head attention divergence.
pub fn compute_attention_divergence(a_curr: &[Vec<f64>], a_prev: &[Vec<f64>]) -> Option<f64> {
    if a_curr.is_empty() || a_prev.is_empty() || a_curr.len() != a_prev.len() || a_curr[0].len() != a_prev[0].len() {
        return None;
    }

    let n_heads = a_curr.len();
    let mut divergences = Vec::with_capacity(n_heads);

    for h in 0..n_heads {
        let head_curr = &a_curr[h];
        let head_prev = &a_prev[h];

        let norm_curr = (head_curr.iter().map(|&x| x * x).sum::<f64>()).sqrt();
        let norm_prev = (head_prev.iter().map(|&x| x * x).sum::<f64>()).sqrt();
        let dot_prod = head_curr.iter().zip(head_prev.iter()).map(|(&x, &y)| x * y).sum::<f64>();

        let cos = dot_prod / (norm_curr * norm_prev + 1e-12);
        divergences.push(1.0 - cos);
    }

    let mean_div = divergences.iter().sum::<f64>() / n_heads as f64;
    Some(mean_div)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runtime State
// ═══════════════════════════════════════════════════════════════════════════════

/// Persistent runtime state — accumulates history across tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    /// Ring buffer of recent entropy values
    pub entropy_history: Vec<f64>,
    /// Ring buffer of recent steering values
    pub steering_history: Vec<f64>,
    /// Rolling mean of routing entropy
    pub routing_stats: RoutingStats,
    /// Per-layer health status
    pub layer_health: Vec<LayerHealth>,
    /// Current precision budget
    pub precision_budget: PrecisionBudget,
    /// Total tokens processed
    pub token_count: usize,
    /// Collapse status
    pub collapse_status: CollapseStatus,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            entropy_history: Vec::with_capacity(64),
            steering_history: Vec::with_capacity(64),
            routing_stats: RoutingStats::default(),
            layer_health: Vec::new(),
            precision_budget: PrecisionBudget::default(),
            token_count: 0,
            collapse_status: CollapseStatus::Healthy,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingStats {
    pub mean_entropy: f64,
    pub mean_active_experts: f64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerHealth {
    pub layer_idx: usize,
    pub skip_count: usize,
    pub run_count: usize,
    pub last_cos: Option<f64>,
    pub divergence_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionBudget {
    pub target_avg_bits: f64,
    pub current_avg_bits: f64,
    pub high_precision_count: usize,
    pub low_precision_count: usize,
}

impl Default for PrecisionBudget {
    fn default() -> Self {
        Self {
            target_avg_bits: 4.5,
            current_avg_bits: 16.0,
            high_precision_count: 0,
            low_precision_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollapseStatus {
    Healthy,
    Warning,
    Critical,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scheduler Configuration
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub family: Family,
    pub backbone: SensitivityDominance,
    pub safe_skip_ceiling: f64,
    pub fusion_ratio: f64,
    pub temporal_stride: usize,
    pub entropy_threshold: EntropyThresholds,
    pub steering_threshold: SteeringThresholds,
    pub collapse_detection: CollapseDetection,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            family: Family::ResidualTransport,
            backbone: SensitivityDominance::AttentionBandwidth,
            safe_skip_ceiling: 0.30,
            fusion_ratio: 0.50,
            temporal_stride: 0,
            entropy_threshold: EntropyThresholds::default(),
            steering_threshold: SteeringThresholds::default(),
            collapse_detection: CollapseDetection::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EntropyThresholds {
    pub stable_max: f64,
    pub transition_min: f64,
    pub collapse_warn: f64,
    pub collapse_critical: f64,
}

impl Default for EntropyThresholds {
    fn default() -> Self {
        Self {
            stable_max: 0.05,
            transition_min: 0.2,
            collapse_warn: 0.1,
            collapse_critical: 0.03,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SteeringThresholds {
    pub stable_max: f64,
    pub steering_min: f64,
    pub transition_min: f64,
}

impl Default for SteeringThresholds {
    fn default() -> Self {
        Self {
            stable_max: 0.4,
            steering_min: 0.5,
            transition_min: 0.6,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CollapseDetection {
    pub entropy_window: usize,
    pub repetition_threshold: usize,
    pub steering_spike: f64,
    pub recovery_entropy_threshold: f64,
    pub collapse_warn_entropy: f64,
    pub collapse_critical_entropy: f64,
}

impl Default for CollapseDetection {
    fn default() -> Self {
        Self {
            entropy_window: 8,
            repetition_threshold: 5,
            steering_spike: 0.8,
            recovery_entropy_threshold: 0.15,
            collapse_warn_entropy: 0.1,
            collapse_critical_entropy: 0.03,
        }
    }
}

/// The dominant sensitivity mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensitivityDominance {
    /// Attention determines transport capacity
    AttentionBandwidth,
    /// FFN coherence is critical
    FfnCoherence,
    /// GQA/Full-attention layers are steering backbone
    SteeringBackbone,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase Policy Table Builder
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a static per-layer policy table from LKO phase structure.
pub fn build_policy_table(n_layers: usize, fusion_ratio: f64,
                          _is_moe: bool, _family: Family) -> Vec<LayerPolicy> {
    let stride = (1.0 / fusion_ratio.max(0.01)).round() as usize;
    let diverge_start = ((n_layers as f64 * 0.7).ceil()) as usize;
    let mut delta_count = 0usize;
    let mut table = Vec::with_capacity(n_layers);

    for l in 0..n_layers {
        let zone = classify_zone(l, n_layers, diverge_start);

        let (is_sacred, is_steering, attn_mode, moe_mode, precision) = match zone {
            LayerZone::Sync => (
                true, false,
                AttnMode::Full, MoeMode::Full, PrecisionMode::Fp16,
            ),
            LayerZone::Unfold => (
                true, false,
                AttnMode::Full, MoeMode::Full, PrecisionMode::Fp16,
            ),
            LayerZone::IsometricLocal | LayerZone::IsometricGlobal => {
                delta_count += 1;
                if delta_count % stride == 0 {
                    // Refresh point: full compute
                    (false, false, AttnMode::Full, MoeMode::Adaptive, PrecisionMode::Q8)
                } else {
                    // Transport: cacheable
                    (false, false, AttnMode::Cached, MoeMode::Sparse, PrecisionMode::Q4)
                }
            }
            LayerZone::Divergent => (
                false, true,
                AttnMode::Full, MoeMode::Full, PrecisionMode::Q8,
            ),
        };

        table.push(LayerPolicy {
            layer_idx: l,
            attn_mode,
            moe_mode,
            precision,
            phase: zone,
            is_sacred,
            is_steering,
            recompute: is_sacred || is_steering,
        });
    }

    table
}

fn classify_zone(l: usize, n_layers: usize, diverge_start: usize) -> LayerZone {
    if l <= 1 {
        LayerZone::Sync
    } else if l == 2 {
        LayerZone::Unfold
    } else if l >= n_layers - 1 {
        LayerZone::Divergent
    } else if l >= diverge_start {
        LayerZone::Divergent
    } else {
        LayerZone::IsometricLocal
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Collapse Detector
// ═══════════════════════════════════════════════════════════════════════════════

/// Detects trajectory collapse and triggers recovery actions.
pub struct CollapseDetector {
    config: CollapseDetection,
    repetition_count: usize,
    prev_token_id: Option<usize>,
}

impl CollapseDetector {
    pub fn new(config: CollapseDetection) -> Self {
        Self {
            config,
            repetition_count: 0,
            prev_token_id: None,
        }
    }

    /// Update internal state with new token.
    pub fn update(&mut self, token_id: usize) {
        if self.prev_token_id == Some(token_id) {
            self.repetition_count += 1;
        } else {
            self.repetition_count = 0;
        }
        self.prev_token_id = Some(token_id);
    }

    /// Check if collapse is detected from observation history.
    pub fn detect(&self, obs: &Observation, state: &RuntimeState) -> CollapseStatus {
        // 1. Repetition lock: same token repeating
        if self.repetition_count >= self.config.repetition_threshold {
            return CollapseStatus::Critical;
        }

        // 2. Entropy collapse: sustained low entropy
        if state.entropy_history.len() >= self.config.entropy_window {
            let recent: Vec<f64> = state.entropy_history
                .iter()
                .rev()
                .take(self.config.entropy_window)
                .copied()
                .collect();
            let mean_entropy: f64 = recent.iter().sum::<f64>() / recent.len() as f64;

            if mean_entropy < self.config.collapse_critical_entropy {
                return CollapseStatus::Critical;
            }
            if mean_entropy < self.config.collapse_warn_entropy {
                return CollapseStatus::Warning;
            }
        }

        // 3. Steering spike: sudden large trajectory change
        if obs.steering > self.config.steering_spike {
            return CollapseStatus::Warning;
        }

        CollapseStatus::Healthy
    }

    /// Recovery actions to take based on collapse status.
    pub fn recovery_actions(&self, status: CollapseStatus) -> RecoveryActions {
        match status {
            CollapseStatus::Healthy => RecoveryActions::default(),
            CollapseStatus::Warning => RecoveryActions {
                force_full_compute: false,
                increase_precision: true,
                disable_skip: false,
            },
            CollapseStatus::Critical => RecoveryActions {
                force_full_compute: true,
                increase_precision: true,
                disable_skip: true,
            },
        }
    }

    pub fn reset(&mut self) {
        self.repetition_count = 0;
        self.prev_token_id = None;
    }
}

#[derive(Debug, Clone, Default)]
pub struct RecoveryActions {
    pub force_full_compute: bool,
    pub increase_precision: bool,
    pub disable_skip: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hysteresis — prevents scheduler thrashing
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HysteresisState {
    pub current_class: TokenClass,
    pub consecutive_stable: usize,
    pub consecutive_unstable: usize,
    pub precision_level: PrecisionMode,
    pub precision_stable_count: usize,
}

impl Default for HysteresisState {
    fn default() -> Self {
        Self {
            current_class: TokenClass::Default,
            consecutive_stable: 0,
            consecutive_unstable: 0,
            precision_level: PrecisionMode::Q8,
            precision_stable_count: 0,
        }
    }
}

impl HysteresisState {
    pub fn classify(&mut self, entropy: f64, steering: f64, is_repeat: bool) -> TokenClass {
        // Repetition has highest priority (no hysteresis needed — absolute signal)
        if is_repeat {
            self.consecutive_unstable += 1;
            self.consecutive_stable = 0;
            self.current_class = TokenClass::Repetitive;
            return self.current_class;
        }

        // Enter TRANSITION: both entropy AND steering spike
        if entropy > 0.22 && steering > 0.7 && self.current_class != TokenClass::Transition {
            self.current_class = TokenClass::Transition;
            self.consecutive_unstable += 1;
            self.consecutive_stable = 0;
            return self.current_class;
        }

        // Stay in TRANSITION: lower leave threshold
        if self.current_class == TokenClass::Transition {
            if steering > 0.5 || entropy > 0.15 {
                self.consecutive_unstable += 1;
                return self.current_class;
            }
            // Leave: both drop below leave thresholds
            self.consecutive_unstable = 0;
        }

        // Enter STEERING: high steering
        if steering > 0.6 && self.current_class != TokenClass::Steering {
            self.current_class = TokenClass::Steering;
            self.consecutive_unstable += 1;
            self.consecutive_stable = 0;
            return self.current_class;
        }

        // Stay in STEERING: lower leave threshold
        if self.current_class == TokenClass::Steering {
            if steering > 0.45 {
                self.consecutive_unstable += 1;
                return self.current_class;
            }
            self.consecutive_unstable = 0;
        }

        // Enter STABLE: sustained low entropy + low steering
        if entropy < 0.04 && steering < 0.35 && self.current_class != TokenClass::Stable {
            self.consecutive_stable += 1;
            if self.consecutive_stable >= 2 {  // need 2 consecutive
                self.current_class = TokenClass::Stable;
                self.consecutive_unstable = 0;
                return self.current_class;
            }
            return self.current_class;
        } else if entropy > 0.06 || steering > 0.4 {
            self.consecutive_stable = 0;
        }

        // Stay in STABLE: wider leave threshold
        if self.current_class == TokenClass::Stable {
            if entropy < 0.08 && steering < 0.5 {
                return self.current_class;
            }
            self.consecutive_stable = 0;
        }

        // Default
        self.current_class = TokenClass::Default;
        self.consecutive_stable = 0;
        self.current_class
    }

    pub fn get_precision(&mut self, target: PrecisionMode) -> PrecisionMode {
        // Allow immediate upgrade (never delay safety)
        if target.bits() > self.precision_level.bits() {
            self.precision_level = target;
            self.precision_stable_count = 0;
            return target;
        }

        // Downgrade requires sustained stability
        if target.bits() < self.precision_level.bits() {
            self.precision_stable_count += 1;
            if self.precision_stable_count >= 3 {  // 3 tokens stable before downgrade
                self.precision_level = self.precision_level.downgrade();
                self.precision_stable_count = 0;
            }
            return self.precision_level;
        }

        self.precision_stable_count = 0;
        self.precision_level
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Collapse Hysteresis — prevents collapse status flapping
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapseHysteresis {
    pub current_status: CollapseStatus,
    pub healthy_count: usize,
    pub warning_count: usize,
}

impl Default for CollapseHysteresis {
    fn default() -> Self {
        Self {
            current_status: CollapseStatus::Healthy,
            healthy_count: 0,
            warning_count: 0,
        }
    }
}

impl CollapseHysteresis {
    pub fn update(&mut self, raw_status: CollapseStatus) -> CollapseStatus {
        // CRITICAL: enter immediately, leave slowly
        if raw_status == CollapseStatus::Critical {
            self.current_status = CollapseStatus::Critical;
            self.healthy_count = 0;
            return self.current_status;
        }

        if self.current_status == CollapseStatus::Critical {
            self.healthy_count += 1;
            if self.healthy_count >= 5 {  // 5 healthy tokens to clear critical
                self.current_status = CollapseStatus::Healthy;
                self.healthy_count = 0;
            }
            return self.current_status;
        }

        // WARNING: enter/leave with debounce
        if raw_status == CollapseStatus::Warning {
            self.warning_count += 1;
            if self.warning_count >= 2 {  // 2 consecutive warnings to enter
                self.current_status = CollapseStatus::Warning;
                self.healthy_count = 0;
            }
            return self.current_status;
        }

        self.warning_count = 0;
        self.healthy_count += 1;
        if self.current_status == CollapseStatus::Warning {
            if self.healthy_count >= 3 {  // 3 healthy to clear warning
                self.current_status = CollapseStatus::Healthy;
            }
        }
        self.current_status
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Collapse Memory — persistent degradation tracking for long-context
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapseMemory {
    pub window_size: usize,
    pub collapse_history: Vec<f64>,
    pub steering_history: Vec<f64>,
    pub entropy_history: Vec<f64>,
    pub repetition_history: Vec<usize>,
    pub risk_score: f64,
    pub conservative_mode: bool,
    pub conservative_mode_entered_at: i32,
    pub total_collapse_tokens: usize,
    pub total_warning_tokens: usize,
}

impl Default for CollapseMemory {
    fn default() -> Self {
        Self {
            window_size: 128,
            collapse_history: Vec::with_capacity(128),
            steering_history: Vec::with_capacity(128),
            entropy_history: Vec::with_capacity(128),
            repetition_history: Vec::with_capacity(128),
            risk_score: 0.0,
            conservative_mode: false,
            conservative_mode_entered_at: -1,
            total_collapse_tokens: 0,
            total_warning_tokens: 0,
        }
    }
}

impl CollapseMemory {
    pub fn update(
        &mut self,
        collapse_status: CollapseStatus,
        steering: f64,
        entropy: f64,
        is_repeat: bool,
        token_idx: usize,
    ) {
        let status_score = match collapse_status {
            CollapseStatus::Healthy => 0.0,
            CollapseStatus::Warning => 0.5,
            CollapseStatus::Critical => 1.0,
        };

        self.collapse_history.push(status_score);
        self.steering_history.push(steering);
        self.entropy_history.push(entropy);
        self.repetition_history.push(if is_repeat { 1 } else { 0 });

        if self.collapse_history.len() > self.window_size {
            self.collapse_history.remove(0);
            self.steering_history.remove(0);
            self.entropy_history.remove(0);
            self.repetition_history.remove(0);
        }

        match collapse_status {
            CollapseStatus::Critical => self.total_collapse_tokens += 1,
            CollapseStatus::Warning => self.total_warning_tokens += 1,
            CollapseStatus::Healthy => {}
        }

        let recent_window = std::cmp::min(32, self.collapse_history.len());
        if recent_window > 0 {
            let start = self.collapse_history.len() - recent_window;
            let mean_collapse = self.collapse_history[start..].iter().sum::<f64>() / recent_window as f64;
            let repeat_rate = self.repetition_history[start..].iter().sum::<usize>() as f64 / recent_window as f64;
            let mean_steering = self.steering_history[start..].iter().sum::<f64>() / recent_window as f64;

            let new_risk = mean_collapse * (1.0 + mean_steering) * (1.0 + repeat_rate * 3.0);
            self.risk_score = 0.8 * self.risk_score + 0.2 * new_risk;
        }

        if self.risk_score > 0.4 && !self.conservative_mode {
            self.conservative_mode = true;
            self.conservative_mode_entered_at = token_idx as i32;
        } else if self.risk_score < 0.15 && self.conservative_mode {
            self.conservative_mode = false;
        }
    }

    pub fn should_force_conservative(&self) -> bool {
        self.conservative_mode
    }

    pub fn reset(&mut self) {
        self.collapse_history.clear();
        self.steering_history.clear();
        self.entropy_history.clear();
        self.repetition_history.clear();
        self.risk_score = 0.0;
        self.conservative_mode = false;
        self.conservative_mode_entered_at = -1;
        self.total_collapse_tokens = 0;
        self.total_warning_tokens = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Precision Governor (DVFS for LLM)
// ═══════════════════════════════════════════════════════════════════════════════

/// Maps token state → precision budget.
///
/// Like CPU frequency scaling (DVFS), but for numerical precision.
pub struct PrecisionGovernor;

impl PrecisionGovernor {
    /// Get target precision for (token_class, policy, collapse_status).
    pub fn get_precision(
        token_class: TokenClass,
        policy: &LayerPolicy,
        collapse: CollapseStatus,
    ) -> PrecisionMode {
        // Sacred and steering layers always get full precision
        if policy.is_sacred || policy.is_steering {
            return PrecisionMode::Fp16;
        }

        // Collapse overrides everything
        if collapse == CollapseStatus::Critical {
            return PrecisionMode::Fp16;
        }
        if collapse == CollapseStatus::Warning {
            return PrecisionMode::Q8;
        }

        // Token-class-based precision
        match token_class {
            TokenClass::Transition => PrecisionMode::Fp16,
            TokenClass::Steering => PrecisionMode::Q8,
            TokenClass::Stable => PrecisionMode::Q4,
            TokenClass::Repetitive => PrecisionMode::Q3,
            TokenClass::Default => PrecisionMode::Q5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Token Budget Allocator
// ═══════════════════════════════════════════════════════════════════════════════

/// Allocates compute budget per token class.
pub struct TokenBudget;

impl TokenBudget {
    /// Get (skip_fraction, use_temporal_stagger) for a token class.
    pub fn get_budget(token_class: TokenClass) -> (f64, bool) {
        match token_class {
            TokenClass::Repetitive => (0.80, true),
            TokenClass::Stable => (0.50, true),
            TokenClass::Default => (0.27, false),
            TokenClass::Steering => (0.0, false),
            TokenClass::Transition => (0.0, false),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scheduler (Kernel)
// ═══════════════════════════════════════════════════════════════════════════════

/// Phase-aware trajectory controller — the OS kernel.
///
/// Replaces the static 'for layer in layers' loop with state-dependent
/// compute allocation: what to run, at what precision, for how long.
pub struct Scheduler {
    pub config: SchedulerConfig,
    pub policy_table: Vec<LayerPolicy>,
    pub state: RuntimeState,
    pub collapse_detector: CollapseDetector,
    #[allow(dead_code)]
    governor: PrecisionGovernor,

    // Hysteresis states
    pub token_hysteresis: HysteresisState,
    pub collapse_hysteresis: CollapseHysteresis,

    // Collapse memory
    pub collapse_memory: CollapseMemory,

    // Counters
    pub layers_run: usize,
    pub layers_skipped: usize,
    pub layers_low_precision: usize,
    pub temporal_skips: usize,
    pub class_oscillations: usize,
    pub last_class: Option<TokenClass>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, n_layers: usize,
               is_moe: bool) -> Self {
        let policy_table = build_policy_table(
            n_layers, config.fusion_ratio, is_moe, config.family);
        let collapse_detector = CollapseDetector::new(
            config.collapse_detection);

        let layer_health: Vec<LayerHealth> = (0..n_layers)
            .map(|i| LayerHealth {
                layer_idx: i,
                skip_count: 0,
                run_count: 0,
                last_cos: None,
                divergence_count: 0,
            })
            .collect();

        let mut state = RuntimeState::default();
        state.layer_health = layer_health;

        Self {
            config,
            policy_table,
            state,
            collapse_detector,
            governor: PrecisionGovernor,
            token_hysteresis: HysteresisState::default(),
            collapse_hysteresis: CollapseHysteresis::default(),
            collapse_memory: CollapseMemory::default(),
            layers_run: 0,
            layers_skipped: 0,
            layers_low_precision: 0,
            temporal_skips: 0,
            class_oscillations: 0,
            last_class: None,
        }
    }

    // ── Token lifecycle ──

    /// Called at the start of each new token.
    pub fn begin_token(&mut self, prev_token_id: Option<usize>,
                       obs: &Observation) -> TokenClass {
        // Update collapse detector
        if let Some(tid) = prev_token_id {
            self.collapse_detector.update(tid);
        }

        // Update history
        self.state.entropy_history.push(obs.entropy);
        if self.state.entropy_history.len() > 64 {
            self.state.entropy_history.remove(0);
        }
        self.state.steering_history.push(obs.steering);
        if self.state.steering_history.len() > 64 {
            self.state.steering_history.remove(0);
        }

        // Classify WITH hysteresis
        let tc = self.token_hysteresis.classify(obs.entropy, obs.steering, obs.is_repeat);

        // Track class oscillations
        if let Some(last) = self.last_class {
            if tc != last {
                self.class_oscillations += 1;
            }
        }
        self.last_class = Some(tc);

        // Check collapse WITH hysteresis
        let raw_cs = self.collapse_detector.detect(obs, &self.state);
        let cs = self.collapse_hysteresis.update(raw_cs);
        self.state.collapse_status = cs;

        // Feed collapse memory (long-context degradation tracking)
        self.collapse_memory.update(cs, obs.steering, obs.entropy, obs.is_repeat, self.state.token_count);

        self.state.token_count += 1;

        tc
    }

    // ── Dispatch ──

    /// Should attention run at this layer for the given token class?
    pub fn should_run_attn(&mut self, layer_idx: usize,
                           token_class: TokenClass) -> bool {
        let policy = &self.policy_table[layer_idx];

        // Sacred and steering layers always run
        if policy.is_sacred || policy.is_steering {
            self.layers_run += 1;
            self.state.layer_health[layer_idx].run_count += 1;
            return true;
        }

        // Collapse detection overrides
        if self.state.collapse_status == CollapseStatus::Critical {
            self.layers_run += 1;
            self.state.layer_health[layer_idx].run_count += 1;
            return true;
        }

        // Long-context: conservative mode overrides all skip
        if self.collapse_memory.should_force_conservative() {
            self.layers_run += 1;
            self.state.layer_health[layer_idx].run_count += 1;
            return true;
        }

        // Temporal stride: stagger compute across tokens
        if self.config.temporal_stride > 1
            && self.state.token_count % self.config.temporal_stride != 0
        {
            self.temporal_skips += 1;
            self.layers_skipped += 1;
            self.state.layer_health[layer_idx].skip_count += 1;
            return false;
        }

        // Token-class-based skip
        let (_skip_frac, _) = TokenBudget::get_budget(token_class);

        match token_class {
            TokenClass::Repetitive => {
                if layer_idx % 4 != 0 {
                    self.layers_skipped += 1;
                    self.state.layer_health[layer_idx].skip_count += 1;
                    return false;
                }
            }
            TokenClass::Stable => {
                if layer_idx % 2 != 0 {
                    self.layers_skipped += 1;
                    self.state.layer_health[layer_idx].skip_count += 1;
                    return false;
                }
            }
            TokenClass::Steering | TokenClass::Transition => {
                // Never skip
            }
            TokenClass::Default => {
                if policy.attn_mode == AttnMode::Cached {
                    self.layers_skipped += 1;
                    self.state.layer_health[layer_idx].skip_count += 1;
                    return false;
                }
            }
        }

        self.layers_run += 1;
        self.state.layer_health[layer_idx].run_count += 1;
        true
    }

    /// Should FFN/MoE run at this layer?
    pub fn should_run_ffn(&self, layer_idx: usize,
                          token_class: TokenClass) -> bool {
        let policy = &self.policy_table[layer_idx];

        if policy.is_sacred || policy.is_steering {
            return true;
        }
        if self.state.collapse_status == CollapseStatus::Critical {
            return true;
        }
        if self.collapse_memory.should_force_conservative() {
            return true;
        }
        if matches!(token_class, TokenClass::Repetitive)
            && matches!(policy.phase, LayerZone::IsometricLocal | LayerZone::IsometricGlobal)
        {
            return false;
        }
        true
    }

    /// Get target precision for this (layer, token_class) pair.
    pub fn get_precision(&self, layer_idx: usize,
                         token_class: TokenClass) -> PrecisionMode {
        let policy = &self.policy_table[layer_idx];
        let collapse = self.state.collapse_status;
        PrecisionGovernor::get_precision(token_class, policy, collapse)
    }

    /// Get the policy for a specific layer.
    pub fn layer_policy(&self, layer_idx: usize) -> &LayerPolicy {
        &self.policy_table[layer_idx]
    }

    // ── Statistics ──

    /// Return current scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        let total = (self.layers_run + self.layers_skipped + self.layers_low_precision).max(1);
        SchedulerStats {
            token_count: self.state.token_count,
            collapse_status: self.state.collapse_status,
            layers_run: self.layers_run,
            layers_skipped: self.layers_skipped,
            layers_low_precision: self.layers_low_precision,
            temporal_skips: self.temporal_skips,
            skip_rate: self.layers_skipped as f64 / total as f64,
            entropy_mean: self.state.entropy_history.iter().sum::<f64>()
                / self.state.entropy_history.len().max(1) as f64,
            steering_mean: self.state.steering_history.iter().sum::<f64>()
                / self.state.steering_history.len().max(1) as f64,
        }
    }

    /// Reset all counters (call between benchmark runs).
    pub fn reset(&mut self) {
        self.layers_run = 0;
        self.layers_skipped = 0;
        self.layers_low_precision = 0;
        self.temporal_skips = 0;
        self.class_oscillations = 0;
        self.last_class = None;
        self.state = RuntimeState::default();
        self.collapse_detector.reset();
        self.token_hysteresis = HysteresisState::default();
        self.collapse_hysteresis = CollapseHysteresis::default();
        self.collapse_memory.reset();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStats {
    pub token_count: usize,
    pub collapse_status: CollapseStatus,
    pub layers_run: usize,
    pub layers_skipped: usize,
    pub layers_low_precision: usize,
    pub temporal_skips: usize,
    pub skip_rate: f64,
    pub entropy_mean: f64,
    pub steering_mean: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runtime IR — Pre-compiled Execution Plans
// ═══════════════════════════════════════════════════════════════════════════════

/// A single layer's execution instruction — no branching at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub layer_idx: usize,
    pub attn_mode: AttnMode,
    pub moe_mode: MoeMode,
    pub precision: PrecisionMode,
    pub is_sacred: bool,
}

/// Pre-compiled execution plan for one token class and collapse status.
///
/// Instead of branching at every layer ("should I run attention? what precision?"),
/// the scheduler pre-compiles ExecutionPlans at init time. At runtime, it's a
/// simple array lookup: plan.steps[layer_idx] tells you exactly what to execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Which token class this plan is for
    pub token_class: TokenClass,
    /// Collapse status this plan assumes
    pub collapse_status: CollapseStatus,
    /// Per-layer execution steps (index = layer_idx)
    pub steps: Vec<ExecutionStep>,
    /// Total layers in this plan
    pub n_layers: usize,
    /// Expected skip rate (0-1)
    pub expected_skip_rate: f64,
    /// Expected average precision bits
    pub expected_avg_precision: f64,
}

/// A complete pre-compiled plan table.
///
/// Indexed by [token_class][collapse_status] → ExecutionPlan.
/// Covers all 5×3 = 15 possible states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanTable {
    pub plans: Vec<ExecutionPlan>,
    pub n_layers: usize,
}

/// Compile all execution plans from a policy table.
pub fn compile_plan_table(policy_table: &[LayerPolicy]) -> PlanTable {
    let n_layers = policy_table.len();
    let token_classes = [
        TokenClass::Repetitive,
        TokenClass::Stable,
        TokenClass::Default,
        TokenClass::Steering,
        TokenClass::Transition,
    ];
    let collapse_statuses = [
        CollapseStatus::Healthy,
        CollapseStatus::Warning,
        CollapseStatus::Critical,
    ];

    let mut plans = Vec::with_capacity(token_classes.len() * collapse_statuses.len());

    for &tc in &token_classes {
        for &cs in &collapse_statuses {
            let steps = compile_plan(policy_table, tc, cs);
            let expected_skip_rate = steps.iter()
                .filter(|s| matches!(s.attn_mode, AttnMode::Cached | AttnMode::Skip))
                .count() as f64 / n_layers.max(1) as f64;
            let expected_avg_precision = steps.iter()
                .map(|s| s.precision.bits() as f64)
                .sum::<f64>() / n_layers.max(1) as f64;

            plans.push(ExecutionPlan {
                token_class: tc,
                collapse_status: cs,
                steps,
                n_layers,
                expected_skip_rate,
                expected_avg_precision,
            });
        }
    }

    PlanTable { plans, n_layers }
}

/// Compile a single execution plan for a specific (token_class, collapse_status).
fn compile_plan(
    policy_table: &[LayerPolicy],
    token_class: TokenClass,
    collapse_status: CollapseStatus,
) -> Vec<ExecutionStep> {
    let (_skip_frac, _temporal_ok) = TokenBudget::get_budget(token_class);

    policy_table.iter().map(|policy| {
        let (attn_mode, moe_mode, precision) = if policy.is_sacred || policy.is_steering {
            // Sacred/steering layers: always full
            (AttnMode::Full, MoeMode::Full, PrecisionMode::Fp16)
        } else if collapse_status == CollapseStatus::Critical {
            // Collapse recovery: force full
            (AttnMode::Full, MoeMode::Full, PrecisionMode::Fp16)
        } else if collapse_status == CollapseStatus::Warning {
            // Warning: bump precision
            (AttnMode::Full, MoeMode::Adaptive, PrecisionMode::Q8)
        } else {
            // Normal: token-class-dependent
            match token_class {
                TokenClass::Repetitive => {
                    let attn = if policy.layer_idx % 4 == 0 { AttnMode::Reduced } else { AttnMode::Skip };
                    let moe = if matches!(policy.phase, LayerZone::IsometricLocal | LayerZone::IsometricGlobal) {
                        MoeMode::Skip
                    } else {
                        MoeMode::Sparse
                    };
                    (attn, moe, PrecisionMode::Q3)
                }
                TokenClass::Stable => {
                    let attn = if policy.layer_idx % 2 == 0 { AttnMode::Cached } else { AttnMode::Skip };
                    (attn, MoeMode::Sparse, PrecisionMode::Q4)
                }
                TokenClass::Steering => {
                    (AttnMode::Full, MoeMode::Full, PrecisionMode::Q8)
                }
                TokenClass::Transition => {
                    (AttnMode::Full, MoeMode::Full, PrecisionMode::Fp16)
                }
                TokenClass::Default => {
                    // Phase-dependent: use policy table's compiled mode
                    let attn = match policy.attn_mode {
                        AttnMode::Cached => AttnMode::Cached,
                        _ => AttnMode::Full,
                    };
                    (attn, MoeMode::Adaptive, policy.precision)
                }
            }
        };

        ExecutionStep {
            layer_idx: policy.layer_idx,
            attn_mode,
            moe_mode,
            precision,
            is_sacred: policy.is_sacred,
        }
    }).collect()
}

/// Look up a pre-compiled plan from the table.
pub fn lookup_plan(plan_table: &PlanTable, token_class: TokenClass,
                   collapse_status: CollapseStatus) -> &ExecutionPlan {
    plan_table.plans.iter()
        .find(|p| p.token_class == token_class && p.collapse_status == collapse_status)
        .expect("PlanTable must contain all 5×3 plans")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Token Trace — Recordable/Replayable Execution History
// ═══════════════════════════════════════════════════════════════════════════════

/// A complete record of one token's execution — for replay and research.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTrace {
    /// Token index in the sequence
    pub token_idx: usize,
    /// Token ID
    pub token_id: usize,
    /// Observation at this token
    pub entropy: f64,
    pub steering: f64,
    pub top1_logit: f64,
    pub is_repeat: bool,
    /// Classification result
    pub token_class: TokenClass,
    pub collapse_status: CollapseStatus,
    /// Which plan was used
    pub plan_index: usize,
    /// Per-layer execution record
    pub layer_actions: Vec<LayerAction>,
    /// Timing
    pub elapsed_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAction {
    pub layer_idx: usize,
    pub attn_ran: bool,
    pub ffn_ran: bool,
    pub precision_used: u8,
}

/// A full runtime trace — replayable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeTrace {
    pub model: String,
    pub config: SchedulerConfig,
    pub n_layers: usize,
    pub tokens: Vec<TokenTrace>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fault Injection
// ═══════════════════════════════════════════════════════════════════════════════

/// Types of faults that can be injected to test the OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultType {
    /// Force all layers to q3 precision
    ForceQ3,
    /// Force aggressive skip (skip all non-sacred attention)
    ExcessiveSkip,
    /// Drop half of experts (MoE only)
    ExpertDrop,
    /// Inject noise into hidden state
    HiddenNoise,
    /// Force random token classification
    RandomClass,
}

/// A single fault injection event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultInjection {
    pub fault_type: FaultType,
    /// Which token to inject at (None = all tokens)
    pub token_idx: Option<usize>,
    /// Duration in tokens (None = until recovery)
    pub duration: Option<usize>,
    /// Fault intensity (0.0-1.0)
    pub intensity: f64,
}

/// Result of a fault injection test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultTestResult {
    pub fault: FaultInjection,
    /// Did the collapse detector fire?
    pub detected: bool,
    /// Tokens until detection
    pub detection_latency: usize,
    /// Tokens until recovery after fault removed
    pub recovery_latency: usize,
    /// Collapse status sequence during fault
    pub status_sequence: Vec<CollapseStatus>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_classification_stable() {
        let obs = Observation {
            entropy: 0.02,
            steering: 0.1,
            is_repeat: false,
            ..Default::default()
        };
        assert_eq!(obs.classify(), TokenClass::Stable);
    }

    #[test]
    fn test_token_classification_repetitive() {
        let obs = Observation {
            entropy: 0.3,
            steering: 0.2,
            is_repeat: true,
            ..Default::default()
        };
        assert_eq!(obs.classify(), TokenClass::Repetitive);
    }

    #[test]
    fn test_token_classification_steering() {
        let obs = Observation {
            entropy: 0.1,
            steering: 0.55,
            is_repeat: false,
            ..Default::default()
        };
        assert_eq!(obs.classify(), TokenClass::Steering);
    }

    #[test]
    fn test_token_classification_transition() {
        let obs = Observation {
            entropy: 0.25,
            steering: 0.65,
            is_repeat: false,
            ..Default::default()
        };
        assert_eq!(obs.classify(), TokenClass::Transition);
    }

    #[test]
    fn test_token_classification_default() {
        let obs = Observation {
            entropy: 0.1,
            steering: 0.3,
            is_repeat: false,
            ..Default::default()
        };
        assert_eq!(obs.classify(), TokenClass::Default);
    }

    #[test]
    fn test_precision_governor_sacred() {
        let policy = LayerPolicy::sacred(2, LayerZone::Unfold);
        let prec = PrecisionGovernor::get_precision(
            TokenClass::Stable, &policy, CollapseStatus::Healthy);
        assert_eq!(prec, PrecisionMode::Fp16);
    }

    #[test]
    fn test_precision_governor_stable() {
        let policy = LayerPolicy::transport(10, LayerZone::IsometricLocal);
        let prec = PrecisionGovernor::get_precision(
            TokenClass::Stable, &policy, CollapseStatus::Healthy);
        assert_eq!(prec, PrecisionMode::Q4);
    }

    #[test]
    fn test_precision_governor_collapse_override() {
        let policy = LayerPolicy::transport(10, LayerZone::IsometricLocal);
        let prec = PrecisionGovernor::get_precision(
            TokenClass::Stable, &policy, CollapseStatus::Critical);
        assert_eq!(prec, PrecisionMode::Fp16);
    }

    #[test]
    fn test_policy_table_tinyllama() {
        let table = build_policy_table(22, 0.5, false, Family::ResidualTransport);
        assert_eq!(table.len(), 22);
        // L0-L1 sacred
        assert!(table[0].is_sacred);
        assert!(table[1].is_sacred);
        // L2 sacred
        assert!(table[2].is_sacred);
        // L10 not sacred, not steering
        assert!(!table[10].is_sacred);
        assert!(!table[10].is_steering);
        // Late layers are DIVERGENT (steering = true for l >= 0.7*22 ≈ 15)
        // But last layer is also DIVERGENT in our classifier
    }

    #[test]
    fn test_collapse_detector_repetition() {
        let config = CollapseDetection::default();
        let mut detector = CollapseDetector::new(config);
        // Simulate 6 repeats (first call initializes prev_token_id without incrementing)
        for _ in 0..6 {
            detector.update(42);
        }
        let obs = Observation::default();
        let state = RuntimeState::default();
        assert_eq!(detector.detect(&obs, &state), CollapseStatus::Critical);
    }

    #[test]
    fn test_scheduler_basic() {
        let config = SchedulerConfig::default();
        let mut sched = Scheduler::new(config, 22, false);

        let obs = Observation {
            entropy: 0.1,
            steering: 0.3,
            is_repeat: false,
            ..Default::default()
        };
        let tc = sched.begin_token(None, &obs);
        assert_eq!(tc, TokenClass::Default);

        // Sacred layers always run
        assert!(sched.should_run_attn(0, tc));
        assert!(sched.should_run_attn(1, tc));
        assert!(sched.should_run_attn(2, tc));

        let stats = sched.stats();
        assert_eq!(stats.token_count, 1);
    }

    #[test]
    fn test_skip_rate() {
        let config = SchedulerConfig::default();
        let mut sched = Scheduler::new(config, 22, false);

        let obs = Observation {
            entropy: 0.02,
            steering: 0.1,
            is_repeat: false,
            ..Default::default()
        };
        let _tc1 = sched.begin_token(None, &obs);
        let tc = sched.begin_token(None, &obs);
        assert_eq!(tc, TokenClass::Stable);

        // Run all layers, count skips
        for l in 0..22 {
            sched.should_run_attn(l, tc);
        }
        assert!(sched.layers_skipped > 0, "Stable token should skip some layers");
    }

    // ── Runtime IR tests ──

    #[test]
    fn test_compile_plan_table() {
        let table = build_policy_table(22, 0.5, false, Family::ResidualTransport);
        let plan_table = compile_plan_table(&table);
        // 5 token classes × 3 collapse statuses
        assert_eq!(plan_table.plans.len(), 15);
        for plan in &plan_table.plans {
            assert_eq!(plan.steps.len(), 22);
        }
    }

    #[test]
    fn test_lookup_plan() {
        let table = build_policy_table(22, 0.5, false, Family::ResidualTransport);
        let plan_table = compile_plan_table(&table);

        let plan = lookup_plan(&plan_table, TokenClass::Stable, CollapseStatus::Healthy);
        assert_eq!(plan.token_class, TokenClass::Stable);
        // Stable tokens should skip some layers
        assert!(plan.expected_skip_rate > 0.0);

        let plan_crit = lookup_plan(&plan_table, TokenClass::Stable, CollapseStatus::Critical);
        // Critical collapse: all layers full, no skips
        assert_eq!(plan_crit.expected_skip_rate, 0.0);
        assert_eq!(plan_crit.expected_avg_precision, 16.0);
    }

    #[test]
    fn test_sacred_layers_always_full() {
        let table = build_policy_table(22, 0.5, false, Family::ResidualTransport);
        let plan_table = compile_plan_table(&table);

        // Check that sacred layers (L0-L2, L21) always get Full/Fp16
        for plan in &plan_table.plans {
            for &sacred_idx in &[0, 1, 2, 21] {
                let step = &plan.steps[sacred_idx];
                assert!(matches!(step.attn_mode, AttnMode::Full),
                    "L{} should always be Full attn for {:?}/{:?}, got {:?}",
                    sacred_idx, plan.token_class, plan.collapse_status, step.attn_mode);
                assert_eq!(step.precision, PrecisionMode::Fp16,
                    "L{} should always be Fp16, got {:?}", sacred_idx, step.precision);
            }
        }
    }

    #[test]
    fn test_repetitive_plan_ultra_aggressive() {
        let table = build_policy_table(22, 0.5, false, Family::ResidualTransport);
        let plan_table = compile_plan_table(&table);

        let plan = lookup_plan(&plan_table, TokenClass::Repetitive, CollapseStatus::Healthy);
        // Repetitive non-sacred, non-steering layers get Skip + Q3
        // (Sacred L0-2 + steering L16-21 = 9 layers always full)
        assert!(plan.expected_skip_rate > 0.4,
            "Repetitive plan skip rate {} should be >0.4 (got {})", plan.expected_skip_rate, plan.expected_skip_rate);
        let l10 = &plan.steps[10];
        assert_eq!(l10.precision, PrecisionMode::Q3,
            "L10 Repetitive should be Q3, got {:?}", l10.precision);
        assert!(matches!(l10.attn_mode, AttnMode::Skip | AttnMode::Reduced),
            "L10 Repetitive attn should be skip/reduced, got {:?}", l10.attn_mode);
    }

    #[test]
    fn test_transition_plan_full_precision() {
        let table = build_policy_table(22, 0.5, false, Family::ResidualTransport);
        let plan_table = compile_plan_table(&table);

        let plan = lookup_plan(&plan_table, TokenClass::Transition, CollapseStatus::Healthy);
        assert_eq!(plan.expected_skip_rate, 0.0);
        assert_eq!(plan.expected_avg_precision, 16.0);
    }

    #[test]
    fn test_token_trace_serialization() {
        let trace = TokenTrace {
            token_idx: 0,
            token_id: 42,
            entropy: 0.1,
            steering: 0.3,
            top1_logit: 15.0,
            is_repeat: false,
            token_class: TokenClass::Default,
            collapse_status: CollapseStatus::Healthy,
            plan_index: 2,
            layer_actions: vec![
                LayerAction { layer_idx: 0, attn_ran: true, ffn_ran: true, precision_used: 16 },
                LayerAction { layer_idx: 10, attn_ran: false, ffn_ran: true, precision_used: 4 },
            ],
            elapsed_us: 5000,
        };
        let json = serde_json::to_string(&trace).unwrap();
        let trace2: TokenTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(trace2.token_class, TokenClass::Default);
        assert_eq!(trace2.layer_actions.len(), 2);
    }

    #[test]
    fn test_fault_injection_types() {
        let fault = FaultInjection {
            fault_type: FaultType::ForceQ3,
            token_idx: Some(5),
            duration: Some(10),
            intensity: 1.0,
        };
        assert_eq!(fault.fault_type, FaultType::ForceQ3);
        assert_eq!(fault.token_idx, Some(5));
    }

    #[test]
    fn test_hysteresis_and_collapse_memory() {
        let mut hyst = HysteresisState::default();
        
        // 1st token stable: count becomes 1, returns Default (not stable yet)
        let tc1 = hyst.classify(0.02, 0.1, false);
        assert_eq!(tc1, TokenClass::Default);
        assert_eq!(hyst.consecutive_stable, 1);

        // 2nd token stable: count becomes 2, returns Stable
        let tc2 = hyst.classify(0.02, 0.1, false);
        assert_eq!(tc2, TokenClass::Stable);
        assert_eq!(hyst.consecutive_stable, 2);

        // 3rd token stable: remains Stable
        let tc3 = hyst.classify(0.02, 0.1, false);
        assert_eq!(tc3, TokenClass::Stable);

        // Transition: spikes should change class
        let tc4 = hyst.classify(0.25, 0.75, false);
        assert_eq!(tc4, TokenClass::Transition);

        // Repetitive absolute override
        let tc5 = hyst.classify(0.1, 0.1, true);
        assert_eq!(tc5, TokenClass::Repetitive);

        // Test precision get_precision rate limit
        // Immediate upgrade
        let p1 = hyst.get_precision(PrecisionMode::Fp16);
        assert_eq!(p1, PrecisionMode::Fp16);

        // Downgrade requires 3 consecutive stable tokens
        let p2 = hyst.get_precision(PrecisionMode::Q4);
        assert_eq!(p2, PrecisionMode::Fp16); // first stable token: no downgrade
        let p3 = hyst.get_precision(PrecisionMode::Q4);
        assert_eq!(p3, PrecisionMode::Fp16); // second stable token: no downgrade
        let p4 = hyst.get_precision(PrecisionMode::Q4);
        assert_eq!(p4, PrecisionMode::Q8); // third stable token: downgrade by 1 level (Fp16 -> Q8)

        // Test CollapseMemory
        let mut mem = CollapseMemory::default();
        mem.update(CollapseStatus::Warning, 0.2, 0.1, false, 1);
        assert_eq!(mem.total_warning_tokens, 1);
        assert!(!mem.should_force_conservative());

        // Accumulate a bunch of critical risk
        for i in 2..20 {
            mem.update(CollapseStatus::Critical, 0.9, 0.01, true, i);
        }
        assert!(mem.risk_score > 0.4);
        assert!(mem.should_force_conservative());
    }

    #[test]
    fn test_observation_pipeline() {
        let mut pipe = ObservationPipeline::new();

        // 1. observe_logits (entropy, logit, token_id)
        let logits = vec![1.0, 2.0, 1.0, 1.0, 5.0];
        let (ent, max_log, tid) = pipe.observe_logits(&logits);
        assert!(ent > 0.0 && ent < 1.0);
        assert_eq!(max_log, 5.0);
        assert_eq!(tid, 4);

        // 2. observe_hidden (steering)
        let h1 = vec![1.0, 0.0, 0.0];
        let steer1 = pipe.observe_hidden(&h1);
        assert_eq!(steer1, 0.0); // first token has no previous hidden

        let h2 = vec![0.0, 1.0, 0.0];
        let steer2 = pipe.observe_hidden(&h2);
        assert!(steer2 > 0.99); // orthogonal vectors: cos = 0, steer = 1.0

        // 3. observe_attention (divergence)
        let attn_weights = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let div1 = pipe.observe_attention(0, &attn_weights);
        assert!(div1.is_none()); // first call initializes weights

        let attn_weights2 = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let div2 = pipe.observe_attention(0, &attn_weights2);
        assert!(div2.is_some());
        assert!(div2.unwrap() > 0.99); // orthogonal: 1.0

        // 4. Standalone functions
        assert!((compute_steering(&h1, &h1) - 0.0).abs() < 1e-9);
        assert!(compute_steering(&h1, &h2) > 0.99);
    }
}
