use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ── Expert Layout types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertLayout {
    pub schema_version: u32,
    pub model: String,
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub num_layers: u32,
    pub num_experts: u32,
    pub vocab_size: Option<u32>,
    pub layout_kind: ExpertLayoutKind,
    pub logical_routed_expert_count: u64,
    pub experts: Vec<ExpertEntry>,
    pub packed_expert_layers: Vec<PackedExpertLayerEntry>,
    pub shared_experts: Vec<SharedExpertEntry>,
    pub routers: Vec<RouterEntry>,
    pub unknown_tensors: Vec<UnknownTensorEntry>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExpertLayoutKind {
    PerExpert,
    PackedExperts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertEntry {
    pub layer: u32,
    pub expert: u32,
    pub gate: Option<TensorRef>,
    pub up: Option<TensorRef>,
    pub gate_up: Option<TensorRef>,
    pub down: Option<TensorRef>,
    pub source_files: Vec<String>,
    pub complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedExpertLayerEntry {
    pub layer: u32,
    pub num_experts_per_layer: u32,
    pub gate_up: Option<TensorRef>,
    pub down: Option<TensorRef>,
    pub source_files: Vec<String>,
    pub complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedExpertEntry {
    pub layer: u32,
    pub gate: Option<TensorRef>,
    pub up: Option<TensorRef>,
    pub gate_up: Option<TensorRef>,
    pub down: Option<TensorRef>,
    pub shared_gate: Option<TensorRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterEntry {
    pub layer: u32,
    pub tensor: TensorRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnknownTensorEntry {
    pub tensor_name: String,
    pub source_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRef {
    pub tensor_kind: ExpertTensorKind,
    pub tensor_name: String,
    pub source_file: String,
    pub shape: Option<Vec<usize>>,
    pub dtype: Option<String>,
    pub byte_offset: Option<u64>,
    pub byte_len: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExpertTensorKind {
    Gate,
    Up,
    GateUp,
    Down,
    PackedGateUp,
    PackedDown,
    Router,
    Shared,
    Unknown,
}

// ── Importance / Tier types ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExpertTier {
    Hot,
    Warm,
    Cold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertImportance {
    pub schema_version: u32,
    pub experts: Vec<ExpertImportanceEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertImportanceEntry {
    pub layer: u32,
    pub expert: u32,
    pub selected_count: u64,
    pub frequency: f64,
    pub avg_gate_weight: f64,
    pub max_gate_weight: f64,
    pub importance: f64,
    pub tier: ExpertTier,
    pub recommended_format: String,
    pub eviction_priority: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertCoresidency {
    pub schema_version: u32,
    pub pairs: Vec<ExpertCoresidencyPair>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertCoresidencyPair {
    pub layer: u32,
    pub expert_a: u32,
    pub expert_b: u32,
    pub co_count: u64,
    pub co_score: f64,
}

// ── Residency Plan types ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyPlan {
    pub schema_version: u32,
    pub target: String,
    pub resident_cache_capacity_bytes: u64,
    pub initial_hot_experts: Vec<PlannedHotExpert>,
    pub eviction_priority: Vec<EvictionPriorityEntry>,
    pub summary: ResidencyPlanSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedHotExpert {
    pub layer: u32,
    pub expert: u32,
    pub bytes: u64,
    pub importance: f64,
    pub tier: ExpertTier,
    pub bytes_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionPriorityEntry {
    pub layer: u32,
    pub expert: u32,
    pub priority: f64,
    pub tier: ExpertTier,
    pub importance: f64,
    pub selected_count: u64,
    pub avg_gate_weight: f64,
    pub bytes: u64,
    pub bytes_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyPlanSummary {
    pub initial_hot_expert_count: usize,
    pub initial_hot_expert_bytes: u64,
    pub eviction_priority_count: usize,
    pub bytes_fallback_expert_count: usize,
}

// ── Phase Policy types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePolicy {
    pub schema_version: u32,
    /// Classifier method used to assign phases. "heuristic_lko_v1" until a
    /// learned/empirical classifier replaces it.
    pub source: String,
    /// Epistemic confidence in the phase assignments.
    /// "experimental" while rule-based; upgrade to "validated" after ablations.
    pub confidence: String,
    pub layers: Vec<PhasePolicyLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePolicyLayer {
    pub layer: u32,
    pub phase: String,
    pub recommended_policy: RecommendedPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedPolicy {
    pub policy_kind: String,
    pub moe_top_p: f32,
    pub moe_min_experts: u32,
    pub moe_max_experts: u32,
}

// ── Runtime Profile ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProfile {
    pub schema_version: u32,
    pub profile_name: String,
    pub target: String,
    pub backend: String,
    pub policy_kind: String,
    pub moe_top_p: f32,
    pub moe_min_experts: u32,
    pub moe_max_experts: u32,
    pub resident_cache_capacity_bytes: u64,
    pub group_preresolve_top_n: u32,
    pub group_preresolve_max_bytes: u64,
    pub source_model: String,
    pub source_calibration: Option<String>,
}

// ── Manifest types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub pack_type: String,
    pub model_family: String,
    pub model_name: String,
    pub target: String,
    pub created_at: String,
    pub files: Vec<String>,
    pub notes: String,
}

// ── Config / Index parsing types ─────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelConfig {
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,
    pub num_hidden_layers: Option<u32>,
    pub num_experts: Option<u32>,
    pub num_local_experts: Option<u32>,
    pub vocab_size: Option<u32>,
    pub hidden_size: Option<u32>,
    pub intermediate_size: Option<u32>,
    pub text_config: Option<Box<ModelConfig>>,
}

impl ModelConfig {
    pub fn effective_num_hidden_layers(&self) -> Option<u32> {
        self.num_hidden_layers
            .or_else(|| self.text_config.as_ref().and_then(|c| c.num_hidden_layers))
    }

    pub fn effective_num_experts(&self) -> Option<u32> {
        self.num_experts
            .or(self.num_local_experts)
            .or_else(|| self.text_config.as_ref().and_then(|c| c.num_experts.or(c.num_local_experts)))
    }

    pub fn effective_vocab_size(&self) -> Option<u32> {
        self.vocab_size
            .or_else(|| self.text_config.as_ref().and_then(|c| c.vocab_size))
    }

    pub fn effective_hidden_size(&self) -> Option<u32> {
        self.hidden_size
            .or_else(|| self.text_config.as_ref().and_then(|c| c.hidden_size))
    }

    pub fn effective_intermediate_size(&self) -> Option<u32> {
        self.intermediate_size
            .or_else(|| self.text_config.as_ref().and_then(|c| c.intermediate_size))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SafeTensorsIndex {
    pub weight_map: HashMap<String, String>,
    #[serde(default)]
    pub metadata: SafeTensorsIndexMetadata,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SafeTensorsIndexMetadata {
    #[serde(default, deserialize_with = "deserialize_opt_u64_from_number")]
    pub total_size: Option<u64>,
}

fn deserialize_opt_u64_from_number<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(n)) => {
            if let Some(v) = n.as_u64() {
                Ok(Some(v))
            } else if let Some(v) = n.as_f64() {
                Ok(Some(v as u64))
            } else {
                Err(serde::de::Error::custom("unsupported numeric total_size"))
            }
        }
        other => Err(serde::de::Error::custom(format!(
            "invalid total_size value: {other:?}"
        ))),
    }
}

// ── Calibration types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct CalibrationTraceEvent {
    pub token_id: Option<u64>,
    pub layer: u32,
    pub selected_experts: Vec<u32>,
    pub selected_weights: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct ExpertUsageStats {
    pub selected_count: u64,
    pub sum_gate_weight: f64,
    pub max_gate_weight: f64,
}

#[derive(Debug, Deserialize)]
pub struct MoeStatsEnvelope {
    #[serde(default)]
    pub moe_io_events: Vec<MoeIoEvent>,
}

#[derive(Debug, Deserialize)]
pub struct MoeIoEvent {
    pub layer_id: u32,
    pub selected_experts: Vec<u32>,
    #[serde(default)]
    pub selected_weights: Vec<f32>,
}

// ── Internal helper types ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedTensorName {
    pub layer_idx: Option<u32>,
    pub expert_id: Option<u32>,
    pub tensor_kind: ExpertTensorKind,
    pub is_shared: bool,
    pub is_packed_experts: bool,
    pub tensor_name: String,
    pub source_file: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

#[derive(Debug, Clone)]
pub struct PlannedExpert {
    pub layer: u32,
    pub expert: u32,
    pub importance: f64,
    pub tier: ExpertTier,
    pub selected_count: u64,
    pub avg_gate_weight: f64,
    pub bytes: u64,
    pub bytes_source: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedModelFiles {
    pub index_path: PathBuf,
    pub config_path: PathBuf,
}

// ── Specialize-specific types ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityBudget {
    Conservative,
    Balanced,
    Aggressive,
}

impl QualityBudget {
    pub fn mass_loss_threshold(&self) -> f64 {
        match self {
            QualityBudget::Conservative => 0.02,
            QualityBudget::Balanced => 0.05,
            QualityBudget::Aggressive => 0.10,
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "conservative" => Some(QualityBudget::Conservative),
            "balanced" => Some(QualityBudget::Balanced),
            "aggressive" => Some(QualityBudget::Aggressive),
            _ => None,
        }
    }
}

impl std::fmt::Display for QualityBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualityBudget::Conservative => write!(f, "conservative"),
            QualityBudget::Balanced => write!(f, "balanced"),
            QualityBudget::Aggressive => write!(f, "aggressive"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecializeArgs {
    pub model: PathBuf,
    pub calib: PathBuf,
    pub target: String,
    pub task_profile: String,
    pub memory_budget: Option<u64>,
    pub quality_budget: QualityBudget,
    pub out: PathBuf,
}

// ── Quant Plan types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantPlan {
    pub schema_version: u32,
    pub entries: Vec<QuantPlanEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantPlanEntry {
    pub kind: String,
    pub layer: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    pub recommended_format: String,
    pub reason: String,
}

// ── Pruning Plan types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningPlan {
    pub schema_version: u32,
    pub summary: PruningPlanSummary,
    pub experts: Vec<PruningPlanEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningPlanSummary {
    // ── Action counts ───────────────────────────────────────────────────
    pub protect_count: usize,
    pub keep_count: usize,
    pub cold_tier_count: usize,
    pub compress_count: usize,
    pub prune_candidate_count: usize,
    // ── Mass-loss budget ────────────────────────────────────────────────
    pub estimated_routing_mass_loss: f64,
    pub quality_budget: String,
    pub mass_loss_threshold: f64,
    pub safe: bool,
    // ── Epistemic flags ─────────────────────────────────────────────────
    /// True: mass-loss is a routing-trace estimate, not an end-to-end
    /// generation-quality measurement. Do not ship without verification.
    pub estimated_only: bool,
    /// True: a verification pass (smoke + oracle trace) is required before
    /// applying this pruning plan to a production pack.
    pub requires_verification: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningPlanEntry {
    pub layer: u32,
    pub expert: u32,
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_format: Option<String>,
    pub estimated_mass_loss: f64,
    pub reason: String,
}

// ── Verification Plan types ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationPlan {
    pub schema_version: u32,
    pub checks: Vec<VerificationCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCheck {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_cosine: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_allowed: Option<f64>,
}

// ── Calibration stats (output of CalibrationPass) ────────────────────────

#[derive(Debug, Clone)]
pub struct CalibrationStats {
    pub importance: ExpertImportance,
    pub coresidency: ExpertCoresidency,
    pub layer_event_counts: std::collections::BTreeMap<u32, u64>,
    pub total_events: u64,
}
