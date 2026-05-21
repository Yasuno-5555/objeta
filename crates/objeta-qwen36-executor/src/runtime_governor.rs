use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GovernorMode {
    Disabled,
    ObserveOnly,
    ApplyAtTokenBoundary,
}

impl GovernorMode {
    pub fn from_env() -> Self {
        match std::env::var("OBJETA_GOVERNOR_MODE")
            .unwrap_or_else(|_| "disabled".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "observe" | "observe_only" | "observe-only" => Self::ObserveOnly,
            "apply" | "apply_at_token_boundary" | "apply-at-token-boundary" => {
                Self::ApplyAtTokenBoundary
            }
            _ => Self::Disabled,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorPolicy {
    pub high_entropy_threshold: f32,
    pub critical_entropy_threshold: f32,
    pub entropy_hard_floor: f32,
    pub high_memory_pressure_ratio: f32,
    pub critical_memory_pressure_ratio: f32,
    pub io_thrash_miss_threshold: u64,
    pub io_thrash_loaded_bytes_threshold: u64,
}

impl Default for GovernorPolicy {
    fn default() -> Self {
        Self {
            high_entropy_threshold: 2.25,
            critical_entropy_threshold: 2.75,
            entropy_hard_floor: 0.05,
            high_memory_pressure_ratio: 0.90,
            critical_memory_pressure_ratio: 0.98,
            io_thrash_miss_threshold: 8,
            io_thrash_loaded_bytes_threshold: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PressureClass {
    Low,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IoThrashClass {
    Stable,
    Thrashing,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QualityRiskClass {
    Low,
    Elevated,
    High,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GovernorPhase {
    Prefill,
    DecodeWarmup,
    DecodeSteady,
    PostStop,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RepetitionKind {
    SemanticRepetition,
    SpecialTokenLoop,
    StructuralMarkerLoop,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RiskReason {
    HighEntropyUncertainty,
    LowEntropyCollapse,
    Repetition,
    ExcessDroppedRoutingMass,
    ConsecutiveCollapseSignal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum HardRiskKind {
    QualityHard,
    MemoryHard,
    IoHard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeObservation {
    pub step: usize,
    pub token_id: usize,
    pub token_position: usize,
    pub phase: GovernorPhase,
    pub prev_decode_entropy: f32,
    pub rss_mb: f32,
    pub os_memory_pressure_state: String,
    pub repetition_risk: bool,
    pub repetition_kind: Option<RepetitionKind>,
    pub collapse_risk: bool,
    pub resident_capacity_bytes: u64,
    pub resident_bytes: u64,
    pub resident_hit_delta: u64,
    pub resident_miss_delta: u64,
    pub pageouts_delta: u64,
    pub swapout_delta: u64,
    pub confirmed_pageouts_delta: u64,
    pub confirmed_swapouts_delta: u64,
    pub pageout_mb_per_sec: f32,
    pub swapout_mb_per_sec: f32,
    pub pageout_mb_per_token: f32,
    pub swapout_mb_per_token: f32,
    pub actual_bytes_loaded_delta: u64,
    pub resident_bytes_reused_delta: u64,
    pub avg_selected_experts: f32,
    pub avg_routing_mass_kept: f32,
    pub avg_routing_mass_dropped: f32,
    pub avg_routing_mass_kept_pre_renorm: f32,
    pub avg_routing_mass_dropped_pre_renorm: f32,
    pub avg_routing_mass_sum_after_renorm: f32,
    pub forward_wall_ms_delta: f64,
    pub cache_lookup_wall_ms_delta: f64,
    pub eviction_count_delta: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorAppliedAction {
    pub kind: String,
    pub no_op: bool,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub actual_state_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorDecision {
    pub mode: GovernorMode,
    pub phase: GovernorPhase,
    pub memory_pressure: PressureClass,
    pub io_thrash: IoThrashClass,
    pub quality_risk: QualityRiskClass,
    pub risk_reasons: Vec<RiskReason>,
    pub suggested_top_p: Option<f32>,
    pub suggested_min_experts: Option<usize>,
    pub suggested_resident_cache_capacity_bytes: Option<u64>,
    pub suggested_group_preresolve_top_n: Option<usize>,
    pub suggested_group_preresolve_max_bytes: Option<u64>,
    pub consecutive_quality_risk: u32,
    pub applied: bool,
    pub applied_actions: Vec<GovernorAppliedAction>,
    pub actual_state_change_count: u32,
    pub no_op_action_count: u32,
    pub cache_capacity_reduction_count: u32,
    pub offensive_action_count: u32,
    pub rollback_action_count: u32,
    pub hard_risk_count: u32,
    pub soft_risk_count: u32,
    pub stable_for: u32,
    pub stable_for_max: u32,
    pub quality_risky_recently: bool,
    pub stable_for_reset_reasons: Vec<String>,
    pub blocked_offensive_reason: Option<String>,
    pub hard_risk_kind: Option<HardRiskKind>,
    pub quality_hard_count: u32,
    pub memory_hard_count: u32,
    pub io_hard_count: u32,
    pub offensive_blocked_by_memory_count: u32,
    pub offensive_blocked_by_quality_count: u32,
    pub confirmed_pageouts_delta: u64,
    pub confirmed_swapouts_delta: u64,
    pub pageout_mb_per_sec: f32,
    pub swapout_mb_per_sec: f32,
    pub pageout_mb_per_token: f32,
    pub swapout_mb_per_token: f32,
    pub current_effective_top_p: Option<f32>,
    pub current_effective_min_experts: Option<usize>,
    pub action_reason: Option<String>,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorTraceRecord {
    pub observation: RuntimeObservation,
    pub decision: GovernorDecision,
}

#[derive(Debug, Clone)]
pub struct GovernorTraceWriter {
    path: PathBuf,
}

impl GovernorTraceWriter {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn write_record(&self, record: &GovernorTraceRecord) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(record)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
        writeln!(file, "{}", line)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GovernorCounters {
    pub resident_hit_count: u64,
    pub resident_miss_count: u64,
    pub actual_bytes_loaded: u64,
    pub resident_bytes_reused: u64,
    pub eviction_count: u64,
    pub forward_wall_ms: f64,
    pub cache_lookup_wall_ms: f64,
}

#[derive(Debug, Clone)]
pub struct RuntimeGovernor {
    pub mode: GovernorMode,
    pub policy: GovernorPolicy,
    pub trace_writer: Option<GovernorTraceWriter>,
    pub offensive_enabled: bool,
    pub last_counters: GovernorCounters,
    pub consecutive_quality_risk: u32,
    consecutive_semantic_repetition: u32,
    consecutive_collapse_risk: u32,
    consecutive_low_entropy_collapse: u32,
    consecutive_memory_pressure_warning: u32,
    consecutive_swapout_rate_high: u32,
    consecutive_pageout_rate_high: u32,
    cache_reduction_cooldown_remaining: u32,
    cache_capacity_reduction_count_generation: u32,
    offensive_cooldown_remaining: u32,
    offensive_soft_block_remaining: u32,
    stable_for: u32,
    stable_for_max: u32,
    quality_risky_recently_countdown: u32,
    hard_risk_count: u32,
    soft_risk_count: u32,
    quality_hard_count: u32,
    memory_hard_count: u32,
    io_hard_count: u32,
    offensive_blocked_by_memory_count: u32,
    offensive_blocked_by_quality_count: u32,
}

impl Default for RuntimeGovernor {
    fn default() -> Self {
        Self {
            mode: GovernorMode::Disabled,
            policy: GovernorPolicy::default(),
            trace_writer: std::env::var("OBJETA_GOVERNOR_TRACE_PATH")
                .ok()
                .map(GovernorTraceWriter::new),
            offensive_enabled: std::env::var("OBJETA_GOVERNOR_OFFENSIVE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            last_counters: GovernorCounters::default(),
            consecutive_quality_risk: 0,
            consecutive_semantic_repetition: 0,
            consecutive_collapse_risk: 0,
            consecutive_low_entropy_collapse: 0,
            consecutive_memory_pressure_warning: 0,
            consecutive_swapout_rate_high: 0,
            consecutive_pageout_rate_high: 0,
            cache_reduction_cooldown_remaining: 0,
            cache_capacity_reduction_count_generation: 0,
            offensive_cooldown_remaining: 0,
            offensive_soft_block_remaining: 0,
            stable_for: 0,
            stable_for_max: 0,
            quality_risky_recently_countdown: 0,
            hard_risk_count: 0,
            soft_risk_count: 0,
            quality_hard_count: 0,
            memory_hard_count: 0,
            io_hard_count: 0,
            offensive_blocked_by_memory_count: 0,
            offensive_blocked_by_quality_count: 0,
        }
    }
}

impl RuntimeGovernor {
    pub fn record_applied_action(
        &mut self,
        decision: &mut GovernorDecision,
        kind: impl Into<String>,
        old_value: Option<String>,
        new_value: Option<String>,
        actual_state_changed: bool,
    ) {
        let no_op = !actual_state_changed;
        if actual_state_changed {
            decision.actual_state_change_count += 1;
        } else {
            decision.no_op_action_count += 1;
        }
        decision.applied = true;
        decision.applied_actions.push(GovernorAppliedAction {
            kind: kind.into(),
            no_op,
            old_value,
            new_value,
            actual_state_changed,
        });
    }

    pub fn allow_memory_pressure_reduction(
        &self,
        pressure: PressureClass,
    ) -> bool {
        match pressure {
            PressureClass::Critical => true,
            PressureClass::High => self.consecutive_memory_pressure_warning >= 2,
            PressureClass::Low => false,
        }
    }

    pub fn can_reduce_cache_capacity(&self) -> bool {
        self.cache_reduction_cooldown_remaining == 0
            && self.cache_capacity_reduction_count_generation < 2
    }

    pub fn cache_capacity_reduction_count_generation(&self) -> u32 {
        self.cache_capacity_reduction_count_generation
    }

    pub fn note_cache_capacity_reduction(&mut self) {
        self.cache_reduction_cooldown_remaining = 5;
        self.cache_capacity_reduction_count_generation += 1;
    }

    pub fn note_offensive_cooldown(&mut self) {
        self.offensive_cooldown_remaining = 3;
    }

    pub fn can_apply_offensive(&self) -> bool {
        self.offensive_enabled
            && self.offensive_cooldown_remaining == 0
            && self.offensive_soft_block_remaining == 0
    }

    pub fn stable_for(&self) -> u32 {
        self.stable_for
    }

    pub fn consecutive_semantic_repetition(&self) -> u32 {
        self.consecutive_semantic_repetition
    }

    pub fn quality_risky_recently(&self) -> bool {
        self.quality_risky_recently_countdown > 0
    }

    pub fn from_env() -> Self {
        let mut governor = Self::default();
        governor.mode = GovernorMode::from_env();
        governor
    }

    pub fn reset_counters(&mut self) {
        self.last_counters = GovernorCounters::default();
        self.consecutive_quality_risk = 0;
        self.consecutive_semantic_repetition = 0;
        self.consecutive_collapse_risk = 0;
        self.consecutive_low_entropy_collapse = 0;
        self.consecutive_memory_pressure_warning = 0;
        self.consecutive_swapout_rate_high = 0;
        self.consecutive_pageout_rate_high = 0;
        self.cache_reduction_cooldown_remaining = 0;
        self.cache_capacity_reduction_count_generation = 0;
        self.offensive_cooldown_remaining = 0;
        self.offensive_soft_block_remaining = 0;
        self.stable_for = 0;
        self.stable_for_max = 0;
        self.quality_risky_recently_countdown = 0;
        self.hard_risk_count = 0;
        self.soft_risk_count = 0;
        self.quality_hard_count = 0;
        self.memory_hard_count = 0;
        self.io_hard_count = 0;
        self.offensive_blocked_by_memory_count = 0;
        self.offensive_blocked_by_quality_count = 0;
    }

    pub fn write_trace(
        &self,
        observation: &RuntimeObservation,
        decision: &GovernorDecision,
    ) -> std::io::Result<()> {
        if let Some(writer) = &self.trace_writer {
            writer.write_record(&GovernorTraceRecord {
                observation: observation.clone(),
                decision: decision.clone(),
            })
        } else {
            Ok(())
        }
    }

    pub fn observe(&mut self, observation: RuntimeObservation) -> GovernorDecision {
        let resident_ratio = if observation.resident_capacity_bytes > 0 {
            observation.resident_bytes as f32 / observation.resident_capacity_bytes as f32
        } else {
            0.0
        };

        let memory_pressure = if observation.resident_capacity_bytes == 0 {
            PressureClass::Low
        } else if resident_ratio >= self.policy.critical_memory_pressure_ratio {
            PressureClass::Critical
        } else if resident_ratio >= self.policy.high_memory_pressure_ratio {
            PressureClass::High
        } else {
            PressureClass::Low
        };

        if self.cache_reduction_cooldown_remaining > 0 {
            self.cache_reduction_cooldown_remaining -= 1;
        }
        if self.offensive_cooldown_remaining > 0 {
            self.offensive_cooldown_remaining -= 1;
        }
        if self.offensive_soft_block_remaining > 0 {
            self.offensive_soft_block_remaining -= 1;
        }
        if self.quality_risky_recently_countdown > 0 {
            self.quality_risky_recently_countdown -= 1;
        }

        match memory_pressure {
            PressureClass::Low => self.consecutive_memory_pressure_warning = 0,
            PressureClass::High => {
                self.consecutive_memory_pressure_warning =
                    self.consecutive_memory_pressure_warning.saturating_add(1);
            }
            PressureClass::Critical => {}
        }

        if observation.swapout_mb_per_sec > 32.0 {
            self.consecutive_swapout_rate_high =
                self.consecutive_swapout_rate_high.saturating_add(1);
        } else {
            self.consecutive_swapout_rate_high = 0;
        }
        if observation.pageout_mb_per_sec > 8.0 {
            self.consecutive_pageout_rate_high =
                self.consecutive_pageout_rate_high.saturating_add(1);
        } else {
            self.consecutive_pageout_rate_high = 0;
        }

        let io_thrash = if observation.resident_miss_delta >= self.policy.io_thrash_miss_threshold
            && observation.actual_bytes_loaded_delta >= self.policy.io_thrash_loaded_bytes_threshold
        {
            IoThrashClass::Thrashing
        } else {
            IoThrashClass::Stable
        };

        let low_entropy_collapse =
            observation.prev_decode_entropy <= self.policy.entropy_hard_floor;

        self.consecutive_collapse_risk = if matches!(
            observation.phase,
            GovernorPhase::DecodeWarmup | GovernorPhase::DecodeSteady
        ) && observation.collapse_risk
        {
            self.consecutive_collapse_risk.saturating_add(1)
        } else {
            0
        };

        self.consecutive_low_entropy_collapse = if matches!(
            observation.phase,
            GovernorPhase::DecodeWarmup | GovernorPhase::DecodeSteady
        ) && low_entropy_collapse
        {
            self.consecutive_low_entropy_collapse.saturating_add(1)
        } else {
            0
        };

        self.consecutive_semantic_repetition = if matches!(
            observation.phase,
            GovernorPhase::DecodeWarmup | GovernorPhase::DecodeSteady
        ) && matches!(
            observation.repetition_kind,
            Some(RepetitionKind::SemanticRepetition)
        ) {
            self.consecutive_semantic_repetition.saturating_add(1)
        } else {
            0
        };

        let mut risk_reasons = Vec::new();
        if observation.prev_decode_entropy >= self.policy.high_entropy_threshold {
            risk_reasons.push(RiskReason::HighEntropyUncertainty);
        }
        if low_entropy_collapse {
            risk_reasons.push(RiskReason::LowEntropyCollapse);
        }
        if observation.repetition_risk
            && matches!(
                observation.repetition_kind,
                Some(RepetitionKind::SemanticRepetition)
            )
        {
            risk_reasons.push(RiskReason::Repetition);
        }
        if observation.avg_routing_mass_dropped_pre_renorm > 0.15 {
            risk_reasons.push(RiskReason::ExcessDroppedRoutingMass);
        }
        if self.consecutive_low_entropy_collapse >= 2 || self.consecutive_collapse_risk >= 2 {
            risk_reasons.push(RiskReason::ConsecutiveCollapseSignal);
        }

        let steady_decode = matches!(observation.phase, GovernorPhase::DecodeSteady);
        let semantic_repetition_hard =
            steady_decode && self.consecutive_semantic_repetition >= 2;
        let low_entropy_hard = steady_decode && self.consecutive_low_entropy_collapse >= 2;
        let dropped_mass_hard = observation.avg_routing_mass_dropped_pre_renorm > 0.15;
        let critical_memory_hard =
            matches!(observation.phase, GovernorPhase::DecodeWarmup | GovernorPhase::DecodeSteady)
                && matches!(memory_pressure, PressureClass::Critical);
        let swapout_rate_hard = matches!(
            observation.phase,
            GovernorPhase::DecodeWarmup | GovernorPhase::DecodeSteady
        ) && self.consecutive_swapout_rate_high >= 2;
        let pageout_rate_hard = matches!(
            observation.phase,
            GovernorPhase::DecodeWarmup | GovernorPhase::DecodeSteady
        ) && self.consecutive_pageout_rate_high >= 2;

        let quality_hard = semantic_repetition_hard || low_entropy_hard || dropped_mass_hard;
        let memory_hard = critical_memory_hard;
        let io_hard = swapout_rate_hard || pageout_rate_hard;
        let hard_risk = quality_hard || memory_hard || io_hard;
        let hard_risk_kind = if quality_hard {
            Some(HardRiskKind::QualityHard)
        } else if memory_hard {
            Some(HardRiskKind::MemoryHard)
        } else if io_hard {
            Some(HardRiskKind::IoHard)
        } else {
            None
        };

        let elevated_quality_trigger = matches!(
            observation.phase,
            GovernorPhase::DecodeWarmup | GovernorPhase::DecodeSteady
        ) && (observation.prev_decode_entropy >= self.policy.high_entropy_threshold
            || matches!(
                observation.repetition_kind,
                Some(RepetitionKind::SpecialTokenLoop | RepetitionKind::StructuralMarkerLoop)
            )
            || low_entropy_collapse
            || observation.collapse_risk);

        let soft_risk = !hard_risk
            && matches!(
                observation.phase,
                GovernorPhase::DecodeWarmup | GovernorPhase::DecodeSteady
            )
            && (observation.prev_decode_entropy >= self.policy.high_entropy_threshold
                || matches!(
                    observation.repetition_kind,
                    Some(
                        RepetitionKind::SpecialTokenLoop
                            | RepetitionKind::StructuralMarkerLoop
                            | RepetitionKind::SemanticRepetition
                    )
                )
                || low_entropy_collapse
                || observation.collapse_risk);

        let quality_risk = if quality_hard {
            QualityRiskClass::High
        } else if elevated_quality_trigger {
            QualityRiskClass::Elevated
        } else {
            QualityRiskClass::Low
        };

        let mut stable_for_reset_reasons = Vec::new();
        if hard_risk {
            if semantic_repetition_hard {
                stable_for_reset_reasons.push("semantic_repetition".to_string());
            }
            if low_entropy_hard {
                stable_for_reset_reasons.push("low_entropy_collapse".to_string());
            }
            if dropped_mass_hard {
                stable_for_reset_reasons.push("excess_dropped_routing_mass".to_string());
            }
            if critical_memory_hard {
                stable_for_reset_reasons.push("memory_pressure_critical".to_string());
            }
            if swapout_rate_hard {
                stable_for_reset_reasons.push("swapout_rate_hard".to_string());
            }
            if pageout_rate_hard {
                stable_for_reset_reasons.push("pageout_rate_hard".to_string());
            }
            self.stable_for = 0;
            self.quality_risky_recently_countdown = 3;
            self.hard_risk_count = self.hard_risk_count.saturating_add(1);
            if quality_hard {
                self.quality_hard_count = self.quality_hard_count.saturating_add(1);
                if self.offensive_enabled {
                    self.offensive_blocked_by_quality_count =
                        self.offensive_blocked_by_quality_count.saturating_add(1);
                }
            }
            if memory_hard {
                self.memory_hard_count = self.memory_hard_count.saturating_add(1);
                if self.offensive_enabled {
                    self.offensive_blocked_by_memory_count =
                        self.offensive_blocked_by_memory_count.saturating_add(1);
                }
            }
            if io_hard {
                self.io_hard_count = self.io_hard_count.saturating_add(1);
                if self.offensive_enabled {
                    self.offensive_blocked_by_memory_count =
                        self.offensive_blocked_by_memory_count.saturating_add(1);
                }
            }
        } else if steady_decode {
            self.stable_for = self.stable_for.saturating_add(1);
            self.stable_for_max = self.stable_for_max.max(self.stable_for);
            if soft_risk {
                self.offensive_soft_block_remaining = self.offensive_soft_block_remaining.max(2);
                self.soft_risk_count = self.soft_risk_count.saturating_add(1);
            }
        }

        self.consecutive_quality_risk = if matches!(quality_risk, QualityRiskClass::Low) {
            0
        } else {
            self.consecutive_quality_risk.saturating_add(1)
        };

        let mut decision = GovernorDecision {
            mode: self.mode,
            phase: observation.phase,
            memory_pressure,
            io_thrash,
            quality_risk,
            risk_reasons,
            suggested_top_p: None,
            suggested_min_experts: None,
            suggested_resident_cache_capacity_bytes: None,
            suggested_group_preresolve_top_n: None,
            suggested_group_preresolve_max_bytes: None,
            consecutive_quality_risk: self.consecutive_quality_risk,
            applied: false,
            applied_actions: Vec::new(),
            actual_state_change_count: 0,
            no_op_action_count: 0,
            cache_capacity_reduction_count: self.cache_capacity_reduction_count_generation,
            offensive_action_count: 0,
            rollback_action_count: 0,
            hard_risk_count: self.hard_risk_count,
            soft_risk_count: self.soft_risk_count,
            stable_for: self.stable_for,
            stable_for_max: self.stable_for_max,
            quality_risky_recently: self.quality_risky_recently_countdown > 0,
            stable_for_reset_reasons,
            blocked_offensive_reason: None,
            hard_risk_kind,
            quality_hard_count: self.quality_hard_count,
            memory_hard_count: self.memory_hard_count,
            io_hard_count: self.io_hard_count,
            offensive_blocked_by_memory_count: self.offensive_blocked_by_memory_count,
            offensive_blocked_by_quality_count: self.offensive_blocked_by_quality_count,
            confirmed_pageouts_delta: observation.confirmed_pageouts_delta,
            confirmed_swapouts_delta: observation.confirmed_swapouts_delta,
            pageout_mb_per_sec: observation.pageout_mb_per_sec,
            swapout_mb_per_sec: observation.swapout_mb_per_sec,
            pageout_mb_per_token: observation.pageout_mb_per_token,
            swapout_mb_per_token: observation.swapout_mb_per_token,
            current_effective_top_p: None,
            current_effective_min_experts: None,
            action_reason: None,
            rationale: "no-op".to_string(),
        };

        match quality_risk {
            QualityRiskClass::High => {
                decision.suggested_top_p = Some(1.0);
                decision.suggested_min_experts = Some(8);
                decision.rationale = "quality-risk-high".to_string();
            }
            QualityRiskClass::Elevated => {
                if matches!(
                    observation.repetition_kind,
                    Some(RepetitionKind::SpecialTokenLoop | RepetitionKind::StructuralMarkerLoop)
                ) {
                    decision.rationale = "special-token-stop-handling".to_string();
                } else if decision
                    .risk_reasons
                    .iter()
                    .any(|reason| matches!(reason, RiskReason::HighEntropyUncertainty))
                {
                    decision.rationale = "high-entropy-uncertainty-hold".to_string();
                } else {
                    decision.rationale = "quality-risk-elevated-hold".to_string();
                }
            }
            QualityRiskClass::Low => {
                if matches!(observation.phase, GovernorPhase::Prefill) {
                    decision.rationale = "prefill-log-only".to_string();
                }
                if matches!(memory_pressure, PressureClass::Critical)
                    || matches!(io_thrash, IoThrashClass::Thrashing)
                {
                    if matches!(self.mode, GovernorMode::ObserveOnly) {
                        decision.suggested_top_p = Some(0.90);
                        decision.suggested_min_experts = Some(4);
                        decision.rationale = "pressure-or-io-thrash".to_string();
                    } else {
                        decision.rationale = "pressure-or-io-thrash-hold".to_string();
                    }
                }
            }
        }

        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_governor_observe_high_quality_risk() {
        let mut governor = RuntimeGovernor::default();
        governor.mode = GovernorMode::ObserveOnly;
        let elevated = governor.observe(RuntimeObservation {
            step: 1,
            token_id: 42,
            token_position: 1,
            phase: GovernorPhase::DecodeWarmup,
            prev_decode_entropy: 0.01,
            rss_mb: 0.0,
            os_memory_pressure_state: "unknown".to_string(),
            repetition_risk: false,
            repetition_kind: None,
            collapse_risk: true,
            resident_capacity_bytes: 1024,
            resident_bytes: 900,
            resident_hit_delta: 10,
            resident_miss_delta: 1,
            pageouts_delta: 0,
            swapout_delta: 0,
            confirmed_pageouts_delta: 0,
            confirmed_swapouts_delta: 0,
            pageout_mb_per_sec: 0.0,
            swapout_mb_per_sec: 0.0,
            pageout_mb_per_token: 0.0,
            swapout_mb_per_token: 0.0,
            actual_bytes_loaded_delta: 0,
            resident_bytes_reused_delta: 0,
            avg_selected_experts: 8.0,
            avg_routing_mass_kept: 1.0,
            avg_routing_mass_dropped: 0.0,
            avg_routing_mass_kept_pre_renorm: 1.0,
            avg_routing_mass_dropped_pre_renorm: 0.0,
            avg_routing_mass_sum_after_renorm: 1.0,
            forward_wall_ms_delta: 0.0,
            cache_lookup_wall_ms_delta: 0.0,
            eviction_count_delta: 0,
        });
        assert_eq!(elevated.quality_risk, QualityRiskClass::Elevated);
        let decision = governor.observe(RuntimeObservation {
            step: 2,
            token_id: 43,
            token_position: 2,
            phase: GovernorPhase::DecodeSteady,
            prev_decode_entropy: 0.01,
            rss_mb: 0.0,
            os_memory_pressure_state: "unknown".to_string(),
            repetition_risk: false,
            repetition_kind: None,
            collapse_risk: true,
            resident_capacity_bytes: 1024,
            resident_bytes: 900,
            resident_hit_delta: 10,
            resident_miss_delta: 1,
            pageouts_delta: 0,
            swapout_delta: 0,
            confirmed_pageouts_delta: 0,
            confirmed_swapouts_delta: 0,
            pageout_mb_per_sec: 0.0,
            swapout_mb_per_sec: 0.0,
            pageout_mb_per_token: 0.0,
            swapout_mb_per_token: 0.0,
            actual_bytes_loaded_delta: 0,
            resident_bytes_reused_delta: 0,
            avg_selected_experts: 8.0,
            avg_routing_mass_kept: 1.0,
            avg_routing_mass_dropped: 0.0,
            avg_routing_mass_kept_pre_renorm: 1.0,
            avg_routing_mass_dropped_pre_renorm: 0.0,
            avg_routing_mass_sum_after_renorm: 1.0,
            forward_wall_ms_delta: 0.0,
            cache_lookup_wall_ms_delta: 0.0,
            eviction_count_delta: 0,
        });
        assert_eq!(decision.quality_risk, QualityRiskClass::High);
        assert_eq!(decision.suggested_top_p, Some(1.0));
    }

    #[test]
    fn test_special_token_loop_is_elevated_not_high() {
        let mut governor = RuntimeGovernor::default();
        governor.mode = GovernorMode::ObserveOnly;
        let decision = governor.observe(RuntimeObservation {
            step: 5,
            token_id: 248045,
            token_position: 5,
            phase: GovernorPhase::DecodeSteady,
            prev_decode_entropy: 0.2,
            rss_mb: 0.0,
            os_memory_pressure_state: "unknown".to_string(),
            repetition_risk: true,
            repetition_kind: Some(RepetitionKind::StructuralMarkerLoop),
            collapse_risk: false,
            resident_capacity_bytes: 1024,
            resident_bytes: 256,
            resident_hit_delta: 1,
            resident_miss_delta: 0,
            pageouts_delta: 0,
            swapout_delta: 0,
            confirmed_pageouts_delta: 0,
            confirmed_swapouts_delta: 0,
            pageout_mb_per_sec: 0.0,
            swapout_mb_per_sec: 0.0,
            pageout_mb_per_token: 0.0,
            swapout_mb_per_token: 0.0,
            actual_bytes_loaded_delta: 0,
            resident_bytes_reused_delta: 0,
            avg_selected_experts: 8.0,
            avg_routing_mass_kept: 1.0,
            avg_routing_mass_dropped: 0.0,
            avg_routing_mass_kept_pre_renorm: 1.0,
            avg_routing_mass_dropped_pre_renorm: 0.0,
            avg_routing_mass_sum_after_renorm: 1.0,
            forward_wall_ms_delta: 0.0,
            cache_lookup_wall_ms_delta: 0.0,
            eviction_count_delta: 0,
        });
        assert_eq!(decision.quality_risk, QualityRiskClass::Elevated);
        assert_eq!(decision.suggested_top_p, None);
        assert_eq!(decision.rationale, "special-token-stop-handling");
    }
}
