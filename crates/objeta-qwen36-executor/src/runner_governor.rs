use crate::qwen36_forward::Qwen36Runner;

impl Qwen36Runner {
    pub(crate) fn governor_min_resident_cache_bytes(&self) -> u64 {
        const MB: u64 = 1024 * 1024;
        if let Ok(raw) = std::env::var("OBJETA_GOVERNOR_MIN_RESIDENT_CACHE_BYTES") {
            if let Ok(parsed) = raw.trim().parse::<u64>() {
                return parsed.min(self.expert_residency_manager.capacity_bytes);
            }
        }
        self.runtime_profile
            .knobs
            .resident_cache_capacity_bytes
            .unwrap_or(self.expert_residency_manager.capacity_bytes)
            .saturating_sub(512 * MB)
            .min(self.expert_residency_manager.capacity_bytes)
    }

    pub(crate) fn disable_group_preresolve_for_governor(
        &mut self,
        decision: &mut crate::runtime_governor::GovernorDecision,
    ) {
        let old_top_n = std::env::var("OBJETA_GROUP_PRERESOLVE_TOP_N")
            .unwrap_or_else(|_| "0".to_string());
        let old_max_bytes = std::env::var("OBJETA_GROUP_PRERESOLVE_MAX_BYTES")
            .unwrap_or_else(|_| "0".to_string());
        let old_group_size = self.expert_residency_manager.residency_group_size.to_string();
        std::env::set_var("OBJETA_GROUP_PRERESOLVE_TOP_N", "0");
        std::env::set_var("OBJETA_GROUP_PRERESOLVE_MAX_BYTES", "0");
        std::env::set_var("OBJETA_RESIDENCY_GROUP_SIZE", "1");
        self.expert_residency_manager.residency_group_size = 1;
        decision.suggested_group_preresolve_top_n = Some(0);
        decision.suggested_group_preresolve_max_bytes = Some(0);
        let changed = old_top_n != "0" || old_max_bytes != "0" || old_group_size != "1";
        self.runtime_governor.record_applied_action(
            decision,
            "disable_group_preresolve",
            Some(format!(
                "top_n={},max_bytes={},group_size={}",
                old_top_n, old_max_bytes, old_group_size
            )),
            Some("top_n=0,max_bytes=0,group_size=1".to_string()),
            changed,
        );
    }

    pub(crate) fn apply_governor_decision_at_token_boundary(
        &mut self,
        observation: &crate::runtime_governor::RuntimeObservation,
        decision: &mut crate::runtime_governor::GovernorDecision,
    ) {
        use crate::runtime_governor::{
            GovernorMode, HardRiskKind, PressureClass, QualityRiskClass,
        };
        decision.current_effective_top_p = Some(self.effective_moe_top_p());
        decision.current_effective_min_experts = Some(self.effective_moe_min_experts());

        if !matches!(self.runtime_governor.mode, GovernorMode::ApplyAtTokenBoundary) {
            return;
        }

        let offensive_enabled = self.runtime_governor.offensive_enabled;
        let offensive_quality_rollback_trigger = offensive_enabled
            && matches!(observation.phase, crate::runtime_governor::GovernorPhase::DecodeSteady)
            && matches!(decision.hard_risk_kind, Some(HardRiskKind::QualityHard));
        let offensive_memory_block_trigger = offensive_enabled
            && matches!(observation.phase, crate::runtime_governor::GovernorPhase::DecodeSteady)
            && matches!(
                decision.hard_risk_kind,
                Some(HardRiskKind::MemoryHard | HardRiskKind::IoHard)
            );

        if matches!(decision.quality_risk, QualityRiskClass::High) {
            if !matches!(self.expert_policy, crate::strategy::ExpertPolicyConfig::Exact) {
                let current_top_p = self.effective_moe_top_p();
                let current_min_experts = self.effective_moe_min_experts();
                let new_top_p = (current_top_p + 0.05).min(1.0);
                let bump = if self.runtime_governor.consecutive_quality_risk >= 2 {
                    2
                } else {
                    1
                };
                let new_min = (current_min_experts + bump).min(8);
                let new_max = self.effective_moe_max_experts().max(new_min).min(8);
                self.set_expert_policy(crate::strategy::ExpertPolicyConfig::TopP {
                    p: new_top_p,
                    min_experts: new_min,
                    max_experts: new_max,
                });
                decision.action_reason = Some("quality_risk_high".to_string());
                decision.suggested_top_p = Some(new_top_p);
                decision.suggested_min_experts = Some(new_min);
                self.runtime_governor.record_applied_action(
                    decision,
                    "raise_top_p",
                    Some(format!("{current_top_p:.2}")),
                    Some(format!("{new_top_p:.2}")),
                    (new_top_p - current_top_p).abs() > f32::EPSILON,
                );
                self.runtime_governor.record_applied_action(
                    decision,
                    "raise_min_experts",
                    Some(current_min_experts.to_string()),
                    Some(new_min.to_string()),
                    new_min != current_min_experts,
                );
                if (new_top_p - current_top_p).abs() > f32::EPSILON
                    || new_min != current_min_experts
                {
                    decision.rollback_action_count += 1;
                }
            }
            self.disable_group_preresolve_for_governor(decision);
        }

        if offensive_quality_rollback_trigger
            && !matches!(decision.quality_risk, QualityRiskClass::High)
            && !matches!(self.expert_policy, crate::strategy::ExpertPolicyConfig::Exact)
        {
            let current_top_p = self.effective_moe_top_p();
            let current_min_experts = self.effective_moe_min_experts();
            let new_top_p = (current_top_p + 0.05).min(1.0);
            let new_min = (current_min_experts + 1).min(8);
            let new_max = self.effective_moe_max_experts().max(new_min).min(8);
            self.set_expert_policy(crate::strategy::ExpertPolicyConfig::TopP {
                p: new_top_p,
                min_experts: new_min,
                max_experts: new_max,
            });
            decision.action_reason = Some("offensive_rollback".to_string());
            decision.suggested_top_p = Some(new_top_p);
            decision.suggested_min_experts = Some(new_min);
            self.runtime_governor.record_applied_action(
                decision,
                "rollback_raise_top_p",
                Some(format!("{current_top_p:.2}")),
                Some(format!("{new_top_p:.2}")),
                (new_top_p - current_top_p).abs() > f32::EPSILON,
            );
            self.runtime_governor.record_applied_action(
                decision,
                "rollback_raise_min_experts",
                Some(current_min_experts.to_string()),
                Some(new_min.to_string()),
                new_min != current_min_experts,
            );
            if (new_top_p - current_top_p).abs() > f32::EPSILON || new_min != current_min_experts {
                decision.rollback_action_count += 1;
                self.runtime_governor.note_offensive_cooldown();
            }
        }

        let should_reduce_memory = self
            .runtime_governor
            .allow_memory_pressure_reduction(decision.memory_pressure)
            || matches!(
                decision.hard_risk_kind,
                Some(HardRiskKind::MemoryHard | HardRiskKind::IoHard)
            );
        if should_reduce_memory {
            self.disable_group_preresolve_for_governor(decision);
            const DELTA: u64 = 256 * 1024 * 1024;
            let lower_bound = self.governor_min_resident_cache_bytes();
            let current = self.expert_residency_manager.capacity_bytes;
            if self.runtime_governor.can_reduce_cache_capacity() {
                let reduced = current.saturating_sub(DELTA).max(lower_bound);
                if reduced < current {
                    self.expert_residency_manager.capacity_bytes = reduced;
                    let expert_total_bytes =
                        (crate::moe_dispatch::GU_EXPERT_BYTES + crate::moe_dispatch::D_EXPERT_BYTES)
                            as u64;
                    self.expert_cache_size = if expert_total_bytes > 0 {
                        (reduced / expert_total_bytes) as usize
                    } else {
                        0
                    };
                    decision.suggested_resident_cache_capacity_bytes = Some(reduced);
                    self.runtime_governor.record_applied_action(
                        decision,
                        "reduce_resident_cache_capacity",
                        Some(current.to_string()),
                        Some(reduced.to_string()),
                        true,
                    );
                    self.runtime_governor.note_cache_capacity_reduction();
                    decision.cache_capacity_reduction_count =
                        self.runtime_governor.cache_capacity_reduction_count_generation();
                } else {
                    self.runtime_governor.record_applied_action(
                        decision,
                        "reduce_resident_cache_capacity",
                        Some(current.to_string()),
                        Some(reduced.to_string()),
                        false,
                    );
                }
            } else {
                self.runtime_governor.record_applied_action(
                    decision,
                    "reduce_resident_cache_capacity",
                    Some(current.to_string()),
                    Some(current.to_string()),
                    false,
                );
            }
        }

        if offensive_enabled
            && !offensive_quality_rollback_trigger
            && !offensive_memory_block_trigger
            && self.runtime_governor.can_apply_offensive()
            && matches!(observation.phase, crate::runtime_governor::GovernorPhase::DecodeSteady)
            && self.runtime_governor.stable_for() >= 5
            && !matches!(self.expert_policy, crate::strategy::ExpertPolicyConfig::Exact)
        {
            let current_top_p = self.effective_moe_top_p();
            let current_min_experts = self.effective_moe_min_experts();
            let current_max_experts = self.effective_moe_max_experts();
            if self.runtime_governor.stable_for() >= 8
                && current_top_p <= 0.85 + f32::EPSILON
                && current_min_experts > 3
            {
                let new_min = (current_min_experts - 1).max(3);
                self.set_expert_policy(crate::strategy::ExpertPolicyConfig::TopP {
                    p: current_top_p,
                    min_experts: new_min,
                    max_experts: current_max_experts.max(new_min),
                });
                decision.action_reason = Some("offensive_reduce_min_experts".to_string());
                decision.suggested_min_experts = Some(new_min);
                self.runtime_governor.record_applied_action(
                    decision,
                    "offensive_reduce_min_experts",
                    Some(current_min_experts.to_string()),
                    Some(new_min.to_string()),
                    new_min != current_min_experts,
                );
                if new_min != current_min_experts {
                    decision.offensive_action_count += 1;
                    decision.action_reason = Some("stable_reduce_min_experts".to_string());
                    self.runtime_governor.note_offensive_cooldown();
                }
            } else if current_top_p > 0.85 + f32::EPSILON {
                let new_top_p = (current_top_p - 0.05).max(0.85);
                self.set_expert_policy(crate::strategy::ExpertPolicyConfig::TopP {
                    p: new_top_p,
                    min_experts: current_min_experts,
                    max_experts: current_max_experts.max(current_min_experts),
                });
                decision.action_reason = Some("offensive_reduce_top_p".to_string());
                decision.suggested_top_p = Some(new_top_p);
                self.runtime_governor.record_applied_action(
                    decision,
                    "offensive_reduce_top_p",
                    Some(format!("{current_top_p:.2}")),
                    Some(format!("{new_top_p:.2}")),
                    (new_top_p - current_top_p).abs() > f32::EPSILON,
                );
                if (new_top_p - current_top_p).abs() > f32::EPSILON {
                    decision.offensive_action_count += 1;
                    decision.action_reason = Some("stable_reduce_top_p".to_string());
                    self.runtime_governor.note_offensive_cooldown();
                }
            }
        } else if offensive_enabled
            && matches!(observation.phase, crate::runtime_governor::GovernorPhase::DecodeSteady)
            && self.runtime_governor.offensive_enabled
            && self.runtime_governor.stable_for() >= 1
            && !self.runtime_governor.can_apply_offensive()
        {
            decision.blocked_offensive_reason = Some("cooldown_or_soft_block".to_string());
        } else if offensive_enabled
            && matches!(observation.phase, crate::runtime_governor::GovernorPhase::DecodeSteady)
            && offensive_memory_block_trigger
        {
            decision.blocked_offensive_reason = Some("memory_or_io_hard".to_string());
        } else if offensive_enabled
            && matches!(observation.phase, crate::runtime_governor::GovernorPhase::DecodeSteady)
            && self.runtime_governor.stable_for() < 5
        {
            decision.blocked_offensive_reason = Some("stable_for_below_threshold".to_string());
        }

        decision.current_effective_top_p = Some(self.effective_moe_top_p());
        decision.current_effective_min_experts = Some(self.effective_moe_min_experts());
    }

    pub fn collect_runtime_observation(
        &mut self,
        step: usize,
        token_id: usize,
        avg_selected_experts: f32,
        avg_routing_mass_kept: f32,
        avg_routing_mass_dropped: f32,
        avg_routing_mass_kept_pre_renorm: f32,
        avg_routing_mass_dropped_pre_renorm: f32,
        avg_routing_mass_sum_after_renorm: f32,
    ) {
        if self.decode_started {
            let total_cache_lookup_wall_ms: f64 = self
                .moe_stats
                .iter()
                .map(|s| s.total_cache_lookup_wall_sec * 1000.0)
                .sum();
            let counters = crate::runtime_governor::GovernorCounters {
                resident_hit_count: self.expert_residency_manager.resident_hit_count,
                resident_miss_count: self.expert_residency_manager.resident_miss_count,
                actual_bytes_loaded: self.expert_residency_manager.actual_expert_bytes_loaded,
                resident_bytes_reused: self.expert_residency_manager.resident_cache_bytes_reused,
                eviction_count: self.expert_residency_manager.eviction_count,
                forward_wall_ms: self.forward_wall_sec * 1000.0,
                cache_lookup_wall_ms: total_cache_lookup_wall_ms,
            };
            let last = self.runtime_governor.last_counters;
            let os_sample = self.os_telemetry.sample();
            let observation = crate::runtime_governor::RuntimeObservation {
                step,
                token_id,
                token_position: step,
                phase: self.current_governor_phase,
                prev_decode_entropy: self.last_decode_entropy,
                rss_mb: os_sample.rss_mb,
                os_memory_pressure_state: os_sample.memory_pressure_state,
                repetition_risk: self.last_repetition_risk,
                repetition_kind: self.last_repetition_kind,
                collapse_risk: self.last_collapse_risk,
                resident_capacity_bytes: self.expert_residency_manager.capacity_bytes,
                resident_bytes: self.expert_residency_manager.resident_bytes(),
                resident_hit_delta: counters
                    .resident_hit_count
                    .saturating_sub(last.resident_hit_count),
                resident_miss_delta: counters
                    .resident_miss_count
                    .saturating_sub(last.resident_miss_count),
                pageouts_delta: os_sample.pageouts_delta,
                swapout_delta: os_sample.swapouts_delta,
                confirmed_pageouts_delta: os_sample.confirmed_pageouts_delta,
                confirmed_swapouts_delta: os_sample.confirmed_swapouts_delta,
                pageout_mb_per_sec: if counters.forward_wall_ms > last.forward_wall_ms {
                    let dt_s = ((counters.forward_wall_ms - last.forward_wall_ms) / 1000.0)
                        .max(1e-6);
                    (os_sample.confirmed_pageouts_delta as f64 / (1024.0 * 1024.0) / dt_s) as f32
                } else {
                    0.0
                },
                swapout_mb_per_sec: if counters.forward_wall_ms > last.forward_wall_ms {
                    let dt_s = ((counters.forward_wall_ms - last.forward_wall_ms) / 1000.0)
                        .max(1e-6);
                    (os_sample.confirmed_swapouts_delta as f64 / (1024.0 * 1024.0) / dt_s) as f32
                } else {
                    0.0
                },
                pageout_mb_per_token: (os_sample.confirmed_pageouts_delta as f64
                    / (1024.0 * 1024.0)) as f32,
                swapout_mb_per_token: (os_sample.confirmed_swapouts_delta as f64
                    / (1024.0 * 1024.0)) as f32,
                actual_bytes_loaded_delta: counters
                    .actual_bytes_loaded
                    .saturating_sub(last.actual_bytes_loaded),
                resident_bytes_reused_delta: counters
                    .resident_bytes_reused
                    .saturating_sub(last.resident_bytes_reused),
                avg_selected_experts,
                avg_routing_mass_kept,
                avg_routing_mass_dropped,
                avg_routing_mass_kept_pre_renorm,
                avg_routing_mass_dropped_pre_renorm,
                avg_routing_mass_sum_after_renorm,
                forward_wall_ms_delta: counters.forward_wall_ms - last.forward_wall_ms,
                cache_lookup_wall_ms_delta: counters.cache_lookup_wall_ms - last.cache_lookup_wall_ms,
                eviction_count_delta: counters.eviction_count.saturating_sub(last.eviction_count),
            };
            if !matches!(
                self.runtime_governor.mode,
                crate::runtime_governor::GovernorMode::Disabled
            ) {
                let mut decision = self.runtime_governor.observe(observation.clone());
                self.apply_governor_decision_at_token_boundary(&observation, &mut decision);
                let _ = self.runtime_governor.write_trace(&observation, &decision);
            }
            self.runtime_governor.last_counters = counters;
        }
    }
}
