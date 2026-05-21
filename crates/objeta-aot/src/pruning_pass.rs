use crate::phase_pass;
use crate::types::*;

/// Classify each expert into keep/protect/cold_tier/compress/prune_candidate.
/// Estimate routing mass loss. Enforce quality-budget thresholds and safety gates.
///
/// # Epistemic note
/// The mass-loss figures are routing-trace estimates only. They do not measure
/// end-to-end generation quality. The returned `PruningPlanSummary` always sets
/// `estimated_only = true` and `requires_verification = true`.
pub fn run(
    importance: &ExpertImportance,
    phase_policy: &PhasePolicy,
    quality_budget: &QualityBudget,
    importance_coverage: f64,
) -> PruningPlan {
    let mass_loss_threshold = quality_budget.mass_loss_threshold();
    let pruning_disabled = importance_coverage < 0.80;

    let mut entries = Vec::new();
    let mut cumulative_mass_loss = 0.0;
    let mut protect_count = 0usize;
    let mut keep_count = 0usize;
    let mut cold_tier_count = 0usize;
    let mut compress_count = 0usize;
    let mut prune_candidate_count = 0usize;

    // Sort experts by importance ascending so we consider pruning the least important first
    let mut sorted: Vec<&ExpertImportanceEntry> = importance.experts.iter().collect();
    sorted.sort_by(|a, b| {
        a.importance
            .partial_cmp(&b.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for e in &sorted {
        let phase = phase_pass::phase_for_layer(phase_policy, e.layer);
        let (action, reason, target_format, estimated_mass_loss) =
            classify_expert(e, phase, quality_budget, cumulative_mass_loss, pruning_disabled);

        match action.as_str() {
            "protect" => protect_count += 1,
            "keep" => keep_count += 1,
            "cold_tier" => cold_tier_count += 1,
            "compress" => {
                compress_count += 1;
                cumulative_mass_loss += estimated_mass_loss;
            }
            "prune_candidate" => {
                prune_candidate_count += 1;
                cumulative_mass_loss += estimated_mass_loss;
            }
            _ => {}
        }

        entries.push(PruningPlanEntry {
            layer: e.layer,
            expert: e.expert,
            action,
            target_format,
            estimated_mass_loss,
            reason,
        });
    }

    // Re-sort by layer/expert for deterministic output
    entries.sort_by(|a, b| a.layer.cmp(&b.layer).then_with(|| a.expert.cmp(&b.expert)));

    PruningPlan {
        schema_version: 1,
        summary: PruningPlanSummary {
            protect_count,
            keep_count,
            cold_tier_count,
            compress_count,
            prune_candidate_count,
            estimated_routing_mass_loss: cumulative_mass_loss,
            quality_budget: quality_budget.to_string(),
            mass_loss_threshold,
            safe: cumulative_mass_loss <= mass_loss_threshold,
            estimated_only: true,
            requires_verification: true,
        },
        experts: entries,
    }
}

fn classify_expert(
    e: &ExpertImportanceEntry,
    phase: &str,
    quality_budget: &QualityBudget,
    cumulative_mass_loss: f64,
    pruning_disabled: bool,
) -> (String, String, Option<String>, f64) {
    let mass_loss_threshold = quality_budget.mass_loss_threshold();

    // Safety gate: protected phases never get pruned
    if phase == "integrity_frontier" {
        return (
            "protect".to_string(),
            "integrity_frontier_protected".to_string(),
            None,
            0.0,
        );
    }

    // Hot experts are always kept
    if e.tier == ExpertTier::Hot {
        return (
            "keep".to_string(),
            "hot_expert_kept".to_string(),
            None,
            0.0,
        );
    }

    // Steering phase: warm→keep, cold→cold_tier (no prune)
    if phase == "steering" || phase == "projection" {
        if e.tier == ExpertTier::Warm {
            return (
                "keep".to_string(),
                "steering_warm_kept".to_string(),
                None,
                0.0,
            );
        }
        return (
            "cold_tier".to_string(),
            "steering_cold_tier".to_string(),
            None,
            0.0,
        );
    }

    // Pruning disabled by coverage gate
    if pruning_disabled {
        return (
            "keep".to_string(),
            "pruning_disabled_low_coverage".to_string(),
            None,
            0.0,
        );
    }

    // Transport phase: warm→keep, cold→compress/prune_candidate
    if e.tier == ExpertTier::Warm {
        return (
            "keep".to_string(),
            "transport_warm_kept".to_string(),
            None,
            0.0,
        );
    }

    // Cold expert in transport phase — estimate mass loss
    // Estimated routing mass loss = (1 - importance) * avg_gate_weight * frequency
    // This is a conservative proxy; real mass loss requires trace replay
    let estimated_mass = e.frequency * e.avg_gate_weight * (1.0 - e.importance);
    let would_exceed = cumulative_mass_loss + estimated_mass > mass_loss_threshold;

    if would_exceed {
        // Over budget — just cold_tier, don't compress/prune
        return (
            "cold_tier".to_string(),
            "mass_loss_budget_exceeded".to_string(),
            None,
            0.0,
        );
    }

    // Very low importance → prune_candidate; moderate → compress
    if e.importance < 0.15 {
        (
            "prune_candidate".to_string(),
            "cold_low_importance".to_string(),
            Some("iq2".to_string()),
            estimated_mass,
        )
    } else {
        (
            "compress".to_string(),
            "cold_moderate_importance".to_string(),
            Some("iq3".to_string()),
            estimated_mass,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(
        layer: u32,
        expert: u32,
        importance: f64,
        tier: ExpertTier,
        freq: f64,
        avg_gate: f64,
    ) -> ExpertImportanceEntry {
        ExpertImportanceEntry {
            layer,
            expert,
            selected_count: 10,
            frequency: freq,
            avg_gate_weight: avg_gate,
            max_gate_weight: avg_gate,
            importance,
            tier,
            recommended_format: "q4".to_string(),
            eviction_priority: 1.0 - importance,
        }
    }

    #[test]
    fn protected_layers_never_pruned() {
        let importance = ExpertImportance {
            schema_version: 1,
            experts: vec![make_entry(0, 1, 0.05, ExpertTier::Cold, 0.01, 0.01)],
        };
        let phase = phase_pass::run(40);
        let plan = run(&importance, &phase, &QualityBudget::Aggressive, 1.0);
        assert_eq!(plan.experts[0].action, "protect");
    }

    #[test]
    fn conservative_prunes_less_than_aggressive() {
        let mut experts = Vec::new();
        for i in 0..20u32 {
            experts.push(make_entry(
                10,
                i,
                0.05 + (i as f64 * 0.01),
                ExpertTier::Cold,
                0.1,
                0.05,
            ));
        }
        let importance = ExpertImportance {
            schema_version: 1,
            experts,
        };
        let phase = phase_pass::run(40);

        let conservative = run(&importance, &phase, &QualityBudget::Conservative, 1.0);
        let aggressive = run(&importance, &phase, &QualityBudget::Aggressive, 1.0);

        let conservative_pruned = conservative
            .experts
            .iter()
            .filter(|e| e.action == "prune_candidate" || e.action == "compress")
            .count();
        let aggressive_pruned = aggressive
            .experts
            .iter()
            .filter(|e| e.action == "prune_candidate" || e.action == "compress")
            .count();

        assert!(
            conservative_pruned <= aggressive_pruned,
            "conservative={} should be <= aggressive={}",
            conservative_pruned,
            aggressive_pruned
        );
    }

    #[test]
    fn cold_expert_gets_compress_or_prune() {
        let importance = ExpertImportance {
            schema_version: 1,
            experts: vec![make_entry(10, 99, 0.05, ExpertTier::Cold, 0.1, 0.02)],
        };
        let phase = phase_pass::run(40);
        let plan = run(&importance, &phase, &QualityBudget::Aggressive, 1.0);
        let entry = &plan.experts[0];
        assert!(
            entry.action == "prune_candidate" || entry.action == "compress",
            "got action: {}",
            entry.action
        );
    }

    #[test]
    fn coverage_gate_disables_pruning() {
        let importance = ExpertImportance {
            schema_version: 1,
            experts: vec![make_entry(10, 99, 0.05, ExpertTier::Cold, 0.1, 0.02)],
        };
        let phase = phase_pass::run(40);
        let plan = run(&importance, &phase, &QualityBudget::Aggressive, 0.50);
        assert_eq!(plan.experts[0].action, "keep");
        assert!(plan.experts[0].reason.contains("low_coverage"));
    }

    #[test]
    fn mass_loss_gate_blocks_over_budget() {
        // Create many cold experts that would together exceed conservative threshold
        let mut experts = Vec::new();
        for i in 0..100u32 {
            experts.push(make_entry(10, i, 0.05, ExpertTier::Cold, 0.5, 0.1));
        }
        let importance = ExpertImportance {
            schema_version: 1,
            experts,
        };
        let phase = phase_pass::run(40);
        let plan = run(&importance, &phase, &QualityBudget::Conservative, 1.0);
        // Not all 100 should be pruned — mass budget should stop it
        let pruned = plan
            .experts
            .iter()
            .filter(|e| e.action == "prune_candidate" || e.action == "compress")
            .count();
        assert!(pruned < 100, "expected budget gate to limit pruning, got {}", pruned);
    }
}
