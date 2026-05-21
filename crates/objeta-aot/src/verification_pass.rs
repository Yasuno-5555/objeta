use crate::types::*;

/// Generate a verification plan based on quality budget and pruning results.
pub fn run(quality_budget: &QualityBudget, pruning: &PruningPlan) -> VerificationPlan {
    let mut checks = vec![
        VerificationCheck {
            kind: "smoke_generation".to_string(),
            prompt: Some("The capital of France is".to_string()),
            max_tokens: Some(25),
            required_cosine: None,
            max_allowed: None,
        },
        VerificationCheck {
            kind: "oracle_layer_trace".to_string(),
            prompt: None,
            max_tokens: None,
            required_cosine: Some(0.999),
            max_allowed: None,
        },
        VerificationCheck {
            kind: "routing_mass_loss".to_string(),
            prompt: None,
            max_tokens: None,
            required_cosine: None,
            max_allowed: Some(quality_budget.mass_loss_threshold()),
        },
    ];

    if pruning.summary.prune_candidate_count > 0 || pruning.summary.compress_count > 0 {
        checks.push(VerificationCheck {
            kind: "pruning_mass_loss_budget".to_string(),
            prompt: None,
            max_tokens: None,
            required_cosine: None,
            max_allowed: Some(pruning.summary.estimated_routing_mass_loss),
        });
    }

    VerificationPlan {
        schema_version: 1,
        checks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_plan_includes_smoke() {
        let pruning = PruningPlan {
            schema_version: 1,
            summary: PruningPlanSummary {
                protect_count: 0,
                keep_count: 0,
                cold_tier_count: 0,
                prune_candidate_count: 0,
                compress_count: 0,
                estimated_routing_mass_loss: 0.0,
                quality_budget: "conservative".to_string(),
                mass_loss_threshold: 0.02,
                safe: true,
                estimated_only: true,
                requires_verification: true,
            },
            experts: vec![],
        };
        let plan = run(&QualityBudget::Conservative, &pruning);
        assert!(plan.checks.iter().any(|c| c.kind == "smoke_generation"));
        assert!(plan.checks.iter().any(|c| c.kind == "oracle_layer_trace"));
        assert!(plan.checks.iter().any(|c| c.kind == "routing_mass_loss"));
    }

    #[test]
    fn pruning_budget_check_added_when_pruning_exists() {
        let pruning = PruningPlan {
            schema_version: 1,
            summary: PruningPlanSummary {
                protect_count: 0,
                keep_count: 0,
                cold_tier_count: 0,
                prune_candidate_count: 5,
                compress_count: 3,
                estimated_routing_mass_loss: 0.015,
                quality_budget: "balanced".to_string(),
                mass_loss_threshold: 0.05,
                safe: true,
                estimated_only: true,
                requires_verification: true,
            },
            experts: vec![],
        };
        let plan = run(&QualityBudget::Balanced, &pruning);
        assert!(plan
            .checks
            .iter()
            .any(|c| c.kind == "pruning_mass_loss_budget"));
    }
}
