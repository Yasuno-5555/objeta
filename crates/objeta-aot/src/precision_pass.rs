use crate::phase_pass;
use crate::target::TargetHardware;
use crate::types::*;

/// Assign recommended quantization format per expert, router, and layer.
pub fn run(
    importance: &ExpertImportance,
    phase_policy: &PhasePolicy,
    routers: &[RouterEntry],
    target: &TargetHardware,
) -> QuantPlan {
    let mut entries = Vec::new();

    // Router entries — always q8
    for router in routers {
        entries.push(QuantPlanEntry {
            kind: "router".to_string(),
            layer: router.layer,
            expert: None,
            tier: None,
            recommended_format: "q8".to_string(),
            reason: "router_precision_sensitive".to_string(),
        });
    }

    // Expert entries — depends on phase + tier
    for e in &importance.experts {
        let phase = phase_pass::phase_for_layer(phase_policy, e.layer);
        let tier_str = tier_to_string(&e.tier);
        let (format, reason) = recommend_format(phase, &e.tier, target);
        entries.push(QuantPlanEntry {
            kind: "expert".to_string(),
            layer: e.layer,
            expert: Some(e.expert),
            tier: Some(tier_str),
            recommended_format: format,
            reason,
        });
    }

    // Sort for deterministic output
    entries.sort_by(|a, b| {
        a.layer
            .cmp(&b.layer)
            .then_with(|| a.kind.cmp(&b.kind))
            .then_with(|| a.expert.cmp(&b.expert))
    });

    QuantPlan {
        schema_version: 1,
        entries,
    }
}

fn recommend_format(phase: &str, tier: &ExpertTier, target: &TargetHardware) -> (String, String) {
    let (fmt, reason) = match (phase, tier) {
        // integrity_frontier: everything q5+
        ("integrity_frontier", _) => ("q5".into(), "integrity_frontier_protected".into()),

        // steering: hot→q5, warm→q4, cold→q4
        ("steering", ExpertTier::Hot) => ("q5".into(), "steering_hot_expert".into()),
        ("steering", ExpertTier::Warm) => ("q4".into(), "steering_warm_expert".into()),
        ("steering", ExpertTier::Cold) => ("q4".into(), "steering_cold_expert".into()),

        // projection: q5
        ("projection", _) => ("q5".into(), "projection_layer_protected".into()),

        // transport: hot→q4, warm→q4, cold→iq3 candidate
        ("transport", ExpertTier::Hot) => ("q4".into(), "transport_hot_expert".into()),
        ("transport", ExpertTier::Warm) => ("q4".into(), "transport_warm_expert".into()),
        ("transport", ExpertTier::Cold) => ("iq3".into(), "transport_cold_low_precision".into()),

        // fallback
        (_, _) => ("q4".into(), "default_format".into()),
    };

    if target.preferred_quant_formats.iter().any(|f| f == &fmt) {
        return (fmt, reason);
    }

    let fallback = if target.preferred_quant_formats.iter().any(|f| f == "q4") {
        "q4".to_string()
    } else if let Some(first) = target.preferred_quant_formats.first() {
        first.clone()
    } else {
        "q4".to_string()
    };
    (
        fallback.clone(),
        format!("{reason}_target_fallback_from_{fmt}"),
    )
}

fn tier_to_string(tier: &ExpertTier) -> String {
    match tier {
        ExpertTier::Hot => "hot".to_string(),
        ExpertTier::Warm => "warm".to_string(),
        ExpertTier::Cold => "cold".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_importance(entries: Vec<(u32, u32, ExpertTier)>) -> ExpertImportance {
        ExpertImportance {
            schema_version: 1,
            experts: entries
                .into_iter()
                .map(|(layer, expert, tier)| ExpertImportanceEntry {
                    layer,
                    expert,
                    selected_count: 10,
                    frequency: 0.5,
                    avg_gate_weight: 0.3,
                    max_gate_weight: 0.4,
                    importance: 0.8,
                    tier,
                    recommended_format: "q4".to_string(),
                    eviction_priority: 0.2,
                })
                .collect(),
        }
    }

    #[test]
    fn quant_plan_router_always_q8() {
        let importance = make_importance(vec![]);
        let phase = phase_pass::run(40);
        let routers = vec![RouterEntry {
            layer: 5,
            tensor: TensorRef {
                tensor_kind: ExpertTensorKind::Router,
                tensor_name: "router.5".to_string(),
                source_file: "mock.safetensors".to_string(),
                shape: None,
                dtype: None,
                byte_offset: None,
                byte_len: None,
            },
        }];
        let target = TargetHardware::from_name("m1-8gb");
        let plan = run(&importance, &phase, &routers, &target);
        let router_entry = plan.entries.iter().find(|e| e.kind == "router").unwrap();
        assert_eq!(router_entry.recommended_format, "q8");
    }

    #[test]
    fn quant_plan_hot_protected_gets_q5() {
        let importance = make_importance(vec![(0, 42, ExpertTier::Hot)]);
        let phase = phase_pass::run(40);
        let target = TargetHardware::from_name("m1-8gb");
        let plan = run(&importance, &phase, &[], &target);
        let entry = plan
            .entries
            .iter()
            .find(|e| e.layer == 0 && e.expert == Some(42))
            .unwrap();
        assert_eq!(entry.recommended_format, "q5");
        assert!(entry.reason.contains("integrity_frontier"));
    }

    #[test]
    fn quant_plan_transport_cold_gets_iq3() {
        let importance = make_importance(vec![(10, 99, ExpertTier::Cold)]);
        let phase = phase_pass::run(40);
        let target = TargetHardware::from_name("rtx3070-8gb-vram-32gb-ram");
        let plan = run(&importance, &phase, &[], &target);
        let entry = plan
            .entries
            .iter()
            .find(|e| e.layer == 10 && e.expert == Some(99))
            .unwrap();
        assert_eq!(entry.recommended_format, "iq3");
    }

    #[test]
    fn quant_plan_transport_cold_falls_back_on_m1() {
        let importance = make_importance(vec![(10, 99, ExpertTier::Cold)]);
        let phase = phase_pass::run(40);
        let target = TargetHardware::from_name("m1-8gb");
        let plan = run(&importance, &phase, &[], &target);
        let entry = plan
            .entries
            .iter()
            .find(|e| e.layer == 10 && e.expert == Some(99))
            .unwrap();
        assert_eq!(entry.recommended_format, "q4");
        assert!(entry.reason.contains("target_fallback_from_iq3"));
    }
}
