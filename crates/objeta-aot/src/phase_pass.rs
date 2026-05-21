use crate::types::*;

/// Qwen3.6 GQA layer indices (0-indexed).
const QWEN36_GQA_LAYERS: &[u32] = &[3, 7, 11, 15, 19, 23, 27, 31, 35, 39];

/// Assign LKO phase to each layer based on Qwen3.6 architecture.
///
/// The current classifier is heuristic / rule-based (integrity_frontier for
/// early layers, GQA layers → steering, last layers → projection, remainder →
/// transport).  Phase assignments are labelled `heuristic_lko_v1` and
/// `experimental` until an empirical classifier is trained and validated.
pub fn run(num_layers: u32) -> PhasePolicy {
    let mut layers = Vec::with_capacity(num_layers as usize);
    for l in 0..num_layers {
        let phase = classify_layer(l, num_layers);
        let recommended_policy = policy_for_phase(&phase);
        layers.push(PhasePolicyLayer {
            layer: l,
            phase,
            recommended_policy,
        });
    }
    PhasePolicy {
        schema_version: 1,
        source: "heuristic_lko_v1".to_string(),
        confidence: "experimental".to_string(),
        layers,
    }
}

fn classify_layer(layer: u32, num_layers: u32) -> String {
    if layer <= 2 {
        "integrity_frontier".to_string()
    } else if num_layers > 2 && layer >= num_layers - 2 {
        "projection".to_string()
    } else if QWEN36_GQA_LAYERS.contains(&layer) {
        "steering".to_string()
    } else {
        "transport".to_string()
    }
}

fn policy_for_phase(phase: &str) -> RecommendedPolicy {
    match phase {
        "integrity_frontier" => RecommendedPolicy {
            policy_kind: "exact".to_string(),
            moe_top_p: 1.0,
            moe_min_experts: 8,
            moe_max_experts: 8,
        },
        "steering" => RecommendedPolicy {
            policy_kind: "top_p".to_string(),
            moe_top_p: 0.95,
            moe_min_experts: 6,
            moe_max_experts: 8,
        },
        "projection" => RecommendedPolicy {
            policy_kind: "top_p".to_string(),
            moe_top_p: 0.95,
            moe_min_experts: 6,
            moe_max_experts: 8,
        },
        _ => RecommendedPolicy {
            // transport phase — most aggressive pruning allowed
            policy_kind: "top_p".to_string(),
            moe_top_p: 0.90,
            moe_min_experts: 4,
            moe_max_experts: 8,
        },
    }
}

/// Look up the phase for a specific layer from the computed policy.
pub fn phase_for_layer(policy: &PhasePolicy, layer: u32) -> &str {
    policy
        .layers
        .iter()
        .find(|l| l.layer == layer)
        .map(|l| l.phase.as_str())
        .unwrap_or("transport")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_layers_are_integrity_frontier() {
        let policy = run(40);
        assert_eq!(policy.layers[0].phase, "integrity_frontier");
        assert_eq!(policy.layers[1].phase, "integrity_frontier");
        assert_eq!(policy.layers[2].phase, "integrity_frontier");
    }

    #[test]
    fn gqa_layers_are_steering() {
        let policy = run(40);
        assert_eq!(policy.layers[3].phase, "steering");
        assert_eq!(policy.layers[7].phase, "steering");
        assert_eq!(policy.layers[23].phase, "steering");
    }

    #[test]
    fn last_layers_are_projection() {
        let policy = run(40);
        assert_eq!(policy.layers[38].phase, "projection");
        assert_eq!(policy.layers[39].phase, "projection");
    }

    #[test]
    fn remaining_layers_are_transport() {
        let policy = run(40);
        assert_eq!(policy.layers[4].phase, "transport");
        assert_eq!(policy.layers[10].phase, "transport");
        assert_eq!(policy.layers[20].phase, "transport");
    }

    #[test]
    fn integrity_frontier_uses_exact_policy() {
        let policy = run(40);
        assert_eq!(policy.layers[0].recommended_policy.policy_kind, "exact");
        assert_eq!(policy.layers[0].recommended_policy.moe_min_experts, 8);
    }
}
