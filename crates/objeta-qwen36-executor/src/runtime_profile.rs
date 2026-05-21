use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeBackend {
    Legacy,
    FusedRowParallel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePolicyKind {
    Exact,
    TopP,
    LkoAware,
}

impl Default for RuntimePolicyKind {
    fn default() -> Self {
        Self::Exact
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeKnobs {
    #[serde(default)]
    pub backend: Option<RuntimeBackend>,
    #[serde(default)]
    pub moe_top_p: Option<f32>,
    #[serde(default)]
    pub moe_min_experts: Option<usize>,
    #[serde(default)]
    pub moe_max_experts: Option<usize>,
    #[serde(default)]
    pub resident_cache_capacity_bytes: Option<u64>,
    #[serde(default)]
    pub residency_group_size: Option<usize>,
    #[serde(default)]
    pub group_preresolve_top_n: Option<usize>,
    #[serde(default)]
    pub group_preresolve_max_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeProfile {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub policy_kind: RuntimePolicyKind,
    #[serde(default)]
    pub knobs: RuntimeKnobs,
}

pub fn load_runtime_profile(path: &Path) -> io::Result<RuntimeProfile> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

fn resolve_policy_from_profile(
    profile: &RuntimeProfile,
    fallback_top_p: f32,
    fallback_min_experts: usize,
    fallback_max_experts: usize,
) -> (crate::strategy::ExpertPolicyConfig, f32, usize, usize) {
    match profile.policy_kind {
        RuntimePolicyKind::Exact => (crate::strategy::ExpertPolicyConfig::Exact, 1.0, 8, 8),
        RuntimePolicyKind::TopP => {
            let p = profile
                .knobs
                .moe_top_p
                .unwrap_or(fallback_top_p)
                .clamp(0.0, 1.0);
            let min_experts = profile
                .knobs
                .moe_min_experts
                .unwrap_or(fallback_min_experts.max(1))
                .max(1);
            let max_experts = profile
                .knobs
                .moe_max_experts
                .unwrap_or(fallback_max_experts.max(min_experts))
                .max(min_experts);
            (
                crate::strategy::ExpertPolicyConfig::TopP {
                    p,
                    min_experts,
                    max_experts,
                },
                p,
                min_experts,
                max_experts,
            )
        }
        RuntimePolicyKind::LkoAware => (
            crate::strategy::ExpertPolicyConfig::LkoAware,
            1.0,
            fallback_min_experts.max(2),
            fallback_max_experts.max(fallback_min_experts.max(2)),
        ),
    }
}

pub fn apply_runtime_profile(
    runner: &mut crate::qwen36_forward::Qwen36Runner,
    profile: &RuntimeProfile,
) {
    let knobs = &profile.knobs;

    if let Some(backend) = knobs.backend {
        match backend {
            RuntimeBackend::Legacy => {
                runner.use_fused_moe = false;
            }
            RuntimeBackend::FusedRowParallel => {
                runner.use_fused_moe = true;
                runner.fused_down_mode = crate::moe_dispatch::FusedDownMode::RowParallel;
            }
        }
    }

    let (policy, top_p, min_experts, max_experts) = resolve_policy_from_profile(
        profile,
        runner.moe_top_p,
        runner.min_experts,
        runner.max_experts,
    );
    runner.set_expert_policy(policy);
    runner.moe_top_p = top_p;
    runner.min_experts = min_experts;
    runner.max_experts = max_experts;

    if let Some(capacity_bytes) = knobs.resident_cache_capacity_bytes {
        let existing_priorities: Vec<_> = runner
            .expert_residency_manager
            .expert_priorities
            .values()
            .cloned()
            .collect();
        runner.expert_residency_manager =
            crate::expert_cache::ExpertResidencyManager::new(capacity_bytes);
        if !existing_priorities.is_empty() {
            runner
                .expert_residency_manager
                .load_expert_priorities(existing_priorities);
        }
        let expert_total_bytes =
            (crate::moe_dispatch::GU_EXPERT_BYTES + crate::moe_dispatch::D_EXPERT_BYTES) as u64;
        runner.expert_cache_size = if expert_total_bytes > 0 {
            (capacity_bytes / expert_total_bytes) as usize
        } else {
            0
        };
    }

    if let Some(group_size) = knobs.residency_group_size {
        std::env::set_var("OBJETA_RESIDENCY_GROUP_SIZE", group_size.max(1).to_string());
    }
    if let Some(top_n) = knobs.group_preresolve_top_n {
        std::env::set_var("OBJETA_GROUP_PRERESOLVE_TOP_N", top_n.to_string());
    }
    if let Some(max_bytes) = knobs.group_preresolve_max_bytes {
        std::env::set_var("OBJETA_GROUP_PRERESOLVE_MAX_BYTES", max_bytes.to_string());
    }

    runner.runtime_profile = profile.clone();
    runner.note_runtime_config_source(crate::qwen36_forward::RuntimeConfigSource::RuntimeProfile);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_runtime_profile_roundtrip() {
        let file = NamedTempFile::new().unwrap();
        fs::write(
            file.path(),
            r#"{
                "name":"m1-safe",
                "policy_kind":"top_p",
                "knobs":{
                    "backend":"fused_row_parallel",
                    "moe_top_p":0.9,
                    "moe_min_experts":4,
                    "resident_cache_capacity_bytes":3221225472
                }
            }"#,
        )
        .unwrap();
        let profile = load_runtime_profile(file.path()).unwrap();
        assert_eq!(profile.name, "m1-safe");
        assert_eq!(profile.policy_kind, RuntimePolicyKind::TopP);
        assert_eq!(
            profile.knobs.backend,
            Some(RuntimeBackend::FusedRowParallel)
        );
        assert_eq!(profile.knobs.moe_top_p, Some(0.9));
    }

    #[test]
    fn test_exact_profile_preserves_exact_identity() {
        let profile = RuntimeProfile {
            policy_kind: RuntimePolicyKind::Exact,
            knobs: RuntimeKnobs {
                moe_top_p: Some(0.75),
                moe_min_experts: Some(2),
                moe_max_experts: Some(4),
                ..RuntimeKnobs::default()
            },
            ..RuntimeProfile::default()
        };
        let (policy, top_p, min_experts, max_experts) =
            resolve_policy_from_profile(&profile, 0.9, 3, 8);
        assert!(matches!(policy, crate::strategy::ExpertPolicyConfig::Exact));
        assert_eq!(top_p, 1.0);
        assert_eq!(min_experts, 8);
        assert_eq!(max_experts, 8);
    }

    #[test]
    fn test_top_p_identity_is_preserved_at_one_point_zero() {
        let profile = RuntimeProfile {
            policy_kind: RuntimePolicyKind::TopP,
            knobs: RuntimeKnobs {
                moe_top_p: Some(1.0),
                moe_min_experts: Some(4),
                moe_max_experts: Some(8),
                ..RuntimeKnobs::default()
            },
            ..RuntimeProfile::default()
        };
        let (policy, top_p, min_experts, max_experts) =
            resolve_policy_from_profile(&profile, 0.9, 3, 8);
        assert!(matches!(
            policy,
            crate::strategy::ExpertPolicyConfig::TopP {
                p: 1.0,
                min_experts: 4,
                max_experts: 8,
            }
        ));
        assert_eq!(top_p, 1.0);
        assert_eq!(min_experts, 4);
        assert_eq!(max_experts, 8);
    }
}
