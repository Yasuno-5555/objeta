use serde::Deserialize;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct RuntimePackLoadStatus {
    pub runtime_pack_loaded: bool,
    pub runtime_pack_path: Option<String>,
    pub runtime_profile_loaded: bool,
    pub expert_importance_loaded: bool,
    pub residency_plan_loaded: bool,
    pub phase_policy_loaded: bool,
    pub expert_coresidency_loaded: bool,
    pub expert_eviction_policy: String,
    pub initial_hot_expert_count: usize,
    pub initial_hot_expert_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct LoadedRuntimePack {
    pub status: RuntimePackLoadStatus,
    pub runtime_profile: Option<crate::runtime_profile::RuntimeProfile>,
    pub expert_priorities: Vec<crate::expert_cache::ExpertPriority>,
}

#[derive(Debug, Deserialize)]
pub struct RuntimePackManifest {
    pub schema_version: u32,
    pub pack_type: String,
    pub model_family: String,
    pub model_name: String,
    pub target: String,
    pub files: RuntimePackFiles,
}

#[derive(Debug, Deserialize)]
pub struct RuntimePackFiles {
    pub expert_layout: String,
    pub expert_importance: String,
    pub expert_coresidency: String,
    pub residency_plan: String,
    pub phase_policy: String,
    pub runtime_profile: String,
}

#[derive(Debug, Deserialize)]
struct AotRuntimeProfile {
    #[allow(dead_code)]
    schema_version: u32,
    profile_name: String,
    target: String,
    backend: String,
    policy_kind: String,
    moe_top_p: f32,
    moe_min_experts: u32,
    moe_max_experts: u32,
    resident_cache_capacity_bytes: u64,
    group_preresolve_top_n: u32,
    group_preresolve_max_bytes: u64,
    source_model: String,
    source_calibration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AotExpertImportance {
    experts: Vec<AotExpertImportanceEntry>,
}

#[derive(Debug, Deserialize)]
struct AotExpertImportanceEntry {
    layer: u32,
    expert: u32,
    selected_count: u64,
    avg_gate_weight: f64,
    importance: f64,
    tier: String,
    eviction_priority: f64,
}

#[derive(Debug, Deserialize)]
struct AotResidencyPlan {
    initial_hot_experts: Vec<AotHotExpert>,
}

#[derive(Debug, Deserialize)]
struct AotHotExpert {
    bytes: u64,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> io::Result<T> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

fn as_executor_policy_kind(kind: &str) -> crate::runtime_profile::RuntimePolicyKind {
    match kind {
        "top_p" => crate::runtime_profile::RuntimePolicyKind::TopP,
        "lko_aware" => crate::runtime_profile::RuntimePolicyKind::LkoAware,
        _ => crate::runtime_profile::RuntimePolicyKind::Exact,
    }
}

fn as_executor_backend(backend: &str) -> Option<crate::runtime_profile::RuntimeBackend> {
    match backend {
        "fused_row_parallel" => Some(crate::runtime_profile::RuntimeBackend::FusedRowParallel),
        "legacy" => Some(crate::runtime_profile::RuntimeBackend::Legacy),
        _ => None,
    }
}

fn as_expert_tier(tier: &str) -> crate::expert_cache::ExpertTier {
    match tier {
        "hot" => crate::expert_cache::ExpertTier::Hot,
        "warm" => crate::expert_cache::ExpertTier::Warm,
        _ => crate::expert_cache::ExpertTier::Cold,
    }
}

fn manifest_path(pack_path: &Path) -> PathBuf {
    if pack_path.is_dir() {
        pack_path.join("manifest.json")
    } else {
        pack_path.to_path_buf()
    }
}

pub fn load_runtime_pack(pack_path: &Path) -> io::Result<LoadedRuntimePack> {
    let manifest_path = manifest_path(pack_path);
    let pack_root = manifest_path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "runtime pack has no parent"))?;
    let manifest: RuntimePackManifest = read_json(&manifest_path)?;

    let mut status = RuntimePackLoadStatus {
        runtime_pack_loaded: true,
        runtime_pack_path: Some(pack_root.display().to_string()),
        expert_eviction_policy: "lru".to_string(),
        ..RuntimePackLoadStatus::default()
    };

    let runtime_profile_path = pack_root.join(&manifest.files.runtime_profile);
    let runtime_profile = if runtime_profile_path.exists() {
        let aot: AotRuntimeProfile = read_json(&runtime_profile_path)?;
        status.runtime_profile_loaded = true;
        Some(crate::runtime_profile::RuntimeProfile {
            name: aot.profile_name,
            target: aot.target,
            notes: format!(
                "loaded from runtime pack for {} calib={}",
                aot.source_model,
                aot.source_calibration.as_deref().unwrap_or("none")
            ),
            policy_kind: as_executor_policy_kind(&aot.policy_kind),
            knobs: crate::runtime_profile::RuntimeKnobs {
                backend: as_executor_backend(&aot.backend),
                moe_top_p: Some(aot.moe_top_p),
                moe_min_experts: Some(aot.moe_min_experts as usize),
                moe_max_experts: Some(aot.moe_max_experts as usize),
                resident_cache_capacity_bytes: Some(aot.resident_cache_capacity_bytes),
                residency_group_size: Some(1),
                group_preresolve_top_n: Some(aot.group_preresolve_top_n as usize),
                group_preresolve_max_bytes: Some(aot.group_preresolve_max_bytes),
            },
        })
    } else {
        None
    };

    let mut expert_priorities = Vec::new();
    let importance_path = pack_root.join(&manifest.files.expert_importance);
    if importance_path.exists() {
        let importance: AotExpertImportance = read_json(&importance_path)?;
        status.expert_importance_loaded = true;
        status.expert_eviction_policy = "importance_lru".to_string();
        expert_priorities = importance
            .experts
            .into_iter()
            .map(|entry| crate::expert_cache::ExpertPriority {
                layer_idx: entry.layer as usize,
                expert_id: entry.expert as usize,
                eviction_priority: entry.eviction_priority,
                tier: as_expert_tier(&entry.tier),
                importance: entry.importance,
                selected_count: entry.selected_count,
                avg_gate_weight: entry.avg_gate_weight,
            })
            .collect();
    }

    let residency_plan_path = pack_root.join(&manifest.files.residency_plan);
    if residency_plan_path.exists() {
        let plan: AotResidencyPlan = read_json(&residency_plan_path)?;
        status.residency_plan_loaded = true;
        status.initial_hot_expert_count = plan.initial_hot_experts.len();
        status.initial_hot_expert_bytes = plan.initial_hot_experts.iter().map(|e| e.bytes).sum();
    }

    status.phase_policy_loaded = pack_root.join(&manifest.files.phase_policy).exists();
    status.expert_coresidency_loaded = pack_root.join(&manifest.files.expert_coresidency).exists();

    Ok(LoadedRuntimePack {
        status,
        runtime_profile,
        expert_priorities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_mock_runtime_pack() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("manifest.json"),
            serde_json::json!({
                "schema_version": 1,
                "pack_type": "objeta_runtime_pack",
                "model_family": "qwen",
                "model_name": "qwen36",
                "target": "m1-8gb",
                "files": {
                    "expert_layout": "expert_layout.json",
                    "expert_importance": "expert_importance.json",
                    "expert_coresidency": "expert_coresidency.json",
                    "residency_plan": "residency_plan.json",
                    "phase_policy": "phase_policy.json",
                    "runtime_profile": "runtime_profile.json"
                }
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("runtime_profile.json"),
            serde_json::json!({
                "schema_version": 1,
                "profile_name": "qwen36-m1-8gb-planner",
                "target": "m1-8gb",
                "backend": "fused_row_parallel",
                "policy_kind": "top_p",
                "moe_top_p": 0.9,
                "moe_min_experts": 4,
                "moe_max_experts": 8,
                "resident_cache_capacity_bytes": 3221225472u64,
                "group_preresolve_top_n": 0,
                "group_preresolve_max_bytes": 0u64,
                "source_model": "/tmp/mock_model",
                "source_calibration": "/tmp/mock_calib"
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("expert_importance.json"),
            serde_json::json!({
                "schema_version": 1,
                "experts": [{
                    "layer": 31,
                    "expert": 42,
                    "selected_count": 12,
                    "avg_gate_weight": 0.31,
                    "importance": 0.91,
                    "tier": "hot",
                    "eviction_priority": 0.09
                }]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("expert_coresidency.json"),
            serde_json::json!({"schema_version":1,"pairs":[]}).to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("phase_policy.json"),
            serde_json::json!({"schema_version":1,"layers":[]}).to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("residency_plan.json"),
            serde_json::json!({
                "schema_version": 1,
                "target": "m1-8gb",
                "resident_cache_capacity_bytes": 3221225472u64,
                "initial_hot_experts": [{"layer":31,"expert":42,"bytes":2097152,"importance":0.91,"tier":"hot","bytes_source":"target_default_estimate"}],
                "eviction_priority": [],
                "summary": {"initial_hot_expert_count":1,"initial_hot_expert_bytes":2097152u64,"eviction_priority_count":0,"bytes_fallback_expert_count":1}
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("expert_layout.json"),
            serde_json::json!({"schema_version":1,"experts":[]}).to_string(),
        )
        .unwrap();

        let loaded = load_runtime_pack(dir.path()).unwrap();
        assert!(loaded.status.runtime_pack_loaded);
        assert!(loaded.status.runtime_profile_loaded);
        assert!(loaded.status.expert_importance_loaded);
        assert!(loaded.status.residency_plan_loaded);
        assert!(loaded.status.phase_policy_loaded);
        assert!(loaded.status.expert_coresidency_loaded);
        assert_eq!(loaded.status.initial_hot_expert_count, 1);
        assert_eq!(loaded.status.initial_hot_expert_bytes, 2097152);
        assert_eq!(loaded.expert_priorities.len(), 1);
        assert!(matches!(
            loaded.runtime_profile.unwrap().policy_kind,
            crate::runtime_profile::RuntimePolicyKind::TopP
        ));
    }
}

impl crate::qwen36_forward::Qwen36Runner {
    pub fn load_runtime_pack(&mut self, path: &Path) -> std::io::Result<()> {
        let loaded = load_runtime_pack(path)?;
        if let Some(profile) = &loaded.runtime_profile {
            crate::runtime_profile::apply_runtime_profile(self, profile);
        }
        if !loaded.expert_priorities.is_empty() {
            self.expert_residency_manager
                .load_expert_priorities(loaded.expert_priorities);
        }
        self.runtime_pack_status = loaded.status;
        Ok(())
    }
}

