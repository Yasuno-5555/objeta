use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneCandidate {
    pub name: String,
    pub backend: crate::runtime_profile::RuntimeBackend,
    pub moe_top_p: f32,
    pub moe_min_experts: usize,
    pub resident_cache_capacity_bytes: u64,
    pub group_preresolve_top_n: usize,
    pub group_preresolve_max_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTunePolicy {
    pub target_machine: String,
    pub max_candidates: usize,
}

impl Default for AutoTunePolicy {
    fn default() -> Self {
        Self {
            target_machine: "m1_8gb".to_string(),
            max_candidates: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneResult {
    pub candidate: AutoTuneCandidate,
    pub score: f32,
    pub rationale: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<AutoTuneObservedMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneRun {
    pub policy: AutoTunePolicy,
    pub selected: AutoTuneResult,
    pub results: Vec<AutoTuneResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutoTuneObservedMetrics {
    pub tok_s: f32,
    pub forward_wall_ms_avg: f32,
    pub moe_wall_ms_avg: f32,
    pub actual_expert_bytes_loaded: u64,
    pub resident_cache_bytes_reused: u64,
    pub token_window_peak_resident_bytes: u64,
    pub eviction_count_at_token_end: u64,
    pub forward_wall_cv: f32,
    pub cache_lookup_wall_cv: f32,
    pub collapse_risk_count: u32,
    pub repetition_risk_count: u32,
    pub generated_token_ids: Vec<usize>,
}

pub fn default_m1_8gb_candidates() -> Vec<AutoTuneCandidate> {
    vec![
        AutoTuneCandidate {
            name: "legacy_exact_cache_default".to_string(),
            backend: crate::runtime_profile::RuntimeBackend::Legacy,
            moe_top_p: 1.00,
            moe_min_experts: 8,
            resident_cache_capacity_bytes: 4 * 1024 * 1024 * 1024,
            group_preresolve_top_n: 0,
            group_preresolve_max_bytes: 0,
        },
        AutoTuneCandidate {
            name: "fused_top_p_090_cache_2gb".to_string(),
            backend: crate::runtime_profile::RuntimeBackend::FusedRowParallel,
            moe_top_p: 0.90,
            moe_min_experts: 4,
            resident_cache_capacity_bytes: 2 * 1024 * 1024 * 1024,
            group_preresolve_top_n: 0,
            group_preresolve_max_bytes: 0,
        },
        AutoTuneCandidate {
            name: "fused_top_p_090_cache_3gb".to_string(),
            backend: crate::runtime_profile::RuntimeBackend::FusedRowParallel,
            moe_top_p: 0.90,
            moe_min_experts: 4,
            resident_cache_capacity_bytes: 3 * 1024 * 1024 * 1024,
            group_preresolve_top_n: 0,
            group_preresolve_max_bytes: 0,
        },
        AutoTuneCandidate {
            name: "fused_top_p_090_cache_4gb".to_string(),
            backend: crate::runtime_profile::RuntimeBackend::FusedRowParallel,
            moe_top_p: 0.90,
            moe_min_experts: 4,
            resident_cache_capacity_bytes: 4 * 1024 * 1024 * 1024,
            group_preresolve_top_n: 0,
            group_preresolve_max_bytes: 0,
        },
        AutoTuneCandidate {
            name: "fused_top_p_085_cache_3gb".to_string(),
            backend: crate::runtime_profile::RuntimeBackend::FusedRowParallel,
            moe_top_p: 0.85,
            moe_min_experts: 3,
            resident_cache_capacity_bytes: 3 * 1024 * 1024 * 1024,
            group_preresolve_top_n: 0,
            group_preresolve_max_bytes: 0,
        },
        AutoTuneCandidate {
            name: "fused_top_p_085_cache_4gb".to_string(),
            backend: crate::runtime_profile::RuntimeBackend::FusedRowParallel,
            moe_top_p: 0.85,
            moe_min_experts: 3,
            resident_cache_capacity_bytes: 4 * 1024 * 1024 * 1024,
            group_preresolve_top_n: 0,
            group_preresolve_max_bytes: 0,
        },
    ]
}

fn build_candidate_profile(candidate: &AutoTuneCandidate) -> crate::runtime_profile::RuntimeProfile {
    crate::runtime_profile::RuntimeProfile {
        name: format!("autotuned-{}", candidate.name),
        target: "m1_8gb".to_string(),
        notes: "AutoTuner v1 runtime-evaluated profile".to_string(),
        policy_kind: if (candidate.moe_top_p - 1.0).abs() < f32::EPSILON
            && candidate.moe_min_experts >= 8
        {
            crate::runtime_profile::RuntimePolicyKind::Exact
        } else {
            crate::runtime_profile::RuntimePolicyKind::TopP
        },
        knobs: crate::runtime_profile::RuntimeKnobs {
            backend: Some(candidate.backend),
            moe_top_p: Some(candidate.moe_top_p),
            moe_min_experts: Some(candidate.moe_min_experts),
            moe_max_experts: Some(8),
            resident_cache_capacity_bytes: Some(candidate.resident_cache_capacity_bytes),
            residency_group_size: Some(1),
            group_preresolve_top_n: Some(candidate.group_preresolve_top_n),
            group_preresolve_max_bytes: Some(candidate.group_preresolve_max_bytes),
        },
    }
}

fn safe_exact_prompt_ids() -> Vec<usize> {
    #[derive(Deserialize)]
    struct RegistryEntry {
        tokenizer_ids: Option<Vec<usize>>,
    }

    let fallback = vec![
        248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198, 248045, 74455, 198, 248068,
        198,
    ];
    let path = Path::new("runs/oracles/registry.json");
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return fallback,
    };
    let registry: serde_json::Map<String, serde_json::Value> = match serde_json::from_str(&text) {
        Ok(reg) => reg,
        Err(_) => return fallback,
    };
    registry
        .get("safe_exact_chat_prefill")
        .and_then(|value| serde_json::from_value::<RegistryEntry>(value.clone()).ok())
        .and_then(|entry| entry.tokenizer_ids)
        .filter(|ids| !ids.is_empty())
        .unwrap_or(fallback)
}

fn coefficient_of_variation(samples: &[f64]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    if mean.abs() < 1e-12 {
        return 0.0;
    }
    let variance = samples
        .iter()
        .map(|value| {
            let d = value - mean;
            d * d
        })
        .sum::<f64>()
        / samples.len() as f64;
    (variance.sqrt() / mean) as f32
}

fn should_runtime_eval() -> bool {
    if cfg!(test) {
        return false;
    }
    std::env::var("OBJETA_AUTOTUNE_SKIP_RUNTIME_EVAL")
        .map(|v| !(v == "1" || v.eq_ignore_ascii_case("true")))
        .unwrap_or(true)
}

fn evaluate_candidate_runtime(candidate: &AutoTuneCandidate) -> io::Result<AutoTuneObservedMetrics> {
    let prompt_ids = safe_exact_prompt_ids();
    let mut runner = crate::qwen36_forward::Qwen36Runner::new(Path::new("models/qwen36_bin"), 256)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "failed to init Qwen36Runner"))?;
    runner.fusion_ratio = 0.80;
    runner.moe_on_deltanet = true;
    runner.runtime_governor.mode = crate::runtime_governor::GovernorMode::Disabled;
    runner.runtime_governor.reset_counters();
    let profile = build_candidate_profile(candidate);
    crate::runtime_profile::apply_runtime_profile(&mut runner, &profile);
    runner.runtime_governor.mode = crate::runtime_governor::GovernorMode::Disabled;
    runner.runtime_governor.reset_counters();

    let mut generated_token_ids = Vec::new();
    let mut next_token = 0usize;
    let mut forward_deltas_ms = Vec::new();
    let mut cache_lookup_deltas_ms = Vec::new();
    let mut collapse_risk_count = 0u32;
    let mut repetition_risk_count = 0u32;
    let total_start = Instant::now();

    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        let prev_forward = runner.forward_wall_sec;
        let prev_cache_lookup: f64 = runner
            .moe_stats
            .iter()
            .map(|s| s.total_cache_lookup_wall_sec)
            .sum();
        runner.begin_token_residency(token_id);
        let (h, _) = runner.forward_timed(token_id, pos, pos + 1);
        let hn = crate::qwen36_forward::rms_norm(&h, runner.final_norm_weights());
        let (indices, _, _) = runner.lm_head_topk_with_entropy(&hn, 10);
        runner.finish_step();
        next_token = indices.first().copied().unwrap_or_default() as usize;
        forward_deltas_ms.push((runner.forward_wall_sec - prev_forward) * 1000.0);
        let curr_cache_lookup: f64 = runner
            .moe_stats
            .iter()
            .map(|s| s.total_cache_lookup_wall_sec)
            .sum();
        cache_lookup_deltas_ms.push((curr_cache_lookup - prev_cache_lookup) * 1000.0);
        if runner.last_collapse_risk {
            collapse_risk_count += 1;
        }
        if runner.last_repetition_risk {
            repetition_risk_count += 1;
        }
    }

    for decode_idx in 0..5usize {
        let pos = prompt_ids.len() + decode_idx;
        let prev_forward = runner.forward_wall_sec;
        let prev_cache_lookup: f64 = runner
            .moe_stats
            .iter()
            .map(|s| s.total_cache_lookup_wall_sec)
            .sum();
        let token_id = next_token;
        generated_token_ids.push(token_id);
        runner.begin_token_residency(token_id);
        let (h, _) = runner.forward_timed(token_id, pos, pos + 1);
        let hn = crate::qwen36_forward::rms_norm(&h, runner.final_norm_weights());
        let (indices, _, _) = runner.lm_head_topk_with_entropy(&hn, 10);
        runner.finish_step();
        next_token = indices.first().copied().unwrap_or_default() as usize;
        forward_deltas_ms.push((runner.forward_wall_sec - prev_forward) * 1000.0);
        let curr_cache_lookup: f64 = runner
            .moe_stats
            .iter()
            .map(|s| s.total_cache_lookup_wall_sec)
            .sum();
        cache_lookup_deltas_ms.push((curr_cache_lookup - prev_cache_lookup) * 1000.0);
        if runner.last_collapse_risk {
            collapse_risk_count += 1;
        }
        if runner.last_repetition_risk {
            repetition_risk_count += 1;
        }
    }

    let elapsed = total_start.elapsed().as_secs_f32();
    let total_layer_calls: u64 = runner.forward_stats.iter().map(|s| s.calls).sum();
    let total_moe_wall_sec: f64 = runner.forward_stats.iter().map(|s| s.total_moe_wall_sec).sum();
    Ok(AutoTuneObservedMetrics {
        tok_s: if elapsed > 0.0 {
            generated_token_ids.len() as f32 / elapsed
        } else {
            0.0
        },
        forward_wall_ms_avg: if runner.forward_calls > 0 {
            (runner.forward_wall_sec * 1000.0 / runner.forward_calls as f64) as f32
        } else {
            0.0
        },
        moe_wall_ms_avg: if total_layer_calls > 0 {
            (total_moe_wall_sec * 1000.0 / total_layer_calls as f64) as f32
        } else {
            0.0
        },
        actual_expert_bytes_loaded: runner.expert_residency_manager.actual_expert_bytes_loaded,
        resident_cache_bytes_reused: runner.expert_residency_manager.resident_cache_bytes_reused,
        token_window_peak_resident_bytes: runner
            .expert_residency_manager
            .token_window_peak_resident_bytes,
        eviction_count_at_token_end: runner.expert_residency_manager.eviction_count_at_token_end,
        forward_wall_cv: coefficient_of_variation(&forward_deltas_ms),
        cache_lookup_wall_cv: coefficient_of_variation(&cache_lookup_deltas_ms),
        collapse_risk_count,
        repetition_risk_count,
        generated_token_ids,
    })
}

fn score_candidate(
    candidate: &AutoTuneCandidate,
    observed: Option<&AutoTuneObservedMetrics>,
) -> AutoTuneResult {
    let mut score = 0.0f32;
    let mut rationale = Vec::new();

    match candidate.backend {
        crate::runtime_profile::RuntimeBackend::Legacy => {
            rationale.push("legacy-backend: stability bias".to_string());
            score += 1.0;
        }
        crate::runtime_profile::RuntimeBackend::FusedRowParallel => {
            rationale.push("fused-row-parallel: throughput bias".to_string());
            score += 2.0;
        }
    }

    if candidate.moe_top_p >= 0.99 {
        rationale.push("full-top-p: quality bias".to_string());
        score += 1.5;
    } else if candidate.moe_top_p >= 0.90 {
        rationale.push("moderate-pruning: balanced".to_string());
        score += 2.0;
    } else {
        rationale.push("aggressive-pruning: speed bias with quality risk".to_string());
        score += 1.2;
    }

    if candidate.resident_cache_capacity_bytes >= 4 * 1024 * 1024 * 1024 {
        rationale.push("4gb-resident-cache".to_string());
        score += 1.0;
    } else {
        rationale.push("3gb-resident-cache".to_string());
        score += 0.4;
    }

    if candidate.group_preresolve_top_n > 0 {
        rationale.push("group-preresolve-enabled".to_string());
        score += 0.5;
    }

    if let Some(metrics) = observed {
        score += metrics.tok_s * 6.0;
        score -= metrics.forward_wall_cv * 1.5;
        score -= metrics.cache_lookup_wall_cv * 1.25;
        score -= (metrics.eviction_count_at_token_end as f32) * 0.01;
        let capacity = candidate.resident_cache_capacity_bytes.max(1) as f32;
        score -= (metrics.token_window_peak_resident_bytes as f32 / capacity).max(1.0) * 0.5;
        rationale.push(format!("observed-tok-s={:.4}", metrics.tok_s));
        rationale.push(format!("forward-wall-cv={:.4}", metrics.forward_wall_cv));
        rationale.push(format!(
            "cache-lookup-wall-cv={:.4}",
            metrics.cache_lookup_wall_cv
        ));
        rationale.push(format!(
            "eviction-count-at-token-end={}",
            metrics.eviction_count_at_token_end
        ));
        rationale.push(format!(
            "token-window-peak-resident-bytes={}",
            metrics.token_window_peak_resident_bytes
        ));
    }

    AutoTuneResult {
        candidate: candidate.clone(),
        score,
        rationale,
        observed: observed.cloned(),
    }
}

pub fn evaluate_candidates_sequential(
    policy: AutoTunePolicy,
    candidates: &[AutoTuneCandidate],
) -> AutoTuneRun {
    let capped: Vec<_> = candidates
        .iter()
        .take(policy.max_candidates.max(1))
        .cloned()
        .collect();
    let mut results: Vec<_> = capped
        .iter()
        .map(|candidate| {
            let observed = if should_runtime_eval() {
                evaluate_candidate_runtime(candidate).ok()
            } else {
                None
            };
            score_candidate(candidate, observed.as_ref())
        })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected = results
        .first()
        .cloned()
        .unwrap_or_else(|| score_candidate(&default_m1_8gb_candidates()[0], None));
    AutoTuneRun {
        policy,
        selected,
        results,
    }
}

pub fn write_autotune_outputs(output_dir: &Path, run: &AutoTuneRun) -> io::Result<()> {
    let results_dir = output_dir.join("results");
    fs::create_dir_all(&results_dir)?;

    let mut profile = build_candidate_profile(&run.selected.candidate);
    profile.target = run.policy.target_machine.clone();
    profile.notes = "AutoTuner v1 runtime-evaluated selection".to_string();

    fs::write(
        output_dir.join("runtime_profile.json"),
        serde_json::to_vec_pretty(&profile)?,
    )?;
    fs::write(
        results_dir.join("auto_tune_runtime.json"),
        serde_json::to_vec_pretty(run)?,
    )?;

    let mut md = String::new();
    md.push_str("# AutoTuner v1\n\n");
    md.push_str(&format!(
        "Selected: `{}` (score {:.2})\n\n",
        run.selected.candidate.name, run.selected.score
    ));
    md.push_str(
        "| candidate | backend | top_p | min_experts | cache_gb | preresolve_top_n | tok_s | fwd_cv | cache_cv | score |\n",
    );
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for result in &run.results {
        let tok_s = result.observed.as_ref().map(|m| m.tok_s).unwrap_or(0.0);
        let forward_cv = result
            .observed
            .as_ref()
            .map(|m| m.forward_wall_cv)
            .unwrap_or(0.0);
        let cache_cv = result
            .observed
            .as_ref()
            .map(|m| m.cache_lookup_wall_cv)
            .unwrap_or(0.0);
        md.push_str(&format!(
            "| {} | {:?} | {:.2} | {} | {:.1} | {} | {:.4} | {:.4} | {:.4} | {:.2} |\n",
            result.candidate.name,
            result.candidate.backend,
            result.candidate.moe_top_p,
            result.candidate.moe_min_experts,
            result.candidate.resident_cache_capacity_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            result.candidate.group_preresolve_top_n,
            tok_s,
            forward_cv,
            cache_cv,
            result.score
        ));
    }
    fs::write(results_dir.join("auto_tune_runtime.md"), md)?;
    Ok(())
}

pub fn auto_tune_default(output_dir: &Path, max_candidates: usize) -> io::Result<AutoTuneRun> {
    let mut policy = AutoTunePolicy::default();
    policy.max_candidates = max_candidates.max(1);
    let run = evaluate_candidates_sequential(policy, &default_m1_8gb_candidates());
    write_autotune_outputs(output_dir, &run)?;
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_autotune_outputs_are_emitted() {
        let dir = tempdir().unwrap();
        let run = auto_tune_default(dir.path(), 2).unwrap();
        assert!(!run.results.is_empty());
        assert!(dir.path().join("runtime_profile.json").exists());
        assert!(dir.path().join("results/auto_tune_runtime.json").exists());
        assert!(dir.path().join("results/auto_tune_runtime.md").exists());
    }
}
