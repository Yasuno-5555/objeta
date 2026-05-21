use crate::types::*;
use crate::util;
use std::error::Error;
use std::fs;
use std::path::Path;

/// Write all specialization pack files + generate specialization_report.md.
pub fn run(
    out: &Path,
    manifest: &Manifest,
    layout: &ExpertLayout,
    importance: &ExpertImportance,
    coresidency: &ExpertCoresidency,
    phase_policy: &PhasePolicy,
    quant_plan: &QuantPlan,
    pruning_plan: &PruningPlan,
    residency_plan: &ResidencyPlan,
    runtime_profile: &RuntimeProfile,
    verification_plan: &VerificationPlan,
    args_summary: &str,
    importance_coverage: f64,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(out)?;

    util::write_json(out.join("manifest.json"), manifest)?;
    util::write_json(out.join("expert_layout.json"), layout)?;
    util::write_json(out.join("expert_importance.json"), importance)?;
    util::write_json(out.join("expert_coresidency.json"), coresidency)?;
    util::write_json(out.join("phase_policy.json"), phase_policy)?;
    util::write_json(out.join("quant_plan.json"), quant_plan)?;
    util::write_json(out.join("pruning_plan.json"), pruning_plan)?;
    util::write_json(out.join("residency_plan.json"), residency_plan)?;
    util::write_json(out.join("runtime_profile.json"), runtime_profile)?;
    util::write_json(out.join("verification_plan.json"), verification_plan)?;

    // Generate human-readable report
    let report = generate_report(
        manifest,
        layout,
        importance,
        coresidency,
        phase_policy,
        quant_plan,
        pruning_plan,
        residency_plan,
        runtime_profile,
        verification_plan,
        args_summary,
        importance_coverage,
    );
    fs::write(out.join("specialization_report.md"), report)?;

    Ok(())
}

fn generate_report(
    manifest: &Manifest,
    layout: &ExpertLayout,
    importance: &ExpertImportance,
    coresidency: &ExpertCoresidency,
    phase_policy: &PhasePolicy,
    quant_plan: &QuantPlan,
    pruning_plan: &PruningPlan,
    residency_plan: &ResidencyPlan,
    runtime_profile: &RuntimeProfile,
    verification_plan: &VerificationPlan,
    args_summary: &str,
    importance_coverage: f64,
) -> String {
    // ── Tier counts ──────────────────────────────────────────────────────
    let hot_count = importance.experts.iter().filter(|e| e.tier == ExpertTier::Hot).count();
    let warm_count = importance.experts.iter().filter(|e| e.tier == ExpertTier::Warm).count();
    let cold_count = importance.experts.iter().filter(|e| e.tier == ExpertTier::Cold).count();

    // ── Phase counts ─────────────────────────────────────────────────────
    let integrity_count = phase_policy.layers.iter().filter(|l| l.phase == "integrity_frontier").count();
    let steering_count = phase_policy.layers.iter().filter(|l| l.phase == "steering").count();
    let transport_count = phase_policy.layers.iter().filter(|l| l.phase == "transport").count();
    let projection_count = phase_policy.layers.iter().filter(|l| l.phase == "projection").count();

    // ── Quant format counts ──────────────────────────────────────────────
    let q8_count = quant_plan.entries.iter().filter(|e| e.recommended_format == "q8").count();
    let q5_count = quant_plan.entries.iter().filter(|e| e.recommended_format == "q5").count();
    let q4_count = quant_plan.entries.iter().filter(|e| e.recommended_format == "q4").count();
    let iq3_count = quant_plan.entries.iter().filter(|e| e.recommended_format == "iq3").count();
    let iq2_count = quant_plan.entries.iter().filter(|e| e.recommended_format == "iq2").count();

    // ── Coverage ─────────────────────────────────────────────────────────
    let calibrated_count = importance.experts.len();
    let total_expert_count = if layout.logical_routed_expert_count > 0 {
        layout.logical_routed_expert_count as usize
    } else {
        layout.experts.len()
    };
    let pruning_enabled = importance_coverage >= 0.80;
    let coverage_pct = importance_coverage * 100.0;

    // ── Safety gates triggered ───────────────────────────────────────────
    let mut safety_gates = Vec::new();
    if !pruning_enabled {
        safety_gates.push(format!(
            "⚠️  Coverage gate: pruning disabled (coverage={:.1}% < 80%)",
            coverage_pct
        ));
    }
    if pruning_plan.summary.safe {
        safety_gates.push(format!(
            "✅ Mass-loss gate: within budget ({:.4} ≤ {:.4})",
            pruning_plan.summary.estimated_routing_mass_loss,
            pruning_plan.summary.mass_loss_threshold,
        ));
    } else {
        safety_gates.push(format!(
            "⛔ Mass-loss gate: EXCEEDED ({:.4} > {:.4})",
            pruning_plan.summary.estimated_routing_mass_loss,
            pruning_plan.summary.mass_loss_threshold,
        ));
    }

    let gates_text = if safety_gates.is_empty() {
        "None triggered.".to_string()
    } else {
        safety_gates.join("\n")
    };

    // ── Phase classifier provenance ──────────────────────────────────────
    let phase_provenance = format!(
        "source=`{}`, confidence=`{}`",
        phase_policy.source, phase_policy.confidence
    );

    // ── Epistemic flags ──────────────────────────────────────────────────
    let estimated_only_flag = if pruning_plan.summary.estimated_only { "yes ⚠️" } else { "no" };
    let requires_verification_flag = if pruning_plan.summary.requires_verification { "yes ⚠️" } else { "no" };

    format!(
        r#"# Objeta Specialization Report

Generated by objeta-aot specialize v0.1 at {created_at}

> **Note**: Phase classification uses heuristic rules ({phase_provenance}).
> Mass-loss figures are routing-trace estimates only — not end-to-end quality measurements.
> Verification is required before applying this plan to a production pack.

## Arguments

{args_summary}

---

## Target Hardware

| Field | Value |
|-------|-------|
| Target | `{target}` |
| Backend | `{backend}` |
| Cache capacity | {cache_bytes_gb:.2} GB ({cache_bytes} bytes) |

---

## Model

| Field | Value |
|-------|-------|
| Model | `{model_name}` |
| Layers | {num_layers} |
| Routed experts (logical) | {num_experts_routed} |
| Packed expert layers | {num_packed_layers} |
| Shared experts | {num_shared} |
| Routers | {num_routers} |

---

## Calibration Coverage

| Metric | Value |
|--------|-------|
| Calibrated experts | {calibrated_count} |
| Total experts (layout) | {total_expert_count} |
| Coverage ratio | {coverage_pct:.1}% |
| Pruning enabled | {pruning_enabled} |

---

## Phase Summary

Phase classification: {phase_provenance}

| Phase | Layers | Role |
|-------|--------|------|
| integrity_frontier | {integrity_count} | Protected — early token formation |
| steering | {steering_count} | Protected — GQA semantic routing |
| transport | {transport_count} | Optimizable — bulk FFN transport |
| projection | {projection_count} | Protected — late output projection |

---

## Expert Tier Distribution

| Tier | Count |
|------|-------|
| 🔥 Hot | {hot_count} |
| 🌡️ Warm | {warm_count} |
| 🧊 Cold | {cold_count} |

---

## Precision Summary

| Format | Count | Note |
|--------|-------|------|
| q8 | {q8_count} | Routers (always) |
| q5 | {q5_count} | Protected layers, hot steering |
| q4 | {q4_count} | Warm transport, hot transport |
| iq3 | {iq3_count} | Cold transport candidates |
| iq2 | {iq2_count} | Prune candidates only |

---

## Pruning Summary

> ⚠️  **estimated_only**: {estimated_only_flag} — these are routing-trace estimates.
> ⚠️  **requires_verification**: {requires_verification_flag} — smoke + oracle trace must pass before applying.

| Action | Count | Description |
|--------|-------|-------------|
| protect | {protect_count} | Safety-gated: never touched |
| keep | {keep_count} | Retained at current precision |
| cold_tier | {cold_tier_count} | Demoted tier, no weight change |
| compress | {compress_count} | Quantized to iq3 |
| prune_candidate | {prune_candidate_count} | Quantized to iq2 (candidates) |

| Budget Metric | Value |
|---------------|-------|
| Quality budget | {quality_budget} |
| Mass-loss threshold | {mass_threshold:.4} ({mass_threshold_pct:.1}%) |
| Estimated routing mass loss | {mass_loss:.6} ({mass_loss_pct:.2}%) |
| Within budget | {safe} |

---

## Safety Gates

{gates_text}

---

## Residency Plan

| Metric | Value |
|--------|-------|
| Initial hot experts | {hot_expert_count} |
| Initial hot bytes | {hot_expert_bytes_mb:.2} MB |
| Eviction entries | {eviction_count} |

---

## Runtime Profile

| Field | Value |
|-------|-------|
| Profile | `{profile_name}` |
| Backend | `{profile_backend}` |
| Policy | {policy_kind} |
| top_p | {moe_top_p} |
| min experts | {moe_min} |
| max experts | {moe_max} |

---

## Co-residency

{coresidency_count} layer-local co-occurrence pairs recorded.

---

## Verification Plan

{verification_count} checks scheduled:

{verification_list}

---

## Output Files

{file_list}
"#,
        created_at = manifest.created_at,
        phase_provenance = phase_provenance,
        args_summary = args_summary,
        target = manifest.target,
        backend = runtime_profile.backend,
        cache_bytes = residency_plan.resident_cache_capacity_bytes,
        cache_bytes_gb = residency_plan.resident_cache_capacity_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        model_name = manifest.model_name,
        num_layers = layout.num_layers,
        num_experts_routed = total_expert_count,
        num_packed_layers = layout.packed_expert_layers.len(),
        num_shared = layout.shared_experts.len(),
        num_routers = layout.routers.len(),
        calibrated_count = calibrated_count,
        total_expert_count = total_expert_count,
        coverage_pct = coverage_pct,
        pruning_enabled = pruning_enabled,
        integrity_count = integrity_count,
        steering_count = steering_count,
        transport_count = transport_count,
        projection_count = projection_count,
        hot_count = hot_count,
        warm_count = warm_count,
        cold_count = cold_count,
        q8_count = q8_count,
        q5_count = q5_count,
        q4_count = q4_count,
        iq3_count = iq3_count,
        iq2_count = iq2_count,
        estimated_only_flag = estimated_only_flag,
        requires_verification_flag = requires_verification_flag,
        protect_count = pruning_plan.summary.protect_count,
        keep_count = pruning_plan.summary.keep_count,
        cold_tier_count = pruning_plan.summary.cold_tier_count,
        compress_count = pruning_plan.summary.compress_count,
        prune_candidate_count = pruning_plan.summary.prune_candidate_count,
        quality_budget = pruning_plan.summary.quality_budget,
        mass_threshold = pruning_plan.summary.mass_loss_threshold,
        mass_threshold_pct = pruning_plan.summary.mass_loss_threshold * 100.0,
        mass_loss = pruning_plan.summary.estimated_routing_mass_loss,
        mass_loss_pct = pruning_plan.summary.estimated_routing_mass_loss * 100.0,
        safe = pruning_plan.summary.safe,
        gates_text = gates_text,
        hot_expert_count = residency_plan.summary.initial_hot_expert_count,
        hot_expert_bytes_mb = residency_plan.summary.initial_hot_expert_bytes as f64 / (1024.0 * 1024.0),
        eviction_count = residency_plan.summary.eviction_priority_count,
        profile_name = runtime_profile.profile_name,
        profile_backend = runtime_profile.backend,
        policy_kind = runtime_profile.policy_kind,
        moe_top_p = runtime_profile.moe_top_p,
        moe_min = runtime_profile.moe_min_experts,
        moe_max = runtime_profile.moe_max_experts,
        coresidency_count = coresidency.pairs.len(),
        verification_count = verification_plan.checks.len(),
        verification_list = verification_plan.checks.iter()
            .map(|c| format!("- `{}`", c.kind))
            .collect::<Vec<_>>()
            .join("\n"),
        file_list = manifest.files.iter()
            .map(|f| format!("- `{f}`"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}
