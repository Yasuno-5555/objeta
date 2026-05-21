use std::error::Error;
use std::path::Path;

use crate::calibration_pass;
use crate::compile;
use crate::layout_pass;
use crate::pack_pass;
use crate::phase_pass;
use crate::precision_pass;
use crate::pruning_pass;
use crate::target::TargetHardware;
use crate::types::*;
use crate::util;
use crate::verification_pass;

/// Main entry point for the `specialize` subcommand.
pub fn run(args: &SpecializeArgs) -> Result<(), Box<dyn Error>> {
    println!("=== objeta-aot specialize v0.1 ===");
    let target_hw = TargetHardware::from_name(&args.target);
    let cache_capacity = args
        .memory_budget
        .unwrap_or(target_hw.recommended_cache_bytes);

    // ── Pass 1: Parse ─────────────────────────────────────────────────────
    println!("[1/11] ParsePass: reading model metadata...");
    let model_files = util::resolve_model_files(&args.model)?;
    let model_config: ModelConfig =
        serde_json::from_str(&std::fs::read_to_string(&model_files.config_path)?)?;
    let index: SafeTensorsIndex =
        serde_json::from_str(&std::fs::read_to_string(&model_files.index_path)?)?;
    let model_name = util::infer_model_name(&args.model, &model_config);
    let num_layers = model_config.effective_num_hidden_layers().unwrap_or(40);
    println!("  model={}, layers={}", model_name, num_layers);

    // ── Pass 2: Layout ────────────────────────────────────────────────────
    println!("[2/11] LayoutPass: building expert layout...");
    let expert_layout = layout_pass::run(&model_name, &model_config, &index)?;
    println!(
        "  {} routed experts, {} packed layers, {} shared, {} routers",
        expert_layout.logical_routed_expert_count,
        expert_layout.packed_expert_layers.len(),
        expert_layout.shared_experts.len(),
        expert_layout.routers.len()
    );

    // ── Pass 3: Calibration ───────────────────────────────────────────────
    println!("[3/11] CalibrationPass: analyzing calibration traces...");
    let calib_stats = calibration_pass::run(&args.calib)?;
    println!(
        "  {} total events, {} unique experts",
        calib_stats.total_events,
        calib_stats.importance.experts.len()
    );

    // ── Pass 4: Importance (3-term formula) ───────────────────────────────
    println!("[4/11] ImportancePass: computing expert importance...");
    let importance = calibration_pass::compute_importance_3term(&calib_stats);
    let hot = importance
        .experts
        .iter()
        .filter(|e| e.tier == ExpertTier::Hot)
        .count();
    let warm = importance
        .experts
        .iter()
        .filter(|e| e.tier == ExpertTier::Warm)
        .count();
    let cold = importance
        .experts
        .iter()
        .filter(|e| e.tier == ExpertTier::Cold)
        .count();
    println!("  {} hot, {} warm, {} cold", hot, warm, cold);

    // ── Pass 5: Co-residency ──────────────────────────────────────────────
    println!("[5/11] CoResidencyPass: extracting co-residency pairs...");
    let coresidency = &calib_stats.coresidency;
    println!("  {} pairs", coresidency.pairs.len());

    // ── Pass 6: Phase Analysis ────────────────────────────────────────────
    println!("[6/11] PhaseAnalysisPass: assigning LKO phases...");
    let phase_policy = phase_pass::run(num_layers);
    println!("  {} layer policies", phase_policy.layers.len());

    // ── Pass 7: Precision Plan ────────────────────────────────────────────
    println!("[7/11] PrecisionPlanPass: generating quant plan...");
    let quant_plan = precision_pass::run(&importance, &phase_policy, &expert_layout.routers, &target_hw);
    println!("  {} quant entries", quant_plan.entries.len());

    // ── Pass 8: Pruning Plan ──────────────────────────────────────────────
    println!("[8/11] PruningPlanPass: simulating pruning...");
    // Compute importance coverage: fraction of experts with calibration data
    let total_possible_experts = if expert_layout.logical_routed_expert_count > 0 {
        expert_layout.logical_routed_expert_count as usize
    } else {
        expert_layout.experts.len()
    };
    let importance_coverage = if total_possible_experts > 0 {
        importance.experts.len() as f64 / total_possible_experts as f64
    } else {
        0.0
    };
    let pruning_plan = pruning_pass::run(
        &importance,
        &phase_policy,
        &args.quality_budget,
        importance_coverage,
    );
    println!(
        "  {} prune candidates, {} compress, mass_loss={:.6}, safe={}",
        pruning_plan.summary.prune_candidate_count,
        pruning_plan.summary.compress_count,
        pruning_plan.summary.estimated_routing_mass_loss,
        pruning_plan.summary.safe
    );

    // ── Pass 9: Residency Plan ────────────────────────────────────────────
    println!("[9/11] ResidencyPlanPass: planning residency...");
    let residency_plan = compile::build_residency_plan(
        &expert_layout,
        &importance,
        coresidency,
        &args.target,
        cache_capacity,
        Some(target_hw.default_expert_bytes()),
    );
    println!(
        "  {} hot experts, {} bytes",
        residency_plan.summary.initial_hot_expert_count,
        residency_plan.summary.initial_hot_expert_bytes
    );

    // ── Pass 10: Runtime Profile ──────────────────────────────────────────
    println!("[10/11] RuntimeProfilePass: generating runtime profile...");
    let runtime_profile = RuntimeProfile {
        schema_version: 1,
        profile_name: format!(
            "{}-{}-{}-specialize",
            model_name, args.task_profile, args.target
        ),
        target: args.target.clone(),
        backend: target_hw.preferred_backend.clone(),
        policy_kind: "top_p".to_string(),
        moe_top_p: 0.90,
        moe_min_experts: 4,
        moe_max_experts: 8,
        resident_cache_capacity_bytes: cache_capacity,
        group_preresolve_top_n: 0,
        group_preresolve_max_bytes: 0,
        source_model: args.model.display().to_string(),
        source_calibration: Some(args.calib.display().to_string()),
    };

    // ── Pass 11: Verification Plan ────────────────────────────────────────
    println!("[11/11] VerificationPlanPass: generating verification plan...");
    let verification_plan = verification_pass::run(&args.quality_budget, &pruning_plan);

    // ── Pack Emit ─────────────────────────────────────────────────────────
    println!("PackEmitPass: writing specialization pack...");
    let manifest = Manifest {
        schema_version: 1,
        pack_type: "objeta_specialization_pack".to_string(),
        model_family: "qwen".to_string(),
        model_name: model_name.clone(),
        target: args.target.clone(),
        created_at: util::now_rfc3339ish(),
        files: vec![
            "expert_layout.json".to_string(),
            "expert_importance.json".to_string(),
            "expert_coresidency.json".to_string(),
            "phase_policy.json".to_string(),
            "quant_plan.json".to_string(),
            "pruning_plan.json".to_string(),
            "residency_plan.json".to_string(),
            "runtime_profile.json".to_string(),
            "verification_plan.json".to_string(),
            "specialization_report.md".to_string(),
        ],
        notes: format!(
            "Generated by objeta-aot specialize v0.1 (task={}, quality={}, target={})",
            args.task_profile, args.quality_budget, args.target
        ),
    };

    let args_summary = format!(
        "- model: `{}`\n- calib: `{}`\n- target: `{}`\n- task-profile: `{}`\n- quality-budget: `{}`\n- memory-budget: {} bytes\n- out: `{}`",
        args.model.display(),
        args.calib.display(),
        args.target,
        args.task_profile,
        args.quality_budget,
        cache_capacity,
        args.out.display(),
    );

    pack_pass::run(
        &args.out,
        &manifest,
        &expert_layout,
        &importance,
        coresidency,
        &phase_policy,
        &quant_plan,
        &pruning_plan,
        &residency_plan,
        &runtime_profile,
        &verification_plan,
        &args_summary,
        importance_coverage,
    )?;

    println!("\nWrote specialization pack: {}", args.out.display());
    println!("  model:     {}", model_name);
    println!("  target:    {}", args.target);
    println!("  task:      {}", args.task_profile);
    println!("  quality:   {}", args.quality_budget);
    println!("  cache:     {} bytes", cache_capacity);
    println!(
        "  experts:   {} hot / {} warm / {} cold",
        hot, warm, cold
    );
    println!(
        "  pruning:   {} prune / {} compress (mass_loss={:.6})",
        pruning_plan.summary.prune_candidate_count,
        pruning_plan.summary.compress_count,
        pruning_plan.summary.estimated_routing_mass_loss
    );

    Ok(())
}

/// Run specialize from within tests using synthetic data.
pub fn run_from_data(
    model: &Path,
    calib: &Path,
    target: &str,
    task_profile: &str,
    quality_budget: QualityBudget,
    out: &Path,
) -> Result<(), Box<dyn Error>> {
    let args = SpecializeArgs {
        model: model.to_path_buf(),
        calib: calib.to_path_buf(),
        target: target.to_string(),
        task_profile: task_profile.to_string(),
        memory_budget: None,
        quality_budget,
        out: out.to_path_buf(),
    };
    run(&args)
}
