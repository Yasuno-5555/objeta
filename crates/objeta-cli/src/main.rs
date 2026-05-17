//! objeta — MoE Runtime Compiler.
//!
//! Pipeline:
//!   1. `objeta analyze model/`  → phase_profile.json + stability_map.json
//!   2. Future: `objeta compile` → execution plan for MoE runtime

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "objeta", version = "1.0.0", about = "MoE Runtime Compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Static geometry analysis → phase_profile.json + stability_map.json
    Analyze {
        path: PathBuf,
        #[arg(short, long, default_value = "phase_profile.json")]
        output: PathBuf,
        #[arg(short, long)]
        verbose: bool,
        #[arg(long)]
        stability: bool,
    },
    /// MoE routing analysis for Qwen3.6 → execution_plan.json
    MoeAnalyze {
        /// Path to qwen36_bin directory
        path: PathBuf,
        #[arg(short, long, default_value = "execution_plan.json")]
        output: PathBuf,
        #[arg(long, default_value = "300")]
        samples: usize,
    },
    /// Family-aware runtime strategy → strategy.json
    Strategy {
        /// Path to phase_profile.json
        profile: PathBuf,
        #[arg(short, long, default_value = "strategy.json")]
        output: PathBuf,
    },
    /// Phase-adaptive quantization plan → quantization_plan.json
    Quantize {
        /// Path to phase_profile.json
        profile: PathBuf,
        #[arg(short, long, default_value = "quantization_plan.json")]
        output: PathBuf,
        /// Target average bits per weight (default: 4.0 = uniform q4 equivalent)
        #[arg(long, default_value = "4.0")]
        target_avg_bits: f64,
        /// Use static LKO-derived rules instead of Lyapunov-weighted
        #[arg(long)]
        static_rules: bool,
        /// Quantization strategy
        #[arg(long, default_value = "layerwise")]
        mode: String,
        /// Attention Q/O bits (for attention-backbone mode)
        #[arg(long, default_value = "5")]
        attn_qo_bits: u8,
        /// Attention K/V bits (for attention-backbone mode)
        #[arg(long, default_value = "4")]
        attn_kv_bits: u8,
        /// FFN bits (for attention-backbone mode)
        #[arg(long, default_value = "3.5")]
        ffn_bits: f64,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("OBJETA_LOG").unwrap_or_else(|_| "info".to_string())
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Analyze { path, output, verbose, stability } => {
            cmd_analyze(&path, &output, verbose, stability)?;
        }
        Command::MoeAnalyze { path, output, samples } => {
            objeta_moe::run(&path, &output, samples)?;
        }
        Command::Quantize { profile, output, target_avg_bits, static_rules, mode,
                             attn_qo_bits, attn_kv_bits, ffn_bits } => {
            cmd_quantize(&profile, &output, target_avg_bits, static_rules,
                         &mode, attn_qo_bits, attn_kv_bits, ffn_bits)?;
        }
        Command::Strategy {
            profile, output,
        } => {
            cmd_strategy(&profile, &output)?;
        }
    }
    Ok(())
}

fn cmd_analyze(
    model_path: &PathBuf, output_path: &PathBuf, verbose: bool, stability: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use objeta_analysis::analyze_model;
    use objeta_runtime::generate_precision_config;

    let report = analyze_model(model_path.to_str().unwrap())?;
    let profile = &report.profile;

    let json = serde_json::to_string_pretty(profile)?;
    std::fs::write(output_path, &json)?;
    println!("Wrote: {}", output_path.display());

    if stability {
        let config = generate_precision_config(profile);
        let stability_path = output_path.with_file_name("stability_map.json");
        let sjson = serde_json::to_string_pretty(&config)?;
        std::fs::write(&stability_path, &sjson)?;
        println!("Wrote: {}", stability_path.display());
    }

    if verbose {
        print_report(profile);
    } else {
        print_summary(profile);
    }
    Ok(())
}

fn print_summary(profile: &objeta_core::PhaseProfile) {
    println!();
    println!("{}", "=".repeat(60));
    println!("  objeta analyze — {}", profile.model_name);
    println!("{}", "=".repeat(60));
    println!();
    println!("  Phase:           {:?}", profile.phase);
    println!("  Family:          {:?}", profile.family);
    println!("  Layers:          {}", profile.n_layers);
    println!("  Hidden dim:      {}", profile.hidden_dim);
    println!("  FFN dim:         {}", profile.ffn_dim);
    println!("  Eff rank:        {:.1}", profile.ffn_compression_ratio * profile.ffn_dim as f64);
    println!("  Coupling std:    {:.4}", profile.coupling_strength);

    if let Some(onset) = profile.inversion_onset {
        println!("  Inversion onset: L{}", onset);
    }
    if let Some(ra) = profile.realignment_onset {
        println!("  Realignment:     L{}", ra);
    }
    if !profile.inversion_layers.is_empty() {
        println!("  Inversion zone:  L{}-L{}",
                 profile.inversion_layers.first().unwrap(),
                 profile.inversion_layers.last().unwrap());
    }
    println!();
    for policy in &profile.zone_policies {
        if policy.layers.is_empty() { continue; }
        println!("  {:?}: L{}-L{} (precision={}bit, critical={}, full_attn={})",
                 policy.zone,
                 policy.layers.first().unwrap(),
                 policy.layers.last().unwrap(),
                 policy.min_precision_bits,
                 policy.stability_critical,
                 policy.force_full_attention,
        );
    }
    println!();
    println!("  Refresh layers:  {:?}", profile.refresh_layers);
    println!();
}

fn cmd_quantize(
    profile_path: &PathBuf, output_path: &PathBuf, target_avg_bits: f64, static_rules: bool,
    mode: &str, attn_qo_bits: u8, attn_kv_bits: u8, ffn_bits: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(profile_path)?;
    let profile: objeta_core::PhaseProfile = serde_json::from_str(&json)?;

    match mode {
        "attention-backbone" => {
            let plan = objeta_quantize::generate_attention_backbone_with_params(
                &profile, attn_qo_bits, attn_kv_bits, ffn_bits);

            let out_json = serde_json::to_string_pretty(&plan)?;
            std::fs::write(output_path, &out_json)?;
            println!("Wrote: {}", output_path.display());
            println!();
            println!("{}", "=".repeat(60));
            println!("  objeta quantize — Attention Backbone");
            println!("{}", "=".repeat(60));
            println!();
            println!("  Model:     {}", plan.model_name);
            println!("  Phase:     {}", plan.phase);
            println!("  Family:    {}", plan.family);
            println!("  Avg bits:  {:.2}", plan.average_bits);
            println!("  Ratio:     {:.1}x", plan.compression_ratio);
            println!();
            println!("  Component precision:");
            println!("    Attention Q/O:  {}bit ({})", plan.layers[0].attn_qo_bits, plan.layers[0].attn_qo_format);
            println!("    Attention K/V:  {}bit ({})", plan.layers[0].attn_kv_bits, plan.layers[0].attn_kv_format);
            println!("    FFN:            {:.1}bit ({})", ffn_bits, plan.layers[0].ffn_format);
            println!();
            println!("  Weight fractions:");
            println!("    FFN:     {:.0}%", plan.ffn_weight_fraction * 100.0);
            println!("    Attn QO: {:.0}%", plan.attn_qo_weight_fraction * 100.0);
            println!("    Attn KV: {:.0}%", plan.attn_kv_weight_fraction * 100.0);
            println!();
            println!("  Total size:  {:.1} MB", plan.total_bytes as f64 / 1_000_000.0);
            println!("  fp16 size:   {:.1} MB", plan.fp16_bytes as f64 / 1_000_000.0);
            println!("  Compression: {:.1}x", plan.compression_ratio);
            println!();
            println!("  {:<4} {:>12} {:>8} {:>8} {:>8}",
                "L", "zone", "FFN", "AttnQO", "AttnKV");
            println!("  {:-<4} {:-<12} {:-<8} {:-<8} {:-<8}", "", "", "", "", "");
            for cq in &plan.layers {
                println!("  L{:<3} {:>12} {:>5}bit {:>5}bit {:>5}bit",
                    cq.layer_idx, cq.zone, cq.ffn_bits, cq.attn_qo_bits, cq.attn_kv_bits);
            }
        }
        _ => {
            let budget = objeta_quantize::BitBudget {
                target_avg_bits,
                ..Default::default()
            };

            let plan = if static_rules {
                objeta_quantize::generate_static_plan(&profile)
            } else {
                objeta_quantize::generate_plan_with_budget(&profile, &budget)
            };

            let savings = objeta_quantize::compute_savings(&plan);

            let out_json = serde_json::to_string_pretty(&plan)?;
            std::fs::write(output_path, &out_json)?;
            println!("Wrote: {}", output_path.display());
            println!();
            println!("{}", "=".repeat(60));
            println!("  objeta quantize — {}", plan.model_name);
            println!("{}", "=".repeat(60));
            println!();
            println!("  Phase:     {}", plan.phase);
            println!("  Family:    {}", plan.family);
            println!("  Avg bits:  {:.2}", plan.average_bits);
            println!("  Ratio:     {:.1}x", plan.compression_ratio);
            println!();
            println!("{}", savings);
            println!();
            println!("  {:<4} {:>10} {:>10} {:>6} {:>8}  {}",
                "L", "zone", "lyapunov", "bits", "sens", "format");
            println!("  {:-<4} {:-<10} {:-<10} {:-<6} {:-<8}  {:-<20}",
                "", "", "", "", "", "");
            for lq in &plan.layers {
                println!("  L{:<3} {:>10} {:>10.2} {:>4}bit {:>8.3}  {}",
                    lq.layer_idx, lq.zone, lq.lyapunov, lq.bits, lq.sensitivity, lq.format);
            }
        }
    }
    Ok(())
}

fn cmd_strategy(
    profile_path: &PathBuf, output_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(profile_path)?;
    let profile: objeta_core::PhaseProfile = serde_json::from_str(&json)?;

    let strategy = objeta_quantize::generate_runtime_strategy(&profile);

    let out_json = serde_json::to_string_pretty(&strategy)?;
    std::fs::write(output_path, &out_json)?;
    println!("Wrote: {}", output_path.display());
    println!();
    println!("{}", "=".repeat(60));
    println!("  objeta strategy — Family-Aware Runtime");
    println!("{}", "=".repeat(60));
    println!();
    println!("  Model:      {}", strategy.model_name);
    println!("  Family:     {:?}", strategy.family);
    println!("  Phase:      {:?}", strategy.phase);
    println!("  Dominance:  {:?}", strategy.dominance);
    println!("  Confidence: {:.0}%", strategy.confidence * 100.0);
    println!();
    println!("  {}", strategy.description);
    println!();
    println!("  Component precision:");
    println!("    Attn Q/O: {}bit", strategy.component_precision.attn_qo_bits);
    println!("    Attn K/V: {}bit", strategy.component_precision.attn_kv_bits);
    println!("    FFN:      {}bit", strategy.component_precision.ffn_bits);
    println!("    Avg:      {:.1}bit", strategy.component_precision.average_bits);
    println!("    Ratio:    {:.1}x", strategy.component_precision.compression_ratio);
    println!();
    println!("  Estimated performance:");
    let ec = &strategy.executor_config;
    println!("    tok/s:   {:.1}", ec.estimated_tok_per_sec);
    println!("    VRAM:    {:.1} GB", ec.estimated_vram_gb);
    println!("    ΔPPL:    +{:.1}", ec.estimated_ppl_delta);
    println!();
    println!("  Runtime config:");
    println!("    fusion_ratio:      {:.2}", ec.fusion_ratio);
    println!("    moe_on_deltanet:   {}", ec.moe_on_deltanet);
    println!();
    if !strategy.steering_layers.is_empty() {
        println!("  Steering layers: {:?}", strategy.steering_layers);
        println!();
    }
    println!("  Executor-ready config:");
    println!("    ffn_bits:     {:?}", &ec.ffn_bits[..ec.ffn_bits.len().min(8)]);
    println!("    attn_qo_bits: {:?}", &ec.attn_qo_bits[..ec.attn_qo_bits.len().min(8)]);
    println!("    attn_kv_bits: {:?}", &ec.attn_kv_bits[..ec.attn_kv_bits.len().min(8)]);
    Ok(())
}

fn print_report(profile: &objeta_core::PhaseProfile) {
    print_summary(profile);
    println!("  {:<4} {:>10} {:>10} {:>10} {:>12}",
             "L", "steer_cos", "intra_cos", "eff_rank", "lyapunov");
    println!("  {:-<4} {:-<10} {:-<10} {:-<10} {:-<12}", "", "", "", "", "");
    for layer in &profile.layers {
        let sc = layer.steering_cos.map_or("    —".to_string(), |v| format!("{:+.4}", v));
        let ic = layer.intra_cos.map_or("    —".to_string(), |v| format!("{:+.4}", v));
        let ly = layer.lyapunov_estimate.map_or("    —".to_string(), |v| format!("{:.4}", v));
        let z = match layer.zone {
            Some(objeta_core::LayerZone::Sync) => "S",
            Some(objeta_core::LayerZone::Unfold) => "U",
            Some(objeta_core::LayerZone::IsometricLocal) => "I",
            Some(objeta_core::LayerZone::IsometricGlobal) => "G",
            Some(objeta_core::LayerZone::Divergent) => "D",
            None => "?",
        };
        println!("  L{:<3} {:>10} {:>10} {:>10.1} {:>12}  {}",
                 layer.layer_idx, sc, ic, layer.effective_rank, ly, z);
    }
}
