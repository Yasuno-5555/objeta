use clap::{Parser, Subcommand, ValueEnum};
use std::error::Error;
use std::path::PathBuf;

mod calibration_pass;
mod compile;
mod layout_pass;
mod pack_pass;
mod phase_pass;
mod precision_pass;
mod pruning_pass;
mod specialize;
mod target;
mod types;
mod util;
mod verification_pass;

#[derive(Parser)]
#[command(
    name = "objeta-aot",
    version,
    about = "Ahead-of-time runtime pack compiler and model specialization planner for Objeta"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile a runtime pack from checkpoint metadata and optional calibration traces.
    Compile {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        calib: Option<PathBuf>,
        #[arg(long, default_value = "m1-8gb")]
        target: String,
        #[arg(long)]
        cache_capacity: Option<u64>,
        #[arg(long)]
        expert_bytes: Option<u64>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
    },
    /// Generate a specialization plan from checkpoint + calibration traces + target hardware.
    Specialize {
        /// Path to model checkpoint directory or .safetensors.index.json
        #[arg(long)]
        model: PathBuf,
        /// Path to calibration trace (JSONL or moe_stats.json)
        #[arg(long)]
        calib: PathBuf,
        /// Target hardware profile (e.g. m1-8gb, rtx3070-8gb-vram-32gb-ram, cpu-32gb)
        #[arg(long, default_value = "m1-8gb")]
        target: String,
        /// Task profile label (e.g. general, coding, japanese-chat, reasoning)
        #[arg(long, default_value = "general")]
        task_profile: String,
        /// Memory budget for expert cache (e.g. 3GB, 8GB, or raw bytes)
        #[arg(long)]
        memory_budget: Option<String>,
        /// Quality budget: conservative, balanced, or aggressive
        #[arg(long, default_value = "conservative")]
        quality_budget: String,
        /// Output directory for the specialization pack
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Json,
    Compact,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Compile {
            model,
            calib,
            target,
            cache_capacity,
            expert_bytes,
            out,
            format,
        } => compile::compile_runtime_pack(
            &model,
            calib.as_deref(),
            &target,
            cache_capacity,
            expert_bytes,
            &out,
            format,
        )?,
        Command::Specialize {
            model,
            calib,
            target,
            task_profile,
            memory_budget,
            quality_budget,
            out,
        } => {
            let qb = types::QualityBudget::from_str_loose(&quality_budget).ok_or_else(|| {
                format!(
                    "invalid quality-budget '{}': expected conservative, balanced, or aggressive",
                    quality_budget
                )
            })?;
            let mem = memory_budget
                .as_deref()
                .and_then(util::parse_bytes_human);
            let args = types::SpecializeArgs {
                model,
                calib,
                target,
                task_profile,
                memory_budget: mem,
                quality_budget: qb,
                out,
            };
            specialize::run(&args)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use types::*;

    fn parsed(name: &str) -> ParsedTensorName {
        layout_pass::parse_qwen_tensor_name(name, "model-00001-of-00002.safetensors")
    }

    #[test]
    fn parse_gate_up_down_proj_names() {
        let gate = parsed("model.layers.31.mlp.experts.42.gate_proj.weight");
        assert_eq!(gate.layer_idx, Some(31));
        assert_eq!(gate.expert_id, Some(42));
        assert_eq!(gate.tensor_kind, ExpertTensorKind::Gate);

        let up = parsed("model.layers.31.mlp.experts.42.up_proj.weight");
        assert_eq!(up.tensor_kind, ExpertTensorKind::Up);

        let down = parsed("model.layers.31.mlp.experts.42.down_proj.weight");
        assert_eq!(down.tensor_kind, ExpertTensorKind::Down);
    }

    #[test]
    fn parse_fused_gate_up_proj_name() {
        let fused = parsed("model.layers.7.mlp.experts.103.gate_up_proj.weight");
        assert_eq!(fused.layer_idx, Some(7));
        assert_eq!(fused.expert_id, Some(103));
        assert_eq!(fused.tensor_kind, ExpertTensorKind::GateUp);
    }

    #[test]
    fn parse_packed_gate_up_and_down_names() {
        let gate_up = parsed("model.language_model.layers.31.mlp.experts.gate_up_proj");
        assert_eq!(gate_up.layer_idx, Some(31));
        assert_eq!(gate_up.expert_id, None);
        assert!(gate_up.is_packed_experts);
        assert_eq!(gate_up.tensor_kind, ExpertTensorKind::PackedGateUp);

        let down = parsed("model.language_model.layers.31.mlp.experts.down_proj");
        assert_eq!(down.layer_idx, Some(31));
        assert_eq!(down.expert_id, None);
        assert!(down.is_packed_experts);
        assert_eq!(down.tensor_kind, ExpertTensorKind::PackedDown);
    }

    #[test]
    fn router_is_not_treated_as_expert() {
        let router = parsed("model.layers.12.mlp.gate.weight");
        assert_eq!(router.layer_idx, Some(12));
        assert_eq!(router.expert_id, None);
        assert_eq!(router.tensor_kind, ExpertTensorKind::Router);
    }

    #[test]
    fn shared_expert_is_not_normal_routed_expert() {
        let shared = parsed("model.layers.5.mlp.shared_expert.gate_proj.weight");
        assert_eq!(shared.layer_idx, Some(5));
        assert_eq!(shared.expert_id, None);
        assert!(shared.is_shared);
        assert_eq!(shared.tensor_kind, ExpertTensorKind::Gate);
    }

    #[test]
    fn unknown_tensor_falls_back_to_unknown() {
        let unknown = parsed("model.layers.3.some_other_tensor.weight");
        assert_eq!(unknown.layer_idx, Some(3));
        assert_eq!(unknown.tensor_kind, ExpertTensorKind::Unknown);
    }

    #[test]
    fn nested_text_config_fields_are_read() {
        let cfg: ModelConfig = serde_json::from_str(
            r#"{
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {
                    "num_hidden_layers": 40,
                    "num_experts": 256,
                    "vocab_size": 248320,
                    "hidden_size": 2048
                }
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.effective_num_hidden_layers(), Some(40));
        assert_eq!(cfg.effective_num_experts(), Some(256));
        assert_eq!(cfg.effective_vocab_size(), Some(248320));
        assert_eq!(cfg.effective_hidden_size(), Some(2048));
    }

    #[test]
    fn mock_index_generates_expected_expert_layout() {
        let config = ModelConfig {
            model_type: Some("qwen3_moe".to_string()),
            architectures: Some(vec!["Qwen3MoeForCausalLM".to_string()]),
            num_hidden_layers: Some(40),
            num_experts: Some(256),
            num_local_experts: None,
            vocab_size: Some(151936),
            hidden_size: None,
            intermediate_size: None,
            text_config: None,
        };
        let index = SafeTensorsIndex {
            weight_map: HashMap::from([
                (
                    "model.layers.0.mlp.experts.42.gate_proj.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ),
                (
                    "model.layers.0.mlp.experts.42.up_proj.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ),
                (
                    "model.layers.0.mlp.experts.42.down_proj.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ),
                (
                    "model.layers.0.mlp.gate.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ),
                (
                    "model.layers.0.mlp.shared_expert.gate_up_proj.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ),
                (
                    "model.layers.0.mlp.shared_expert.down_proj.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ),
                (
                    "model.layers.0.unknown_blob.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ),
            ]),
            metadata: SafeTensorsIndexMetadata {
                total_size: Some(1234),
            },
        };

        let layout = layout_pass::run("qwen36", &config, &index).unwrap();
        assert_eq!(layout.num_layers, 40);
        assert_eq!(layout.num_experts, 256);
        assert_eq!(layout.experts.len(), 1);
        assert_eq!(layout.layout_kind, ExpertLayoutKind::PerExpert);
        assert_eq!(layout.logical_routed_expert_count, 40 * 256);
        assert!(layout.packed_expert_layers.is_empty());
        assert_eq!(layout.shared_experts.len(), 1);
        assert_eq!(layout.routers.len(), 1);
        assert_eq!(layout.unknown_tensors.len(), 1);
        let expert = &layout.experts[0];
        assert_eq!(expert.layer, 0);
        assert_eq!(expert.expert, 42);
        assert!(expert.complete);
        assert!(expert.gate.is_some());
        assert!(expert.up.is_some());
        assert!(expert.down.is_some());
    }

    #[test]
    fn packed_layout_generates_logical_expert_count() {
        let config: ModelConfig = serde_json::from_str(
            r#"{
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {
                    "num_hidden_layers": 40,
                    "num_experts": 256,
                    "vocab_size": 248320
                }
            }"#,
        )
        .unwrap();
        let index = SafeTensorsIndex {
            weight_map: HashMap::from([
                (
                    "model.language_model.layers.31.mlp.experts.gate_up_proj".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ),
                (
                    "model.language_model.layers.31.mlp.experts.down_proj".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ),
            ]),
            metadata: SafeTensorsIndexMetadata {
                total_size: Some(1234),
            },
        };
        let layout = layout_pass::run("qwen36", &config, &index).unwrap();
        assert_eq!(layout.layout_kind, ExpertLayoutKind::PackedExperts);
        assert_eq!(layout.logical_routed_expert_count, 10240);
        assert_eq!(layout.packed_expert_layers.len(), 1);
        assert_eq!(layout.experts.len(), 0);
        assert!(layout.warnings.is_empty());
    }

    #[test]
    fn compile_generates_real_expert_layout_from_mock_model_dir() {
        let root =
            std::env::temp_dir().join(format!("objeta_aot_mock_{}", util::now_rfc3339ish()));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            serde_json::json!({
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 40,
                "num_local_experts": 256,
                "vocab_size": 151936
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": {"total_size": 999},
                "weight_map": {
                    "model.layers.3.mlp.experts.7.gate_up_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.3.mlp.experts.7.down_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.3.mlp.gate.weight": "model-00001-of-00002.safetensors"
                }
            })
            .to_string(),
        )
        .unwrap();
        let out = root.join("out_pack");
        compile::compile_runtime_pack(&root, None, "m1-8gb", None, None, &out, OutputFormat::Json)
            .unwrap();
        let layout: ExpertLayout =
            serde_json::from_str(&fs::read_to_string(out.join("expert_layout.json")).unwrap())
                .unwrap();
        assert_eq!(layout.experts.len(), 1);
        assert_eq!(layout.experts[0].layer, 3);
        assert_eq!(layout.experts[0].expert, 7);
        assert!(layout.experts[0].gate_up.is_some());
        assert!(layout.experts[0].down.is_some());
        assert_eq!(layout.routers.len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn parse_jsonl_calibration_event() {
        let root =
            std::env::temp_dir().join(format!("objeta_aot_calib_{}", util::now_rfc3339ish()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("calib_trace.jsonl");
        fs::write(
            &path,
            r#"{"token_id":0,"layer":31,"selected_experts":[42,7,103],"selected_weights":[0.31,0.22,0.14]}"#,
        )
        .unwrap();
        let events = calibration_pass::load_calibration_events(&path).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].token_id, Some(0));
        assert_eq!(events[0].layer, 31);
        assert_eq!(events[0].selected_experts, vec![42, 7, 103]);
        assert_eq!(events[0].selected_weights, vec![0.31, 0.22, 0.14]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn aggregate_selected_count_and_avg_gate_weight() {
        let root =
            std::env::temp_dir().join(format!("objeta_aot_stats_{}", util::now_rfc3339ish()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("calib_trace.jsonl");
        fs::write(
            &path,
            [
                r#"{"token_id":0,"layer":31,"selected_experts":[42,7],"selected_weights":[0.30,0.20]}"#,
                r#"{"token_id":1,"layer":31,"selected_experts":[42,8],"selected_weights":[0.10,0.05]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let stats = calibration_pass::run(&path).unwrap();
        let e42 = stats
            .importance
            .experts
            .iter()
            .find(|e| e.layer == 31 && e.expert == 42)
            .unwrap();
        assert_eq!(e42.selected_count, 2);
        assert!((e42.avg_gate_weight - 0.20).abs() < 1e-6);
        assert!((e42.max_gate_weight - 0.30).abs() < 1e-6);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn importance_ordering_from_synthetic_trace() {
        let root =
            std::env::temp_dir().join(format!("objeta_aot_rank_{}", util::now_rfc3339ish()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("calib_trace.jsonl");
        fs::write(
            &path,
            [
                r#"{"token_id":0,"layer":2,"selected_experts":[1,2],"selected_weights":[0.50,0.10]}"#,
                r#"{"token_id":1,"layer":2,"selected_experts":[1,3],"selected_weights":[0.40,0.20]}"#,
                r#"{"token_id":2,"layer":2,"selected_experts":[2,3],"selected_weights":[0.10,0.05]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let stats = calibration_pass::run(&path).unwrap();
        let mut layer2: Vec<_> = stats
            .importance
            .experts
            .iter()
            .filter(|e| e.layer == 2)
            .collect();
        layer2.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert_eq!(layer2[0].expert, 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn hot_warm_cold_tier_assignment() {
        let root =
            std::env::temp_dir().join(format!("objeta_aot_tiers_{}", util::now_rfc3339ish()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("calib_trace.jsonl");
        let mut lines = Vec::new();
        for expert in 0..10u32 {
            lines.push(format!(
                r#"{{"token_id":{},"layer":5,"selected_experts":[{}],"selected_weights":[{}]}}"#,
                expert,
                expert,
                1.0 - (expert as f32 * 0.05)
            ));
        }
        fs::write(&path, lines.join("\n")).unwrap();
        let stats = calibration_pass::run(&path).unwrap();
        let layer5: Vec<_> = stats.importance.experts.iter().filter(|e| e.layer == 5).collect();
        let hot = layer5.iter().filter(|e| e.tier == ExpertTier::Hot).count();
        let warm = layer5.iter().filter(|e| e.tier == ExpertTier::Warm).count();
        let cold = layer5.iter().filter(|e| e.tier == ExpertTier::Cold).count();
        assert_eq!(hot, 1);
        assert_eq!(warm, 3);
        assert_eq!(cold, 6);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn coresidency_pair_counting() {
        let root =
            std::env::temp_dir().join(format!("objeta_aot_pairs_{}", util::now_rfc3339ish()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("calib_trace.jsonl");
        fs::write(
            &path,
            [
                r#"{"token_id":0,"layer":9,"selected_experts":[1,2,3],"selected_weights":[0.4,0.3,0.2]}"#,
                r#"{"token_id":1,"layer":9,"selected_experts":[1,2],"selected_weights":[0.5,0.2]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let stats = calibration_pass::run(&path).unwrap();
        let pair = stats
            .coresidency
            .pairs
            .iter()
            .find(|p| p.layer == 9 && p.expert_a == 1 && p.expert_b == 2)
            .unwrap();
        assert_eq!(pair.co_count, 2);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn compile_with_synthetic_calib_produces_non_placeholder_outputs() {
        let root =
            std::env::temp_dir().join(format!("objeta_aot_full_{}", util::now_rfc3339ish()));
        let model = root.join("mock_model");
        fs::create_dir_all(&model).unwrap();
        fs::write(
            model.join("config.json"),
            serde_json::json!({
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 40,
                "num_local_experts": 256,
                "vocab_size": 151936
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            model.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": {"total_size": 999},
                "weight_map": {
                    "model.layers.31.mlp.experts.42.gate_up_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.31.mlp.experts.42.down_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.31.mlp.experts.7.gate_up_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.31.mlp.experts.7.down_proj.weight": "model-00001-of-00002.safetensors"
                }
            })
            .to_string(),
        )
        .unwrap();
        let calib = root.join("calib_trace.jsonl");
        fs::write(
            &calib,
            [
                r#"{"token_id":0,"layer":31,"selected_experts":[42,7,103],"selected_weights":[0.31,0.22,0.14]}"#,
                r#"{"token_id":1,"layer":31,"selected_experts":[42,7],"selected_weights":[0.20,0.10]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let out = root.join("out_pack");
        compile::compile_runtime_pack(
            &model,
            Some(&calib),
            "m1-8gb",
            None,
            None,
            &out,
            OutputFormat::Json,
        )
        .unwrap();

        let importance: ExpertImportance = serde_json::from_str(
            &fs::read_to_string(out.join("expert_importance.json")).unwrap(),
        )
        .unwrap();
        let coresidency: ExpertCoresidency = serde_json::from_str(
            &fs::read_to_string(out.join("expert_coresidency.json")).unwrap(),
        )
        .unwrap();
        assert!(!importance.experts.is_empty());
        assert!(!coresidency.pairs.is_empty());
        assert!(importance.experts.iter().any(|e| e.expert == 42));
        assert!(coresidency
            .pairs
            .iter()
            .any(|p| p.layer == 31 && p.expert_a == 7 && p.expert_b == 42));
        let residency_plan: ResidencyPlan =
            serde_json::from_str(&fs::read_to_string(out.join("residency_plan.json")).unwrap())
                .unwrap();
        assert!(!residency_plan.initial_hot_experts.is_empty());
        let hot_sum: u64 = residency_plan.initial_hot_experts.iter().map(|e| e.bytes).sum();
        assert!(hot_sum <= residency_plan.resident_cache_capacity_bytes);
        let runtime_profile: RuntimeProfile =
            serde_json::from_str(&fs::read_to_string(out.join("runtime_profile.json")).unwrap())
                .unwrap();
        assert_eq!(runtime_profile.backend, "fused_row_parallel");
        assert_eq!(runtime_profile.policy_kind, "top_p");
        assert_eq!(
            runtime_profile.resident_cache_capacity_bytes,
            residency_plan.resident_cache_capacity_bytes
        );
        let _ = fs::remove_dir_all(root);
    }

    fn importance_entry(
        layer: u32,
        expert: u32,
        importance: f64,
        tier: ExpertTier,
        selected_count: u64,
        avg_gate_weight: f64,
    ) -> ExpertImportanceEntry {
        ExpertImportanceEntry {
            layer,
            expert,
            selected_count,
            frequency: 0.0,
            avg_gate_weight,
            max_gate_weight: avg_gate_weight,
            importance,
            tier,
            recommended_format: "q4".to_string(),
            eviction_priority: 1.0 - importance,
        }
    }

    fn routed_expert(layer: u32, expert: u32, byte_len: Option<u64>) -> ExpertEntry {
        ExpertEntry {
            layer,
            expert,
            gate: Some(TensorRef {
                tensor_kind: ExpertTensorKind::Gate,
                tensor_name: format!("gate.{layer}.{expert}"),
                source_file: "mock.safetensors".to_string(),
                shape: None,
                dtype: None,
                byte_offset: None,
                byte_len,
            }),
            up: Some(TensorRef {
                tensor_kind: ExpertTensorKind::Up,
                tensor_name: format!("up.{layer}.{expert}"),
                source_file: "mock.safetensors".to_string(),
                shape: None,
                dtype: None,
                byte_offset: None,
                byte_len: None,
            }),
            gate_up: None,
            down: Some(TensorRef {
                tensor_kind: ExpertTensorKind::Down,
                tensor_name: format!("down.{layer}.{expert}"),
                source_file: "mock.safetensors".to_string(),
                shape: None,
                dtype: None,
                byte_offset: None,
                byte_len: None,
            }),
            source_files: vec!["mock.safetensors".to_string()],
            complete: true,
        }
    }

    #[test]
    fn residency_plan_respects_capacity_invariant() {
        let layout = ExpertLayout {
            schema_version: 1, model: "qwen36".to_string(), model_type: None,
            architectures: vec![], num_layers: 1, num_experts: 3, vocab_size: None,
            layout_kind: ExpertLayoutKind::PerExpert, logical_routed_expert_count: 3,
            experts: vec![routed_expert(0, 1, Some(100)), routed_expert(0, 2, Some(100)), routed_expert(0, 3, Some(100))],
            packed_expert_layers: vec![],
            shared_experts: vec![], routers: vec![], unknown_tensors: vec![], warnings: vec![],
        };
        let importance = ExpertImportance { schema_version: 1, experts: vec![
            importance_entry(0, 1, 0.95, ExpertTier::Hot, 10, 0.3),
            importance_entry(0, 2, 0.75, ExpertTier::Warm, 8, 0.2),
            importance_entry(0, 3, 0.10, ExpertTier::Cold, 1, 0.1),
        ]};
        let plan = compile::build_residency_plan(&layout, &importance, &ExpertCoresidency { schema_version: 1, pairs: vec![] }, "m1-8gb", 200, None);
        let sum: u64 = plan.initial_hot_experts.iter().map(|e| e.bytes).sum();
        assert!(sum <= 200);
        assert_eq!(plan.initial_hot_experts.len(), 2);
    }

    #[test]
    fn hot_experts_selected_before_warm_and_cold() {
        let layout = ExpertLayout {
            schema_version: 1, model: "qwen36".to_string(), model_type: None,
            architectures: vec![], num_layers: 1, num_experts: 3, vocab_size: None,
            layout_kind: ExpertLayoutKind::PerExpert, logical_routed_expert_count: 3,
            experts: vec![routed_expert(0, 1, Some(64)), routed_expert(0, 2, Some(64)), routed_expert(0, 3, Some(64))],
            packed_expert_layers: vec![],
            shared_experts: vec![], routers: vec![], unknown_tensors: vec![], warnings: vec![],
        };
        let importance = ExpertImportance { schema_version: 1, experts: vec![
            importance_entry(0, 1, 0.90, ExpertTier::Hot, 10, 0.3),
            importance_entry(0, 2, 0.50, ExpertTier::Warm, 5, 0.2),
            importance_entry(0, 3, 0.20, ExpertTier::Cold, 2, 0.1),
        ]};
        let plan = compile::build_residency_plan(&layout, &importance, &ExpertCoresidency { schema_version: 1, pairs: vec![] }, "m1-8gb", 128, None);
        let selected: Vec<u32> = plan.initial_hot_experts.iter().map(|e| e.expert).collect();
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn low_importance_cold_expert_evicted_before_hot_expert() {
        let layout = ExpertLayout {
            schema_version: 1, model: "qwen36".to_string(), model_type: None,
            architectures: vec![], num_layers: 1, num_experts: 3, vocab_size: None,
            layout_kind: ExpertLayoutKind::PerExpert, logical_routed_expert_count: 3,
            experts: vec![routed_expert(0, 1, Some(64)), routed_expert(0, 2, Some(64)), routed_expert(0, 3, Some(64))],
            packed_expert_layers: vec![],
            shared_experts: vec![], routers: vec![], unknown_tensors: vec![], warnings: vec![],
        };
        let importance = ExpertImportance { schema_version: 1, experts: vec![
            importance_entry(0, 1, 0.95, ExpertTier::Hot, 10, 0.3),
            importance_entry(0, 2, 0.45, ExpertTier::Warm, 5, 0.2),
            importance_entry(0, 3, 0.05, ExpertTier::Cold, 1, 0.1),
        ]};
        let plan = compile::build_residency_plan(&layout, &importance, &ExpertCoresidency { schema_version: 1, pairs: vec![] }, "m1-8gb", 128, None);
        assert_eq!(plan.eviction_priority.first().map(|e| e.expert), Some(3));
        assert_eq!(plan.eviction_priority.last().map(|e| e.expert), Some(1));
    }

    #[test]
    fn missing_byte_len_uses_estimated_expert_bytes() {
        let layout = ExpertLayout {
            schema_version: 1, model: "qwen36".to_string(), model_type: None,
            architectures: vec![], num_layers: 1, num_experts: 1, vocab_size: None,
            layout_kind: ExpertLayoutKind::PerExpert, logical_routed_expert_count: 1,
            experts: vec![routed_expert(0, 1, None)],
            packed_expert_layers: vec![],
            shared_experts: vec![], routers: vec![], unknown_tensors: vec![], warnings: vec![],
        };
        let importance = ExpertImportance { schema_version: 1, experts: vec![
            importance_entry(0, 1, 0.9, ExpertTier::Hot, 10, 0.3),
        ]};
        let plan = compile::build_residency_plan(&layout, &importance, &ExpertCoresidency { schema_version: 1, pairs: vec![] }, "m1-8gb", 1024, Some(512));
        assert_eq!(plan.initial_hot_experts[0].bytes, 512);
        assert_eq!(plan.initial_hot_experts[0].bytes_source, "expert_bytes_override");
    }

    #[test]
    fn cache_capacity_override_works() {
        let layout = ExpertLayout {
            schema_version: 1, model: "qwen36".to_string(), model_type: None,
            architectures: vec![], num_layers: 1, num_experts: 4, vocab_size: None,
            layout_kind: ExpertLayoutKind::PerExpert, logical_routed_expert_count: 4,
            experts: vec![routed_expert(0, 1, Some(100)), routed_expert(0, 2, Some(100)), routed_expert(0, 3, Some(100)), routed_expert(0, 4, Some(100))],
            packed_expert_layers: vec![],
            shared_experts: vec![], routers: vec![], unknown_tensors: vec![], warnings: vec![],
        };
        let importance = ExpertImportance { schema_version: 1, experts: vec![
            importance_entry(0, 1, 0.9, ExpertTier::Hot, 10, 0.3),
            importance_entry(0, 2, 0.8, ExpertTier::Hot, 9, 0.28),
            importance_entry(0, 3, 0.7, ExpertTier::Warm, 8, 0.2),
            importance_entry(0, 4, 0.1, ExpertTier::Cold, 1, 0.05),
        ]};
        let plan = compile::build_residency_plan(&layout, &importance, &ExpertCoresidency { schema_version: 1, pairs: vec![] }, "m1-8gb", 250, None);
        assert_eq!(plan.resident_cache_capacity_bytes, 250);
        let sum: u64 = plan.initial_hot_experts.iter().map(|e| e.bytes).sum();
        assert!(sum <= 250);
    }

    // ── Specialize integration test ──────────────────────────────────────
    #[test]
    fn specialize_produces_all_plan_files() {
        let root = std::env::temp_dir().join(format!("objeta_aot_specialize_{}", util::now_rfc3339ish()));
        let model = root.join("mock_model");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), serde_json::json!({
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_hidden_layers": 40,
            "num_local_experts": 256,
            "vocab_size": 151936
        }).to_string()).unwrap();
        fs::write(model.join("model.safetensors.index.json"), serde_json::json!({
            "metadata": {"total_size": 999},
            "weight_map": {
                "model.layers.10.mlp.experts.42.gate_up_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.10.mlp.experts.42.down_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.10.mlp.experts.7.gate_up_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.10.mlp.experts.7.down_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.10.mlp.gate.weight": "model-00001-of-00002.safetensors"
            }
        }).to_string()).unwrap();
        let calib = root.join("calib_trace.jsonl");
        fs::write(&calib, [
            r#"{"token_id":0,"layer":10,"selected_experts":[42,7],"selected_weights":[0.31,0.22]}"#,
            r#"{"token_id":1,"layer":10,"selected_experts":[42,7],"selected_weights":[0.20,0.10]}"#,
        ].join("\n")).unwrap();
        let out = root.join("spec_pack");
        specialize::run_from_data(
            &model, &calib, "m1-8gb", "coding",
            QualityBudget::Conservative, &out,
        ).unwrap();

        // Verify all expected files exist
        assert!(out.join("manifest.json").exists());
        assert!(out.join("expert_layout.json").exists());
        assert!(out.join("expert_importance.json").exists());
        assert!(out.join("expert_coresidency.json").exists());
        assert!(out.join("phase_policy.json").exists());
        assert!(out.join("quant_plan.json").exists());
        assert!(out.join("pruning_plan.json").exists());
        assert!(out.join("residency_plan.json").exists());
        assert!(out.join("runtime_profile.json").exists());
        assert!(out.join("verification_plan.json").exists());
        assert!(out.join("specialization_report.md").exists());

        // Verify quant plan has entries
        let qp: QuantPlan = serde_json::from_str(
            &fs::read_to_string(out.join("quant_plan.json")).unwrap()
        ).unwrap();
        assert!(!qp.entries.is_empty());

        // Verify pruning plan exists and is safe
        let pp: PruningPlan = serde_json::from_str(
            &fs::read_to_string(out.join("pruning_plan.json")).unwrap()
        ).unwrap();
        assert!(pp.summary.safe);

        // Verify verification plan has smoke check
        let vp: VerificationPlan = serde_json::from_str(
            &fs::read_to_string(out.join("verification_plan.json")).unwrap()
        ).unwrap();
        assert!(vp.checks.iter().any(|c| c.kind == "smoke_generation"));

        let _ = fs::remove_dir_all(root);
    }

    // ── Phase C Tests ────────────────────────────────────────────────────

    #[test]
    fn rtx3070_target_profile_differs_from_m1() {
        use crate::target::TargetHardware;
        let m1 = TargetHardware::from_name("m1-8gb");
        let rtx = TargetHardware::from_name("rtx3070-8gb-vram-32gb-ram");

        // RTX has 4× more RAM, 8 GB VRAM that m1 has none of
        assert!(rtx.ram_bytes > m1.ram_bytes);
        assert!(rtx.vram_bytes > 0);
        assert_eq!(m1.vram_bytes, 0);

        // GPU support differs
        assert!(rtx.supports_gpu);
        assert!(!m1.supports_gpu);

        // Backend differs
        assert_ne!(rtx.preferred_backend, m1.preferred_backend);
        assert_eq!(rtx.preferred_backend, "cuda_fused");
        assert_eq!(m1.preferred_backend, "fused_row_parallel");

        // RTX has higher cache capacity
        assert!(rtx.recommended_cache_bytes > m1.recommended_cache_bytes);

        // RTX supports iq3/iq2 in preferred formats; m1 does not
        assert!(rtx.preferred_quant_formats.iter().any(|f| f == "iq3"));
        assert!(!m1.preferred_quant_formats.iter().any(|f| f == "iq3"));
    }

    #[test]
    fn pruning_plan_has_estimated_only_and_requires_verification() {
        let root = std::env::temp_dir().join(format!("objeta_aot_prun_flags_{}", util::now_rfc3339ish()));
        let model = root.join("m");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), serde_json::json!({
            "model_type": "qwen3_moe", "num_hidden_layers": 40, "num_local_experts": 256
        }).to_string()).unwrap();
        fs::write(model.join("model.safetensors.index.json"), serde_json::json!({
            "metadata": {"total_size": 1},
            "weight_map": {
                "model.layers.10.mlp.experts.42.gate_up_proj.weight": "f.safetensors",
                "model.layers.10.mlp.experts.42.down_proj.weight": "f.safetensors"
            }
        }).to_string()).unwrap();
        let calib = root.join("c.jsonl");
        fs::write(&calib, r#"{"token_id":0,"layer":10,"selected_experts":[42],"selected_weights":[0.5]}"#).unwrap();
        let out = root.join("out");

        specialize::run_from_data(&model, &calib, "m1-8gb", "general",
            QualityBudget::Balanced, &out).unwrap();

        let pp: PruningPlan = serde_json::from_str(
            &fs::read_to_string(out.join("pruning_plan.json")).unwrap()
        ).unwrap();

        assert!(pp.summary.estimated_only,
            "pruning_plan.summary.estimated_only must be true");
        assert!(pp.summary.requires_verification,
            "pruning_plan.summary.requires_verification must be true");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn phase_policy_has_source_and_confidence() {
        let root = std::env::temp_dir().join(format!("objeta_aot_phase_meta_{}", util::now_rfc3339ish()));
        let model = root.join("m");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), serde_json::json!({
            "model_type": "qwen3_moe", "num_hidden_layers": 40, "num_local_experts": 256
        }).to_string()).unwrap();
        fs::write(model.join("model.safetensors.index.json"), serde_json::json!({
            "metadata": {"total_size": 1},
            "weight_map": {
                "model.layers.5.mlp.experts.1.gate_up_proj.weight": "f.safetensors",
                "model.layers.5.mlp.experts.1.down_proj.weight": "f.safetensors"
            }
        }).to_string()).unwrap();
        let calib = root.join("c.jsonl");
        fs::write(&calib, r#"{"token_id":0,"layer":5,"selected_experts":[1],"selected_weights":[0.5]}"#).unwrap();
        let out = root.join("out");

        specialize::run_from_data(&model, &calib, "m1-8gb", "general",
            QualityBudget::Conservative, &out).unwrap();

        let pp: PhasePolicy = serde_json::from_str(
            &fs::read_to_string(out.join("phase_policy.json")).unwrap()
        ).unwrap();
        assert_eq!(pp.source, "heuristic_lko_v1",
            "phase_policy.source must be heuristic_lko_v1");
        assert_eq!(pp.confidence, "experimental",
            "phase_policy.confidence must be experimental");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn specialization_report_contains_required_sections() {
        let root = std::env::temp_dir().join(format!("objeta_aot_report_check_{}", util::now_rfc3339ish()));
        let model = root.join("m");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), serde_json::json!({
            "model_type": "qwen3_moe", "num_hidden_layers": 40, "num_local_experts": 256
        }).to_string()).unwrap();
        fs::write(model.join("model.safetensors.index.json"), serde_json::json!({
            "metadata": {"total_size": 1},
            "weight_map": {
                "model.layers.10.mlp.experts.42.gate_up_proj.weight": "f.safetensors",
                "model.layers.10.mlp.experts.42.down_proj.weight": "f.safetensors",
                "model.layers.10.mlp.experts.7.gate_up_proj.weight": "f.safetensors",
                "model.layers.10.mlp.experts.7.down_proj.weight": "f.safetensors"
            }
        }).to_string()).unwrap();
        let calib = root.join("c.jsonl");
        fs::write(&calib, [
            r#"{"token_id":0,"layer":10,"selected_experts":[42,7],"selected_weights":[0.4,0.2]}"#,
            r#"{"token_id":1,"layer":10,"selected_experts":[42],"selected_weights":[0.3]}"#,
        ].join("\n")).unwrap();
        let out = root.join("out");

        specialize::run_from_data(&model, &calib, "m1-8gb", "coding",
            QualityBudget::Conservative, &out).unwrap();

        let report = fs::read_to_string(out.join("specialization_report.md")).unwrap();

        // Target hardware section
        assert!(report.contains("## Target Hardware"), "missing Target Hardware section");
        assert!(report.contains("m1-8gb"), "missing target name");

        // Calibration coverage section
        assert!(report.contains("## Calibration Coverage"), "missing Calibration Coverage section");
        assert!(report.contains("Coverage ratio"), "missing coverage ratio row");

        // Phase section with provenance
        assert!(report.contains("## Phase Summary"), "missing Phase Summary section");
        assert!(report.contains("heuristic_lko_v1"), "missing phase source");
        assert!(report.contains("experimental"), "missing phase confidence");
        assert!(report.contains("integrity_frontier"), "missing integrity_frontier phase");

        // Precision section
        assert!(report.contains("## Precision Summary"), "missing Precision Summary section");
        assert!(report.contains("q8"), "missing q8 row");
        assert!(report.contains("q4"), "missing q4 row");

        // Pruning section with epistemic warnings
        assert!(report.contains("## Pruning Summary"), "missing Pruning Summary section");
        assert!(report.contains("estimated_only"), "missing estimated_only flag");
        assert!(report.contains("requires_verification"), "missing requires_verification flag");
        assert!(report.contains("protect"), "missing protect row");
        assert!(report.contains("Mass-loss threshold"), "missing mass-loss threshold row");

        // Safety gates
        assert!(report.contains("## Safety Gates"), "missing Safety Gates section");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn coverage_589_over_10240_is_about_five_point_seven_five_percent() {
        let coverage = 589.0 / 10240.0;
        let pct = coverage * 100.0;
        assert!((pct - 5.751953125_f64).abs() < 1e-9);
        assert!(pct < 100.0);
    }
}
