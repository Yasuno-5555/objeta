//! DeepSeek V4 Flash — inventory sanity report.
//!
//! Reads the five JSON files produced by `parse_deepseek_v4_flash` and
//! emits both a human-readable terminal summary and a structured
//! `SanityReport` that can be serialised to JSON.
//!
//! No tensor payloads are ever loaded.

use std::path::Path;

use objeta_core::{ObjetaError, Result};
use serde::{Deserialize, Serialize};

use crate::deepseek::{DeepseekLayout, ExpertLayout, InventorySummary, RouterLayout};

// ── Constants ─────────────────────────────────────────────────────────────

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

// ── Output structures ─────────────────────────────────────────────────────

/// Byte-level working set estimates for VRAM planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingSetEstimates {
    /// Bytes for one expert's gate+up+down tensors in a single layer.
    pub single_expert_bytes: u64,
    /// Bytes for top_k selected experts in the current layer.
    pub current_layer_selected_bytes: u64,
    /// 2 × current_layer_selected_bytes (prefetch window).
    pub prefetch_window_2_layers_bytes: u64,
    /// 4 × current_layer_selected_bytes (prefetch window).
    pub prefetch_window_4_layers_bytes: u64,
    /// top_k × single_expert × num_layers (full forward-pass resident set).
    pub full_pass_selected_bytes: u64,
}

/// Whether the full forward-pass resident set fits in standard VRAM budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheFitEstimates {
    #[serde(rename = "1GB")]
    pub fit_1gb: bool,
    #[serde(rename = "2GB")]
    pub fit_2gb: bool,
    #[serde(rename = "4GB")]
    pub fit_4gb: bool,
    #[serde(rename = "8GB")]
    pub fit_8gb: bool,
}

/// Categorised tensor counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCounts {
    /// Router / gate tensors (from router_layout).
    pub router: usize,
    /// Expert gate / up / down / gate_up tensors (from expert_layout).
    pub expert: usize,
    /// Shared expert tensors (from expert_layout).
    pub shared_expert: usize,
    /// Everything else (attention, embedding, norm, uncategorised).
    pub other: usize,
}

/// Complete sanity report for a parsed DeepSeek V4 Flash checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanityReport {
    // ── Layout metadata ─────────────────────────────────────────────────
    pub layout_kind: String,
    pub num_layers: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,

    // ── Tensor inventory ─────────────────────────────────────────────────
    pub tensor_counts: TensorCounts,

    // ── Working set ───────────────────────────────────────────────────────
    pub working_set: WorkingSetEstimates,

    // ── Cache fit ─────────────────────────────────────────────────────────
    pub cache_fit: CacheFitEstimates,

    // ── Quantized storage ─────────────────────────────────────────────────
    /// True when expert weights are stored in fp4-packed I8 format.
    pub fp4_expert_storage_detected: bool,
    /// True when fp4 dequantization is implemented (always false for now).
    pub fp4_decode_supported: bool,

    // ── Routing ───────────────────────────────────────────────────────────
    /// Routing mode: "learned_gate", "hash_plus_learned_gate", "unknown_hash_plus_gate"
    pub routing_mode: String,

    // ── Diagnostic output ─────────────────────────────────────────────────
    pub warnings: Vec<String>,

    /// True when tensor layout is classified correctly (explicit_experts, routers found).
    pub metadata_compatible: bool,
    /// True when the model can be executed end-to-end on objeta-cuda.
    pub execution_ready: bool,
}

// ── Core logic ────────────────────────────────────────────────────────────

/// Load and deserialise a JSON file.
fn read_json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(ObjetaError::Io)?;
    serde_json::from_str(&content).map_err(|e| {
        ObjetaError::Parse(format!(
            "JSON parse error in {}: {}",
            path.display(),
            e
        ))
    })
}

/// Estimate the bytes for one selected expert in one layer.
///
/// For explicit experts:  avg of bytes_per_expert map / num_layers.
/// For packed experts or fallback: total_expert_bytes / (num_layers × num_experts).
fn compute_single_expert_bytes(
    summary: &InventorySummary,
    num_layers: usize,
    num_experts: usize,
) -> u64 {
    let layers = num_layers.max(1) as u64;
    let experts = num_experts.max(1) as u64;

    if let Some(ref bpe) = summary.bytes_per_expert {
        if !bpe.is_empty() {
            // bytes_per_expert[e_id] = total bytes for that expert across all
            // layers in the checkpoint.  Average it then divide by num_layers.
            let sum: u64 = bpe.values().sum();
            let avg = sum / bpe.len() as u64;
            return avg / layers;
        }
    }

    summary.total_expert_bytes / (layers * experts).max(1)
}

/// Read the four required JSON files from `input_dir` and produce a
/// [`SanityReport`] without loading any tensor payloads.
pub fn run_sanity_report(input_dir: &Path) -> Result<SanityReport> {
    let layout: DeepseekLayout =
        read_json(input_dir.join("deepseek_v4_flash_layout.json"))?;
    let expert_layout: ExpertLayout =
        read_json(input_dir.join("deepseek_v4_flash_expert_layout.json"))?;
    let router_layout: RouterLayout =
        read_json(input_dir.join("deepseek_v4_flash_router_layout.json"))?;
    let summary: InventorySummary =
        read_json(input_dir.join("deepseek_v4_flash_inventory_summary.json"))?;

    let mut warnings: Vec<String> = Vec::new();

    // ── Tensor counts ─────────────────────────────────────────────────────
    let router_count = router_layout.routers.len();
    let mut expert_count = 0usize;
    let mut shared_count = 0usize;

    for t in &expert_layout.tensors {
        match t.tensor_kind.as_str() {
            "gate" | "up" | "down" | "gate_up" => expert_count += 1,
            "shared_expert" => shared_count += 1,
            _ => {} // edge-case unknown kind inside expert_layout
        }
    }

    // "other" = tensors not classified as expert / shared-expert / router
    // (attention, embeddings, norms, uncategorised, …)
    let other_count = layout
        .tensor_count
        .saturating_sub(router_count + expert_count + shared_count);

    // ── Parameters ────────────────────────────────────────────────────────
    let num_layers = layout.num_layers;
    let num_experts = layout.num_experts;
    let top_k = layout.top_k;
    let layout_kind = expert_layout.layout_kind.clone();

    // ── Working set estimates ─────────────────────────────────────────────
    let single_expert_bytes =
        compute_single_expert_bytes(&summary, num_layers, num_experts);

    let current = single_expert_bytes.saturating_mul(top_k as u64);
    let prefetch2 = current.saturating_mul(2);
    let prefetch4 = current.saturating_mul(4);
    let full_pass = current.saturating_mul(num_layers as u64);

    // ── Cache fit ─────────────────────────────────────────────────────────
    let fit_1gb = full_pass <= GB;
    let fit_2gb = full_pass <= 2 * GB;
    let fit_4gb = full_pass <= 4 * GB;
    let fit_8gb = full_pass <= 8 * GB;

    // ── Warnings ──────────────────────────────────────────────────────────

    // Propagate any warnings emitted by the parser itself.
    for w in &router_layout.warnings {
        warnings.push(format!("[parser] {w}"));
    }

    if top_k == 0 {
        warnings.push(
            "top_k is 0 or missing — cannot compute working set.".into(),
        );
    }

    if layout_kind == "unknown" {
        warnings.push(
            "Expert layout kind is 'unknown' — unable to identify \
             explicit or packed expert tensors."
                .into(),
        );
    }

    if layout_kind == "packed_experts" && num_experts == 0 {
        warnings.push(
            "Packed expert layout detected but num_experts is 0 — \
             cannot compute per-expert slicing metadata."
                .into(),
        );
    }

    if router_count == 0 {
        warnings.push(
            "No router tensors found — model may not use MoE routing, \
             or router tensors were not classified."
                .into(),
        );
    } else if num_layers > 0 && router_count < num_layers {
        warnings.push(format!(
            "Missing router tensors: found {router_count} but \
             num_layers = {num_layers}."
        ));
    }

    if single_expert_bytes > GB {
        warnings.push(format!(
            "Single expert weight set is {:.1} MB — exceeds the 1 GB cache tier.",
            single_expert_bytes as f64 / MB as f64
        ));
    }

    // Current-layer working set vs cache tiers (most restrictive first).
    if top_k > 0 && current > 4 * GB {
        warnings.push(format!(
            "Current-layer working set ({:.2} GB) exceeds the 4 GB cache tier.",
            current as f64 / GB as f64
        ));
    } else if top_k > 0 && current > 2 * GB {
        warnings.push(format!(
            "Current-layer working set ({:.2} GB) exceeds the 2 GB cache tier.",
            current as f64 / GB as f64
        ));
    } else if top_k > 0 && current > GB {
        warnings.push(format!(
            "Current-layer working set ({:.1} MB) exceeds the 1 GB cache tier.",
            current as f64 / MB as f64
        ));
    }

    // ── FP4 detection ─────────────────────────────────────────────────────
    let fp4_detected = expert_layout.fp4_expert_storage_detected;
    let fp4_decode_supported = true; // semantics confirmed, CPU decode implemented

    if fp4_detected {
        warnings.push(
            "FP4 expert weight storage detected — CPU decode implemented but not yet \
             integrated into CUDA MoE execution path.".into(),
        );
    }

    // ── Routing mode ──────────────────────────────────────────────────────
    let has_hc_tensors = router_layout.routers.iter().any(|r| {
        matches!(r.tensor_kind.as_deref(), Some("router_hc_base" | "router_hc_fn" | "router_hc_scale"))
    });
    let has_learned_gate = router_layout.routers.iter().any(|r| {
        matches!(r.tensor_kind.as_deref(), Some("router"))
    });
    let has_tid2eid = router_layout.routers.iter().any(|r| {
        matches!(r.tensor_kind.as_deref(), Some("router_tid2eid"))
    });

    let routing_mode = if has_hc_tensors && (has_learned_gate || has_tid2eid) {
        "hash_plus_learned_gate".to_string()
    } else if has_hc_tensors {
        "unknown_hash_routing".to_string()
    } else if has_learned_gate || router_count > 0 {
        "learned_gate".to_string()
    } else {
        "unknown".to_string()
    };

    if has_hc_tensors {
        warnings.push(
            "Hash routing tensors detected (hc_ffn_*) — routing execution not \
             implemented.".into(),
        );
    }

    // ── Compatibility ─────────────────────────────────────────────────────
    let metadata_compatible = layout_kind != "unknown"
        && num_experts > 0
        && top_k > 0
        && expert_count > 0
        && router_count > 0;

    let execution_ready = metadata_compatible
        && !fp4_detected
        && !has_hc_tensors;

    Ok(SanityReport {
        layout_kind,
        num_layers,
        num_experts,
        top_k,
        hidden_size: layout.hidden_size,
        intermediate_size: layout.intermediate_size,
        tensor_counts: TensorCounts {
            router: router_count,
            expert: expert_count,
            shared_expert: shared_count,
            other: other_count,
        },
        working_set: WorkingSetEstimates {
            single_expert_bytes,
            current_layer_selected_bytes: current,
            prefetch_window_2_layers_bytes: prefetch2,
            prefetch_window_4_layers_bytes: prefetch4,
            full_pass_selected_bytes: full_pass,
        },
        cache_fit: CacheFitEstimates {
            fit_1gb,
            fit_2gb,
            fit_4gb,
            fit_8gb,
        },
        fp4_expert_storage_detected: fp4_detected,
        fp4_decode_supported,
        routing_mode,
        warnings,
        metadata_compatible,
        execution_ready,
    })
}

// ── Human-readable printer ────────────────────────────────────────────────

/// Print a human-readable sanity report to stdout.
pub fn print_sanity_report(report: &SanityReport) {
    let sep = "=".repeat(64);
    println!();
    println!("{sep}");
    println!("  DeepSeek V4 Flash — objeta-cuda MoE Sanity Report");
    println!("{sep}");
    println!();
    println!("  Layout kind        : {}", report.layout_kind);
    println!("  Layers             : {}", report.num_layers);
    println!("  Experts / layer    : {}", report.num_experts);
    println!("  top_k              : {}", report.top_k);
    println!("  Hidden size        : {}", report.hidden_size);
    println!("  Intermediate size  : {}", report.intermediate_size);
    println!();
    println!("  Tensor inventory:");
    println!(
        "    router tensors       : {}",
        report.tensor_counts.router
    );
    println!(
        "    expert tensors       : {}",
        report.tensor_counts.expert
    );
    println!(
        "    shared expert tensors: {}",
        report.tensor_counts.shared_expert
    );
    println!(
        "    other / unknown      : {}",
        report.tensor_counts.other
    );
    println!();
    println!("  Working set estimates:");
    println!(
        "    single expert (1 layer)   : {}",
        fmt_bytes(report.working_set.single_expert_bytes)
    );
    println!(
        "    current layer (top_k sel) : {}",
        fmt_bytes(report.working_set.current_layer_selected_bytes)
    );
    println!(
        "    2-layer prefetch window   : {}",
        fmt_bytes(report.working_set.prefetch_window_2_layers_bytes)
    );
    println!(
        "    4-layer prefetch window   : {}",
        fmt_bytes(report.working_set.prefetch_window_4_layers_bytes)
    );
    println!(
        "    full forward pass         : {}",
        fmt_bytes(report.working_set.full_pass_selected_bytes)
    );
    println!();
    println!("  Cache fit (full forward pass):");
    println!("    1 GB : {}", fit_label(report.cache_fit.fit_1gb));
    println!("    2 GB : {}", fit_label(report.cache_fit.fit_2gb));
    println!("    4 GB : {}", fit_label(report.cache_fit.fit_4gb));
    println!("    8 GB : {}", fit_label(report.cache_fit.fit_8gb));
    println!();
    println!("  FP4 expert storage  : {}", if report.fp4_expert_storage_detected { "detected" } else { "none" });
    println!("  FP4 decode supported: {}", if report.fp4_decode_supported { "yes" } else { "no" });
    println!("  Routing mode        : {}", report.routing_mode);
    println!();
    if report.warnings.is_empty() {
        println!("  Warnings : none");
    } else {
        println!("  Warnings ({}):", report.warnings.len());
        for w in &report.warnings {
            println!("    [!] {w}");
        }
    }
    println!();
    println!("  Metadata compatible  : {}", if report.metadata_compatible { "YES" } else { "NO" });
    println!("  Execution ready      : {}", if report.execution_ready { "YES" } else { "NO" });
    println!("{sep}");
    println!();
}

fn fmt_bytes(b: u64) -> String {
    if b == 0 {
        return "0 B  (unknown — check top_k / num_experts)".into();
    }
    if b >= GB {
        format!("{:.2} GB  ({b} B)", b as f64 / GB as f64)
    } else if b >= MB {
        format!("{:.1} MB  ({b} B)", b as f64 / MB as f64)
    } else if b >= 1024 {
        format!("{:.1} KB  ({b} B)", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

fn fit_label(fits: bool) -> &'static str {
    if fits { "fits" } else { "does not fit" }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    // ── Test fixture builder ───────────────────────────────────────────────

    /// Write the four required JSON files into `dir` using synthetic data.
    ///
    /// `bytes_per_expert_per_layer` is the combined gate+up+down byte count
    /// for **one expert in one layer** (e.g. 12_582_912 for three 4 MB
    /// tensors).
    fn write_test_json_files(
        dir: &Path,
        num_layers: usize,
        num_experts: usize,
        top_k: usize,
        bytes_per_expert_per_layer: u64,
    ) {
        let bytes_per_tensor = bytes_per_expert_per_layer / 3; // gate / up / down evenly split
        let tensor_count = num_layers * num_experts * 3 + num_layers; // experts×3 + routers

        // ── 1. layout JSON ────────────────────────────────────────────────
        let layout = json!({
            "model_name": "test_deepseek",
            "num_layers": num_layers,
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "num_experts": num_experts,
            "top_k": top_k,
            "vocab_size": 32000,
            "dtype": "bfloat16",
            "quant_dtype": null,
            "tensor_count": tensor_count,
            "shard_count": 1,
            "total_byte_size": (num_layers * num_experts) as u64 * bytes_per_expert_per_layer,
            "largest_tensor": null,
            "tensor_name_patterns": []
        });

        // ── 2. expert_layout JSON ─────────────────────────────────────────
        let mut tensors = Vec::new();
        for l in 0..num_layers {
            for e in 0..num_experts {
                for kind in ["gate", "up", "down"] {
                    tensors.push(json!({
                        "name": format!("model.layers.{l}.mlp.experts.{e}.{kind}_proj.weight"),
                        "layer_id": l,
                        "expert_id": e,
                        "tensor_kind": kind,
                        "shape": [1024usize, 2048usize],
                        "dtype": "BF16",
                        "byte_length": bytes_per_tensor,
                        "source_file": "model.safetensors"
                    }));
                }
            }
        }
        let expert_layout = json!({
            "layout_kind": "explicit_experts",
            "tensors": tensors
        });

        // ── 3. router_layout JSON ─────────────────────────────────────────
        let routers: Vec<_> = (0..num_layers)
            .map(|l| json!({
                "name": format!("model.layers.{l}.mlp.gate.weight"),
                "layer_id": l,
                "tensor_kind": "router",
                "shape": [num_experts, 2048usize],
                "dtype": "BF16",
                "byte_length": num_experts * 2048 * 2usize,
                "source_file": "model.safetensors"
            }))
            .collect();

        let router_layout = json!({
            "num_experts": num_experts,
            "top_k": top_k,
            "warnings": [],
            "routers": routers
        });

        // ── 4. inventory_summary JSON ─────────────────────────────────────
        let total_expert: u64 =
            num_layers as u64 * num_experts as u64 * bytes_per_expert_per_layer;

        // bytes_per_expert[e_id] = bytes for expert e across ALL layers
        let bytes_per_expert: serde_json::Map<_, _> = (0..num_experts)
            .map(|e| {
                (
                    e.to_string(),
                    json!(num_layers as u64 * bytes_per_expert_per_layer),
                )
            })
            .collect();

        let expert_bytes_per_layer: serde_json::Map<_, _> = (0..num_layers)
            .map(|l| {
                (
                    l.to_string(),
                    json!(num_experts as u64 * bytes_per_expert_per_layer),
                )
            })
            .collect();

        let inventory = json!({
            "total_expert_bytes": total_expert,
            "expert_bytes_per_layer": expert_bytes_per_layer,
            "bytes_per_expert": bytes_per_expert,
            "bytes_by_tensor_kind": {
                "gate": total_expert / 3,
                "up":   total_expert / 3,
                "down": total_expert / 3,
                "gate_up": 0,
                "router": 0,
                "shared": 0,
                "attention": 0,
                "other": 0
            },
            "largest_expert_tensor": null,
            "largest_layer_by_expert_bytes": null,
            "fits_in_cache": {
                "1GB": total_expert <= GB,
                "2GB": total_expert <= 2 * GB,
                "4GB": total_expert <= 4 * GB,
                "8GB": total_expert <= 8 * GB
            }
        });

        let write = |name: &str, val: &serde_json::Value| {
            std::fs::write(
                dir.join(name),
                serde_json::to_string_pretty(val).unwrap(),
            )
            .unwrap();
        };
        write("deepseek_v4_flash_layout.json", &layout);
        write("deepseek_v4_flash_expert_layout.json", &expert_layout);
        write("deepseek_v4_flash_router_layout.json", &router_layout);
        write("deepseek_v4_flash_inventory_summary.json", &inventory);
    }

    // ── Tests ──────────────────────────────────────────────────────────────

    /// Small synthetic model: 4 layers, 8 experts, top_k=2,
    /// 12 MB per expert per layer.
    #[test]
    fn test_sanity_report_small_explicit() {
        let dir = std::env::temp_dir().join("objeta_sanity_small");
        std::fs::create_dir_all(&dir).unwrap();

        // 3 tensors × 4 194 304 B = 12 582 912 B per expert per layer
        write_test_json_files(&dir, 4, 8, 2, 12_582_912);

        let report = run_sanity_report(&dir).unwrap();

        assert_eq!(report.layout_kind, "explicit_experts");
        assert_eq!(report.num_layers, 4);
        assert_eq!(report.num_experts, 8);
        assert_eq!(report.top_k, 2);
        assert_eq!(report.hidden_size, 2048);
        assert_eq!(report.intermediate_size, 1024);

        // Tensor counts: 4 layers × 8 experts × 3 = 96 expert tensors
        //               4 router tensors
        assert_eq!(report.tensor_counts.router, 4);
        assert_eq!(report.tensor_counts.expert, 96);
        assert_eq!(report.tensor_counts.shared_expert, 0);

        // bytes_per_expert[e] = 4 layers × 12_582_912 = 50_331_648
        // avg / num_layers = 50_331_648 / 4 = 12_582_912  ← single expert per layer
        assert_eq!(report.working_set.single_expert_bytes, 12_582_912);

        // current = top_k × single = 2 × 12_582_912 = 25_165_824
        assert_eq!(report.working_set.current_layer_selected_bytes, 25_165_824);

        // prefetch_2 = 50_331_648
        assert_eq!(
            report.working_set.prefetch_window_2_layers_bytes,
            50_331_648
        );

        // full_pass = current × num_layers = 25_165_824 × 4 = 100_663_296
        assert_eq!(report.working_set.full_pass_selected_bytes, 100_663_296);

        // 100 663 296 B ≈ 96 MB → fits in all tiers
        assert!(report.cache_fit.fit_1gb);
        assert!(report.cache_fit.fit_2gb);
        assert!(report.cache_fit.fit_4gb);
        assert!(report.cache_fit.fit_8gb);

        assert!(report.warnings.is_empty(), "unexpected warnings: {:?}", report.warnings);
        assert!(report.metadata_compatible);
        assert!(report.execution_ready);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Missing router tensors must trigger a warning and set compatible=false.
    #[test]
    fn test_sanity_report_missing_routers() {
        let dir = std::env::temp_dir().join("objeta_sanity_no_router");
        std::fs::create_dir_all(&dir).unwrap();

        write_test_json_files(&dir, 4, 8, 2, 12_582_912);

        // Override with empty router list
        let empty_routers = json!({
            "num_experts": 8,
            "top_k": 2,
            "warnings": [],
            "routers": []
        });
        std::fs::write(
            dir.join("deepseek_v4_flash_router_layout.json"),
            serde_json::to_string_pretty(&empty_routers).unwrap(),
        )
        .unwrap();

        let report = run_sanity_report(&dir).unwrap();

        assert!(!report.warnings.is_empty());
        assert!(
            report.warnings.iter().any(|w| w.contains("No router tensors")),
            "expected router warning, got: {:?}",
            report.warnings
        );
        assert!(!report.metadata_compatible);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Unknown layout kind must trigger a warning and set compatible=false.
    #[test]
    fn test_sanity_report_unknown_layout() {
        let dir = std::env::temp_dir().join("objeta_sanity_unknown");
        std::fs::create_dir_all(&dir).unwrap();

        write_test_json_files(&dir, 4, 8, 2, 12_582_912);

        // Override with "unknown" layout
        let unknown_expert_layout = json!({
            "layout_kind": "unknown",
            "tensors": []
        });
        std::fs::write(
            dir.join("deepseek_v4_flash_expert_layout.json"),
            serde_json::to_string_pretty(&unknown_expert_layout).unwrap(),
        )
        .unwrap();

        let report = run_sanity_report(&dir).unwrap();

        assert!(
            report.warnings.iter().any(|w| w.contains("unknown")),
            "expected unknown-layout warning, got: {:?}",
            report.warnings
        );
        assert!(!report.metadata_compatible);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Missing top_k (=0) must emit a warning.
    #[test]
    fn test_sanity_report_missing_top_k() {
        let dir = std::env::temp_dir().join("objeta_sanity_no_topk");
        std::fs::create_dir_all(&dir).unwrap();

        write_test_json_files(&dir, 4, 8, 2, 12_582_912);

        // Overwrite layout with top_k = 0
        let layout_no_topk = json!({
            "model_name": "test_deepseek",
            "num_layers": 4,
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "num_experts": 8,
            "top_k": 0,
            "vocab_size": 32000,
            "dtype": "bfloat16",
            "quant_dtype": null,
            "tensor_count": 100,
            "shard_count": 1,
            "total_byte_size": 1000000,
            "largest_tensor": null,
            "tensor_name_patterns": []
        });
        std::fs::write(
            dir.join("deepseek_v4_flash_layout.json"),
            serde_json::to_string_pretty(&layout_no_topk).unwrap(),
        )
        .unwrap();

        let report = run_sanity_report(&dir).unwrap();

        assert!(
            report.warnings.iter().any(|w| w.contains("top_k is 0")),
            "expected top_k warning, got: {:?}",
            report.warnings
        );
        // working set should be zero when top_k=0
        assert_eq!(report.working_set.current_layer_selected_bytes, 0);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Large expert tensors (> 1 GB per expert) must trigger a size warning.
    #[test]
    fn test_sanity_report_oversized_expert_warning() {
        let dir = std::env::temp_dir().join("objeta_sanity_big");
        std::fs::create_dir_all(&dir).unwrap();

        // 1.5 GB per expert per layer → single_expert_bytes > 1 GB
        write_test_json_files(&dir, 2, 2, 1, 1_500_000_000);

        let report = run_sanity_report(&dir).unwrap();

        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("exceeds the 1 GB cache tier")),
            "expected size warning, got: {:?}",
            report.warnings
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Packed expert layout without num_experts must produce a slicing warning.
    #[test]
    fn test_sanity_report_packed_no_experts_warning() {
        let dir = std::env::temp_dir().join("objeta_sanity_packed");
        std::fs::create_dir_all(&dir).unwrap();

        write_test_json_files(&dir, 4, 8, 2, 12_582_912);

        // Override layout with num_experts = 0
        let layout_packed = json!({
            "model_name": "test_deepseek",
            "num_layers": 4,
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "num_experts": 0,
            "top_k": 2,
            "vocab_size": 32000,
            "dtype": "bfloat16",
            "quant_dtype": null,
            "tensor_count": 100,
            "shard_count": 1,
            "total_byte_size": 1000000,
            "largest_tensor": null,
            "tensor_name_patterns": []
        });
        std::fs::write(
            dir.join("deepseek_v4_flash_layout.json"),
            serde_json::to_string_pretty(&layout_packed).unwrap(),
        )
        .unwrap();

        // Override expert_layout with "packed_experts"
        let packed_expert_layout = json!({
            "layout_kind": "packed_experts",
            "tensors": [{
                "name": "model.layers.0.mlp.experts.gate_proj.weight",
                "layer_id": 0,
                "expert_id": null,
                "tensor_kind": "gate",
                "shape": [8, 1024, 2048],
                "dtype": "BF16",
                "byte_length": 33554432,
                "source_file": "model.safetensors"
            }]
        });
        std::fs::write(
            dir.join("deepseek_v4_flash_expert_layout.json"),
            serde_json::to_string_pretty(&packed_expert_layout).unwrap(),
        )
        .unwrap();

        let report = run_sanity_report(&dir).unwrap();

        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("num_experts is 0")),
            "expected packed slicing warning, got: {:?}",
            report.warnings
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
