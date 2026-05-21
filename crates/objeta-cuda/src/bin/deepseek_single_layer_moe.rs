//! Real DeepSeek V4 Flash single-layer MoE proof.
//!
//! Loads parser metadata, lazily reads only the router + selected expert
//! tensors from safetensors shards, runs CPU fp32 reference and CUDA Q4_0
//! selected-expert MoE, then compares outputs.
//!
//! Hard rules:
//! - Only explicit_experts layout is supported for execution.
//! - packed_experts refuses to run (needs slicing metadata).
//! - No full model loading, no attention, no generation.

use objeta_cuda::{
    compare_outputs, q4_quantize_matrix_cpu, selected_moe_cpu_fp32,
    BytesByTensorKind, CudaBackendBuilder, CudaError, CudaErrorKind, CudaExpertCache,
    ExpertWeights, ExpertWeightsFp32, MoeExecutor, QGemvNumerics,
    QGemvShape, QuantBackend, QuantFormat, Result,
};
use objeta_parser::deepseek::decode_deepseek_fp4_to_f32;
use objeta_parser::ModelWeights;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

// ── Constants ───────────────────────────────────────────────────────────────

const SOURCE_LABEL: &str = "real_deepseek_v4_flash_single_layer_moe";

// ── Parser JSON schemas (subsets of what we need) ───────────────────────────

#[derive(Debug, Deserialize)]
struct LayoutJson {
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    #[allow(dead_code)]
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct ExpertLayoutJson {
    layout_kind: String,
    tensors: Vec<ExpertTensorEntry>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
struct ExpertTensorEntry {
    name: String,
    layer_id: Option<usize>,
    expert_id: Option<Option<usize>>,
    tensor_kind: String,
    shape: Vec<usize>,
    dtype: String,
    byte_length: usize,
    source_file: String,
    // FP4 storage metadata
    #[serde(default)]
    storage_dtype: Option<String>,
    #[serde(default)]
    logical_dtype: Option<String>,
    #[serde(default)]
    scale_tensor_name: Option<String>,
    #[serde(default)]
    scale_dtype: Option<String>,
    #[serde(default)]
    logical_shape: Option<Vec<usize>>,
    #[serde(default)]
    block_size: Option<usize>,
    #[serde(default)]
    packed_values_per_byte: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RouterLayoutJson {
    routers: Vec<RouterTensorEntry>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct RouterTensorEntry {
    name: String,
    layer_id: Option<usize>,
    shape: Vec<usize>,
    dtype: String,
}

// ── Parser metadata (all loaded from JSON) ──────────────────────────────────

struct ParserMetadata {
    layout: LayoutJson,
    expert_layout: ExpertLayoutJson,
    router_layout: RouterLayoutJson,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let content = std::fs::read_to_string(path).map_err(|err| {
        CudaError::new(
            CudaErrorKind::Io,
            format!("read {}", path.display()),
            err.to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })?;
    serde_json::from_str(&content).map_err(|err| {
        CudaError::new(
            CudaErrorKind::InvalidInput,
            format!("parse {}", path.display()),
            err.to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })
}

fn load_parser_metadata(parse_dir: &Path) -> Result<ParserMetadata> {
    let layout: LayoutJson = read_json(&parse_dir.join("deepseek_v4_flash_layout.json"))?;
    let expert_layout: ExpertLayoutJson =
        read_json(&parse_dir.join("deepseek_v4_flash_expert_layout.json"))?;
    let router_layout: RouterLayoutJson =
        read_json(&parse_dir.join("deepseek_v4_flash_router_layout.json"))?;

    Ok(ParserMetadata {
        layout,
        expert_layout,
        router_layout,
    })
}

// ── Validation ──────────────────────────────────────────────────────────────

fn validate_layer(meta: &ParserMetadata, layer: usize) -> Result<()> {
    if layer >= meta.layout.num_layers {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "validate layer id",
            format!(
                "layer {} out of range: num_layers={}",
                layer, meta.layout.num_layers
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    Ok(())
}

fn require_explicit_experts(meta: &ParserMetadata) -> Result<()> {
    match meta.expert_layout.layout_kind.as_str() {
        "explicit_experts" => Ok(()),
        "packed_experts" => Err(CudaError::new(
            CudaErrorKind::Unsupported,
            "require explicit_experts layout",
            "packed_experts requires explicit slicing metadata; refusing to guess offsets".to_string(),
            file!(),
            line!(),
            module_path!(),
        )),
        other => Err(CudaError::new(
            CudaErrorKind::Unsupported,
            "require explicit_experts layout",
            format!(
                "unsupported layout_kind '{}': only explicit_experts is supported",
                other
            ),
            file!(),
            line!(),
            module_path!(),
        )),
    }
}

// ── Router tensor identification ────────────────────────────────────────────

#[derive(Debug)]
struct RouterTensorInfo {
    name: String,
    dtype: String,
}

fn find_router_tensor(meta: &ParserMetadata, layer: usize) -> Result<RouterTensorInfo> {
    let candidates: Vec<&RouterTensorEntry> = meta
        .router_layout
        .routers
        .iter()
        .filter(|r| r.layer_id == Some(layer))
        .collect();

    if candidates.is_empty() {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "find router tensor",
            format!("no router tensor found for layer {}", layer),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    if candidates.len() > 1 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "find router tensor",
            format!(
                "{} router tensors found for layer {}: {:?} — ambiguous",
                candidates.len(),
                layer,
                candidates.iter().map(|r| &r.name).collect::<Vec<_>>()
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    let rt = candidates[0];
    if rt.shape.len() != 2 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "validate router tensor shape",
            format!(
                "router tensor '{}' has shape {:?}, expected [num_experts, hidden_size]",
                rt.name, rt.shape
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    let num_experts_from_shape = rt.shape[0];
    let hidden_from_shape = rt.shape[1];

    if num_experts_from_shape != meta.layout.num_experts {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "validate router tensor shape",
            format!(
                "router tensor '{}' expert dim {} != num_experts {}",
                rt.name, num_experts_from_shape, meta.layout.num_experts
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    if hidden_from_shape != meta.layout.hidden_size {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "validate router tensor shape",
            format!(
                "router tensor '{}' hidden dim {} != hidden_size {}",
                rt.name, hidden_from_shape, meta.layout.hidden_size
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    Ok(RouterTensorInfo {
        name: rt.name.clone(),
        dtype: rt.dtype.clone(),
    })
}

// ── Expert tensor identification ────────────────────────────────────────────

#[derive(Debug)]
struct ExpertTensorSet {
    gate_name: String,
    up_name: String,
    down_name: String,
}

fn find_expert_tensors(
    meta: &ParserMetadata,
    layer: usize,
    expert_id: usize,
) -> Result<ExpertTensorSet> {
    let mut gate: Option<&str> = None;
    let mut up: Option<&str> = None;
    let mut down: Option<&str> = None;

    for t in &meta.expert_layout.tensors {
        if t.layer_id != Some(layer) {
            continue;
        }
        let tid = match t.expert_id {
            Some(Some(id)) => id,
            _ => continue,
        };
        if tid != expert_id {
            continue;
        }
        match t.tensor_kind.as_str() {
            "gate" => gate = Some(t.name.as_str()),
            "up" => up = Some(t.name.as_str()),
            "down" => down = Some(t.name.as_str()),
            _ => {}
        }
    }

    let missing: Vec<&str> = [
        ("gate", gate.is_none()),
        ("up", up.is_none()),
        ("down", down.is_none()),
    ]
    .iter()
    .filter_map(|(kind, is_missing)| if *is_missing { Some(*kind) } else { None })
    .collect();

    if !missing.is_empty() {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "find expert tensors",
            format!(
                "layer {} expert {} missing tensors: {:?}",
                layer, expert_id, missing
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    Ok(ExpertTensorSet {
        gate_name: gate.unwrap().to_string(),
        up_name: up.unwrap().to_string(),
        down_name: down.unwrap().to_string(),
    })
}

// ── FP4 expert weight loading ──────────────────────────────────────────────

#[derive(Debug)]
struct Fp4ExpertTensorSet {
    gate_name: String,
    gate_scale_name: String,
    up_name: String,
    up_scale_name: String,
    down_name: String,
    down_scale_name: String,
    gate_block_size: usize,
    up_block_size: usize,
    down_block_size: usize,
    gate_logical_shape: Vec<usize>,
    up_logical_shape: Vec<usize>,
    down_logical_shape: Vec<usize>,
}

fn find_fp4_expert_tensors(
    meta: &ParserMetadata,
    layer: usize,
    expert_id: usize,
) -> Result<Fp4ExpertTensorSet> {
    let mut gate_weight: Option<&ExpertTensorEntry> = None;
    let mut up_weight: Option<&ExpertTensorEntry> = None;
    let mut down_weight: Option<&ExpertTensorEntry> = None;

    for t in &meta.expert_layout.tensors {
        if t.layer_id != Some(layer) {
            continue;
        }
        let tid = match t.expert_id {
            Some(Some(id)) => id,
            _ => continue,
        };
        if tid != expert_id {
            continue;
        }
        match t.tensor_kind.as_str() {
            "gate" => gate_weight = Some(t),
            "up" => up_weight = Some(t),
            "down" => down_weight = Some(t),
            _ => {}
        }
    }

    let missing: Vec<&str> = [
        ("gate", gate_weight.is_none()),
        ("up", up_weight.is_none()),
        ("down", down_weight.is_none()),
    ]
    .iter()
    .filter_map(|(kind, is_missing)| if *is_missing { Some(*kind) } else { None })
    .collect();

    if !missing.is_empty() {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "find fp4 expert tensors",
            format!(
                "layer {} expert {} missing tensors: {:?}",
                layer, expert_id, missing
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    let gw = gate_weight.unwrap();
    let uw = up_weight.unwrap();
    let dw = down_weight.unwrap();

    let gate_scale_name = gw.scale_tensor_name.clone().ok_or_else(|| {
        CudaError::new(
            CudaErrorKind::InvalidInput,
            "find fp4 scale tensor",
            format!(
                "gate weight '{}' has no scale_tensor_name — not FP4?",
                gw.name
            ),
            file!(),
            line!(),
            module_path!(),
        )
    })?;
    let up_scale_name = uw.scale_tensor_name.clone().ok_or_else(|| {
        CudaError::new(
            CudaErrorKind::InvalidInput,
            "find fp4 scale tensor",
            format!("up weight '{}' has no scale_tensor_name — not FP4?", uw.name),
            file!(),
            line!(),
            module_path!(),
        )
    })?;
    let down_scale_name = dw.scale_tensor_name.clone().ok_or_else(|| {
        CudaError::new(
            CudaErrorKind::InvalidInput,
            "find fp4 scale tensor",
            format!(
                "down weight '{}' has no scale_tensor_name — not FP4?",
                dw.name
            ),
            file!(),
            line!(),
            module_path!(),
        )
    })?;

    let gate_block_size = gw.block_size.ok_or_else(|| {
        CudaError::new(CudaErrorKind::InvalidInput, "fp4 block_size", "missing".to_string(), file!(), line!(), module_path!())
    })?;
    let up_block_size = uw.block_size.ok_or_else(|| {
        CudaError::new(CudaErrorKind::InvalidInput, "fp4 block_size", "missing".to_string(), file!(), line!(), module_path!())
    })?;
    let down_block_size = dw.block_size.ok_or_else(|| {
        CudaError::new(CudaErrorKind::InvalidInput, "fp4 block_size", "missing".to_string(), file!(), line!(), module_path!())
    })?;

    let gate_logical = gw.logical_shape.clone().ok_or_else(|| {
        CudaError::new(CudaErrorKind::InvalidInput, "fp4 logical_shape", "missing".to_string(), file!(), line!(), module_path!())
    })?;
    let up_logical = uw.logical_shape.clone().ok_or_else(|| {
        CudaError::new(CudaErrorKind::InvalidInput, "fp4 logical_shape", "missing".to_string(), file!(), line!(), module_path!())
    })?;
    let down_logical = dw.logical_shape.clone().ok_or_else(|| {
        CudaError::new(CudaErrorKind::InvalidInput, "fp4 logical_shape", "missing".to_string(), file!(), line!(), module_path!())
    })?;

    Ok(Fp4ExpertTensorSet {
        gate_name: gw.name.clone(),
        gate_scale_name,
        up_name: uw.name.clone(),
        up_scale_name,
        down_name: dw.name.clone(),
        down_scale_name,
        gate_block_size,
        up_block_size,
        down_block_size,
        gate_logical_shape: gate_logical,
        up_logical_shape: up_logical,
        down_logical_shape: down_logical,
    })
}

fn load_fp4_expert_weight(
    model_weights: &ModelWeights,
    weight_name: &str,
    scale_name: &str,
    physical_shape: &[usize],
    logical_shape: &[usize],
    block_size: usize,
) -> Result<Vec<f32>> {
    let weight_bytes = model_weights.get_raw(weight_name).map_err(|e| {
        CudaError::new(
            CudaErrorKind::Io,
            format!("load fp4 weight '{}'", weight_name),
            e.to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })?;
    let scale_bytes = model_weights.get_raw(scale_name).map_err(|e| {
        CudaError::new(
            CudaErrorKind::Io,
            format!("load fp4 scale '{}'", scale_name),
            e.to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })?;

    Ok(decode_deepseek_fp4_to_f32(
        weight_bytes,
        scale_bytes,
        physical_shape,
        logical_shape,
        block_size,
    ))
}

fn validate_manual_expert_ids(expert_ids: &[usize], num_experts: usize, top_k: usize) -> Result<()> {
    if expert_ids.is_empty() {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "validate expert ids",
            "expert-ids must not be empty".to_string(),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    if expert_ids.len() != top_k {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "validate expert ids",
            format!(
                "expected {} expert ids (top_k={}), got {}",
                top_k,
                top_k,
                expert_ids.len()
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    for &eid in expert_ids {
        if eid >= num_experts {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate expert ids",
                format!(
                    "expert id {} out of range: num_experts={}",
                    eid, num_experts
                ),
                file!(),
                line!(),
                module_path!(),
            ));
        }
    }
    Ok(())
}

fn validate_manual_expert_weights(weights: &[f32], top_k: usize) -> Result<()> {
    if weights.len() != top_k {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "validate expert weights",
            format!(
                "expected {} expert weights (top_k={}), got {}",
                top_k,
                top_k,
                weights.len()
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    let sum: f32 = weights.iter().sum();
    if (sum - 1.0).abs() > 0.01 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "validate expert weights",
            format!("expert weights sum to {}, expected ~1.0", sum),
            file!(),
            line!(),
            module_path!(),
        ));
    }
    Ok(())
}

// ── Hidden vector ───────────────────────────────────────────────────────────

fn seeded_f32s(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((state >> 32) as u32) | 1;
        let unit = (bits as f32) / (u32::MAX as f32);
        out.push((unit * 2.0) - 1.0);
    }
    out
}

// ── CPU router ──────────────────────────────────────────────────────────────

fn cpu_router(
    router_weights: &[f32],
    hidden: &[f32],
    num_experts: usize,
    hidden_size: usize,
    top_k: usize,
) -> Result<(Vec<usize>, Vec<f32>)> {
    if router_weights.len() != num_experts * hidden_size {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "cpu_router",
            format!(
                "router weights len {} != num_experts {} * hidden_size {}",
                router_weights.len(),
                num_experts,
                hidden_size
            ),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    // logits = W @ hidden
    let mut logits = vec![0.0f32; num_experts];
    for e in 0..num_experts {
        let mut sum = 0.0f32;
        let base = e * hidden_size;
        for h in 0..hidden_size {
            sum += router_weights[base + h] * hidden[h];
        }
        logits[e] = sum;
    }

    // Softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; num_experts];
    let mut sum_exp = 0.0f32;
    for e in 0..num_experts {
        let val = (logits[e] - max_logit).exp();
        probs[e] = val;
        sum_exp += val;
    }
    for p in probs.iter_mut() {
        *p /= sum_exp;
    }

    // Select top_k
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    let selected_ids: Vec<usize> = indexed.iter().map(|(id, _)| *id).collect();
    let selected_weights: Vec<f32> = indexed.iter().map(|(_, w)| *w).collect();

    Ok((selected_ids, selected_weights))
}

// ── Dtype validation ────────────────────────────────────────────────────────

fn validate_dtype(dtype: &str) -> Result<()> {
    match dtype.to_uppercase().as_str() {
        "F32" | "FLOAT32" | "F16" | "FLOAT16" | "BF16" | "BFLOAT16" => Ok(()),
        other => Err(CudaError::new(
            CudaErrorKind::Unsupported,
            "validate dtype",
            format!(
                "unsupported dtype '{}': only F32, F16, BF16 are supported",
                other
            ),
            file!(),
            line!(),
            module_path!(),
        )),
    }
}

// ── Output report ───────────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
struct M9Report {
    source: String,
    model_dir: String,
    parse_dir: String,
    layer_id: usize,
    layout_kind: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    selected_expert_ids: Vec<usize>,
    selected_expert_weights: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    router_tensor_name: Option<String>,
    expert_tensor_names_used: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scale_tensor_names_used: Option<Vec<String>>,
    source_dtypes: HashMap<String, String>,
    quant_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    fp4_decode_ms: Option<f32>,
    logical_expert_bytes_requested: usize,
    actual_expert_bytes_loaded: usize,
    resident_cache_bytes_reused: usize,
    resident_cache_resident_bytes: usize,
    dequantized_scratch_bytes: usize,
    selected_working_set_bytes: usize,
    bytes_per_expert: usize,
    bytes_by_tensor_kind: BytesByTensorKind,
    cache_hit_count: usize,
    cache_miss_count: usize,
    cache_eviction_count: usize,
    cache_insert_attempt_count: usize,
    cache_insert_accept_count: usize,
    cache_insert_bypass_count: usize,
    oversized_tensor_bypass_count: usize,
    oversized_expert_bypass_count: usize,
    self_eviction_risk_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    router_ms: Option<f32>,
    tensor_load_ms: f32,
    quantize_ms: f32,
    h2d_ms: f32,
    gate_up_qgemv_ms: f32,
    activation_ms: f32,
    down_qgemv_ms: f32,
    accum_ms: f32,
    total_ms: f32,
    quant_vs_fp32: QGemvNumerics,
}

// ── CLI parsing ─────────────────────────────────────────────────────────────

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn parse_comma_separated_usize(raw: &str) -> Result<Vec<usize>> {
    raw.split(',')
        .map(|s| s.trim().parse::<usize>().map_err(|_| {
            CudaError::new(
                CudaErrorKind::InvalidInput,
                "parse comma-separated usize",
                format!("invalid integer: '{}'", s),
                file!(),
                line!(),
                module_path!(),
            )
        }))
        .collect()
}

fn parse_comma_separated_f32(raw: &str) -> Result<Vec<f32>> {
    raw.split(',')
        .map(|s| s.trim().parse::<f32>().map_err(|_| {
            CudaError::new(
                CudaErrorKind::InvalidInput,
                "parse comma-separated f32",
                format!("invalid float: '{}'", s),
                file!(),
                line!(),
                module_path!(),
            )
        }))
        .collect()
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let parse_dir = parse_flag(&args, "--parse-dir").ok_or_else(|| {
        CudaError::new(
            CudaErrorKind::InvalidInput,
            "parse args",
            "--parse-dir is required".to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })?;
    let model_dir = parse_flag(&args, "--model-dir").ok_or_else(|| {
        CudaError::new(
            CudaErrorKind::InvalidInput,
            "parse args",
            "--model-dir is required".to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })?;
    let layer = parse_flag(&args, "--layer")
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| {
            CudaError::new(
                CudaErrorKind::InvalidInput,
                "parse args",
                "--layer is required and must be a non-negative integer".to_string(),
                file!(),
                line!(),
                module_path!(),
            )
        })?;
    let seed = parse_flag(&args, "--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(42);
    let cache_bytes = parse_flag(&args, "--cache-bytes")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let bypass_oversized_experts = has_flag(&args, "--bypass-oversized-experts");

    // Manual expert mode
    let manual_ids: Option<Vec<usize>> = parse_flag(&args, "--expert-ids")
        .map(|raw| parse_comma_separated_usize(&raw))
        .transpose()?;
    let manual_weights_raw: Option<String> = parse_flag(&args, "--expert-weights");
    let manual_mode = manual_ids.is_some();

    // ── 1. Load parser metadata ─────────────────────────────────────────
    let parse_dir_path = Path::new(&parse_dir);
    let meta = load_parser_metadata(parse_dir_path)?;

    // ── 2. Validate ─────────────────────────────────────────────────────
    validate_layer(&meta, layer)?;
    require_explicit_experts(&meta)?;

    let hidden_size = meta.layout.hidden_size;
    let intermediate_size = meta.layout.intermediate_size;
    let num_experts = meta.layout.num_experts;
    let top_k = meta.layout.top_k;

    if top_k == 0 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "validate top_k",
            "top_k is 0".to_string(),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    // ── 4. Generate hidden vector ───────────────────────────────────────
    let hidden = seeded_f32s(hidden_size, seed);

    // ── 5. Open model weights lazily ────────────────────────────────────
    let t_load_start = Instant::now();
    let model_weights = ModelWeights::open(&model_dir).map_err(|e| {
        CudaError::new(
            CudaErrorKind::Io,
            format!("open model weights from {}", model_dir),
            e.to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })?;

    let (selected_ids, selected_weights, router_tensor_name, router_ms) = if manual_mode {
        // ── Manual expert selection ────────────────────────────────────
        let ids = manual_ids.unwrap();
        validate_manual_expert_ids(&ids, num_experts, top_k)?;

        let weights = if let Some(ref raw) = manual_weights_raw {
            if raw == "uniform" {
                let w = 1.0 / top_k as f32;
                vec![w; top_k]
            } else {
                let w = parse_comma_separated_f32(raw)?;
                validate_manual_expert_weights(&w, top_k)?;
                w
            }
        } else {
            let w = 1.0 / top_k as f32;
            vec![w; top_k]
        };

        (ids, weights, None, None)
    } else {
        // ── Router-based expert selection ───────────────────────────────
        // ── 3. Find router tensor ───────────────────────────────────────
        let router_info = find_router_tensor(&meta, layer)?;
        validate_dtype(&router_info.dtype)?;

        // ── 6. Load router tensor ───────────────────────────────────────
        let mut router_fp32 = Vec::new();
        model_weights
            .get_f32(&router_info.name, &mut router_fp32)
            .map_err(|e| {
                CudaError::new(
                    CudaErrorKind::Io,
                    format!("load router tensor '{}'", router_info.name),
                    e.to_string(),
                    file!(),
                    line!(),
                    module_path!(),
                )
            })?;

        let expected_router_elems = num_experts * hidden_size;
        if router_fp32.len() != expected_router_elems {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "validate router tensor",
                format!(
                    "router tensor '{}' has {} elements, expected {}",
                    router_info.name,
                    router_fp32.len(),
                    expected_router_elems
                ),
                file!(),
                line!(),
                module_path!(),
            ));
        }

        // ── 7. CPU router ───────────────────────────────────────────────
        let router_start = Instant::now();
        let (ids, weights) =
            cpu_router(&router_fp32, &hidden, num_experts, hidden_size, top_k)?;
        let rms = router_start.elapsed().as_secs_f32() * 1000.0;

        (ids, weights, Some(router_info.name), Some(rms))
    };

    // ── 8. Find and load selected expert tensors ────────────────────────
    let mut expert_tensor_names: Vec<String> = Vec::new();
    let mut scale_tensor_names: Vec<String> = Vec::new();
    let mut source_dtypes: HashMap<String, String> = HashMap::new();
    let mut expert_fp32_set: Vec<ExpertWeightsFp32> = Vec::new();
    let mut fp4_decode_ms: Option<f32> = None;

    // Detect whether the first expert uses FP4 storage
    let first_expert_uses_fp4 = if !selected_ids.is_empty() {
        meta.expert_layout.tensors.iter().any(|t| {
            t.layer_id == Some(layer)
                && t.expert_id == Some(Some(selected_ids[0]))
                && t.storage_dtype.as_deref() == Some("I8")
        })
    } else {
        false
    };

    if first_expert_uses_fp4 {
        // FP4 decode path
        let fp4_start = Instant::now();
        for &eid in &selected_ids {
            let fp4_tensors = find_fp4_expert_tensors(&meta, layer, eid)?;

            // Record dtypes
            source_dtypes.insert("gate".into(), "I8".into());
            source_dtypes.insert("up".into(), "I8".into());
            source_dtypes.insert("down".into(), "I8".into());
            source_dtypes.insert("gate_scale".into(), "F8_E8M0".into());
            source_dtypes.insert("up_scale".into(), "F8_E8M0".into());
            source_dtypes.insert("down_scale".into(), "F8_E8M0".into());

            // Decode gate
            let gate_logical = fp4_tensors.gate_logical_shape.clone();
            let gate_phys = vec![gate_logical[0], gate_logical[1] / 2];
            let gate_fp32 = load_fp4_expert_weight(
                &model_weights,
                &fp4_tensors.gate_name,
                &fp4_tensors.gate_scale_name,
                &gate_phys,
                &gate_logical,
                fp4_tensors.gate_block_size,
            )?;
            if gate_fp32.len() != intermediate_size * hidden_size {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate fp4 gate shape",
                    format!(
                        "'{}' decoded to {} elements, expected {}",
                        fp4_tensors.gate_name,
                        gate_fp32.len(),
                        intermediate_size * hidden_size
                    ),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }

            // Decode up
            let up_logical = fp4_tensors.up_logical_shape.clone();
            let up_phys = vec![up_logical[0], up_logical[1] / 2];
            let up_fp32 = load_fp4_expert_weight(
                &model_weights,
                &fp4_tensors.up_name,
                &fp4_tensors.up_scale_name,
                &up_phys,
                &up_logical,
                fp4_tensors.up_block_size,
            )?;
            if up_fp32.len() != intermediate_size * hidden_size {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate fp4 up shape",
                    format!(
                        "'{}' decoded to {} elements, expected {}",
                        fp4_tensors.up_name,
                        up_fp32.len(),
                        intermediate_size * hidden_size
                    ),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }

            // Decode down
            let down_logical = fp4_tensors.down_logical_shape.clone();
            let down_phys = vec![down_logical[0], down_logical[1] / 2];
            let down_fp32 = load_fp4_expert_weight(
                &model_weights,
                &fp4_tensors.down_name,
                &fp4_tensors.down_scale_name,
                &down_phys,
                &down_logical,
                fp4_tensors.down_block_size,
            )?;
            if down_fp32.len() != hidden_size * intermediate_size {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate fp4 down shape",
                    format!(
                        "'{}' decoded to {} elements, expected {}",
                        fp4_tensors.down_name,
                        down_fp32.len(),
                        hidden_size * intermediate_size
                    ),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }

            expert_tensor_names.push(fp4_tensors.gate_name.clone());
            expert_tensor_names.push(fp4_tensors.up_name.clone());
            expert_tensor_names.push(fp4_tensors.down_name.clone());
            scale_tensor_names.push(fp4_tensors.gate_scale_name.clone());
            scale_tensor_names.push(fp4_tensors.up_scale_name.clone());
            scale_tensor_names.push(fp4_tensors.down_scale_name.clone());

            expert_fp32_set.push(ExpertWeightsFp32 {
                w_gate: gate_fp32,
                w_up: up_fp32,
                w_down: down_fp32,
            });
        }
        fp4_decode_ms = Some(fp4_start.elapsed().as_secs_f32() * 1000.0);
    } else {
        // Non-FP4 path (original BF16/F32/F16 loading)
        for &eid in &selected_ids {
            let tensors = find_expert_tensors(&meta, layer, eid)?;

            // Find dtype from metadata
            for t in &meta.expert_layout.tensors {
                if t.layer_id == Some(layer)
                    && t.expert_id == Some(Some(eid))
                {
                    let kind = t.tensor_kind.clone();
                    if !source_dtypes.contains_key(&kind) {
                        source_dtypes.insert(kind, t.dtype.clone());
                    }
                }
            }

            // Load gate
            let mut gate_fp32 = Vec::new();
            model_weights.get_f32(&tensors.gate_name, &mut gate_fp32).map_err(|e| {
                CudaError::new(
                    CudaErrorKind::Io,
                    format!("load '{}'", tensors.gate_name),
                    e.to_string(),
                    file!(),
                    line!(),
                    module_path!(),
                )
            })?;
            if gate_fp32.len() != intermediate_size * hidden_size {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert tensor",
                    format!(
                        "'{}' has {} elements, expected {}",
                        tensors.gate_name,
                        gate_fp32.len(),
                        intermediate_size * hidden_size
                    ),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }

            // Load up
            let mut up_fp32 = Vec::new();
            model_weights.get_f32(&tensors.up_name, &mut up_fp32).map_err(|e| {
                CudaError::new(
                    CudaErrorKind::Io,
                    format!("load '{}'", tensors.up_name),
                    e.to_string(),
                    file!(),
                    line!(),
                    module_path!(),
                )
            })?;
            if up_fp32.len() != intermediate_size * hidden_size {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert tensor",
                    format!(
                        "'{}' has {} elements, expected {}",
                        tensors.up_name,
                        up_fp32.len(),
                        intermediate_size * hidden_size
                    ),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }

            // Load down
            let mut down_fp32 = Vec::new();
            model_weights.get_f32(&tensors.down_name, &mut down_fp32).map_err(|e| {
                CudaError::new(
                    CudaErrorKind::Io,
                    format!("load '{}'", tensors.down_name),
                    e.to_string(),
                    file!(),
                    line!(),
                    module_path!(),
                )
            })?;
            if down_fp32.len() != hidden_size * intermediate_size {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "validate expert tensor",
                    format!(
                        "'{}' has {} elements, expected {}",
                        tensors.down_name,
                        down_fp32.len(),
                        hidden_size * intermediate_size
                    ),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }

            expert_tensor_names.push(tensors.gate_name.clone());
            expert_tensor_names.push(tensors.up_name.clone());
            expert_tensor_names.push(tensors.down_name.clone());

            expert_fp32_set.push(ExpertWeightsFp32 {
                w_gate: gate_fp32,
                w_up: up_fp32,
                w_down: down_fp32,
            });
        }
    };

    let tensor_load_ms = t_load_start.elapsed().as_secs_f32() * 1000.0;

    // ── 9. CPU fp32 MoE reference ───────────────────────────────────────
    let ref_out = selected_moe_cpu_fp32(
        &expert_fp32_set,
        &selected_ids
            .iter()
            .zip(selected_weights.iter())
            .map(|(id, w)| (*id, *w))
            .collect::<Vec<_>>(),
        &hidden,
        hidden_size,
        intermediate_size,
        hidden_size,
    )?;

    // ── 10. Quantize to Q4_0 ────────────────────────────────────────────
    let quantize_start = Instant::now();
    let shape_gate_up = QGemvShape::new(QuantFormat::Q4_0, intermediate_size, hidden_size);
    let shape_down = QGemvShape::new(QuantFormat::Q4_0, hidden_size, intermediate_size);

    let mut expert_q4_set: Vec<ExpertWeights> = Vec::new();
    for ef in &expert_fp32_set {
        let w_gate = q4_quantize_matrix_cpu(&ef.w_gate, shape_gate_up)?;
        let w_up = q4_quantize_matrix_cpu(&ef.w_up, shape_gate_up)?;
        let w_down = q4_quantize_matrix_cpu(&ef.w_down, shape_down)?;
        expert_q4_set.push(ExpertWeights {
            w_gate,
            w_up,
            w_down,
        });
    }
    let quantize_ms = quantize_start.elapsed().as_secs_f32() * 1000.0;

    // ── 11. CUDA selected MoE ───────────────────────────────────────────
    let backend = CudaBackendBuilder::new().stream_count(1).build()?;
    let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
    let moe_executor = MoeExecutor::new(backend.context().clone(), backend.device_info().clone());
    let stream = backend.stream_pool().stream(0)?;

    quant.compile_format(QuantFormat::Q4_0)?;
    moe_executor.compile()?;

    let mut cache = if cache_bytes > 0 {
        let mut c = CudaExpertCache::new(cache_bytes);
        c.bypass_oversized_experts = bypass_oversized_experts;
        Some(c)
    } else {
        None
    };

    let selected_pairs: Vec<(usize, f32)> = selected_ids
        .iter()
        .zip(selected_weights.iter())
        .map(|(id, w)| (*id, *w))
        .collect();

    let (cuda_out, moe_telemetry) = moe_executor.execute_selected_moe_cuda(
        &quant,
        stream,
        &expert_q4_set,
        &selected_pairs,
        &hidden,
        hidden_size,
        intermediate_size,
        hidden_size,
        layer,
        cache.as_mut(),
    )?;

    // ── 12. Compare outputs ─────────────────────────────────────────────
    let quant_vs_fp32 = compare_outputs(&ref_out, &cuda_out)?;

    // ── 13. Build report ────────────────────────────────────────────────
    let source_label = if manual_mode {
        "real_deepseek_v4_flash_manual_expert_single_layer_moe"
    } else {
        SOURCE_LABEL
    };

    let cache_counters = if let Some(ref c) = cache {
        (
            c.hit_count,
            c.miss_count,
            c.eviction_count,
            c.cache_insert_attempt_count,
            c.cache_insert_accept_count,
            c.cache_insert_bypass_count,
            c.oversized_tensor_bypass_count,
            c.oversized_expert_bypass_count,
            c.self_eviction_risk_count,
        )
    } else {
        (0, 0, 0, 0, 0, 0, 0, 0, 0)
    };

    let scale_tensor_names_used = if scale_tensor_names.is_empty() {
        None
    } else {
        Some(scale_tensor_names)
    };

    let report = M9Report {
        source: source_label.to_string(),
        model_dir,
        parse_dir,
        layer_id: layer,
        layout_kind: meta.expert_layout.layout_kind.clone(),
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        selected_expert_ids: selected_ids,
        selected_expert_weights: selected_weights,
        router_tensor_name,
        expert_tensor_names_used: expert_tensor_names,
        scale_tensor_names_used,
        source_dtypes,
        quant_format: "Q4_0",
        fp4_decode_ms,
        logical_expert_bytes_requested: moe_telemetry.logical_expert_bytes_requested,
        actual_expert_bytes_loaded: moe_telemetry.actual_expert_bytes_loaded,
        resident_cache_bytes_reused: moe_telemetry.resident_cache_bytes_reused,
        resident_cache_resident_bytes: moe_telemetry.resident_cache_resident_bytes,
        dequantized_scratch_bytes: moe_telemetry.dequantized_scratch_bytes,
        selected_working_set_bytes: moe_telemetry.selected_working_set_bytes,
        bytes_per_expert: moe_telemetry.bytes_per_expert,
        bytes_by_tensor_kind: moe_telemetry.bytes_by_tensor_kind,
        cache_hit_count: cache_counters.0,
        cache_miss_count: cache_counters.1,
        cache_eviction_count: cache_counters.2,
        cache_insert_attempt_count: cache_counters.3,
        cache_insert_accept_count: cache_counters.4,
        cache_insert_bypass_count: cache_counters.5,
        oversized_tensor_bypass_count: cache_counters.6,
        oversized_expert_bypass_count: cache_counters.7,
        self_eviction_risk_count: cache_counters.8,
        router_ms,
        tensor_load_ms,
        quantize_ms,
        h2d_ms: moe_telemetry.h2d_ms,
        gate_up_qgemv_ms: moe_telemetry.gate_up_qgemv_ms,
        activation_ms: moe_telemetry.activation_ms,
        down_qgemv_ms: moe_telemetry.down_qgemv_ms,
        accum_ms: moe_telemetry.accum_ms,
        total_ms: moe_telemetry.total_ms,
        quant_vs_fp32,
    };

    println!("{}", serde_json::to_string_pretty(&report).unwrap());

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a minimal safetensors file with given tensors.
    /// Each tensor entry: (name, dtype_string, shape, byte_offset_range)
    fn write_mock_safetensors(
        path: &Path,
        tensors: &HashMap<String, (String, Vec<usize>, (usize, usize), Vec<u8>)>,
    ) -> std::io::Result<()> {
        let mut header_map = serde_json::Map::new();
        let mut max_offset = 0usize;
        let mut data_blocks: Vec<(usize, &Vec<u8>)> = Vec::new();

        for (name, (dtype, shape, offsets, data)) in tensors {
            let mut entry = serde_json::Map::new();
            entry.insert("dtype".to_string(), serde_json::Value::String(dtype.clone()));
            entry.insert(
                "shape".to_string(),
                serde_json::Value::Array(
                    shape.iter().map(|&s| serde_json::Value::Number(s.into())).collect(),
                ),
            );
            entry.insert(
                "data_offsets".to_string(),
                serde_json::Value::Array(vec![
                    serde_json::Value::Number(offsets.0.into()),
                    serde_json::Value::Number(offsets.1.into()),
                ]),
            );
            header_map.insert(name.clone(), serde_json::Value::Object(entry));
            if offsets.1 > max_offset {
                max_offset = offsets.1;
            }
            data_blocks.push((offsets.0, data));
        }

        let header_json = serde_json::to_string(&header_map).unwrap();
        // Pad header JSON so that data_start = 8 + header_len is 8-byte aligned.
        // ModelWeights::open accesses data via f32 pointers which need 4-byte alignment.
        let pad = (8 - (header_json.as_bytes().len() % 8)) % 8;
        let header_json_padded = format!("{}{}", header_json, " ".repeat(pad));
        let header_bytes = header_json_padded.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file = std::fs::File::create(path)?;
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(header_bytes)?;

        // Write data section with actual content at specified offsets
        let mut data_buf = vec![0u8; max_offset];
        for (offset, data) in &data_blocks {
            let end = offset + data.len();
            data_buf[*offset..end].copy_from_slice(data);
        }
        file.write_all(&data_buf)?;

        Ok(())
    }

    /// Build a minimal parser output directory for testing.
    fn write_parser_json_files(
        dir: &Path,
        num_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
        dtype: &str,
        tensors: &[ExpertTensorEntry],
        routers: &[RouterTensorEntry],
    ) {
        std::fs::create_dir_all(dir).unwrap();

        let layout = serde_json::json!({
            "model_name": "test_m9",
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "vocab_size": 1000,
            "dtype": dtype,
            "quant_dtype": null,
            "tensor_count": tensors.len() + routers.len(),
            "shard_count": 1,
            "total_byte_size": 0,
            "largest_tensor": null,
            "tensor_name_patterns": []
        });
        std::fs::write(
            dir.join("deepseek_v4_flash_layout.json"),
            serde_json::to_string_pretty(&layout).unwrap(),
        )
        .unwrap();

        let expert_layout = serde_json::json!({
            "layout_kind": "explicit_experts",
            "tensors": tensors
        });
        std::fs::write(
            dir.join("deepseek_v4_flash_expert_layout.json"),
            serde_json::to_string_pretty(&expert_layout).unwrap(),
        )
        .unwrap();

        let router_layout = serde_json::json!({
            "num_experts": num_experts,
            "top_k": top_k,
            "warnings": [],
            "routers": routers
        });
        std::fs::write(
            dir.join("deepseek_v4_flash_router_layout.json"),
            serde_json::to_string_pretty(&router_layout).unwrap(),
        )
        .unwrap();

        // Write minimal inventory summary and tensor index too (not used by m9
        // but the parser output convention includes them)
        let inventory = serde_json::json!({
            "total_expert_bytes": 0,
            "expert_bytes_per_layer": {},
            "bytes_per_expert": {},
            "bytes_by_tensor_kind": {"gate":0,"up":0,"down":0,"gate_up":0,"router":0,"shared":0,"attention":0,"other":0},
            "largest_expert_tensor": null,
            "largest_layer_by_expert_bytes": null,
            "fits_in_cache": {"1GB": true, "2GB": true, "4GB": true, "8GB": true}
        });
        std::fs::write(
            dir.join("deepseek_v4_flash_inventory_summary.json"),
            serde_json::to_string_pretty(&inventory).unwrap(),
        )
        .unwrap();

        // Build tensor index from the same tensor entries
        let mut index_map = serde_json::Map::new();
        for t in tensors {
            index_map.insert(
                t.name.clone(),
                serde_json::json!({
                    "shape": t.shape,
                    "dtype": t.dtype,
                    "byte_length": t.byte_length,
                    "offset": 0,
                    "source_file": "model.safetensors"
                }),
            );
        }
        for r in routers {
            index_map.insert(
                r.name.clone(),
                serde_json::json!({
                    "shape": r.shape,
                    "dtype": r.dtype,
                    "byte_length": 0,
                    "offset": 0,
                    "source_file": "model.safetensors"
                }),
            );
        }
        std::fs::write(
            dir.join("deepseek_v4_flash_tensor_index.json"),
            serde_json::to_string_pretty(&index_map).unwrap(),
        )
        .unwrap();
    }

    /// Build synthetic fp32 data as little-endian bytes.
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    #[test]
    fn test_explicit_single_layer_works() -> Result<()> {
        let tmp = std::env::temp_dir().join("objeta_m9_test_explicit");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let hidden = 64usize;
        let intermediate = 32usize;
        let num_experts = 4usize;
        let top_k = 2usize;

        // Build synthetic tensors with known values
        // Router: [num_experts, hidden] — all zeros so logits are uniform
        let router_data = vec![0.1f32; num_experts * hidden];
        let router_bytes = f32_bytes(&router_data);

        // Expert 0 gate/up/down
        let gate0 = vec![0.01f32; intermediate * hidden];
        let up0 = vec![0.02f32; intermediate * hidden];
        let down0 = vec![0.03f32; hidden * intermediate];
        let gate0_bytes = f32_bytes(&gate0);
        let up0_bytes = f32_bytes(&up0);
        let down0_bytes = f32_bytes(&down0);

        // Expert 1 gate/up/down
        let gate1 = vec![0.01f32; intermediate * hidden];
        let up1 = vec![0.02f32; intermediate * hidden];
        let down1 = vec![0.03f32; hidden * intermediate];
        let gate1_bytes = f32_bytes(&gate1);
        let up1_bytes = f32_bytes(&up1);
        let down1_bytes = f32_bytes(&down1);

        // Expert 2 gate/up/down
        let gate2 = vec![0.01f32; intermediate * hidden];
        let up2 = vec![0.02f32; intermediate * hidden];
        let down2 = vec![0.03f32; hidden * intermediate];
        let gate2_bytes = f32_bytes(&gate2);
        let up2_bytes = f32_bytes(&up2);
        let down2_bytes = f32_bytes(&down2);

        // Expert 3 gate/up/down
        let gate3 = vec![0.01f32; intermediate * hidden];
        let up3 = vec![0.02f32; intermediate * hidden];
        let down3 = vec![0.03f32; hidden * intermediate];
        let gate3_bytes = f32_bytes(&gate3);
        let up3_bytes = f32_bytes(&up3);
        let down3_bytes = f32_bytes(&down3);

        // Compute byte offsets for each tensor in the safetensors data section
        let router_off = (0, router_bytes.len());
        let gate0_off = (router_off.1, router_off.1 + gate0_bytes.len());
        let up0_off = (gate0_off.1, gate0_off.1 + up0_bytes.len());
        let down0_off = (up0_off.1, up0_off.1 + down0_bytes.len());
        let gate1_off = (down0_off.1, down0_off.1 + gate1_bytes.len());
        let up1_off = (gate1_off.1, gate1_off.1 + up1_bytes.len());
        let down1_off = (up1_off.1, up1_off.1 + down1_bytes.len());
        let gate2_off = (down1_off.1, down1_off.1 + gate2_bytes.len());
        let up2_off = (gate2_off.1, gate2_off.1 + up2_bytes.len());
        let down2_off = (up2_off.1, up2_off.1 + down2_bytes.len());
        let gate3_off = (down2_off.1, down2_off.1 + gate3_bytes.len());
        let up3_off = (gate3_off.1, gate3_off.1 + up3_bytes.len());
        let down3_off = (up3_off.1, up3_off.1 + down3_bytes.len());

        let mut sf_tensors = HashMap::new();
        sf_tensors.insert(
            "model.layers.0.mlp.gate.weight".to_string(),
            (
                "F32".to_string(),
                vec![num_experts, hidden],
                router_off,
                router_bytes,
            ),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
            ("F32".to_string(), vec![intermediate, hidden], gate0_off, gate0_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.0.up_proj.weight".to_string(),
            ("F32".to_string(), vec![intermediate, hidden], up0_off, up0_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.0.down_proj.weight".to_string(),
            ("F32".to_string(), vec![hidden, intermediate], down0_off, down0_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
            ("F32".to_string(), vec![intermediate, hidden], gate1_off, gate1_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.1.up_proj.weight".to_string(),
            ("F32".to_string(), vec![intermediate, hidden], up1_off, up1_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.1.down_proj.weight".to_string(),
            ("F32".to_string(), vec![hidden, intermediate], down1_off, down1_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.2.gate_proj.weight".to_string(),
            ("F32".to_string(), vec![intermediate, hidden], gate2_off, gate2_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.2.up_proj.weight".to_string(),
            ("F32".to_string(), vec![intermediate, hidden], up2_off, up2_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.2.down_proj.weight".to_string(),
            ("F32".to_string(), vec![hidden, intermediate], down2_off, down2_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.3.gate_proj.weight".to_string(),
            ("F32".to_string(), vec![intermediate, hidden], gate3_off, gate3_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.3.up_proj.weight".to_string(),
            ("F32".to_string(), vec![intermediate, hidden], up3_off, up3_bytes),
        );
        sf_tensors.insert(
            "model.layers.0.mlp.experts.3.down_proj.weight".to_string(),
            ("F32".to_string(), vec![hidden, intermediate], down3_off, down3_bytes),
        );

        let model_dir = tmp.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        write_mock_safetensors(&model_dir.join("model.safetensors"), &sf_tensors).unwrap();

        // Write parser JSONs
        let parse_dir = tmp.join("parse");
        let expert_entries: Vec<ExpertTensorEntry> = (0..num_experts)
            .flat_map(|e| {
                vec![
                    ExpertTensorEntry {
                        name: format!("model.layers.0.mlp.experts.{}.gate_proj.weight", e),
                        layer_id: Some(0),
                        expert_id: Some(Some(e)),
                        tensor_kind: "gate".into(),
                        shape: vec![intermediate, hidden],
                        dtype: "F32".into(),
                        byte_length: intermediate * hidden * 4,
                        source_file: "model.safetensors".into(),
                        ..Default::default()
                    },
                    ExpertTensorEntry {
                        name: format!("model.layers.0.mlp.experts.{}.up_proj.weight", e),
                        layer_id: Some(0),
                        expert_id: Some(Some(e)),
                        tensor_kind: "up".into(),
                        shape: vec![intermediate, hidden],
                        dtype: "F32".into(),
                        byte_length: intermediate * hidden * 4,
                        source_file: "model.safetensors".into(),
                        ..Default::default()
                    },
                    ExpertTensorEntry {
                        name: format!("model.layers.0.mlp.experts.{}.down_proj.weight", e),
                        layer_id: Some(0),
                        expert_id: Some(Some(e)),
                        tensor_kind: "down".into(),
                        shape: vec![hidden, intermediate],
                        dtype: "F32".into(),
                        byte_length: hidden * intermediate * 4,
                        source_file: "model.safetensors".into(),
                        ..Default::default()
                    },
                ]
            })
            .collect();

        let router_entries = vec![RouterTensorEntry {
            name: "model.layers.0.mlp.gate.weight".into(),
            layer_id: Some(0),
            shape: vec![num_experts, hidden],
            dtype: "F32".into(),
        }];

        write_parser_json_files(
            &parse_dir,
            1,
            hidden,
            intermediate,
            num_experts,
            top_k,
            "float32",
            &expert_entries,
            &router_entries,
        );

        // Run the main logic programmatically
        let meta = load_parser_metadata(&parse_dir)?;
        validate_layer(&meta, 0)?;
        require_explicit_experts(&meta)?;
        assert_eq!(meta.layout.hidden_size, hidden);
        assert_eq!(meta.layout.intermediate_size, intermediate);
        assert_eq!(meta.layout.num_experts, num_experts);
        assert_eq!(meta.layout.top_k, top_k);

        let router_info = find_router_tensor(&meta, 0)?;
        assert_eq!(router_info.name, "model.layers.0.mlp.gate.weight");

        let hidden_vec = seeded_f32s(hidden, 42);

        // Load router
        let model_weights = ModelWeights::open(&model_dir).map_err(|e| {
            CudaError::new(
                CudaErrorKind::Io,
                "open model",
                e.to_string(),
                file!(),
                line!(),
                module_path!(),
            )
        })?;
        let mut router_fp32 = Vec::new();
        model_weights.get_f32(&router_info.name, &mut router_fp32).map_err(|e| {
            CudaError::new(
                CudaErrorKind::Io,
                "load router",
                e.to_string(),
                file!(),
                line!(),
                module_path!(),
            )
        })?;

        // Run router
        let (selected_ids, selected_weights) =
            cpu_router(&router_fp32, &hidden_vec, num_experts, hidden, top_k)?;
        assert_eq!(selected_ids.len(), top_k);
        assert_eq!(selected_weights.len(), top_k);

        // Find and load expert tensors
        for &eid in &selected_ids {
            let tensors = find_expert_tensors(&meta, 0, eid)?;
            assert!(tensors.gate_name.contains(&format!("experts.{}", eid)));
            assert!(tensors.up_name.contains(&format!("experts.{}", eid)));
            assert!(tensors.down_name.contains(&format!("experts.{}", eid)));

            let mut gate = Vec::new();
            model_weights.get_f32(&tensors.gate_name, &mut gate).unwrap();
            assert_eq!(gate.len(), intermediate * hidden);

            let mut up = Vec::new();
            model_weights.get_f32(&tensors.up_name, &mut up).unwrap();
            assert_eq!(up.len(), intermediate * hidden);

            let mut down = Vec::new();
            model_weights.get_f32(&tensors.down_name, &mut down).unwrap();
            assert_eq!(down.len(), hidden * intermediate);
        }

        // Verify only selected expert tensors were loaded (not all 4 experts)
        // The test loaded only top_k experts' tensors

        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }

    #[test]
    fn test_missing_router_fails_clearly() {
        let tmp = std::env::temp_dir().join("objeta_m9_test_no_router");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let parse_dir = tmp.join("parse");
        write_parser_json_files(
            &parse_dir,
            1,
            64,
            32,
            4,
            2,
            "float32",
            &[], // no expert tensors
            &[], // no router tensors
        );

        let meta = load_parser_metadata(&parse_dir).unwrap();
        let err = find_router_tensor(&meta, 0).unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::InvalidInput);
        assert!(err.source_message.contains("no router tensor"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_missing_expert_tensor_fails_clearly() -> Result<()> {
        let tmp = std::env::temp_dir().join("objeta_m9_test_missing_expert");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let hidden = 64;
        let intermediate = 32;
        let num_experts = 4;

        let parse_dir = tmp.join("parse");
        let router_entries = vec![RouterTensorEntry {
            name: "model.layers.0.mlp.gate.weight".into(),
            layer_id: Some(0),
            shape: vec![num_experts, hidden],
            dtype: "F32".into(),
        }];

        // Only expert 0 has gate+up+down; expert 1 has only gate (missing up, down)
        let expert_entries = vec![
            ExpertTensorEntry {
                name: "model.layers.0.mlp.experts.0.gate_proj.weight".into(),
                layer_id: Some(0),
                expert_id: Some(Some(0)),
                tensor_kind: "gate".into(),
                shape: vec![intermediate, hidden],
                dtype: "F32".into(),
                byte_length: intermediate * hidden * 4,
                source_file: "model.safetensors".into(),
                ..Default::default()
            },
            ExpertTensorEntry {
                name: "model.layers.0.mlp.experts.0.up_proj.weight".into(),
                layer_id: Some(0),
                expert_id: Some(Some(0)),
                tensor_kind: "up".into(),
                shape: vec![intermediate, hidden],
                dtype: "F32".into(),
                byte_length: intermediate * hidden * 4,
                source_file: "model.safetensors".into(),
                ..Default::default()
            },
            ExpertTensorEntry {
                name: "model.layers.0.mlp.experts.0.down_proj.weight".into(),
                layer_id: Some(0),
                expert_id: Some(Some(0)),
                tensor_kind: "down".into(),
                shape: vec![hidden, intermediate],
                dtype: "F32".into(),
                byte_length: hidden * intermediate * 4,
                source_file: "model.safetensors".into(),
                ..Default::default()
            },
            // Expert 1: only gate, missing up and down
            ExpertTensorEntry {
                name: "model.layers.0.mlp.experts.1.gate_proj.weight".into(),
                layer_id: Some(0),
                expert_id: Some(Some(1)),
                tensor_kind: "gate".into(),
                shape: vec![intermediate, hidden],
                dtype: "F32".into(),
                byte_length: intermediate * hidden * 4,
                source_file: "model.safetensors".into(),
                ..Default::default()
            },
        ];

        write_parser_json_files(
            &parse_dir,
            1,
            hidden,
            intermediate,
            num_experts,
            2,
            "float32",
            &expert_entries,
            &router_entries,
        );

        let meta = load_parser_metadata(&parse_dir).unwrap();

        // Expert 0 should be found
        let t0 = find_expert_tensors(&meta, 0, 0)?;
        assert!(t0.gate_name.contains("experts.0"));
        assert!(t0.up_name.contains("experts.0"));
        assert!(t0.down_name.contains("experts.0"));

        // Expert 1 should fail (missing up and down)
        let err = find_expert_tensors(&meta, 0, 1).unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::InvalidInput);
        assert!(err.source_message.contains("missing tensors"));

        // Expert 2 should fail (no tensors at all)
        let err = find_expert_tensors(&meta, 0, 2).unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::InvalidInput);
        assert!(err.source_message.contains("missing tensors"));

        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }

    #[test]
    fn test_packed_experts_refuses_to_run() {
        let tmp = std::env::temp_dir().join("objeta_m9_test_packed_refuse");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let parse_dir = tmp.join("parse");
        std::fs::create_dir_all(&parse_dir).unwrap();

        // Write layout with explicit_experts in layout but override expert_layout
        // to be packed_experts
        let layout = serde_json::json!({
            "model_name": "test",
            "num_layers": 1,
            "hidden_size": 64,
            "intermediate_size": 32,
            "num_experts": 4,
            "top_k": 2,
            "vocab_size": 1000,
            "dtype": "float32",
            "quant_dtype": null,
            "tensor_count": 1,
            "shard_count": 1,
            "total_byte_size": 0,
            "largest_tensor": null,
            "tensor_name_patterns": []
        });
        std::fs::write(
            parse_dir.join("deepseek_v4_flash_layout.json"),
            serde_json::to_string_pretty(&layout).unwrap(),
        )
        .unwrap();

        let expert_layout = serde_json::json!({
            "layout_kind": "packed_experts",
            "tensors": [{
                "name": "model.layers.0.mlp.experts.gate_proj.weight",
                "layer_id": 0,
                "expert_id": null,
                "tensor_kind": "gate",
                "shape": [4, 32, 64],
                "dtype": "F32",
                "byte_length": 32768,
                "source_file": "model.safetensors"
            }]
        });
        std::fs::write(
            parse_dir.join("deepseek_v4_flash_expert_layout.json"),
            serde_json::to_string_pretty(&expert_layout).unwrap(),
        )
        .unwrap();

        let router_layout = serde_json::json!({
            "num_experts": 4,
            "top_k": 2,
            "warnings": [],
            "routers": [{
                "name": "model.layers.0.mlp.gate.weight",
                "layer_id": 0,
                "shape": [4, 64],
                "dtype": "F32",
                "byte_length": 512,
                "source_file": "model.safetensors"
            }]
        });
        std::fs::write(
            parse_dir.join("deepseek_v4_flash_router_layout.json"),
            serde_json::to_string_pretty(&router_layout).unwrap(),
        )
        .unwrap();

        // Also write inventory and tensor index (not used but parser convention)
        let inventory = serde_json::json!({"total_expert_bytes":0,"expert_bytes_per_layer":{},"bytes_per_expert":{},"bytes_by_tensor_kind":{"gate":0,"up":0,"down":0,"gate_up":0,"router":0,"shared":0,"attention":0,"other":0},"largest_expert_tensor":null,"largest_layer_by_expert_bytes":null,"fits_in_cache":{"1GB":true,"2GB":true,"4GB":true,"8GB":true}});
        std::fs::write(
            parse_dir.join("deepseek_v4_flash_inventory_summary.json"),
            serde_json::to_string_pretty(&inventory).unwrap(),
        )
        .unwrap();
        std::fs::write(
            parse_dir.join("deepseek_v4_flash_tensor_index.json"),
            "{}",
        )
        .unwrap();

        let meta = load_parser_metadata(&parse_dir).unwrap();
        let err = require_explicit_experts(&meta).unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::Unsupported);
        assert!(err.source_message.contains("packed_experts"));
        assert!(err.source_message.contains("refusing to guess offsets"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_unsupported_dtype_fails() {
        assert!(validate_dtype("F32").is_ok());
        assert!(validate_dtype("F16").is_ok());
        assert!(validate_dtype("BF16").is_ok());
        assert!(validate_dtype("float32").is_ok());
        assert!(validate_dtype("bfloat16").is_ok());

        let err = validate_dtype("INT8").unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::Unsupported);
        assert!(err.source_message.contains("INT8"));
    }

    #[test]
    fn test_router_shape_mismatch_fails() {
        let tmp = std::env::temp_dir().join("objeta_m9_test_shape_mismatch");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let parse_dir = tmp.join("parse");
        let router_entries = vec![RouterTensorEntry {
            name: "model.layers.0.mlp.gate.weight".into(),
            layer_id: Some(0),
            // Shape says 8 experts but layout says 4
            shape: vec![8, 64],
            dtype: "F32".into(),
        }];
        write_parser_json_files(
            &parse_dir,
            1,
            64,
            32,
            4, // num_experts=4 in layout, but router shape says 8
            2,
            "float32",
            &[],
            &router_entries,
        );

        let meta = load_parser_metadata(&parse_dir).unwrap();
        let err = find_router_tensor(&meta, 0).unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::InvalidInput);
        assert!(err.source_message.contains("expert dim"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_layer_out_of_range_fails() {
        let tmp = std::env::temp_dir().join("objeta_m9_test_layer_range");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let parse_dir = tmp.join("parse");
        write_parser_json_files(
            &parse_dir,
            2, // 2 layers (0, 1)
            64,
            32,
            4,
            2,
            "float32",
            &[],
            &[],
        );

        let meta = load_parser_metadata(&parse_dir).unwrap();
        assert!(validate_layer(&meta, 0).is_ok());
        assert!(validate_layer(&meta, 1).is_ok());

        let err = validate_layer(&meta, 2).unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::InvalidInput);
        assert!(err.source_message.contains("out of range"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_cpu_router_top_k_selection() -> Result<()> {
        // Create router weights that strongly favor expert 1 and 3
        let hidden = 4;
        let num_experts = 4;
        let top_k = 2;

        let mut router = vec![0.0f32; num_experts * hidden];
        // Expert 0: all zeros -> logit = 0
        // Expert 1: ones -> logit = sum(hidden)
        for h in 0..hidden {
            router[1 * hidden + h] = 1.0;
        }
        // Expert 2: zeros -> logit = 0
        // Expert 3: twos -> logit = 2 * sum(hidden)
        for h in 0..hidden {
            router[3 * hidden + h] = 2.0;
        }

        // Hidden vector: all 0.5
        let hidden_vec = vec![0.5f32; hidden];

        let (ids, weights) = cpu_router(&router, &hidden_vec, num_experts, hidden, top_k)?;

        assert_eq!(ids.len(), 2);
        // Expert 3 should be first (highest logit), then expert 1
        assert_eq!(ids[0], 3);
        assert_eq!(ids[1], 1);
        assert!(weights[0] > weights[1]);

        Ok(())
    }

    // ── Manual-expert mode tests ──────────────────────────────────────────

    #[test]
    fn test_manual_expert_ids_skip_router() {
        // Verify that parse_comma_separated_usize works correctly
        let ids = parse_comma_separated_usize("0,1,2,3,4,5").unwrap();
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 5]);

        // Verify single value
        let ids = parse_comma_separated_usize("42").unwrap();
        assert_eq!(ids, vec![42]);

        // Verify with whitespace
        let ids = parse_comma_separated_usize(" 0 , 1 , 2 ").unwrap();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_uniform_expert_weights_sum_to_one() {
        // Uniform weights for top_k=6: each 1/6
        let top_k = 6;
        let w = 1.0 / top_k as f32;
        let weights = vec![w; top_k];
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // Validate with validate_manual_expert_weights
        assert!(validate_manual_expert_weights(&weights, top_k).is_ok());
    }

    #[test]
    fn test_explicit_expert_weights_validation() {
        // Valid weights
        let weights = vec![0.166, 0.166, 0.166, 0.166, 0.166, 0.170];
        assert!(validate_manual_expert_weights(&weights, 6).is_ok());

        // Wrong count
        let err = validate_manual_expert_weights(&[0.5, 0.5], 6).unwrap_err();
        assert!(err.source_message.contains("expected 6"));

        // Sum != 1.0
        let err = validate_manual_expert_weights(&[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 6).unwrap_err();
        assert!(err.source_message.contains("sum to"));
    }

    #[test]
    fn test_invalid_expert_id_fails() {
        // Valid: within range
        assert!(validate_manual_expert_ids(&[0, 1, 2, 3, 4, 5], 256, 6).is_ok());

        // Out of range
        let err = validate_manual_expert_ids(&[0, 1, 256], 256, 3).unwrap_err();
        assert!(err.source_message.contains("out of range"));

        // Wrong count
        let err = validate_manual_expert_ids(&[0, 1], 256, 6).unwrap_err();
        assert!(err.source_message.contains("expected 6"));

        // Empty
        let err = validate_manual_expert_ids(&[], 256, 6).unwrap_err();
        assert!(err.source_message.contains("must not be empty"));
    }

    #[test]
    fn test_missing_scale_tensor_fails() -> Result<()> {
        let tmp = std::env::temp_dir().join("objeta_m9_test_missing_scale");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let hidden = 64;
        let intermediate = 32;

        let parse_dir = tmp.join("parse");
        let expert_entries = vec![
            // Expert 0: weight has storage_dtype=I8 but no scale_tensor_name
            ExpertTensorEntry {
                name: "layers.0.ffn.experts.0.w1.weight".into(),
                layer_id: Some(0),
                expert_id: Some(Some(0)),
                tensor_kind: "gate".into(),
                shape: vec![intermediate, hidden / 2],
                dtype: "I8".into(),
                byte_length: intermediate * hidden / 2,
                source_file: "model.safetensors".into(),
                storage_dtype: Some("I8".into()),
                logical_dtype: Some("FP4".into()),
                // scale_tensor_name is None — this should fail
                ..Default::default()
            },
            ExpertTensorEntry {
                name: "layers.0.ffn.experts.0.w2.weight".into(),
                layer_id: Some(0),
                expert_id: Some(Some(0)),
                tensor_kind: "up".into(),
                shape: vec![intermediate, hidden / 2],
                dtype: "I8".into(),
                byte_length: intermediate * hidden / 2,
                source_file: "model.safetensors".into(),
                storage_dtype: Some("I8".into()),
                logical_dtype: Some("FP4".into()),
                scale_tensor_name: Some("layers.0.ffn.experts.0.w2.scale".into()),
                scale_dtype: Some("F8_E8M0".into()),
                logical_shape: Some(vec![intermediate, hidden]),
                block_size: Some(32),
                ..Default::default()
            },
            ExpertTensorEntry {
                name: "layers.0.ffn.experts.0.w3.weight".into(),
                layer_id: Some(0),
                expert_id: Some(Some(0)),
                tensor_kind: "down".into(),
                shape: vec![hidden, intermediate / 2],
                dtype: "I8".into(),
                byte_length: hidden * intermediate / 2,
                source_file: "model.safetensors".into(),
                storage_dtype: Some("I8".into()),
                logical_dtype: Some("FP4".into()),
                scale_tensor_name: Some("layers.0.ffn.experts.0.w3.scale".into()),
                scale_dtype: Some("F8_E8M0".into()),
                logical_shape: Some(vec![hidden, intermediate]),
                block_size: Some(32),
                ..Default::default()
            },
        ];

        let router_entries = vec![RouterTensorEntry {
            name: "layers.0.ffn.gate.weight".into(),
            layer_id: Some(0),
            shape: vec![4, hidden],
            dtype: "BF16".into(),
        }];

        write_parser_json_files(
            &parse_dir,
            1,
            hidden,
            intermediate,
            4,
            2,
            "bfloat16",
            &expert_entries,
            &router_entries,
        );

        let meta = load_parser_metadata(&parse_dir)?;

        // find_fp4_expert_tensors should fail because gate weight has no scale_tensor_name
        let err = find_fp4_expert_tensors(&meta, 0, 0).unwrap_err();
        assert!(err.source_message.contains("scale_tensor_name"));
        assert!(err.source_message.contains("not FP4"));

        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }

    #[test]
    fn test_fp4_decode_flow_with_synthetic_tensors() -> Result<()> {
        let tmp = std::env::temp_dir().join("objeta_m9_test_fp4_decode_flow");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let hidden = 64;
        let intermediate = 32;
        let num_experts = 4;
        let top_k = 2;
        let block_size = 32;

        // Create synthetic FP4 (I8) weights and F8_E8M0 scales
        // Expert 0 gate: logical [32, 64], physical [32, 32]
        let gate_phys_rows = intermediate;
        let gate_phys_cols = hidden / 2; // 32
        let gate_logical = vec![intermediate, hidden];
        let gate_scale_cols = hidden / block_size; // 64/32 = 2
        let gate_bytes = vec![0x21u8; gate_phys_rows * gate_phys_cols]; // all 0.5,1.0 pattern
        let gate_scale_bytes = vec![127u8; gate_phys_rows * gate_scale_cols]; // scale=1.0

        // Expert 0 up: same shape as gate
        let up_bytes = vec![0x21u8; gate_phys_rows * gate_phys_cols];
        let up_scale_bytes = vec![127u8; gate_phys_rows * gate_scale_cols];

        // Expert 0 down: logical [64, 32], physical [64, 16]
        let down_phys_rows = hidden;
        let down_phys_cols = intermediate / 2; // 16
        let down_logical = vec![hidden, intermediate];
        let down_scale_cols = intermediate / block_size; // 32/32 = 1
        let down_bytes = vec![0x21u8; down_phys_rows * down_phys_cols];
        let down_scale_bytes = vec![127u8; down_phys_rows * down_scale_cols];

        // Expert 1: same patterns
        let gate1_bytes = vec![0x21u8; gate_phys_rows * gate_phys_cols];
        let gate1_scale_bytes = vec![127u8; gate_phys_rows * gate_scale_cols];
        let up1_bytes = vec![0x21u8; gate_phys_rows * gate_phys_cols];
        let up1_scale_bytes = vec![127u8; gate_phys_rows * gate_scale_cols];
        let down1_bytes = vec![0x21u8; down_phys_rows * down_phys_cols];
        let down1_scale_bytes = vec![127u8; down_phys_rows * down_scale_cols];

        // Write mock safetensors with all expert tensors
        let mut sf_tensors = HashMap::new();
        let ex0 = 0;
        let ex1 = 1;
        let g0 = gate_bytes.len();
        let gs0 = g0 + gate_scale_bytes.len();
        let u0 = gs0 + up_bytes.len();
        let us0 = u0 + up_scale_bytes.len();
        let d0 = us0 + down_bytes.len();
        let ds0 = d0 + down_scale_bytes.len();
        let g1 = ds0 + gate1_bytes.len();
        let gs1 = g1 + gate1_scale_bytes.len();
        let u1 = gs1 + up1_bytes.len();
        let us1 = u1 + up1_scale_bytes.len();
        let d1 = us1 + down1_bytes.len();
        let ds1 = d1 + down1_scale_bytes.len();

        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w1.weight", ex0), ("I8".into(), vec![intermediate, hidden / 2], (0, g0), gate_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w1.scale", ex0), ("F8_E8M0".into(), vec![intermediate, gate_scale_cols], (g0, gs0), gate_scale_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w2.weight", ex0), ("I8".into(), vec![intermediate, hidden / 2], (gs0, u0), up_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w2.scale", ex0), ("F8_E8M0".into(), vec![intermediate, gate_scale_cols], (u0, us0), up_scale_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w3.weight", ex0), ("I8".into(), vec![hidden, intermediate / 2], (us0, d0), down_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w3.scale", ex0), ("F8_E8M0".into(), vec![hidden, down_scale_cols], (d0, ds0), down_scale_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w1.weight", ex1), ("I8".into(), vec![intermediate, hidden / 2], (ds0, g1), gate1_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w1.scale", ex1), ("F8_E8M0".into(), vec![intermediate, gate_scale_cols], (g1, gs1), gate1_scale_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w2.weight", ex1), ("I8".into(), vec![intermediate, hidden / 2], (gs1, u1), up1_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w2.scale", ex1), ("F8_E8M0".into(), vec![intermediate, gate_scale_cols], (u1, us1), up1_scale_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w3.weight", ex1), ("I8".into(), vec![hidden, intermediate / 2], (us1, d1), down1_bytes));
        sf_tensors.insert(format!("layers.0.ffn.experts.{}.w3.scale", ex1), ("F8_E8M0".into(), vec![hidden, down_scale_cols], (d1, ds1), down1_scale_bytes));

        let model_dir = tmp.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        write_mock_safetensors(&model_dir.join("model.safetensors"), &sf_tensors).unwrap();

        // Write parser JSONs with FP4 metadata
        let parse_dir = tmp.join("parse");
        let mut expert_entries = Vec::new();
        for eid in [0, 1] {
            expert_entries.push(ExpertTensorEntry {
                name: format!("layers.0.ffn.experts.{}.w1.weight", eid),
                layer_id: Some(0),
                expert_id: Some(Some(eid)),
                tensor_kind: "gate".into(),
                shape: vec![intermediate, hidden / 2],
                dtype: "I8".into(),
                byte_length: intermediate * hidden / 2,
                source_file: "model.safetensors".into(),
                storage_dtype: Some("I8".into()),
                logical_dtype: Some("FP4".into()),
                scale_tensor_name: Some(format!("layers.0.ffn.experts.{}.w1.scale", eid)),
                scale_dtype: Some("F8_E8M0".into()),
                logical_shape: Some(gate_logical.clone()),
                block_size: Some(block_size),
                packed_values_per_byte: Some(2),
            });
            expert_entries.push(ExpertTensorEntry {
                name: format!("layers.0.ffn.experts.{}.w2.weight", eid),
                layer_id: Some(0),
                expert_id: Some(Some(eid)),
                tensor_kind: "up".into(),
                shape: vec![intermediate, hidden / 2],
                dtype: "I8".into(),
                byte_length: intermediate * hidden / 2,
                source_file: "model.safetensors".into(),
                storage_dtype: Some("I8".into()),
                logical_dtype: Some("FP4".into()),
                scale_tensor_name: Some(format!("layers.0.ffn.experts.{}.w2.scale", eid)),
                scale_dtype: Some("F8_E8M0".into()),
                logical_shape: Some(gate_logical.clone()),
                block_size: Some(block_size),
                packed_values_per_byte: Some(2),
            });
            expert_entries.push(ExpertTensorEntry {
                name: format!("layers.0.ffn.experts.{}.w3.weight", eid),
                layer_id: Some(0),
                expert_id: Some(Some(eid)),
                tensor_kind: "down".into(),
                shape: vec![hidden, intermediate / 2],
                dtype: "I8".into(),
                byte_length: hidden * intermediate / 2,
                source_file: "model.safetensors".into(),
                storage_dtype: Some("I8".into()),
                logical_dtype: Some("FP4".into()),
                scale_tensor_name: Some(format!("layers.0.ffn.experts.{}.w3.scale", eid)),
                scale_dtype: Some("F8_E8M0".into()),
                logical_shape: Some(down_logical.clone()),
                block_size: Some(block_size),
                packed_values_per_byte: Some(2),
            });
        }

        write_parser_json_files(
            &parse_dir, 1, hidden, intermediate, num_experts, top_k, "bfloat16",
            &expert_entries,
            &[RouterTensorEntry {
                name: "layers.0.ffn.gate.weight".into(),
                layer_id: Some(0),
                shape: vec![num_experts, hidden],
                dtype: "BF16".into(),
            }],
        );

        // Load metadata and validate
        let meta = load_parser_metadata(&parse_dir)?;
        validate_layer(&meta, 0)?;
        require_explicit_experts(&meta)?;

        let model_weights = ModelWeights::open(&model_dir).map_err(|e| {
            CudaError::new(CudaErrorKind::Io, "open model", e.to_string(), file!(), line!(), module_path!())
        })?;

        // Test FP4 decode for expert 0
        let fp4_tensors = find_fp4_expert_tensors(&meta, 0, 0)?;
        let gate_fp32 = load_fp4_expert_weight(
            &model_weights,
            &fp4_tensors.gate_name,
            &fp4_tensors.gate_scale_name,
            &[fp4_tensors.gate_logical_shape[0], fp4_tensors.gate_logical_shape[1] / 2],
            &fp4_tensors.gate_logical_shape,
            fp4_tensors.gate_block_size,
        )?;

        // Verify decoded shape
        assert_eq!(gate_fp32.len(), intermediate * hidden);
        // With scale=1.0 and fp4 values [0.5, 1.0, ...], first two values should be 0.5 and 1.0
        assert!((gate_fp32[0] - 0.5).abs() < 1e-6, "expected 0.5, got {}", gate_fp32[0]);
        assert!((gate_fp32[1] - 1.0).abs() < 1e-6, "expected 1.0, got {}", gate_fp32[1]);

        let up_fp32 = load_fp4_expert_weight(
            &model_weights,
            &fp4_tensors.up_name,
            &fp4_tensors.up_scale_name,
            &[fp4_tensors.up_logical_shape[0], fp4_tensors.up_logical_shape[1] / 2],
            &fp4_tensors.up_logical_shape,
            fp4_tensors.up_block_size,
        )?;
        assert_eq!(up_fp32.len(), intermediate * hidden);

        let down_fp32 = load_fp4_expert_weight(
            &model_weights,
            &fp4_tensors.down_name,
            &fp4_tensors.down_scale_name,
            &[fp4_tensors.down_logical_shape[0], fp4_tensors.down_logical_shape[1] / 2],
            &fp4_tensors.down_logical_shape,
            fp4_tensors.down_block_size,
        )?;
        assert_eq!(down_fp32.len(), hidden * intermediate);

        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }
}
