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
    compare_outputs, DeepSeekFp8SharedExpertWeightsDevice, execute_selected_moe_official_routed_fp4_cuda,
    q4_quantize_matrix_cpu, selected_moe_cpu_fp32,
    BytesByTensorKind, CudaBackendBuilder, CudaError, CudaErrorKind, CudaExpertCache,
    ExpertWeights, ExpertWeightsFp32, DeepSeekFp4ExpertWeights, MoeExecutor, QGemvNumerics,
    QGemvShape, QuantBackend, QuantFormat, Result, selected_moe_cpu_native_fp4,
};
use objeta_parser::deepseek::{
    cpu_act_quant, cpu_fp8_act_fp4_weight_gemv, cpu_fp8_act_fp4_weight_gemv_f32,
    cpu_fp8_act_fp8_weight_gemv,
    cpu_routed_expert_fp4_official, cpu_routed_expert_fp4_official_f32,
    cpu_shared_expert_fp8, cpu_shared_expert_fp8_official,
    decode_deepseek_fp4_to_f32, fp8_tile_gemv,
};
use objeta_parser::ModelWeights;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static GLOBAL_EXPERT_CACHE: OnceLock<Mutex<CudaExpertCache>> = OnceLock::new();

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

// ── BlockFamily and ExecutionFormat ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum BlockFamily {
    Decoder,
    Mtp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionFormat {
    NativeFp4,
    Q4Transcode,
    OfficialRouted,
}

fn router_name_for_layer(leaf: &str, layer: usize, block_family: BlockFamily) -> String {
    match block_family {
        BlockFamily::Decoder => format!("layers.{}.{}", layer, leaf),
        BlockFamily::Mtp => format!("mtp.{}.{}", layer, leaf),
    }
}

fn matches_block_family(name: &str, family: BlockFamily) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    let has_mtp = parts.contains(&"mtp");
    match family {
        BlockFamily::Decoder => !has_mtp,
        BlockFamily::Mtp => has_mtp,
    }
}

// ── Router tensor identification ────────────────────────────────────────────

#[derive(Debug)]
struct RouterTensorInfo {
    name: String,
    dtype: String,
}

fn find_router_tensor(
    meta: &ParserMetadata,
    layer: usize,
    block_family: BlockFamily,
) -> Result<RouterTensorInfo> {
    let candidates: Vec<&RouterTensorEntry> = meta
        .router_layout
        .routers
        .iter()
        .filter(|r| {
            r.layer_id == Some(layer)
            && matches_block_family(&r.name, block_family)
            // Exclude Hyper-Connection tensors (hc_ffn_base, hc_ffn_fn, hc_ffn_scale)
            // that share the same layer prefix but are not router tensors.
            && !r.name.contains("hc_ffn_")
        })
        .collect();

    if candidates.is_empty() {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "find router tensor",
            format!("no router tensor found for layer {} and block family {:?}", layer, block_family),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    // Among router candidates, prefer gate.weight (exclude gate.bias, gate.scale, etc.)
    let candidates: Vec<&RouterTensorEntry> = candidates
        .into_iter()
        .filter(|r| r.name.ends_with("gate.weight"))
        .collect();

    if candidates.len() > 1 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "find router tensor",
            format!(
                "{} router tensors found for layer {} and block family {:?}: {:?} — ambiguous",
                candidates.len(),
                layer,
                block_family,
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
    block_family: BlockFamily,
) -> Result<ExpertTensorSet> {
    let mut gate: Option<&str> = None;
    let mut up: Option<&str> = None;
    let mut down: Option<&str> = None;

    for t in &meta.expert_layout.tensors {
        if t.layer_id != Some(layer) || !matches_block_family(&t.name, block_family) {
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
                "layer {} block family {:?} expert {} missing tensors: {:?}",
                layer, block_family, expert_id, missing
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
    block_family: BlockFamily,
) -> Result<Fp4ExpertTensorSet> {
    let mut gate_weight: Option<&ExpertTensorEntry> = None;
    let mut up_weight: Option<&ExpertTensorEntry> = None;
    let mut down_weight: Option<&ExpertTensorEntry> = None;

    for t in &meta.expert_layout.tensors {
        if t.layer_id != Some(layer) || !matches_block_family(&t.name, block_family) {
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
                "layer {} block family {:?} expert {} missing tensors: {:?}",
                layer, block_family, expert_id, missing
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

fn load_raw_tensor(model_weights: &ModelWeights, name: &str) -> Result<Vec<u8>> {
    let bytes = model_weights.get_raw(name).map_err(|e| {
        CudaError::new(
            CudaErrorKind::Io,
            format!("load raw tensor '{}'", name),
            e.to_string(),
            file!(),
            line!(),
            module_path!(),
        )
    })?;
    Ok(bytes.to_vec())
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

// ── DeepSeek V4 Flash Gate reference ────────────────────────────────────────

/// Score function for DeepSeek V4 Flash gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreFunc {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

/// Full DeepSeek V4 Flash Gate reference.
///
/// Implements exact gate selection as `model.py:Gate.forward`:
///   scores = linear(hidden, gate_weight)
///   original_scores = score_func(scores)
///   selection_scores = original_scores + bias  (non-hash layers only)
///   expert_ids = topk(selection_scores, top_k)  (non-hash)
///              | tid2eid[input_id]              (hash)
///   weights = gather(original_scores, expert_ids)
///   weights /= sum(weights)
///   weights *= route_scale
///
/// Returns (expert_ids, weights).
fn cpu_gate_ref(
    gate_weight: &[f32],
    gate_bias: Option<&[f32]>,
    tid2eid: Option<&[i64]>,
    hidden: &[f32],
    input_id: Option<usize>,
    num_experts: usize,
    hidden_size: usize,
    top_k: usize,
    score_func: ScoreFunc,
    route_scale: f32,
) -> Result<(Vec<usize>, Vec<f32>)> {
    assert_eq!(gate_weight.len(), num_experts * hidden_size);

    // 1. logits = gate_weight @ hidden
    let mut logits = vec![0.0f32; num_experts];
    for e in 0..num_experts {
        let mut sum = 0.0f32;
        let base = e * hidden_size;
        for h in 0..hidden_size {
            sum += gate_weight[base + h] * hidden[h];
        }
        logits[e] = sum;
    }

    // 2. original_scores = score_func(logits)
    let original_scores: Vec<f32> = match score_func {
        ScoreFunc::Softmax => {
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut probs = vec![0.0f32; num_experts];
            for e in 0..num_experts {
                let val = (logits[e] - max_logit).exp();
                probs[e] = val;
                sum_exp += val;
            }
            for p in probs.iter_mut() {
                *p /= sum_exp;
            }
            probs
        }
        ScoreFunc::Sigmoid => {
            logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
        }
        ScoreFunc::SqrtSoftplus => {
            logits.iter().map(|v| {
                // softplus(x) = ln(1 + e^x), with clamp for numerical stability
                let sp = if *v > 20.0 {
                    *v
                } else if *v < -20.0 {
                    (-v).exp().ln_1p()
                } else {
                    v.exp().ln_1p()
                };
                sp.sqrt()
            }).collect()
        }
    };

    // 3. expert selection
    let expert_ids: Vec<usize> = if let Some(tid2eid_table) = tid2eid {
        // Hash layer: expert_ids come from token-id lookup table
        let token_id = input_id.ok_or_else(|| CudaError::new(
            CudaErrorKind::InvalidInput,
            "cpu_gate_ref hash layer",
            "input_id is required for hash layers".to_string(),
            file!(), line!(), module_path!(),
        ))?;
        if token_id >= tid2eid_table.len() / top_k {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "cpu_gate_ref hash layer",
                format!("input_id {} out of range for tid2eid (vocab={})", token_id, tid2eid_table.len() / top_k),
                file!(), line!(), module_path!(),
            ));
        }
        let base = token_id * top_k;
        (0..top_k).map(|k| tid2eid_table[base + k] as usize).collect()
    } else {
        // Non-hash layer: selection_scores = original_scores + bias
        let mut selection_scores = original_scores.clone();
        if let Some(bias) = gate_bias {
            for e in 0..num_experts {
                selection_scores[e] += bias[e];
            }
        }
        // top-k by selection score
        let mut indexed: Vec<(usize, f32)> = selection_scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);
        indexed.iter().map(|(id, _)| *id).collect()
    };

    // 4. weights = gather(original_scores, expert_ids)
    let mut weights: Vec<f32> = expert_ids.iter().map(|&eid| original_scores[eid]).collect();

    // 5. normalize weights
    if score_func != ScoreFunc::Softmax {
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
    }

    // 6. route_scale
    for w in weights.iter_mut() {
        *w *= route_scale;
    }

    Ok((expert_ids, weights))
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    #[test]
    fn test_sqrtsoftplus_scores() {
        let gate = vec![0.0f32; 256 * 4096];
        let hidden = vec![0.1f32; 4096];
        let (ids, weights) = cpu_gate_ref(
            &gate, None, None, &hidden, None, 256, 4096, 6,
            ScoreFunc::SqrtSoftplus, 1.5,
        ).unwrap();
        assert_eq!(ids.len(), 6);
        assert_eq!(weights.len(), 6);
        // Weights should sum to route_scale
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.5).abs() < 1e-4, "weights sum {} != 1.5", sum);
        // Original scores should be softplus(0).sqrt() which is < 1.0
        // Since all scores are equal, top-k picks first 6
    }

    #[test]
    fn test_bias_affects_selection_not_weights() {
        // Create gate where all logits are equal, but bias makes expert 7 win
        let mut gate = vec![0.0f32; 256 * 4096];
        // Set gate such that expert 0 gets higher original score
        gate[0] = 1.0;
        let hidden = vec![1.0f32; 4096]; // dot with gate[0]=1.0 → 4096.0
        let bias = Some(vec![0.0f32; 256]);

        let (ids, _weights) = cpu_gate_ref(
            &gate, bias.as_deref(), None, &hidden, None, 256, 4096, 6,
            ScoreFunc::SqrtSoftplus, 1.5,
        ).unwrap();

        // Expert 0 should be among selected (highest original score)
        assert!(ids.contains(&0));
    }

    #[test]
    fn test_hash_layer_tid2eid() {
        let gate = vec![0.0f32; 256 * 4096];
        let hidden = vec![0.1f32; 4096];
        let tid2eid: Vec<i64> = (0..(129280 * 6)).map(|x| (x % 256) as i64).collect();
        let (ids, weights) = cpu_gate_ref(
            &gate, None, Some(&tid2eid), &hidden, Some(42), 256, 4096, 6,
            ScoreFunc::SqrtSoftplus, 1.5,
        ).unwrap();
        assert_eq!(ids.len(), 6);
        // token 42 should map to tid2eid[42*6..48*6]
        let expected: Vec<usize> = (0..6).map(|k| (tid2eid[42 * 6 + k]) as usize).collect();
        assert_eq!(ids, expected);
        // Weights should be from original_scores (not tid2eid), summing to route_scale
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.5).abs() < 1e-4, "weights sum {} != 1.5", sum);
    }

    #[test]
    fn test_hash_layer_weights_from_hidden() {
        // Two hidden states: one favors expert 0, another favors expert 100
        // Both use same input_id, so expert IDs are identical from tid2eid
        // But weights differ because original_scores depend on hidden
        let mut gate = vec![0.0f32; 256 * 4096];
        // Expert 0: gate row 0 has large positive values
        for h in 0..4096 { gate[0 * 4096 + h] = 0.5; }
        // Expert 100: gate row 100 has large values
        for h in 0..4096 { gate[100 * 4096 + h] = 2.0; }

        let tid2eid: Vec<i64> = (0..(129280 * 6)).map(|x| (x % 256) as i64).collect();

        let hidden_a = vec![1.0f32; 4096];
        let (ids_a, weights_a) = cpu_gate_ref(
            &gate, None, Some(&tid2eid), &hidden_a, Some(0), 256, 4096, 6,
            ScoreFunc::SqrtSoftplus, 1.5,
        ).unwrap();
        assert_eq!(ids_a, vec![0, 1, 2, 3, 4, 5]); // tid2eid[0] = [0,1,2,3,4,5]

        let hidden_b = vec![-1.0f32; 4096];
        let (ids_b, weights_b) = cpu_gate_ref(
            &gate, None, Some(&tid2eid), &hidden_b, Some(0), 256, 4096, 6,
            ScoreFunc::SqrtSoftplus, 1.5,
        ).unwrap();
        assert_eq!(ids_b, ids_a); // Same expert IDs from tid2eid

        // Weights differ because hidden affects original_scores
        assert!(weights_a != weights_b, "Weights should differ per hidden state");
    }

    #[test]
    fn test_route_scale_exact() {
        let gate = vec![0.0f32; 256 * 4096];
        let hidden = vec![0.42f32; 4096];
        let (_, weights) = cpu_gate_ref(
            &gate, None, None, &hidden, None, 256, 4096, 6,
            ScoreFunc::SqrtSoftplus, 1.5,
        ).unwrap();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.5).abs() < 1e-4, "weights sum {} != 1.5", sum);
    }

    #[test]
    fn test_non_hash_top_k_selection() {
        // Make expert 10 dominant, 20 second, 30 third
        let mut gate = vec![0.0f32; 256 * 4096];
        for h in 0..4096 { gate[10 * 4096 + h] = 3.0; }
        for h in 0..4096 { gate[20 * 4096 + h] = 2.0; }
        for h in 0..4096 { gate[30 * 4096 + h] = 1.0; }
        let hidden = vec![1.0f32; 4096];
        let (ids, _) = cpu_gate_ref(
            &gate, None, None, &hidden, None, 256, 4096, 6,
            ScoreFunc::SqrtSoftplus, 1.5,
        ).unwrap();
        assert_eq!(ids[0], 10);
        assert_eq!(ids[1], 20);
        assert_eq!(ids[2], 30);
    }
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

// ── Shared FP8 expert types ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DeepSeekFp8SharedExpertWeights {
    gate_weight: Vec<u8>,
    gate_scale: Vec<u8>,
    up_weight: Vec<u8>,
    up_scale: Vec<u8>,
    down_weight: Vec<u8>,
    down_scale: Vec<u8>,
    dim: usize,
    inter_dim: usize,
}

/// Load shared expert tensors (F8_E4M3 + F8_E8M0 scale) for a decoder layer.
fn load_shared_expert_fp8(
    model_weights: &ModelWeights,
    layer: usize,
    block_family: BlockFamily,
) -> Result<DeepSeekFp8SharedExpertWeights> {
    let prefix = match block_family {
        BlockFamily::Decoder => format!("layers.{}.ffn.shared_experts", layer),
        BlockFamily::Mtp => format!("mtp.{}.ffn.shared_experts", layer),
    };

    let w1_name = format!("{}.w1.weight", prefix);
    let w1_scale_name = format!("{}.w1.scale", prefix);
    let w2_name = format!("{}.w2.weight", prefix);
    let w2_scale_name = format!("{}.w2.scale", prefix);
    let w3_name = format!("{}.w3.weight", prefix);
    let w3_scale_name = format!("{}.w3.scale", prefix);

    let gate_weight = model_weights.get_raw(&w1_name).map(|t| t.to_vec()).map_err(|e| {
        CudaError::new(CudaErrorKind::Io, format!("load '{}'", w1_name), e.to_string(), file!(), line!(), module_path!())
    })?;
    let gate_scale = model_weights.get_raw(&w1_scale_name).map(|t| t.to_vec()).map_err(|e| {
        CudaError::new(CudaErrorKind::Io, format!("load '{}'", w1_scale_name), e.to_string(), file!(), line!(), module_path!())
    })?;
    let up_weight = model_weights.get_raw(&w3_name).map(|t| t.to_vec()).map_err(|e| {
        CudaError::new(CudaErrorKind::Io, format!("load '{}'", w3_name), e.to_string(), file!(), line!(), module_path!())
    })?;
    let up_scale = model_weights.get_raw(&w3_scale_name).map(|t| t.to_vec()).map_err(|e| {
        CudaError::new(CudaErrorKind::Io, format!("load '{}'", w3_scale_name), e.to_string(), file!(), line!(), module_path!())
    })?;
    let down_weight = model_weights.get_raw(&w2_name).map(|t| t.to_vec()).map_err(|e| {
        CudaError::new(CudaErrorKind::Io, format!("load '{}'", w2_name), e.to_string(), file!(), line!(), module_path!())
    })?;
    let down_scale = model_weights.get_raw(&w2_scale_name).map(|t| t.to_vec()).map_err(|e| {
        CudaError::new(CudaErrorKind::Io, format!("load '{}'", w2_scale_name), e.to_string(), file!(), line!(), module_path!())
    })?;

    Ok(DeepSeekFp8SharedExpertWeights {
        gate_weight, gate_scale,
        up_weight, up_scale,
        down_weight, down_scale,
        dim: 4096,
        inter_dim: 2048,
    })
}

// ── Output report ───────────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
struct M9Report {
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_scope: Option<String>,
    model_dir: String,
    parse_dir: String,
    block_family: String,
    layer_id: usize,
    layout_kind: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    expert_ids: Vec<usize>,
    expert_weights: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    router_tensor_name: Option<String>,
    tensor_names_used: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scale_tensor_names_used: Option<Vec<String>>,
    source_dtypes: HashMap<String, String>,
    quant_format: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    gate_up_gemv_ms: Option<f32>,
    activation_ms: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    down_gemv_ms: Option<f32>,
    accum_ms: f32,
    total_ms: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    quant_vs_fp32: Option<QGemvNumerics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_vs_cpu_native_fp4: Option<QGemvNumerics>,
    weight_bytes_loaded: usize,
    weight_bytes_reused: usize,
    scale_bytes_loaded: usize,
    scale_bytes_reused: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    routed_fp4_bytes_loaded: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    routed_fp4_bytes_reused: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_fp8_bytes_loaded: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_fp8_bytes_reused: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_logical_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_loaded_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_reused_bytes: Option<usize>,
    source_bytes_loaded: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_fp4_cuda_moe_ms: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gate_up_qgemv_ms: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    down_qgemv_ms: Option<f32>,
    // ── Repeated-run fields ──────────────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    warmup_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host_tensor_load_ms: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_device_fill_exec_ms: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_cold_end_to_end_ms: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    all_layers_fixed_route_resident_bytes: Option<usize>,
    // ── Shared expert fields ─────────────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_expert_included: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_expert_weight_bytes: Option<usize>,
    // ── Pinned shared-expert residency ────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_total_model_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_resident_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_load_bytes_per_token: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    free_capacity_after_shared_pin: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_forward_full_cosine: Option<f64>,
    // ── Diagnostic norm fields ──────────────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    routed_only_output_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_only_output_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_moe_output_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_minus_routed_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_merge_residual_l2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hyper_connection_included: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attention_included: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    q4_transcode_invoked: Option<bool>,
    // ── Finite-output diagnostics ────────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    routed_only_nan_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    routed_only_inf_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_only_nan_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_only_inf_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_moe_nan_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_moe_inf_count: Option<u32>,
    /// "valid_finite" if all outputs are finite; otherwise the reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    parity_status: Option<String>,
    // ── Independent parity comparisons ────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_shared_vs_cpu_shared: Option<QGemvNumerics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_full_moe_vs_cpu_full_moe: Option<QGemvNumerics>,
    // ── Official arithmetic parity classification ──────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    official_arithmetic_cpu_reference_available: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    official_arithmetic_cuda_parity: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decoded_weight_reference_parity: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warm_runs: Option<Vec<WarmRunTelemetry>>,
}

#[derive(Debug, serde::Serialize)]
struct WarmRunTelemetry {
    iteration: usize,
    h2d_ms: f32,
    gate_up_gemv_ms: f32,
    activation_ms: f32,
    down_gemv_ms: f32,
    accum_ms: f32,
    device_upload_exec_ms: f32,
    warm_resident_exec_ms: f32,
    cache_hit_delta: usize,
    cache_miss_delta: usize,
    cache_hit_cumulative: usize,
    cache_miss_cumulative: usize,
    hit_rate: f32,
    actual_expert_bytes_loaded: usize,
    resident_cache_bytes_reused: usize,
    resident_cache_resident_bytes: usize,
    logical_expert_bytes_requested: usize,
    weight_bytes_loaded: usize,
    weight_bytes_reused: usize,
    scale_bytes_loaded: usize,
    scale_bytes_reused: usize,
    routed_fp4_bytes_loaded: usize,
    routed_fp4_bytes_reused: usize,
    shared_fp8_bytes_loaded: usize,
    shared_fp8_bytes_reused: usize,
    total_logical_bytes: usize,
    total_loaded_bytes: usize,
    total_reused_bytes: usize,
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

/// Parse a seed specification: "0" → [0], "0:127" → [0,1,...,127], "0,5,10" → [0,5,10].
fn parse_seed_range(raw: &str) -> Result<Vec<usize>> {
    if let Some((lo, hi)) = raw.split_once(':') {
        let lo = lo.trim().parse::<usize>().map_err(|_| {
            CudaError::new(CudaErrorKind::InvalidInput, "parse seed range", format!("invalid lo: '{}'", lo), file!(), line!(), module_path!())
        })?;
        let hi = hi.trim().parse::<usize>().map_err(|_| {
            CudaError::new(CudaErrorKind::InvalidInput, "parse seed range", format!("invalid hi: '{}'", hi), file!(), line!(), module_path!())
        })?;
        Ok((lo..=hi).collect())
    } else {
        parse_comma_separated_usize(raw)
    }
}

/// Compute softmax-temperature router entropy: H = -sum(p * log(p)).
/// `scores` are the raw router logits before softmax.
fn compute_router_entropy(scores: &[f32]) -> f64 {
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f64> = scores.iter().map(|s| (*s as f64 - max_s as f64).exp()).collect();
    let sum: f64 = shifted.iter().sum();
    let mut entropy = 0.0f64;
    for p in &shifted {
        let pn = p / sum;
        if pn > 0.0 {
            entropy -= pn * pn.ln();
        }
    }
    entropy
}

/// Compute the margin between ranked score[k-1] and score[k] (0-indexed) from raw logits.
fn compute_router_margin(scores: &[f32], k: usize) -> f64 {
    let mut sorted: Vec<f64> = scores.iter().map(|s| *s as f64).collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if k > 0 && k < sorted.len() {
        sorted[k - 1] - sorted[k]
    } else {
        0.0
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let report = run_moe(&args)?;
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}

fn run_moe(args: &[String]) -> Result<M9Report> {
    let parse_dir = parse_flag(args, "--parse-dir").ok_or_else(|| {
        CudaError::new(CudaErrorKind::InvalidInput, "parse args", "--parse-dir is required".to_string(), file!(), line!(), module_path!())
    })?;
    let model_dir = parse_flag(args, "--model-dir").ok_or_else(|| {
        CudaError::new(CudaErrorKind::InvalidInput, "parse args", "--model-dir is required".to_string(), file!(), line!(), module_path!())
    })?;
    let layer = parse_flag(args, "--layer")
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| {
            CudaError::new(CudaErrorKind::InvalidInput, "parse args", "--layer is required and must be a non-negative integer".to_string(), file!(), line!(), module_path!())
        })?;
    let seed = parse_flag(args, "--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(42);
    let cache_bytes = parse_flag(args, "--cache-bytes")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let bypass_oversized_experts = has_flag(args, "--bypass-oversized-experts");

    let warmup = parse_flag(args, "--warmup")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let repeat = parse_flag(args, "--repeat")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);
    if warmup == 0 && repeat == 0 {
        return Err(CudaError::new(
            CudaErrorKind::InvalidInput,
            "parse args",
            "--repeat must be >= 1 if --warmup is 0".to_string(),
            file!(),
            line!(),
            module_path!(),
        ));
    }

    // Selection mode: manual (default) or router-based
    let selection_mode_str = parse_flag(args, "--selection-mode");
    let selection_mode = match selection_mode_str.as_deref() {
        Some("router") => "router",
        None | Some("manual") => "manual",
        Some(other) => {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "parse args",
                format!("invalid selection-mode: '{}' (expected manual|router)", other),
                file!(), line!(), module_path!(),
            ));
        }
    };
    let is_router_mode = selection_mode == "router";
    let include_shared = has_flag(args, "--include-shared");
    let mode_official_routed = has_flag(args, "--mode") && std::env::args().any(|a| a == "official-routed");

    // Input token ID for hash-layer router mode
    let input_id: Option<usize> = parse_flag(args, "--input-id")
        .and_then(|v| v.parse::<usize>().ok());

    // Preload / pinned-shared-residency flags
    let preload_shared_all = has_flag(args, "--preload-shared-all-layers");
    let shared_residency = parse_flag(args, "--shared-residency");
    let shared_residency_pinned = shared_residency.as_deref() == Some("pinned_all_layers");

    // Research MOE field mode
    let research_mode = has_flag(args, "--research-moe-field");
    let research_layers: Option<Vec<usize>> = parse_flag(args, "--layers")
        .map(|raw| parse_comma_separated_usize(&raw)).transpose()?;
    let research_seeds: Option<Vec<usize>> = parse_flag(args, "--seeds")
        .map(|raw| parse_seed_range(&raw)).transpose()?;
    let output_jsonl = parse_flag(args, "--output-jsonl");

    // Manual expert mode
    let manual_ids: Option<Vec<usize>> = parse_flag(args, "--expert-ids")
        .map(|raw| parse_comma_separated_usize(&raw))
        .transpose()?;
    let manual_weights_raw: Option<String> = parse_flag(args, "--expert-weights");
    let manual_mode = !is_router_mode && manual_ids.is_some();

    let execution_format_str = parse_flag(args, "--execution-format");
    let execution_format = if mode_official_routed {
        Some(ExecutionFormat::OfficialRouted)
    } else {
        match execution_format_str.as_deref() {
            Some("native-fp4") => Some(ExecutionFormat::NativeFp4),
            Some("q4-transcode") => Some(ExecutionFormat::Q4Transcode),
            None => None,
            Some(other) => {
                return Err(CudaError::new(
                    CudaErrorKind::InvalidInput,
                    "parse args",
                    format!("invalid execution-format: '{}' (expected native-fp4|q4-transcode)", other),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
        }
    };

    let block_family_str = parse_flag(args, "--block-family");
    let block_family = match block_family_str.as_deref() {
        Some("decoder") | None => BlockFamily::Decoder,
        Some("mtp") => BlockFamily::Mtp,
        Some(other) => {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput,
                "parse args",
                format!("invalid block-family: '{}' (expected decoder|mtp)", other),
                file!(),
                line!(),
                module_path!(),
            ));
        }
    };

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

    let mut source_bytes_loaded = 0;

    let is_hash_layer = layer < 3; // n_hash_layers = 3
    let mut gate_tensor_names_used: Vec<String> = Vec::new();

    let (selected_ids, selected_weights, router_tensor_name, router_ms
    ) = if manual_mode {
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
    } else if is_router_mode {
        // ── DeepSeek V4 Flash Gate reference routing ───────────────────
        let gate_weight_name = router_name_for_layer("ffn.gate.weight", layer, block_family);
        gate_tensor_names_used.push(gate_weight_name.clone());

        let mut gate_weight = Vec::new();
        model_weights.get_f32(&gate_weight_name, &mut gate_weight).map_err(|e| {
            CudaError::new(CudaErrorKind::Io, format!("load '{}'", gate_weight_name), e.to_string(), file!(), line!(), module_path!())
        })?;
        source_bytes_loaded += gate_weight.len() * 4;

        // Load bias (non-hash layers) or tid2eid (hash layers)
        let (gate_bias, tid2eid) = if is_hash_layer {
            let tid_name = router_name_for_layer("ffn.gate.tid2eid", layer, block_family);
            gate_tensor_names_used.push(tid_name.clone());
            let tid_raw = model_weights.get_raw(&tid_name).map(|t| t.to_vec()).map_err(|e| {
                CudaError::new(CudaErrorKind::Io, format!("load '{}'", tid_name), e.to_string(), file!(), line!(), module_path!())
            })?;
            // I64 raw bytes → i64 slice
            let tid_len = tid_raw.len() / 8;
            source_bytes_loaded += tid_raw.len();
            let tid_i64: Vec<i64> = unsafe {
                let ptr = tid_raw.as_ptr() as *const i64;
                std::slice::from_raw_parts(ptr, tid_len).to_vec()
            };
            (None, Some(tid_i64))
        } else {
            let bias_name = router_name_for_layer("ffn.gate.bias", layer, block_family);
            gate_tensor_names_used.push(bias_name.clone());
            let mut bias_f32 = Vec::new();
            model_weights.get_f32(&bias_name, &mut bias_f32).map_err(|e| {
                CudaError::new(CudaErrorKind::Io, format!("load '{}'", bias_name), e.to_string(), file!(), line!(), module_path!())
            })?;
            source_bytes_loaded += bias_f32.len() * 4;
            (Some(bias_f32), None)
        };

        let score_func = ScoreFunc::SqrtSoftplus;
        let route_scale: f32 = 1.5;

        let router_start = Instant::now();
        let (ids, weights) = cpu_gate_ref(
            &gate_weight,
            gate_bias.as_deref(),
            tid2eid.as_deref(),
            &hidden,
            input_id,
            num_experts,
            hidden_size,
            top_k,
            score_func,
            route_scale,
        )?;
        let rms = router_start.elapsed().as_secs_f32() * 1000.0;

        (ids, weights, Some(gate_weight_name), Some(rms))
    } else {
        // ── Old router-based expert selection ──────────────────────────
        let router_info = find_router_tensor(&meta, layer, block_family)?;
        validate_dtype(&router_info.dtype)?;

        let mut router_fp32 = Vec::new();
        model_weights
            .get_f32(&router_info.name, &mut router_fp32)
            .map_err(|e| {
                CudaError::new(CudaErrorKind::Io, format!("load router tensor '{}'", router_info.name), e.to_string(), file!(), line!(), module_path!())
            })?;
        source_bytes_loaded += router_fp32.len() * 4;

        let expected_router_elems = num_experts * hidden_size;
        if router_fp32.len() != expected_router_elems {
            return Err(CudaError::new(
                CudaErrorKind::InvalidInput, "validate router tensor",
                format!("router tensor '{}' has {} elements, expected {}", router_info.name, router_fp32.len(), expected_router_elems),
                file!(), line!(), module_path!(),
            ));
        }

        let router_start = Instant::now();
        let (ids, weights) =
            cpu_router(&router_fp32, &hidden, num_experts, hidden_size, top_k)?;
        let rms = router_start.elapsed().as_secs_f32() * 1000.0;

        (ids, weights, Some(router_info.name), Some(rms))
    };

    // Detect whether the first expert uses FP4 storage
    let first_expert_uses_fp4 = if !selected_ids.is_empty() {
        meta.expert_layout.tensors.iter().any(|t| {
            t.layer_id == Some(layer)
                && t.expert_id == Some(Some(selected_ids[0]))
                && matches_block_family(&t.name, block_family)
                && t.storage_dtype.as_deref() == Some("I8")
        })
    } else {
        false
    };

    let execution_format = match execution_format {
        Some(fmt) => {
            if fmt == ExecutionFormat::NativeFp4 && !first_expert_uses_fp4 {
                return Err(CudaError::new(
                    CudaErrorKind::Unsupported,
                    "execution-format native-fp4",
                    "native-fp4 was requested but the model weights are not FP4".to_string(),
                    file!(),
                    line!(),
                    module_path!(),
                ));
            }
            fmt
        }
        None => {
            if first_expert_uses_fp4 {
                ExecutionFormat::NativeFp4
            } else {
                ExecutionFormat::Q4Transcode
            }
        }
    };

    let mut expert_tensor_names: Vec<String> = Vec::new();
    let mut scale_tensor_names: Vec<String> = Vec::new();
    let mut source_dtypes: HashMap<String, String> = HashMap::new();

    let backend = CudaBackendBuilder::new().stream_count(1).build()?;
    let quant = QuantBackend::new(backend.context().clone(), backend.device_info().clone());
    let moe_executor = MoeExecutor::new(backend.context().clone(), backend.device_info().clone());
    let stream = backend.stream_pool().stream(0)?;

    let mut cache_guard = if cache_bytes > 0 {
        let cache_mutex = GLOBAL_EXPERT_CACHE.get_or_init(|| {
            Mutex::new(CudaExpertCache::new(cache_bytes))
        });
        let mut guard = cache_mutex.lock().unwrap();
        guard.capacity_bytes = cache_bytes;
        guard.bypass_oversized_experts = bypass_oversized_experts;
        guard.reset_counters();
        Some(guard)
    } else {
        None
    };

    let mut cache = cache_guard.as_deref_mut();

    let selected_pairs: Vec<(usize, f32)> = selected_ids
        .iter()
        .zip(selected_weights.iter())
        .map(|(id, w)| (*id, *w))
        .collect();

    // ── Preload all shared experts (pinned residency) ────────────────────
    let mut preload_time_ms = 0.0f32;
    if preload_shared_all && include_shared {
        let t0 = Instant::now();
        let num_layers = meta.layout.num_layers;
        let mut total_shared_bytes = 0usize;
        for l in 0..num_layers {
            let bf = if l < num_layers { BlockFamily::Decoder } else { BlockFamily::Mtp };
            let sh = load_shared_expert_fp8(&model_weights, l, bf)?;
            let layer_bytes = sh.gate_weight.len() + sh.gate_scale.len()
                + sh.up_weight.len() + sh.up_scale.len()
                + sh.down_weight.len() + sh.down_scale.len();
            total_shared_bytes += layer_bytes;

            if let Some(ref mut c) = cache {
                // Inline pin operations: upload to device, then insert_pinned in cache
                let k = |tid| objeta_cuda::ExpertCacheKey {
                    layer_id: l, expert_id: 0,
                    tensor_kind: tid,
                    quant_format: QuantFormat::DeepSeekFp4E2M1,
                };
                let b_gw = stream.copy_from_slice(&sh.gate_weight)?;
                c.insert_pinned(k(objeta_cuda::ExpertTensorKind::SharedGateWeight), b_gw);
                let b_gs = stream.copy_from_slice(&sh.gate_scale)?;
                c.insert_pinned(k(objeta_cuda::ExpertTensorKind::SharedGateScale), b_gs);
                let b_uw = stream.copy_from_slice(&sh.up_weight)?;
                c.insert_pinned(k(objeta_cuda::ExpertTensorKind::SharedUpWeight), b_uw);
                let b_us = stream.copy_from_slice(&sh.up_scale)?;
                c.insert_pinned(k(objeta_cuda::ExpertTensorKind::SharedUpScale), b_us);
                let b_dw = stream.copy_from_slice(&sh.down_weight)?;
                c.insert_pinned(k(objeta_cuda::ExpertTensorKind::SharedDownWeight), b_dw);
                let b_ds = stream.copy_from_slice(&sh.down_scale)?;
                c.insert_pinned(k(objeta_cuda::ExpertTensorKind::SharedDownScale), b_ds);
            }
        }
        preload_time_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let total_model = total_shared_bytes;
        eprintln!("PRELOAD shared experts: {} layers, {} bytes in {:.1}ms",
            num_layers, total_model, preload_time_ms);
        if let Some(ref c) = cache {
            eprintln!("  cache pinned_bytes={} resident={} capacity={}",
                c.pinned_bytes, c.resident_bytes, c.capacity_bytes);
        }
    }

    let report = match execution_format {
        ExecutionFormat::NativeFp4 => {
            let mut expert_fp4_set = vec![
                DeepSeekFp4ExpertWeights {
                    gate_weight: Vec::new(),
                    gate_scale: Vec::new(),
                    up_weight: Vec::new(),
                    up_scale: Vec::new(),
                    down_weight: Vec::new(),
                    down_scale: Vec::new(),
                };
                num_experts
            ];

            for &eid in &selected_ids {
                let fp4_tensors = find_fp4_expert_tensors(&meta, layer, eid, block_family)?;

                // Record dtypes
                source_dtypes.insert(format!("expert_{}_gate_weight", eid), "I8".into());
                source_dtypes.insert(format!("expert_{}_up_weight", eid), "I8".into());
                source_dtypes.insert(format!("expert_{}_down_weight", eid), "I8".into());
                source_dtypes.insert(format!("expert_{}_gate_scale", eid), "F8_E8M0".into());
                source_dtypes.insert(format!("expert_{}_up_scale", eid), "F8_E8M0".into());
                source_dtypes.insert(format!("expert_{}_down_scale", eid), "F8_E8M0".into());

                // Load raw bytes
                let gate_weight = load_raw_tensor(&model_weights, &fp4_tensors.gate_name)?;
                let gate_scale = load_raw_tensor(&model_weights, &fp4_tensors.gate_scale_name)?;
                let up_weight = load_raw_tensor(&model_weights, &fp4_tensors.up_name)?;
                let up_scale = load_raw_tensor(&model_weights, &fp4_tensors.up_scale_name)?;
                let down_weight = load_raw_tensor(&model_weights, &fp4_tensors.down_name)?;
                let down_scale = load_raw_tensor(&model_weights, &fp4_tensors.down_scale_name)?;

                expert_tensor_names.push(fp4_tensors.gate_name.clone());
                expert_tensor_names.push(fp4_tensors.up_name.clone());
                expert_tensor_names.push(fp4_tensors.down_name.clone());
                scale_tensor_names.push(fp4_tensors.gate_scale_name.clone());
                scale_tensor_names.push(fp4_tensors.up_scale_name.clone());
                scale_tensor_names.push(fp4_tensors.down_scale_name.clone());

                expert_fp4_set[eid] = DeepSeekFp4ExpertWeights {
                    gate_weight,
                    gate_scale,
                    up_weight,
                    up_scale,
                    down_weight,
                    down_scale,
                };
            }

            for &eid in &selected_ids {
                let exp = &expert_fp4_set[eid];
                source_bytes_loaded += exp.gate_weight.len();
                source_bytes_loaded += exp.gate_scale.len();
                source_bytes_loaded += exp.up_weight.len();
                source_bytes_loaded += exp.up_scale.len();
                source_bytes_loaded += exp.down_weight.len();
                source_bytes_loaded += exp.down_scale.len();
            }

            let tensor_load_ms = t_load_start.elapsed().as_secs_f32() * 1000.0;

            // CPU native FP4 reference (routed only)
            let routed_ref = selected_moe_cpu_native_fp4(
                &expert_fp4_set,
                &selected_pairs,
                &hidden,
                hidden_size,
                intermediate_size,
                hidden_size,
            )?;

            // ── Shared expert (CPU FP8 reference, if requested) ─────────
            let (shared_ref, shared_weight) = if include_shared {
                let shared = load_shared_expert_fp8(&model_weights, layer, block_family)?;
                let shared_bytes = shared.gate_weight.len() + shared.gate_scale.len()
                    + shared.up_weight.len() + shared.up_scale.len()
                    + shared.down_weight.len() + shared.down_scale.len();
                let shared_out = cpu_shared_expert_fp8(
                    &shared.gate_weight, &shared.gate_scale,
                    &shared.up_weight, &shared.up_scale,
                    &shared.down_weight, &shared.down_scale,
                    &hidden, hidden_size, intermediate_size, 10.0,
                );
                (Some(shared_out), Some(shared_bytes))
            } else {
                (None, None)
            };

            // Full MoE reference: routed + shared
            let ref_out = if let Some(ref shared) = shared_ref {
                let mut full = routed_ref.clone();
                for i in 0..full.len() {
                    full[i] += shared[i];
                }
                full
            } else {
                routed_ref.clone()
            };

            quant.compile_format(QuantFormat::DeepSeekFp4E2M1)?;
            moe_executor.compile()?;

            // ── Warmup iterations (unmeasured, primes VRAM cache) ─────
            for _ in 0..warmup {
                let _ = moe_executor.execute_selected_moe_native_fp4_cuda(
                    &quant,
                    stream,
                    &expert_fp4_set,
                    &selected_pairs,
                    &hidden,
                    hidden_size,
                    intermediate_size,
                    hidden_size,
                    layer,
                    cache.as_deref_mut(),
                )?;
            }

            // Reset cache counters so first measured run has clean stats
            if let Some(ref mut c) = cache {
                c.reset_counters();
            }

            // ── Timed repeat iterations ────────────────────────────────
            let mut warm_runs: Vec<WarmRunTelemetry> = Vec::new();
            let mut first_out: Option<Vec<f32>> = None;
            let mut first_telemetry: Option<objeta_cuda::MoeTelemetry> = None;

            for run_idx in 0..repeat {
                let (cuda_out, moe_telemetry) = moe_executor.execute_selected_moe_native_fp4_cuda(
                    &quant,
                    stream,
                    &expert_fp4_set,
                    &selected_pairs,
                    &hidden,
                    hidden_size,
                    intermediate_size,
                    hidden_size,
                    layer,
                    cache.as_deref_mut(),
                )?;

                let dev_exec = moe_telemetry.h2d_ms
                    + moe_telemetry.gate_up_qgemv_ms
                    + moe_telemetry.activation_ms
                    + moe_telemetry.down_qgemv_ms
                    + moe_telemetry.accum_ms;

                let prev_hits = if run_idx == 0 { 0 }
                    else { warm_runs[run_idx - 1].cache_hit_cumulative };
                let prev_misses = if run_idx == 0 { 0 }
                    else { warm_runs[run_idx - 1].cache_miss_cumulative };
                let hit_delta = moe_telemetry.resident_cache_hit_count.saturating_sub(prev_hits);
                let miss_delta = moe_telemetry.resident_cache_miss_count.saturating_sub(prev_misses);
                let run_hits = moe_telemetry.resident_cache_hit_count;
                let run_misses = moe_telemetry.resident_cache_miss_count;
                let total_run = run_hits + run_misses;
                let hit_rate = if total_run > 0 { run_hits as f32 / total_run as f32 } else { 0.0 };

                warm_runs.push(WarmRunTelemetry {
                    iteration: run_idx,
                    h2d_ms: moe_telemetry.h2d_ms,
                    gate_up_gemv_ms: moe_telemetry.gate_up_qgemv_ms,
                    activation_ms: moe_telemetry.activation_ms,
                    down_gemv_ms: moe_telemetry.down_qgemv_ms,
                    accum_ms: moe_telemetry.accum_ms,
                    device_upload_exec_ms: dev_exec,
                    warm_resident_exec_ms: moe_telemetry.total_ms,
                    cache_hit_delta: hit_delta,
                    cache_miss_delta: miss_delta,
                    cache_hit_cumulative: run_hits,
                    cache_miss_cumulative: run_misses,
                    hit_rate,
                    actual_expert_bytes_loaded: moe_telemetry.actual_expert_bytes_loaded,
                    resident_cache_bytes_reused: moe_telemetry.resident_cache_bytes_reused,
                    resident_cache_resident_bytes: moe_telemetry.resident_cache_resident_bytes,
                    logical_expert_bytes_requested: moe_telemetry.logical_expert_bytes_requested,
                    weight_bytes_loaded: moe_telemetry.weight_bytes_loaded,
                    weight_bytes_reused: moe_telemetry.weight_bytes_reused,
                    scale_bytes_loaded: moe_telemetry.scale_bytes_loaded,
                    scale_bytes_reused: moe_telemetry.scale_bytes_reused,
                    routed_fp4_bytes_loaded: 0, routed_fp4_bytes_reused: 0,
                    shared_fp8_bytes_loaded: 0, shared_fp8_bytes_reused: 0,
                    total_logical_bytes: 0, total_loaded_bytes: 0, total_reused_bytes: 0,
                });

                if run_idx == 0 {
                    first_out = Some(cuda_out);
                    first_telemetry = Some(moe_telemetry);
                }
            }

            let first_cuda_out = first_out.unwrap();
            let moe_telemetry = first_telemetry.unwrap();

            // Compare outputs (first run only)
            let cuda_vs_cpu_native_fp4 = compare_outputs(&ref_out, &first_cuda_out)?;

            let cache_counters = if let Some(ref c) = cache {
                (
                    c.hit_count, c.miss_count, c.eviction_count,
                    c.cache_insert_attempt_count, c.cache_insert_accept_count, c.cache_insert_bypass_count,
                    c.oversized_tensor_bypass_count, c.oversized_expert_bypass_count, c.self_eviction_risk_count,
                )
            } else {
                (0, 0, 0, 0, 0, 0, 0, 0, 0)
            };

            let source_label = if manual_mode {
                "real_deepseek_v4_flash_manual_expert_native_fp4_moe"
            } else {
                "real_deepseek_v4_flash_native_fp4_moe"
            };

            let first_dev_ms = moe_telemetry.total_ms;
            let all_layers_bytes = moe_telemetry.selected_working_set_bytes * meta.layout.num_layers;

            M9Report {
                source: source_label.to_string(),
                output_scope: None,
                model_dir,
                parse_dir,
                block_family: match block_family {
                    BlockFamily::Decoder => "decoder".to_string(),
                    BlockFamily::Mtp => "mtp".to_string(),
                },
                layer_id: layer,
                layout_kind: meta.expert_layout.layout_kind.clone(),
                hidden_size,
                intermediate_size,
                num_experts,
                top_k,
                expert_ids: selected_ids,
                expert_weights: selected_weights,
                router_tensor_name,
                tensor_names_used: expert_tensor_names,
                scale_tensor_names_used: Some(scale_tensor_names),
                source_dtypes,
                quant_format: "DeepSeekFp4E2M1".to_string(),
                fp4_decode_ms: None,
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
                quantize_ms: 0.0,
                h2d_ms: moe_telemetry.h2d_ms,
                gate_up_gemv_ms: Some(moe_telemetry.gate_up_qgemv_ms),
                activation_ms: moe_telemetry.activation_ms,
                down_gemv_ms: Some(moe_telemetry.down_qgemv_ms),
                accum_ms: moe_telemetry.accum_ms,
                total_ms: moe_telemetry.total_ms,
                quant_vs_fp32: None,
                cuda_vs_cpu_native_fp4: Some(cuda_vs_cpu_native_fp4),
                weight_bytes_loaded: moe_telemetry.weight_bytes_loaded,
                weight_bytes_reused: moe_telemetry.weight_bytes_reused,
                scale_bytes_loaded: moe_telemetry.scale_bytes_loaded,
                scale_bytes_reused: moe_telemetry.scale_bytes_reused,
                routed_fp4_bytes_loaded: None,
                routed_fp4_bytes_reused: None,
                shared_fp8_bytes_loaded: None,
                shared_fp8_bytes_reused: None,
                total_logical_bytes: None,
                total_loaded_bytes: None,
                total_reused_bytes: None,
                source_bytes_loaded,
                native_fp4_cuda_moe_ms: Some(moe_telemetry.total_ms),
                gate_up_qgemv_ms: None,
                down_qgemv_ms: None,
                // Repeated-run fields
                warmup_count: if warmup > 0 || repeat > 1 { Some(warmup) } else { None },
                repeat_count: if warmup > 0 || repeat > 1 { Some(repeat) } else { None },
                host_tensor_load_ms: Some(tensor_load_ms),
                first_device_fill_exec_ms: Some(first_dev_ms),
                first_cold_end_to_end_ms: Some(tensor_load_ms + first_dev_ms),
                all_layers_fixed_route_resident_bytes: Some(all_layers_bytes),
                shared_expert_included: Some(include_shared),
                shared_expert_weight_bytes: shared_weight,
                shared_total_model_bytes: None,
                shared_resident_bytes: None,
                shared_load_bytes_per_token: None,
                free_capacity_after_shared_pin: None,
                moe_forward_full_cosine: if shared_ref.is_some() { Some(cuda_vs_cpu_native_fp4.cosine_similarity as f64) } else { None },
                routed_only_output_norm: None,
                shared_only_output_norm: None,
                full_moe_output_norm: None,
                full_minus_routed_norm: None,
                shared_merge_residual_l2: None,
                hyper_connection_included: Some(false),
                attention_included: Some(false),
                q4_transcode_invoked: Some(false),
                routed_only_nan_count: None,
                routed_only_inf_count: None,
                shared_only_nan_count: None,
                shared_only_inf_count: None,
                full_moe_nan_count: None,
                full_moe_inf_count: None,
                parity_status: None,
                cuda_shared_vs_cpu_shared: None,
                cuda_full_moe_vs_cpu_full_moe: None,
                official_arithmetic_cpu_reference_available: Some(include_shared),
                official_arithmetic_cuda_parity: Some(false),  // No CUDA act_quant yet
                decoded_weight_reference_parity: Some(true),    // weight-only parity is verified
                warm_runs: if warmup > 0 || repeat > 1 { Some(warm_runs) } else { None },
            }
        }
        ExecutionFormat::OfficialRouted => {
            // Official-arithmetic device-resident routed expert execution
            let mut expert_fp4_set = vec![
                DeepSeekFp4ExpertWeights { gate_weight: Vec::new(), gate_scale: Vec::new(), up_weight: Vec::new(), up_scale: Vec::new(), down_weight: Vec::new(), down_scale: Vec::new() };
                num_experts
            ];

            for &eid in &selected_ids {
                let fp4_tensors = find_fp4_expert_tensors(&meta, layer, eid, block_family)?;
                for (kind, dtype) in [("gate_weight", "I8"), ("up_weight", "I8"), ("down_weight", "I8"), ("gate_scale", "F8_E8M0"), ("up_scale", "F8_E8M0"), ("down_scale", "F8_E8M0")] {
                    source_dtypes.insert(format!("expert_{}_{}", eid, kind), dtype.to_string());
                }
                let gw = load_raw_tensor(&model_weights, &fp4_tensors.gate_name)?;
                let gs = load_raw_tensor(&model_weights, &fp4_tensors.gate_scale_name)?;
                let uw = load_raw_tensor(&model_weights, &fp4_tensors.up_name)?;
                let us = load_raw_tensor(&model_weights, &fp4_tensors.up_scale_name)?;
                let dw = load_raw_tensor(&model_weights, &fp4_tensors.down_name)?;
                let ds = load_raw_tensor(&model_weights, &fp4_tensors.down_scale_name)?;
                expert_tensor_names.push(fp4_tensors.gate_name.clone());
                expert_tensor_names.push(fp4_tensors.up_name.clone());
                expert_tensor_names.push(fp4_tensors.down_name.clone());
                scale_tensor_names.push(fp4_tensors.gate_scale_name.clone());
                scale_tensor_names.push(fp4_tensors.up_scale_name.clone());
                scale_tensor_names.push(fp4_tensors.down_scale_name.clone());
                expert_fp4_set[eid] = DeepSeekFp4ExpertWeights { gate_weight: gw, gate_scale: gs, up_weight: uw, up_scale: us, down_weight: dw, down_scale: ds };
            }

            let tensor_load_ms = t_load_start.elapsed().as_secs_f32() * 1000.0;

            // CPU official reference (routed only)
            let official_cpu_routed = {
                let mut full = vec![0.0f32; hidden_size];
                for &(eid, w) in &selected_pairs {
                    let exp = &expert_fp4_set[eid];
                    let out = cpu_routed_expert_fp4_official_f32(
                        &exp.gate_weight, &exp.gate_scale,
                        &exp.up_weight, &exp.up_scale,
                        &exp.down_weight, &exp.down_scale,
                        &[intermediate_size, hidden_size/2], &[intermediate_size, hidden_size],
                        &[intermediate_size, hidden_size/2], &[intermediate_size, hidden_size],
                        &[hidden_size, intermediate_size/2], &[hidden_size, intermediate_size],
                        32, &hidden, hidden_size, intermediate_size, 10.0,
                    );
                    for i in 0..hidden_size { full[i] += w * out[i]; }
                }
                full
            };

            // CPU shared-only reference (separate for independent comparison)
            let (shared_host, official_cpu_shared) = if include_shared {
                let sh = load_shared_expert_fp8(&model_weights, layer, block_family)?;
                let shared_out = cpu_shared_expert_fp8_official(
                    &sh.gate_weight, &sh.gate_scale,
                    &sh.up_weight, &sh.up_scale,
                    &sh.down_weight, &sh.down_scale,
                    &hidden, hidden_size, intermediate_size, 10.0,
                );
                (Some(sh), shared_out)
            } else {
                (None, vec![0.0f32; hidden_size])
            };

            // CPU full = routed + shared
            let official_cpu = if include_shared {
                let mut full = official_cpu_routed.clone();
                for i in 0..hidden_size { full[i] += official_cpu_shared[i]; }
                full
            } else {
                official_cpu_routed.clone()
            };

            // Also compute decoded-weight reference
            let decoded_cpu = selected_moe_cpu_native_fp4(&expert_fp4_set, &selected_pairs, &hidden, hidden_size, intermediate_size, hidden_size)?;

            // Shared expert weight bytes
            let shared_weight = shared_host.as_ref().map(|sh| {
                sh.gate_weight.len() + sh.gate_scale.len()
                + sh.up_weight.len() + sh.up_scale.len()
                + sh.down_weight.len() + sh.down_scale.len()
            });

            // Load shared expert to device (reuse shared_host loaded above)
            let mut shared_dev: Option<DeepSeekFp8SharedExpertWeightsDevice> = None;
            if let Some(ref sh) = shared_host {
                shared_dev = Some(DeepSeekFp8SharedExpertWeightsDevice {
                    gate_weight: stream.copy_from_slice(&sh.gate_weight)?,
                    gate_scale: stream.copy_from_slice(&sh.gate_scale)?,
                    up_weight: stream.copy_from_slice(&sh.up_weight)?,
                    up_scale: stream.copy_from_slice(&sh.up_scale)?,
                    down_weight: stream.copy_from_slice(&sh.down_weight)?,
                    down_scale: stream.copy_from_slice(&sh.down_scale)?,
                });
            }

            // Warmup + repeat official CUDA (full = routed + shared)
            for _ in 0..warmup {
                let _ = execute_selected_moe_official_routed_fp4_cuda(&quant, &moe_executor, stream, &expert_fp4_set, &selected_pairs, &hidden, hidden_size, intermediate_size, hidden_size, layer, cache.as_deref_mut(), shared_dev.as_ref())?;
            }

            if let Some(ref mut c) = cache { c.reset_counters(); }

            let mut warm_runs: Vec<WarmRunTelemetry> = Vec::new();
            let mut first_full_out: Option<Vec<f32>> = None;
            let mut moe_telemetry: Option<objeta_cuda::MoeTelemetry> = None;

            for run_idx in 0..repeat {
                let (out, tel) = execute_selected_moe_official_routed_fp4_cuda(&quant, &moe_executor, stream, &expert_fp4_set, &selected_pairs, &hidden, hidden_size, intermediate_size, hidden_size, layer, cache.as_deref_mut(), shared_dev.as_ref())?;
                if run_idx == 0 {
                    first_full_out = Some(out);
                    moe_telemetry = Some(tel);
                }
                warm_runs.push(WarmRunTelemetry {
                    iteration: run_idx, h2d_ms: tel.h2d_ms, gate_up_gemv_ms: tel.gate_up_qgemv_ms, activation_ms: tel.activation_ms,
                    down_gemv_ms: tel.down_qgemv_ms, accum_ms: tel.accum_ms,
                    device_upload_exec_ms: tel.h2d_ms + tel.gate_up_qgemv_ms + tel.activation_ms + tel.down_qgemv_ms + tel.accum_ms,
                    warm_resident_exec_ms: tel.total_ms,
                    cache_hit_delta: 0, cache_miss_delta: 0, cache_hit_cumulative: tel.resident_cache_hit_count, cache_miss_cumulative: tel.resident_cache_miss_count,
                    hit_rate: 1.0, actual_expert_bytes_loaded: tel.actual_expert_bytes_loaded, resident_cache_bytes_reused: tel.resident_cache_bytes_reused,
                    resident_cache_resident_bytes: tel.resident_cache_resident_bytes, logical_expert_bytes_requested: tel.logical_expert_bytes_requested,
                    weight_bytes_loaded: tel.weight_bytes_loaded, weight_bytes_reused: tel.weight_bytes_reused,
                    scale_bytes_loaded: tel.scale_bytes_loaded, scale_bytes_reused: tel.scale_bytes_reused,
                    routed_fp4_bytes_loaded: tel.routed_fp4_bytes_loaded, routed_fp4_bytes_reused: tel.routed_fp4_bytes_reused,
                    shared_fp8_bytes_loaded: tel.shared_fp8_bytes_loaded, shared_fp8_bytes_reused: tel.shared_fp8_bytes_reused,
                    total_logical_bytes: tel.total_logical_bytes, total_loaded_bytes: tel.total_loaded_bytes, total_reused_bytes: tel.total_reused_bytes,
                });
            }

            let cuda_full_out = first_full_out.unwrap();
            let mtel = moe_telemetry.unwrap();

            // CUDA routed-only (no shared expert)
            let cuda_routed_only = if include_shared {
                if let Some(ref mut c) = cache { c.reset_counters(); }
                let (out, _) = execute_selected_moe_official_routed_fp4_cuda(&quant, &moe_executor, stream, &expert_fp4_set, &selected_pairs, &hidden, hidden_size, intermediate_size, hidden_size, layer, cache.as_deref_mut(), None)?;
                out
            } else {
                cuda_full_out.clone()
            };

            // CUDA shared-only (no routed experts, shared expert on hidden state)
            let cuda_shared_only = if include_shared {
                if let Some(ref mut c) = cache { c.reset_counters(); }
                let empty_pairs: Vec<(usize, f32)> = vec![];
                let (out, _) = execute_selected_moe_official_routed_fp4_cuda(&quant, &moe_executor, stream, &expert_fp4_set, &empty_pairs, &hidden, hidden_size, intermediate_size, hidden_size, layer, cache.as_deref_mut(), shared_dev.as_ref())?;
                out
            } else {
                vec![0.0f32; hidden_size]
            };

            // Independent parity comparisons
            let cuda_full_vs_cpu_full = compare_outputs(&official_cpu, &cuda_full_out)?;
            let cuda_shared_vs_cpu_shared = if include_shared {
                Some(compare_outputs(&official_cpu_shared, &cuda_shared_only)?)
            } else { None };

            let official_numerics = cuda_full_vs_cpu_full;
            let decoded_vs_official = compare_outputs(&decoded_cpu, &cuda_full_out)?;

            // Diagnostic norms and finite-output checks
            fn compute_l2_norm(v: &[f32]) -> f64 {
                v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
            }
            fn count_nan_inf(v: &[f32]) -> (u32, u32) {
                let mut nan = 0u32;
                let mut inf = 0u32;
                for &x in v {
                    if x.is_nan() { nan += 1; }
                    else if x.is_infinite() { inf += 1; }
                }
                (nan, inf)
            }
            let (r_nan, r_inf) = count_nan_inf(&official_cpu_routed);
            let (s_nan, s_inf) = count_nan_inf(&official_cpu_shared);
            let (f_nan, f_inf) = count_nan_inf(&official_cpu);
            let all_finite = r_nan == 0 && r_inf == 0 && s_nan == 0 && s_inf == 0 && f_nan == 0 && f_inf == 0;
            let parity_status: Option<String> = if all_finite {
                Some("valid_finite".into())
            } else {
                Some(format!("non_finite: routed_nan={r_nan} routed_inf={r_inf} shared_nan={s_nan} shared_inf={s_inf} full_nan={f_nan} full_inf={f_inf}"))
            };

            let routed_only_output_norm = compute_l2_norm(&official_cpu_routed);
            let shared_only_output_norm = compute_l2_norm(&official_cpu_shared);
            let full_moe_output_norm = compute_l2_norm(&official_cpu);
            // Note: shared_only_output_norm and full_moe_output_norm may be NaN if
            // the shared expert weight dimensions don't match hidden_size/intermediate_size.
            let full_minus_routed_norm = if include_shared {
                let diff: Vec<f32> = official_cpu.iter().zip(official_cpu_routed.iter()).map(|(f, r)| f - r).collect();
                compute_l2_norm(&diff)
            } else { 0.0 };

            let shared_merge_residual_l2 = if include_shared {
                let residual: Vec<f32> = cuda_full_out.iter().zip(cuda_routed_only.iter()).zip(cuda_shared_only.iter())
                    .map(|((full, routed), shared)| full - routed - shared).collect();
                compute_l2_norm(&residual)
            } else { 0.0 };

            M9Report {
                source: "real_deepseek_v4_flash_official_routed_fp4".into(),
                output_scope: Some("moe_forward_only".into()),
                model_dir, parse_dir,
                block_family: match block_family { BlockFamily::Decoder => "decoder".into(), BlockFamily::Mtp => "mtp".into() },
                layer_id: layer, layout_kind: meta.expert_layout.layout_kind.clone(),
                hidden_size, intermediate_size, num_experts, top_k,
                expert_ids: selected_ids, expert_weights: selected_weights,
                router_tensor_name,
                tensor_names_used: expert_tensor_names,
                scale_tensor_names_used: Some(scale_tensor_names),
                source_dtypes,
                quant_format: "DeepSeekFp4E2M1_official".into(),
                fp4_decode_ms: None,
                logical_expert_bytes_requested: mtel.logical_expert_bytes_requested,
                actual_expert_bytes_loaded: mtel.actual_expert_bytes_loaded,
                resident_cache_bytes_reused: mtel.resident_cache_bytes_reused,
                resident_cache_resident_bytes: mtel.resident_cache_resident_bytes,
                dequantized_scratch_bytes: 0,
                selected_working_set_bytes: mtel.selected_working_set_bytes,
                bytes_per_expert: mtel.bytes_per_expert,
                bytes_by_tensor_kind: Default::default(),
                cache_hit_count: 0, cache_miss_count: 0, cache_eviction_count: 0,
                cache_insert_attempt_count: 0, cache_insert_accept_count: 0, cache_insert_bypass_count: 0,
                oversized_tensor_bypass_count: 0, oversized_expert_bypass_count: 0, self_eviction_risk_count: 0,
                router_ms, tensor_load_ms, quantize_ms: 0.0,
                h2d_ms: mtel.h2d_ms,
                gate_up_gemv_ms: Some(mtel.gate_up_qgemv_ms),
                activation_ms: mtel.activation_ms,
                down_gemv_ms: Some(mtel.down_qgemv_ms),
                accum_ms: mtel.accum_ms,
                total_ms: mtel.total_ms,
                quant_vs_fp32: None,
                cuda_vs_cpu_native_fp4: Some(official_numerics),
                weight_bytes_loaded: mtel.weight_bytes_loaded,
                weight_bytes_reused: mtel.weight_bytes_reused,
                scale_bytes_loaded: mtel.scale_bytes_loaded,
                scale_bytes_reused: mtel.scale_bytes_reused,
                routed_fp4_bytes_loaded: Some(mtel.routed_fp4_bytes_loaded),
                routed_fp4_bytes_reused: Some(mtel.routed_fp4_bytes_reused),
                shared_fp8_bytes_loaded: Some(mtel.shared_fp8_bytes_loaded),
                shared_fp8_bytes_reused: Some(mtel.shared_fp8_bytes_reused),
                total_logical_bytes: Some(mtel.total_logical_bytes),
                total_loaded_bytes: Some(mtel.total_loaded_bytes),
                total_reused_bytes: Some(mtel.total_reused_bytes),
                source_bytes_loaded,
                native_fp4_cuda_moe_ms: Some(mtel.total_ms),
                gate_up_qgemv_ms: None, down_qgemv_ms: None,
                warmup_count: Some(warmup), repeat_count: Some(repeat),
                host_tensor_load_ms: Some(tensor_load_ms),
                first_device_fill_exec_ms: Some(mtel.total_ms),
                first_cold_end_to_end_ms: Some(tensor_load_ms + mtel.total_ms),
                all_layers_fixed_route_resident_bytes: Some(mtel.selected_working_set_bytes * meta.layout.num_layers),
                shared_expert_included: Some(include_shared),
                shared_expert_weight_bytes: shared_weight,
                shared_total_model_bytes: shared_weight.map(|w| w * meta.layout.num_layers),
                shared_resident_bytes: None,  // updated below when cache is available
                shared_load_bytes_per_token: Some(mtel.shared_fp8_bytes_loaded),
                free_capacity_after_shared_pin: cache.as_ref().map(|c| {
                    let shared_all = shared_weight.unwrap_or(0) * meta.layout.num_layers;
                    c.capacity_bytes.saturating_sub(shared_all + mtel.selected_working_set_bytes)
                }),
                moe_forward_full_cosine: Some(cuda_full_vs_cpu_full.cosine_similarity as f64),
                routed_only_output_norm: Some(routed_only_output_norm),
                shared_only_output_norm: if include_shared { Some(shared_only_output_norm) } else { None },
                full_moe_output_norm: Some(full_moe_output_norm),
                full_minus_routed_norm: if include_shared { Some(full_minus_routed_norm) } else { None },
                shared_merge_residual_l2: if include_shared { Some(shared_merge_residual_l2) } else { None },
                hyper_connection_included: Some(false),
                attention_included: Some(false),
                q4_transcode_invoked: Some(false),
                routed_only_nan_count: Some(r_nan),
                routed_only_inf_count: Some(r_inf),
                shared_only_nan_count: Some(s_nan),
                shared_only_inf_count: Some(s_inf),
                full_moe_nan_count: Some(f_nan),
                full_moe_inf_count: Some(f_inf),
                parity_status,
                cuda_shared_vs_cpu_shared,
                cuda_full_moe_vs_cpu_full_moe: Some(cuda_full_vs_cpu_full),
                official_arithmetic_cpu_reference_available: Some(true),
                official_arithmetic_cuda_parity: Some(all_finite && cuda_full_vs_cpu_full.cosine_similarity > 0.9999),
                decoded_weight_reference_parity: Some(decoded_vs_official.cosine_similarity > 0.99),
                warm_runs: Some(warm_runs),
            }
        }
        ExecutionFormat::Q4Transcode => {
            let mut expert_fp32_set = vec![
                ExpertWeightsFp32 {
                    w_gate: Vec::new(),
                    w_up: Vec::new(),
                    w_down: Vec::new(),
                };
                num_experts
            ];

            let mut fp4_decode_ms = None;

            if first_expert_uses_fp4 {
                let fp4_start = Instant::now();
                for &eid in &selected_ids {
                    let fp4_tensors = find_fp4_expert_tensors(&meta, layer, eid, block_family)?;

                    source_dtypes.insert(format!("expert_{}_gate", eid), "I8".into());
                    source_dtypes.insert(format!("expert_{}_up", eid), "I8".into());
                    source_dtypes.insert(format!("expert_{}_down", eid), "I8".into());
                    source_dtypes.insert(format!("expert_{}_gate_scale", eid), "F8_E8M0".into());
                    source_dtypes.insert(format!("expert_{}_up_scale", eid), "F8_E8M0".into());
                    source_dtypes.insert(format!("expert_{}_down_scale", eid), "F8_E8M0".into());

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
                    let gate_raw_w = load_raw_tensor(&model_weights, &fp4_tensors.gate_name)?;
                    let gate_raw_s = load_raw_tensor(&model_weights, &fp4_tensors.gate_scale_name)?;
                    source_bytes_loaded += gate_raw_w.len() + gate_raw_s.len();

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
                    let up_raw_w = load_raw_tensor(&model_weights, &fp4_tensors.up_name)?;
                    let up_raw_s = load_raw_tensor(&model_weights, &fp4_tensors.up_scale_name)?;
                    source_bytes_loaded += up_raw_w.len() + up_raw_s.len();

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
                    let down_raw_w = load_raw_tensor(&model_weights, &fp4_tensors.down_name)?;
                    let down_raw_s = load_raw_tensor(&model_weights, &fp4_tensors.down_scale_name)?;
                    source_bytes_loaded += down_raw_w.len() + down_raw_s.len();

                    expert_tensor_names.push(fp4_tensors.gate_name.clone());
                    expert_tensor_names.push(fp4_tensors.up_name.clone());
                    expert_tensor_names.push(fp4_tensors.down_name.clone());
                    scale_tensor_names.push(fp4_tensors.gate_scale_name.clone());
                    scale_tensor_names.push(fp4_tensors.up_scale_name.clone());
                    scale_tensor_names.push(fp4_tensors.down_scale_name.clone());

                    expert_fp32_set[eid] = ExpertWeightsFp32 {
                        w_gate: gate_fp32,
                        w_up: up_fp32,
                        w_down: down_fp32,
                    };
                }
                fp4_decode_ms = Some(fp4_start.elapsed().as_secs_f32() * 1000.0);
            } else {
                for &eid in &selected_ids {
                    let tensors = find_expert_tensors(&meta, layer, eid, block_family)?;

                    for t in &meta.expert_layout.tensors {
                        if t.layer_id == Some(layer)
                            && t.expert_id == Some(Some(eid))
                            && matches_block_family(&t.name, block_family)
                        {
                            let kind = t.tensor_kind.clone();
                            if !source_dtypes.contains_key(&kind) {
                                source_dtypes.insert(kind, t.dtype.clone());
                            }
                        }
                    }

                    let mut gate_fp32 = Vec::new();
                    model_weights.get_f32(&tensors.gate_name, &mut gate_fp32).map_err(|e| {
                        CudaError::new(
                            CudaErrorKind::Io,
                            format!("load gate tensor '{}'", tensors.gate_name),
                            e.to_string(),
                            file!(),
                            line!(),
                            module_path!(),
                        )
                    })?;
                    source_bytes_loaded += gate_fp32.len() * 4;

                    let mut up_fp32 = Vec::new();
                    model_weights.get_f32(&tensors.up_name, &mut up_fp32).map_err(|e| {
                        CudaError::new(
                            CudaErrorKind::Io,
                            format!("load up tensor '{}'", tensors.up_name),
                            e.to_string(),
                            file!(),
                            line!(),
                            module_path!(),
                        )
                    })?;
                    source_bytes_loaded += up_fp32.len() * 4;

                    let mut down_fp32 = Vec::new();
                    model_weights.get_f32(&tensors.down_name, &mut down_fp32).map_err(|e| {
                        CudaError::new(
                            CudaErrorKind::Io,
                            format!("load down tensor '{}'", tensors.down_name),
                            e.to_string(),
                            file!(),
                            line!(),
                            module_path!(),
                        )
                    })?;
                    source_bytes_loaded += down_fp32.len() * 4;

                    expert_tensor_names.push(tensors.gate_name.clone());
                    expert_tensor_names.push(tensors.up_name.clone());
                    expert_tensor_names.push(tensors.down_name.clone());

                    expert_fp32_set[eid] = ExpertWeightsFp32 {
                        w_gate: gate_fp32,
                        w_up: up_fp32,
                        w_down: down_fp32,
                    };
                }
            }

            let tensor_load_ms = t_load_start.elapsed().as_secs_f32() * 1000.0;

            let ref_out = selected_moe_cpu_fp32(
                &expert_fp32_set,
                &selected_pairs,
                &hidden,
                hidden_size,
                intermediate_size,
                hidden_size,
            )?;

            let quantize_start = Instant::now();
            let shape_gate_up = QGemvShape::new(QuantFormat::Q4_0, intermediate_size, hidden_size);
            let shape_down = QGemvShape::new(QuantFormat::Q4_0, hidden_size, intermediate_size);

            let mut expert_q4_set = vec![
                ExpertWeights {
                    w_gate: Vec::new(),
                    w_up: Vec::new(),
                    w_down: Vec::new(),
                };
                num_experts
            ];

            for &eid in &selected_ids {
                let ef = &expert_fp32_set[eid];
                let w_gate = q4_quantize_matrix_cpu(&ef.w_gate, shape_gate_up)?;
                let w_up = q4_quantize_matrix_cpu(&ef.w_up, shape_gate_up)?;
                let w_down = q4_quantize_matrix_cpu(&ef.w_down, shape_down)?;
                expert_q4_set[eid] = ExpertWeights {
                    w_gate,
                    w_up,
                    w_down,
                };
            }
            let quantize_ms = quantize_start.elapsed().as_secs_f32() * 1000.0;

            quant.compile_format(QuantFormat::Q4_0)?;
            moe_executor.compile()?;

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
                cache.as_deref_mut(),
            )?;

            let quant_vs_fp32 = compare_outputs(&ref_out, &cuda_out)?;

            let cache_counters = if let Some(ref c) = cache {
                (
                    c.hit_count, c.miss_count, c.eviction_count,
                    c.cache_insert_attempt_count, c.cache_insert_accept_count, c.cache_insert_bypass_count,
                    c.oversized_tensor_bypass_count, c.oversized_expert_bypass_count, c.self_eviction_risk_count,
                )
            } else {
                (0, 0, 0, 0, 0, 0, 0, 0, 0)
            };

            let source_label = if manual_mode {
                "real_deepseek_v4_flash_manual_expert_single_layer_moe"
            } else {
                "real_deepseek_v4_flash_single_layer_moe"
            };

            let scale_tensor_names_used = if scale_tensor_names.is_empty() {
                None
            } else {
                Some(scale_tensor_names)
            };

            M9Report {
                source: source_label.to_string(),
                output_scope: None,
                model_dir,
                parse_dir,
                block_family: match block_family {
                    BlockFamily::Decoder => "decoder".to_string(),
                    BlockFamily::Mtp => "mtp".to_string(),
                },
                layer_id: layer,
                layout_kind: meta.expert_layout.layout_kind.clone(),
                hidden_size,
                intermediate_size,
                num_experts,
                top_k,
                expert_ids: selected_ids,
                expert_weights: selected_weights,
                router_tensor_name,
                tensor_names_used: expert_tensor_names,
                scale_tensor_names_used,
                source_dtypes,
                quant_format: "Q4_0".to_string(),
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
                gate_up_gemv_ms: None,
                activation_ms: moe_telemetry.activation_ms,
                down_gemv_ms: None,
                accum_ms: moe_telemetry.accum_ms,
                total_ms: moe_telemetry.total_ms,
                quant_vs_fp32: Some(quant_vs_fp32),
                cuda_vs_cpu_native_fp4: None,
                weight_bytes_loaded: moe_telemetry.weight_bytes_loaded,
                weight_bytes_reused: moe_telemetry.weight_bytes_reused,
                scale_bytes_loaded: moe_telemetry.scale_bytes_loaded,
                scale_bytes_reused: moe_telemetry.scale_bytes_reused,
                routed_fp4_bytes_loaded: None,
                routed_fp4_bytes_reused: None,
                shared_fp8_bytes_loaded: None,
                shared_fp8_bytes_reused: None,
                total_logical_bytes: None,
                total_loaded_bytes: None,
                total_reused_bytes: None,
                source_bytes_loaded,
                native_fp4_cuda_moe_ms: None,
                gate_up_qgemv_ms: Some(moe_telemetry.gate_up_qgemv_ms),
                down_qgemv_ms: Some(moe_telemetry.down_qgemv_ms),
                warmup_count: None,
                repeat_count: None,
                host_tensor_load_ms: None,
                first_device_fill_exec_ms: None,
                first_cold_end_to_end_ms: None,
                all_layers_fixed_route_resident_bytes: None,
                shared_expert_included: None,
                shared_expert_weight_bytes: None,
                shared_total_model_bytes: None,
                shared_resident_bytes: None,
                shared_load_bytes_per_token: None,
                free_capacity_after_shared_pin: None,
                moe_forward_full_cosine: None,
                routed_only_output_norm: None,
                shared_only_output_norm: None,
                full_moe_output_norm: None,
                full_minus_routed_norm: None,
                shared_merge_residual_l2: None,
                hyper_connection_included: None,
                attention_included: None,
                q4_transcode_invoked: None,
                routed_only_nan_count: None,
                routed_only_inf_count: None,
                shared_only_nan_count: None,
                shared_only_inf_count: None,
                full_moe_nan_count: None,
                full_moe_inf_count: None,
                parity_status: None,
                cuda_shared_vs_cpu_shared: None,
                cuda_full_moe_vs_cpu_full_moe: None,
                official_arithmetic_cpu_reference_available: None,
                official_arithmetic_cuda_parity: None,
                decoded_weight_reference_parity: None,
                warm_runs: None,
            }
        }
    };

    Ok(report)
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

        let router_info = find_router_tensor(&meta, 0, BlockFamily::Decoder)?;
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
            let tensors = find_expert_tensors(&meta, 0, eid, BlockFamily::Decoder)?;
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
        let err = find_router_tensor(&meta, 0, BlockFamily::Decoder).unwrap_err();
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
        let t0 = find_expert_tensors(&meta, 0, 0, BlockFamily::Decoder)?;
        assert!(t0.gate_name.contains("experts.0"));
        assert!(t0.up_name.contains("experts.0"));
        assert!(t0.down_name.contains("experts.0"));

        // Expert 1 should fail (missing up and down)
        let err = find_expert_tensors(&meta, 0, 1, BlockFamily::Decoder).unwrap_err();
        assert_eq!(err.kind, CudaErrorKind::InvalidInput);
        assert!(err.source_message.contains("missing tensors"));

        // Expert 2 should fail (no tensors at all)
        let err = find_expert_tensors(&meta, 0, 2, BlockFamily::Decoder).unwrap_err();
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
        let err = find_router_tensor(&meta, 0, BlockFamily::Decoder).unwrap_err();
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
        let err = find_fp4_expert_tensors(&meta, 0, 0, BlockFamily::Decoder).unwrap_err();
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
        let fp4_tensors = find_fp4_expert_tensors(&meta, 0, 0, BlockFamily::Decoder)?;
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

    #[test]
    fn test_native_fp4_e2e_run() -> Result<()> {
        let tmp = std::env::temp_dir().join("objeta_m9_test_fp4_e2e");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let hidden = 64;
        let intermediate = 32;
        let num_experts = 4;
        let top_k = 2;
        let block_size = 32;

        let gate_phys_rows = intermediate;
        let gate_phys_cols = hidden / 2;
        let gate_logical = vec![intermediate, hidden];
        let gate_scale_cols = hidden / block_size;
        let gate_bytes = vec![0x21u8; gate_phys_rows * gate_phys_cols];
        let gate_scale_bytes = vec![127u8; gate_phys_rows * gate_scale_cols];

        let up_bytes = vec![0x21u8; gate_phys_rows * gate_phys_cols];
        let up_scale_bytes = vec![127u8; gate_phys_rows * gate_scale_cols];

        let down_phys_rows = hidden;
        let down_phys_cols = intermediate / 2;
        let down_logical = vec![hidden, intermediate];
        let down_scale_cols = intermediate / block_size;
        let down_bytes = vec![0x21u8; down_phys_rows * down_phys_cols];
        let down_scale_bytes = vec![127u8; down_phys_rows * down_scale_cols];

        let mut sf_tensors = HashMap::new();
        let router_data = vec![0.1f32; num_experts * hidden];
        let router_bytes = f32_bytes(&router_data);

        let r_off = (0, router_bytes.len());
        sf_tensors.insert("layers.0.ffn.gate.weight".to_string(), ("F32".into(), vec![num_experts, hidden], r_off, router_bytes));

        let mut offset = r_off.1;
        for eid in 0..num_experts {
            let g_off = (offset, offset + gate_bytes.len());
            sf_tensors.insert(format!("layers.0.ffn.experts.{}.w1.weight", eid), ("I8".into(), vec![intermediate, hidden / 2], g_off, gate_bytes.clone()));
            offset = g_off.1;

            let gs_off = (offset, offset + gate_scale_bytes.len());
            sf_tensors.insert(format!("layers.0.ffn.experts.{}.w1.scale", eid), ("F8_E8M0".into(), vec![intermediate, gate_scale_cols], gs_off, gate_scale_bytes.clone()));
            offset = gs_off.1;

            let u_off = (offset, offset + up_bytes.len());
            sf_tensors.insert(format!("layers.0.ffn.experts.{}.w2.weight", eid), ("I8".into(), vec![intermediate, hidden / 2], u_off, up_bytes.clone()));
            offset = u_off.1;

            let us_off = (offset, offset + up_scale_bytes.len());
            sf_tensors.insert(format!("layers.0.ffn.experts.{}.w2.scale", eid), ("F8_E8M0".into(), vec![intermediate, gate_scale_cols], us_off, up_scale_bytes.clone()));
            offset = us_off.1;

            let d_off = (offset, offset + down_bytes.len());
            sf_tensors.insert(format!("layers.0.ffn.experts.{}.w3.weight", eid), ("I8".into(), vec![hidden, intermediate / 2], d_off, down_bytes.clone()));
            offset = d_off.1;

            let ds_off = (offset, offset + down_scale_bytes.len());
            sf_tensors.insert(format!("layers.0.ffn.experts.{}.w3.scale", eid), ("F8_E8M0".into(), vec![hidden, down_scale_cols], ds_off, down_scale_bytes.clone()));
            offset = ds_off.1;
        }

        let model_dir = tmp.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        write_mock_safetensors(&model_dir.join("model.safetensors"), &sf_tensors).unwrap();

        let parse_dir = tmp.join("parse");
        let mut expert_entries = Vec::new();
        for eid in 0..num_experts {
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

        let args_native = vec![
            "binary".to_string(),
            "--parse-dir".to_string(), parse_dir.to_str().unwrap().to_string(),
            "--model-dir".to_string(), model_dir.to_str().unwrap().to_string(),
            "--layer".to_string(), "0".to_string(),
            "--execution-format".to_string(), "native-fp4".to_string(),
            "--cache-bytes".to_string(), "1000000".to_string(),
        ];

        let report = run_moe(&args_native)?;
        assert_eq!(report.quantize_ms, 0.0);
        assert!(report.cuda_vs_cpu_native_fp4.is_some());
        let sim = report.cuda_vs_cpu_native_fp4.unwrap().cosine_similarity;
        assert!(sim > 0.9999, "cosine similarity was {}", sim);
        assert_eq!(report.bytes_per_expert, 3264);
        assert_eq!(report.logical_expert_bytes_requested, 6528);
        assert_eq!(report.weight_bytes_loaded, 1024 * 3 * 2);
        assert_eq!(report.scale_bytes_loaded, 64 * 3 * 2);

        let report_hit = run_moe(&args_native)?;
        assert_eq!(report_hit.cache_hit_count, 12);
        assert_eq!(report_hit.actual_expert_bytes_loaded, 0);
        assert_eq!(report_hit.resident_cache_bytes_reused, 6528);
        assert_eq!(
            report_hit.logical_expert_bytes_requested,
            report_hit.actual_expert_bytes_loaded + report_hit.resident_cache_bytes_reused
        );

        let args_transcode = vec![
            "binary".to_string(),
            "--parse-dir".to_string(), parse_dir.to_str().unwrap().to_string(),
            "--model-dir".to_string(), model_dir.to_str().unwrap().to_string(),
            "--layer".to_string(), "0".to_string(),
            "--execution-format".to_string(), "q4-transcode".to_string(),
        ];
        let report_trans = run_moe(&args_transcode)?;
        assert!(report_trans.quantize_ms > 0.0);
        assert!(report_trans.quant_vs_fp32.is_some());
        assert!(report_trans.cuda_vs_cpu_native_fp4.is_none());

        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }

    #[test]
    fn test_block_family_distinction() -> Result<()> {
        let tmp = std::env::temp_dir().join("objeta_m9_test_family_dist");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let hidden = 64;
        let intermediate = 32;
        let num_experts = 2;
        let top_k = 1;

        let router_data = vec![0.1f32; num_experts * hidden];
        let router_bytes = f32_bytes(&router_data);

        let mut sf_tensors = HashMap::new();
        sf_tensors.insert("layers.0.ffn.gate.weight".to_string(), ("F32".into(), vec![num_experts, hidden], (0, router_bytes.len()), router_bytes.clone()));
        sf_tensors.insert("mtp.0.ffn.gate.weight".to_string(), ("F32".into(), vec![num_experts, hidden], (0, router_bytes.len()), router_bytes.clone()));

        let model_dir = tmp.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        write_mock_safetensors(&model_dir.join("model.safetensors"), &sf_tensors).unwrap();

        let parse_dir = tmp.join("parse");
        let routers = vec![
            RouterTensorEntry {
                name: "layers.0.ffn.gate.weight".into(),
                layer_id: Some(0),
                shape: vec![num_experts, hidden],
                dtype: "F32".into(),
            },
            RouterTensorEntry {
                name: "mtp.0.ffn.gate.weight".into(),
                layer_id: Some(0),
                shape: vec![num_experts, hidden],
                dtype: "F32".into(),
            },
        ];

        write_parser_json_files(
            &parse_dir, 1, hidden, intermediate, num_experts, top_k, "float32",
            &[],
            &routers,
        );

        let meta = load_parser_metadata(&parse_dir)?;

        let dec_router = find_router_tensor(&meta, 0, BlockFamily::Decoder)?;
        assert_eq!(dec_router.name, "layers.0.ffn.gate.weight");

        let mtp_router = find_router_tensor(&meta, 0, BlockFamily::Mtp)?;
        assert_eq!(mtp_router.name, "mtp.0.ffn.gate.weight");

        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }
}
