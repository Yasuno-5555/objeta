use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use objeta_core::{ObjetaError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct DeepseekConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    
    // Hidden layers
    #[serde(alias = "num_hidden_layers", alias = "num_layers")]
    pub num_hidden_layers: Option<usize>,
    
    pub hidden_size: Option<usize>,
    
    #[serde(alias = "intermediate_size", alias = "moe_intermediate_size", alias = "ffn_dim")]
    pub intermediate_size: Option<usize>,
    
    // Experts
    #[serde(alias = "n_routed_experts", alias = "num_local_experts", alias = "num_experts")]
    pub num_experts: Option<usize>,
    
    #[serde(alias = "num_experts_per_tok", alias = "top_k")]
    pub top_k: Option<usize>,
    
    pub vocab_size: Option<usize>,
    
    #[serde(alias = "torch_dtype")]
    pub dtype: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekLayout {
    pub model_name: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    pub dtype: String,
    pub quant_dtype: Option<String>,
    pub tensor_count: usize,
    pub shard_count: usize,
    pub total_byte_size: u64,
    pub largest_tensor: Option<LargestTensorInfo>,
    pub tensor_name_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargestTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorIndexEntry {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub byte_length: usize,
    pub offset: usize,
    pub source_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertLayout {
    pub layout_kind: String, // "explicit_experts", "packed_experts", "unknown"
    pub tensors: Vec<ExpertTensorClassification>,
    #[serde(default)]
    pub fp4_expert_storage_detected: bool,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertTensorClassification {
    pub name: String,
    pub layer_id: Option<usize>,
    pub expert_id: Option<Option<usize>>, // explicit_id if explicit, or None
    pub tensor_kind: String, // "gate", "up", "down", "gate_up", "router", "shared_expert", "unknown"
    pub shape: Vec<usize>,
    pub dtype: String,
    pub byte_length: usize,
    pub source_file: String,
    // Quantized storage metadata (populated when fp4/fp8 storage is detected)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_tensor_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_shape: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_shape: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packed_values_per_byte: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterLayout {
    pub num_experts: Option<usize>,
    pub top_k: Option<usize>,
    pub warnings: Vec<String>,
    pub routers: Vec<RouterTensorClassification>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterTensorClassification {
    pub name: String,
    pub layer_id: Option<usize>,
    pub tensor_kind: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub byte_length: usize,
    pub source_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventorySummary {
    pub total_expert_bytes: u64,
    pub expert_bytes_per_layer: HashMap<String, u64>,
    pub bytes_per_expert: Option<HashMap<String, u64>>,
    pub bytes_by_tensor_kind: BytesByTensorKindSummary,
    pub largest_expert_tensor: Option<LargestTensorInfo>,
    pub largest_layer_by_expert_bytes: Option<LargestLayerSummary>,
    pub fits_in_cache: FitsInCacheSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BytesByTensorKindSummary {
    pub gate: u64,
    pub up: u64,
    pub down: u64,
    pub gate_up: u64,
    #[serde(default)]
    pub gate_scale: u64,
    #[serde(default)]
    pub up_scale: u64,
    #[serde(default)]
    pub down_scale: u64,
    pub router: u64,
    pub shared: u64,
    pub attention: u64,
    pub other: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargestLayerSummary {
    pub layer_id: usize,
    pub expert_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitsInCacheSummary {
    #[serde(rename = "1GB")]
    pub fit_1gb: bool,
    #[serde(rename = "2GB")]
    pub fit_2gb: bool,
    #[serde(rename = "4GB")]
    pub fit_4gb: bool,
    #[serde(rename = "8GB")]
    pub fit_8gb: bool,
}

// Helper to classify tensor names
pub fn classify_tensor(name: &str) -> (Option<usize>, Option<usize>, String) {
    let parts: Vec<&str> = name.split('.').collect();

    let mut layer_id = None;
    if let Some(pos) = parts.iter().position(|&x| x == "layers") {
        if pos + 1 < parts.len() {
            if let Ok(l_id) = parts[pos + 1].parse::<usize>() {
                layer_id = Some(l_id);
            }
        }
    }

    // Also parse layer_id from "mtp.{L}.*" pattern (multi-token prediction heads)
    if layer_id.is_none() {
        if let Some(pos) = parts.iter().position(|&x| x == "mtp") {
            if pos + 1 < parts.len() {
                if let Ok(l_id) = parts[pos + 1].parse::<usize>() {
                    layer_id = Some(l_id);
                }
            }
        }
    }

    let is_ffn = parts.contains(&"mlp") || parts.contains(&"ffn");

    if is_ffn {
        let is_shared = parts.contains(&"shared_experts") || parts.contains(&"shared_expert");
        let is_experts = parts.contains(&"experts") || parts.contains(&"expert");

        if is_shared {
            return (layer_id, None, "shared_expert".to_string());
        }

        if is_experts {
            let mut expert_id = None;
            if let Some(pos) = parts.iter().position(|&x| x == "experts" || x == "expert") {
                if pos + 1 < parts.len() {
                    if let Ok(e_id) = parts[pos + 1].parse::<usize>() {
                        expert_id = Some(e_id);
                    }
                }
            }

            // Detect scale tensors
            let is_scale = name.ends_with(".scale");

            // DeepSeek V4 Flash convention: w1=gate, w2=down, w3=up
            if name.contains(".w1.") {
                return (layer_id, expert_id, if is_scale { "gate_scale" } else { "gate" }.to_string());
            }
            if name.contains(".w3.") {
                return (layer_id, expert_id, if is_scale { "up_scale" } else { "up" }.to_string());
            }
            if name.contains(".w2.") {
                return (layer_id, expert_id, if is_scale { "down_scale" } else { "down" }.to_string());
            }

            // Standard HF convention
            let kind = if name.contains("gate_up_proj") || (name.contains("gate_proj") && name.contains("up_proj")) {
                "gate_up"
            } else if name.contains("gate_proj") {
                "gate"
            } else if name.contains("up_proj") {
                "up"
            } else if name.contains("down_proj") {
                "down"
            } else {
                "unknown"
            };
            return (layer_id, expert_id, kind.to_string());
        }

        // Router / gate in ffn path (but not inside experts/shared_experts)
        let is_gate = parts.contains(&"gate") || parts.contains(&"router");
        if is_gate {
            if name.ends_with(".tid2eid") {
                return (layer_id, None, "router_tid2eid".to_string());
            }
            return (layer_id, None, "router".to_string());
        }
    }

    // Highway Crossing (hash routing) tensors: hc_ffn_*, hc_attn_*
    if name.contains("hc_ffn_base") {
        return (layer_id, None, "router_hc_base".to_string());
    }
    if name.contains("hc_ffn_fn") {
        return (layer_id, None, "router_hc_fn".to_string());
    }
    if name.contains("hc_ffn_scale") {
        return (layer_id, None, "router_hc_scale".to_string());
    }

    (layer_id, None, "unknown".to_string())
}

fn get_tensor_pattern(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let mut pattern_parts = Vec::new();
    for part in parts {
        if part.parse::<usize>().is_ok() {
            pattern_parts.push("*");
        } else {
            pattern_parts.push(part);
        }
    }
    pattern_parts.join(".")
}

#[derive(Deserialize, Debug)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

pub fn parse_deepseek_v4_flash(model_dir: &Path, output_dir: &Path) -> Result<()> {
    // 1. Load config.json if present
    let config_path = model_dir.join("config.json");
    let mut warnings = Vec::new();
    
    let config = if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        match serde_json::from_str::<DeepseekConfig>(&content) {
            Ok(cfg) => cfg,
            Err(e) => {
                warnings.push(format!("Failed to parse config.json: {}", e));
                DeepseekConfig {
                    model_type: None,
                    architectures: None,
                    num_hidden_layers: None,
                    hidden_size: None,
                    intermediate_size: None,
                    num_experts: None,
                    top_k: None,
                    vocab_size: None,
                    dtype: None,
                }
            }
        }
    } else {
        warnings.push("config.json not found in model directory".to_string());
        DeepseekConfig {
            model_type: None,
            architectures: None,
            num_hidden_layers: None,
            hidden_size: None,
            intermediate_size: None,
            num_experts: None,
            top_k: None,
            vocab_size: None,
            dtype: None,
        }
    };

    // 2. Scan directory for .safetensors files
    let mut sf_files = Vec::new();
    if model_dir.exists() && model_dir.is_dir() {
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "safetensors") {
                sf_files.push(path);
            }
        }
    }
    sf_files.sort();

    if sf_files.is_empty() {
        return Err(ObjetaError::Parse("No .safetensors files found in model directory".into()));
    }

    // 3. Parse headers of all safetensors files (Inspect only, no full loading/mmap of data)
    let mut tensor_index: HashMap<String, TensorIndexEntry> = HashMap::new();
    let mut total_byte_size = 0u64;
    let mut largest_tensor: Option<LargestTensorInfo> = None;
    let mut pattern_set = std::collections::HashSet::new();

    for file_path in &sf_files {
        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();
        let mut file = File::open(file_path)?;
        
        let mut len_bytes = [0u8; 8];
        if file.read_exact(&mut len_bytes).is_err() {
            warnings.push(format!("File {} is too short to contain safetensors header length prefix", file_name));
            continue;
        }
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        
        let mut header_buf = vec![0u8; header_len];
        if file.read_exact(&mut header_buf).is_err() {
            warnings.push(format!("Failed to read header of size {} from {}", header_len, file_name));
            continue;
        }
        
        let header_json = match std::str::from_utf8(&header_buf) {
            Ok(s) => s,
            Err(e) => {
                warnings.push(format!("Invalid UTF-8 in header of {}: {}", file_name, e));
                continue;
            }
        };

        let raw: HashMap<String, serde_json::Value> = match serde_json::from_str(header_json) {
            Ok(r) => r,
            Err(e) => {
                warnings.push(format!("Invalid JSON in header of {}: {}", file_name, e));
                continue;
            }
        };

        for (name, val) in raw {
            if name == "__metadata__" {
                continue;
            }
            if let Ok(entry) = serde_json::from_value::<RawTensorEntry>(val) {
                let byte_length = entry.data_offsets.1 - entry.data_offsets.0;
                total_byte_size += byte_length as u64;

                let pattern = get_tensor_pattern(&name);
                pattern_set.insert(pattern);

                if largest_tensor.is_none() || byte_length as u64 > largest_tensor.as_ref().unwrap().size_bytes {
                    largest_tensor = Some(LargestTensorInfo {
                        name: name.clone(),
                        shape: entry.shape.clone(),
                        dtype: entry.dtype.clone(),
                        size_bytes: byte_length as u64,
                    });
                }

                tensor_index.insert(name, TensorIndexEntry {
                    shape: entry.shape,
                    dtype: entry.dtype,
                    byte_length,
                    offset: 8 + header_len + entry.data_offsets.0,
                    source_file: file_name.clone(),
                });
            }
        }
    }

    // 4. Classify tensors to build expert layout
    let mut expert_tensors = Vec::new();
    let mut router_tensors = Vec::new();
    let mut has_explicit_experts = false;
    let mut has_packed_experts = false;
    let mut fp4_detected = false;

    let mut bytes_by_tensor_kind = BytesByTensorKindSummary {
        gate: 0,
        up: 0,
        down: 0,
        gate_up: 0,
        gate_scale: 0,
        up_scale: 0,
        down_scale: 0,
        router: 0,
        shared: 0,
        attention: 0,
        other: 0,
    };

    let mut detected_num_experts = 0;
    let mut inferred_num_layers = 0;

    // First pass: collect all classified tensors
    let mut classified: Vec<(String, Option<usize>, Option<usize>, String, TensorIndexEntry)> = Vec::new();
    for (name, tensor) in &tensor_index {
        let (layer_id, expert_id, kind) = classify_tensor(name);
        if let Some(l_id) = layer_id {
            if l_id >= inferred_num_layers {
                inferred_num_layers = l_id + 1;
            }
        }
        classified.push((name.clone(), layer_id, expert_id, kind, tensor.clone()));
    }

    for (name, layer_id, expert_id, kind, tensor) in &classified {
        let kind = kind.as_str();

        if name.contains("self_attn") || name.contains("attn") {
            bytes_by_tensor_kind.attention += tensor.byte_length as u64;
        } else {
            match kind {
                "gate" => bytes_by_tensor_kind.gate += tensor.byte_length as u64,
                "up" => bytes_by_tensor_kind.up += tensor.byte_length as u64,
                "down" => bytes_by_tensor_kind.down += tensor.byte_length as u64,
                "gate_up" => bytes_by_tensor_kind.gate_up += tensor.byte_length as u64,
                "gate_scale" => bytes_by_tensor_kind.gate_scale += tensor.byte_length as u64,
                "up_scale" => bytes_by_tensor_kind.up_scale += tensor.byte_length as u64,
                "down_scale" => bytes_by_tensor_kind.down_scale += tensor.byte_length as u64,
                "router" => bytes_by_tensor_kind.router += tensor.byte_length as u64,
                "shared_expert" => bytes_by_tensor_kind.shared += tensor.byte_length as u64,
                _ => bytes_by_tensor_kind.other += tensor.byte_length as u64,
            }
        }

        match kind {
            "router" | "router_tid2eid" | "router_hc_base" | "router_hc_fn" | "router_hc_scale" => {
                router_tensors.push(RouterTensorClassification {
                    name: name.clone(),
                    layer_id: *layer_id,
                    tensor_kind: Some(kind.to_string()),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype.clone(),
                    byte_length: tensor.byte_length,
                    source_file: tensor.source_file.clone(),
                });

                if kind == "router" && !tensor.shape.is_empty() {
                    let experts_dim = tensor.shape[0];
                    if experts_dim > detected_num_experts {
                        detected_num_experts = experts_dim;
                    }
                }
            }
            "gate" | "up" | "down" | "gate_up" | "shared_expert" | "gate_scale" | "up_scale" | "down_scale" => {
                if expert_id.is_some() {
                    if kind == "gate" || kind == "up" || kind == "down" {
                        has_explicit_experts = true;
                    }
                } else if kind != "shared_expert" && !kind.ends_with("_scale") {
                    has_packed_experts = true;
                }

                // Build quantized storage metadata for fp4-packed weights
                let mut storage_dtype: Option<String> = None;
                let mut logical_dtype = None;
                let mut scale_tensor_name: Option<String> = None;
                let mut scale_dtype = None;
                let mut logical_shape = None;
                let mut packed_values_per_byte = None;
                let mut block_size = None;

                let is_weight = matches!(kind, "gate" | "up" | "down");
                if is_weight && tensor.dtype == "I8" {
                    // Potential fp4-packed weight: look for matching .scale tensor
                    let scale_name = name.replace(".weight", ".scale");
                    if let Some(scale_entry) = tensor_index.get(&scale_name) {
                        fp4_detected = true;
                        storage_dtype = Some("I8".to_string());
                        logical_dtype = Some("FP4".to_string());
                        scale_tensor_name = Some(scale_name.clone());
                        scale_dtype = Some(scale_entry.dtype.clone());
                        packed_values_per_byte = Some(2);

                        // Infer logical shape: I8 packs 2 fp4 values per byte
                        let phys_nelem: usize = tensor.shape.iter().product();
                        // Assume the packed dimension is the last one (cols)
                        let mut log_shape = tensor.shape.clone();
                        if let Some(last) = log_shape.last_mut() {
                            *last *= 2;
                        }
                        logical_shape = Some(log_shape);

                        // Infer block_size (logical fp4 elements per scale value)
                        // phys_nelem is I8 bytes; each packs 2 fp4 values
                        let scale_nelem: usize = scale_entry.shape.iter().product();
                        let logical_nelem = phys_nelem * 2;
                        if scale_nelem > 0 && logical_nelem % scale_nelem == 0 {
                            block_size = Some(logical_nelem / scale_nelem);
                        }
                    }
                }

                let has_storage = storage_dtype.is_some();
                expert_tensors.push(ExpertTensorClassification {
                    name: name.clone(),
                    layer_id: *layer_id,
                    expert_id: if expert_id.is_some() { Some(*expert_id) } else { None },
                    tensor_kind: kind.to_string(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype.clone(),
                    byte_length: tensor.byte_length,
                    source_file: tensor.source_file.clone(),
                    storage_dtype,
                    logical_dtype,
                    scale_tensor_name,
                    scale_dtype,
                    physical_shape: if has_storage { Some(tensor.shape.clone()) } else { None },
                    logical_shape,
                    packed_values_per_byte,
                    block_size,
                });
            }
            _ => {} // unknown tensors not added to expert/ router layouts
        }
    }

    // Determine layout kind
    let layout_kind = if has_explicit_experts {
        "explicit_experts"
    } else if has_packed_experts {
        "packed_experts"
    } else {
        "unknown"
    };

    // 5. Build High-Level config/layout metadata fields
    let config_num_experts = config.num_experts.unwrap_or(detected_num_experts);
    let inferred_num_experts = if config_num_experts > 0 { config_num_experts } else { detected_num_experts };
    let inferred_top_k = config.top_k.unwrap_or(0);
    
    if inferred_top_k == 0 {
        warnings.push("top_k could not be inferred from config, defaulting to null".to_string());
    }
    if inferred_num_experts == 0 {
        warnings.push("num_experts could not be inferred from config or router shape, defaulting to null".to_string());
    }

    let final_num_layers = config.num_hidden_layers.unwrap_or(inferred_num_layers);
    let final_vocab_size = config.vocab_size.unwrap_or(0);
    let final_hidden_size = config.hidden_size.unwrap_or(0);
    let final_intermediate_size = config.intermediate_size.unwrap_or(0);

    let primary_dtype = config.dtype.clone().unwrap_or_else(|| {
        if let Some(ref entry) = largest_tensor {
            entry.dtype.clone()
        } else {
            "unknown".to_string()
        }
    });

    let mut patterns: Vec<String> = pattern_set.into_iter().collect();
    patterns.sort();

    let layout_summary = DeepseekLayout {
        model_name: config.model_type.clone().unwrap_or_else(|| {
            if let Some(ref archs) = config.architectures {
                if !archs.is_empty() {
                    return archs[0].clone();
                }
            }
            "deepseek_v4_flash".to_string()
        }),
        num_layers: final_num_layers,
        hidden_size: final_hidden_size,
        intermediate_size: final_intermediate_size,
        num_experts: inferred_num_experts,
        top_k: inferred_top_k,
        vocab_size: final_vocab_size,
        dtype: primary_dtype,
        quant_dtype: None, // Will be filled dynamically if we parse quantization details
        tensor_count: tensor_index.len(),
        shard_count: sf_files.len(),
        total_byte_size,
        largest_tensor: largest_tensor.clone(),
        tensor_name_patterns: patterns,
    };

    // Sort outputs for determinism
    expert_tensors.sort_by(|a, b| a.name.cmp(&b.name));
    router_tensors.sort_by(|a, b| a.name.cmp(&b.name));

    // 6. Build Expert layout
    let expert_layout = ExpertLayout {
        layout_kind: layout_kind.to_string(),
        tensors: expert_tensors.clone(),
        fp4_expert_storage_detected: fp4_detected,
        warnings: warnings.clone(),
    };

    // 7. Build Router layout
    let router_layout = RouterLayout {
        num_experts: if inferred_num_experts > 0 { Some(inferred_num_experts) } else { None },
        top_k: if inferred_top_k > 0 { Some(inferred_top_k) } else { None },
        warnings: warnings.clone(),
        routers: router_tensors,
    };

    // 8. Build Inventory summary
    let total_expert_bytes: u64 = expert_tensors.iter()
        .filter(|t| t.tensor_kind != "shared_expert")
        .map(|t| t.byte_length as u64)
        .sum();

    let mut expert_bytes_per_layer = HashMap::new();
    for t in &expert_tensors {
        if t.tensor_kind != "shared_expert" {
            let layer_key = t.layer_id.unwrap_or(0).to_string();
            *expert_bytes_per_layer.entry(layer_key).or_insert(0) += t.byte_length as u64;
        }
    }

    let mut bytes_per_expert = None;
    if layout_kind == "explicit_experts" {
        let mut map = HashMap::new();
        for t in &expert_tensors {
            if let Some(Some(e_id)) = t.expert_id {
                *map.entry(e_id.to_string()).or_insert(0) += t.byte_length as u64;
            }
        }
        bytes_per_expert = Some(map);
    } else if layout_kind == "packed_experts" && inferred_num_experts > 0 {
        let mut map = HashMap::new();
        let per_expert = total_expert_bytes / inferred_num_experts as u64;
        for i in 0..inferred_num_experts {
            map.insert(i.to_string(), per_expert);
        }
        bytes_per_expert = Some(map);
    }

    let mut largest_expert_tensor: Option<LargestTensorInfo> = None;
    for t in &expert_tensors {
        if largest_expert_tensor.is_none() || t.byte_length as u64 > largest_expert_tensor.as_ref().unwrap().size_bytes {
            largest_expert_tensor = Some(LargestTensorInfo {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: t.dtype.clone(),
                size_bytes: t.byte_length as u64,
            });
        }
    }

    let mut largest_layer_by_expert_bytes = None;
    let mut max_layer_expert_bytes = 0u64;
    for (layer_str, bytes) in &expert_bytes_per_layer {
        if let Ok(layer_id) = layer_str.parse::<usize>() {
            if *bytes > max_layer_expert_bytes {
                max_layer_expert_bytes = *bytes;
                largest_layer_by_expert_bytes = Some(LargestLayerSummary {
                    layer_id,
                    expert_bytes: *bytes,
                });
            }
        }
    }

    // Compute working set and fits-in-cache
    let mut single_expert_bytes_per_layer = 0;
    let num_layers_val = final_num_layers.max(1);
    let num_experts_val = inferred_num_experts.max(1);

    if layout_kind == "explicit_experts" {
        if let Some(first_expert) = expert_tensors.iter().find(|t| t.expert_id.is_some() && t.layer_id.is_some()) {
            let target_layer = first_expert.layer_id;
            let target_expert = first_expert.expert_id;
            single_expert_bytes_per_layer = expert_tensors.iter()
                .filter(|t| t.layer_id == target_layer && t.expert_id == target_expert)
                .map(|t| t.byte_length as u64)
                .sum();
        }
    } else if layout_kind == "packed_experts" {
        if let Some(first_expert) = expert_tensors.iter().find(|t| t.layer_id.is_some()) {
            let target_layer = first_expert.layer_id;
            let layer_total: u64 = expert_tensors.iter()
                .filter(|t| t.layer_id == target_layer)
                .map(|t| t.byte_length as u64)
                .sum();
            if inferred_num_experts > 0 {
                single_expert_bytes_per_layer = layer_total / inferred_num_experts as u64;
            }
        }
    }

    if single_expert_bytes_per_layer == 0 {
        single_expert_bytes_per_layer = total_expert_bytes / (num_layers_val * num_experts_val) as u64;
    }

    let working_set_bytes = num_layers_val as u64 * inferred_top_k as u64 * single_expert_bytes_per_layer;

    let fit_1gb = working_set_bytes <= 1024 * 1024 * 1024;
    let fit_2gb = working_set_bytes <= 2 * 1024 * 1024 * 1024;
    let fit_4gb = working_set_bytes <= 4 * 1024 * 1024 * 1024;
    let fit_8gb = working_set_bytes <= 8 * 1024 * 1024 * 1024;

    let inventory_summary = InventorySummary {
        total_expert_bytes,
        expert_bytes_per_layer,
        bytes_per_expert,
        bytes_by_tensor_kind,
        largest_expert_tensor,
        largest_layer_by_expert_bytes,
        fits_in_cache: FitsInCacheSummary {
            fit_1gb,
            fit_2gb,
            fit_4gb,
            fit_8gb,
        },
    };

    // 9. Write JSON files
    std::fs::create_dir_all(output_dir)?;
    
    let path_layout = output_dir.join("deepseek_v4_flash_layout.json");
    let json_layout = serde_json::to_string_pretty(&layout_summary).unwrap();
    std::fs::write(&path_layout, json_layout)?;

    let path_index = output_dir.join("deepseek_v4_flash_tensor_index.json");
    let json_index = serde_json::to_string_pretty(&tensor_index).unwrap();
    std::fs::write(&path_index, json_index)?;

    let path_expert = output_dir.join("deepseek_v4_flash_expert_layout.json");
    let json_expert = serde_json::to_string_pretty(&expert_layout).unwrap();
    std::fs::write(&path_expert, json_expert)?;

    let path_router = output_dir.join("deepseek_v4_flash_router_layout.json");
    let json_router = serde_json::to_string_pretty(&router_layout).unwrap();
    std::fs::write(&path_router, json_router)?;

    let path_summary = output_dir.join("deepseek_v4_flash_inventory_summary.json");
    let json_summary = serde_json::to_string_pretty(&inventory_summary).unwrap();
    std::fs::write(&path_summary, json_summary)?;

    Ok(())
}

// ── FP4 decode ───────────────────────────────────────────────────────────

/// FP4 E2M1FN decode lookup table, confirmed from DeepSeek V4 Flash inference/convert.py.
const FP4_E2M1FN_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode an F8_E8M0 scale byte to f32: 2^(raw - 127).
/// Confirmed from DeepSeek V4 Flash inference/kernel.py `fast_pow2`.
#[inline]
pub fn f8e8m0_to_f32(raw: u8) -> f32 {
    let exponent = raw as i32 - 127;
    // 2^exponent via bit-level fp32 construction
    if exponent < -126 {
        // Subnormal: result = 2^exponent
        // Minimum representable is 2^-149; below that is zero
        if exponent < -149 {
            return 0.0;
        }
        // Subnormal: mantissa bit shifted right by (-126 - exponent)
        let mantissa = 1u32 << (23 + (exponent + 126));
        f32::from_bits(mantissa)
    } else if exponent > 127 {
        f32::INFINITY
    } else {
        let bits = ((exponent + 127) as u32) << 23;
        f32::from_bits(bits)
    }
}

/// Decode a DeepSeek FP4-packed expert weight to f32.
///
/// Packing: 2 FP4 values per I8 byte along the last dimension.
/// Low nibble (bits 0-3) = first fp4 value, high nibble (bits 4-7) = second.
/// Each block of `block_size` logical fp4 elements shares one F8_E8M0 scale value.
/// Block size is 32 for DeepSeek V4 Flash.
///
/// Weight matrix orientation: `[rows, logical_cols]` row-major,
/// where `logical_cols = physical_cols * 2`.
///
/// This is a CPU-only reference implementation. It does not require CUDA.
pub fn decode_deepseek_fp4_to_f32(
    weight_i8: &[u8],
    scale_f8e8m0: &[u8],
    physical_shape: &[usize],
    logical_shape: &[usize],
    block_size: usize,
) -> Vec<f32> {
    assert_eq!(physical_shape.len(), 2, "weight must be 2D");
    assert_eq!(logical_shape.len(), 2, "logical shape must be 2D");
    assert_eq!(physical_shape[0], logical_shape[0], "row count must match");

    let rows = logical_shape[0];
    let logical_cols = logical_shape[1];
    let phys_cols = physical_shape[1];
    let scale_cols = logical_cols / block_size;
    let fp4_per_byte = 2;

    assert_eq!(
        weight_i8.len(),
        rows * phys_cols,
        "weight bytes must match physical shape"
    );
    assert_eq!(
        scale_f8e8m0.len(),
        rows * scale_cols,
        "scale bytes must match rows * (logical_cols / block_size)"
    );
    assert_eq!(
        logical_cols % block_size,
        0,
        "logical columns must be divisible by block_size"
    );

    let mut out = vec![0.0f32; rows * logical_cols];

    for row in 0..rows {
        for col_group in 0..phys_cols {
            let byte_val = weight_i8[row * phys_cols + col_group];
            let low = byte_val & 0x0F;
            let high = (byte_val >> 4) & 0x0F;

            let logical_col_lo = col_group * fp4_per_byte;
            let logical_col_hi = logical_col_lo + 1;

            let scale_idx_lo = row * scale_cols + logical_col_lo / block_size;
            let scale_idx_hi = row * scale_cols + logical_col_hi / block_size;

            let s_lo = f8e8m0_to_f32(scale_f8e8m0[scale_idx_lo]);
            let s_hi = f8e8m0_to_f32(scale_f8e8m0[scale_idx_hi]);

            out[row * logical_cols + logical_col_lo] = FP4_E2M1FN_TABLE[low as usize] * s_lo;
            out[row * logical_cols + logical_col_hi] = FP4_E2M1FN_TABLE[high as usize] * s_hi;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_mock_safetensors(
        path: &Path,
        tensors: &HashMap<String, (String, Vec<usize>, (usize, usize))>,
    ) -> std::io::Result<()> {
        let mut header_map = serde_json::Map::new();
        let mut max_offset = 0;
        
        for (name, (dtype, shape, offsets)) in tensors {
            let mut entry = serde_json::Map::new();
            entry.insert("dtype".to_string(), serde_json::Value::String(dtype.clone()));
            entry.insert(
                "shape".to_string(),
                serde_json::Value::Array(shape.iter().map(|&s| serde_json::Value::Number(s.into())).collect()),
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
        }
        
        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;
        
        let mut file = File::create(path)?;
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(header_bytes)?;
        
        let dummy_data = vec![0u8; max_offset];
        file.write_all(&dummy_data)?;
        
        Ok(())
    }

    #[test]
    fn test_classify_tensor() {
        // Standard mlp convention (unchanged)
        assert_eq!(
            classify_tensor("model.layers.5.mlp.experts.12.gate_proj.weight"),
            (Some(5), Some(12), "gate".to_string())
        );
        assert_eq!(
            classify_tensor("model.layers.2.mlp.experts.3.up_proj.weight"),
            (Some(2), Some(3), "up".to_string())
        );
        assert_eq!(
            classify_tensor("model.layers.0.mlp.experts.1.down_proj.weight"),
            (Some(0), Some(1), "down".to_string())
        );
        assert_eq!(
            classify_tensor("model.layers.1.mlp.shared_experts.gate_proj.weight"),
            (Some(1), None, "shared_expert".to_string())
        );
        assert_eq!(
            classify_tensor("model.layers.3.mlp.gate.weight"),
            (Some(3), None, "router".to_string())
        );
        assert_eq!(
            classify_tensor("model.layers.3.self_attn.q_proj.weight"),
            (Some(3), None, "unknown".to_string())
        );

        // DeepSeek V4 Flash ffn convention
        assert_eq!(
            classify_tensor("layers.0.ffn.experts.5.w1.weight"),
            (Some(0), Some(5), "gate".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.ffn.experts.5.w2.weight"),
            (Some(0), Some(5), "down".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.ffn.experts.5.w3.weight"),
            (Some(0), Some(5), "up".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.ffn.experts.5.w1.scale"),
            (Some(0), Some(5), "gate_scale".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.ffn.experts.5.w2.scale"),
            (Some(0), Some(5), "down_scale".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.ffn.experts.5.w3.scale"),
            (Some(0), Some(5), "up_scale".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.ffn.gate.weight"),
            (Some(0), None, "router".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.ffn.gate.tid2eid"),
            (Some(0), None, "router_tid2eid".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.ffn.shared_experts.w1.weight"),
            (Some(0), None, "shared_expert".to_string())
        );

        // Highway Crossing (hash routing) tensors
        assert_eq!(
            classify_tensor("layers.0.hc_ffn_base"),
            (Some(0), None, "router_hc_base".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.hc_ffn_fn"),
            (Some(0), None, "router_hc_fn".to_string())
        );
        assert_eq!(
            classify_tensor("layers.0.hc_ffn_scale"),
            (Some(0), None, "router_hc_scale".to_string())
        );

        // Attention tensors still classified as unknown (not in ffn path)
        assert_eq!(
            classify_tensor("layers.0.attn.wq_a.weight"),
            (Some(0), None, "unknown".to_string())
        );
    }

    #[test]
    fn test_explicit_expert_layout_parser() -> std::io::Result<()> {
        let temp_dir = std::env::temp_dir().join("objeta_test_explicit");
        std::fs::create_dir_all(&temp_dir)?;

        // Write config.json
        let config_data = serde_json::json!({
            "model_type": "deepseek_v4_flash",
            "num_hidden_layers": 2,
            "hidden_size": 1024,
            "moe_intermediate_size": 512,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "vocab_size": 10000,
            "torch_dtype": "bfloat16"
        });
        std::fs::write(temp_dir.join("config.json"), serde_json::to_string_pretty(&config_data).unwrap())?;

        // Write mock safetensors
        let mut tensors = HashMap::new();
        tensors.insert("model.layers.0.mlp.experts.0.gate_proj.weight".to_string(), ("BF16".to_string(), vec![512, 1024], (0, 1024)));
        tensors.insert("model.layers.0.mlp.experts.0.up_proj.weight".to_string(), ("BF16".to_string(), vec![512, 1024], (1024, 2048)));
        tensors.insert("model.layers.0.mlp.experts.0.down_proj.weight".to_string(), ("BF16".to_string(), vec![1024, 512], (2048, 3072)));
        tensors.insert("model.layers.0.mlp.gate.weight".to_string(), ("BF16".to_string(), vec![4, 1024], (3072, 4096)));
        tensors.insert("model.layers.1.mlp.experts.1.gate_proj.weight".to_string(), ("BF16".to_string(), vec![512, 1024], (0, 1024)));
        tensors.insert("model.layers.1.mlp.shared_experts.gate_proj.weight".to_string(), ("BF16".to_string(), vec![512, 1024], (1024, 2048)));
        tensors.insert("model.layers.1.self_attn.q_proj.weight".to_string(), ("BF16".to_string(), vec![1024, 1024], (2048, 4144)));

        write_mock_safetensors(&temp_dir.join("model.safetensors"), &tensors)?;

        // Run parser
        let output_dir = temp_dir.join("output");
        parse_deepseek_v4_flash(&temp_dir, &output_dir).unwrap();

        // Verify output files exist
        assert!(output_dir.join("deepseek_v4_flash_layout.json").exists());
        assert!(output_dir.join("deepseek_v4_flash_tensor_index.json").exists());
        assert!(output_dir.join("deepseek_v4_flash_expert_layout.json").exists());
        assert!(output_dir.join("deepseek_v4_flash_router_layout.json").exists());
        assert!(output_dir.join("deepseek_v4_flash_inventory_summary.json").exists());

        // Verify layout
        let layout_str = std::fs::read_to_string(output_dir.join("deepseek_v4_flash_layout.json"))?;
        let layout: DeepseekLayout = serde_json::from_str(&layout_str).unwrap();
        assert_eq!(layout.model_name, "deepseek_v4_flash");
        assert_eq!(layout.num_layers, 2);
        assert_eq!(layout.hidden_size, 1024);
        assert_eq!(layout.num_experts, 4);
        assert_eq!(layout.top_k, 2);
        assert_eq!(layout.vocab_size, 10000);
        assert_eq!(layout.dtype, "bfloat16");
        assert_eq!(layout.tensor_count, 7);

        // Verify expert layout kind
        let expert_str = std::fs::read_to_string(output_dir.join("deepseek_v4_flash_expert_layout.json"))?;
        let expert_layout: ExpertLayout = serde_json::from_str(&expert_str).unwrap();
        assert_eq!(expert_layout.layout_kind, "explicit_experts");
        assert!(expert_layout.tensors.iter().any(|t| t.expert_id == Some(Some(0))));

        // Clean up
        std::fs::remove_dir_all(&temp_dir)?;

        Ok(())
    }

    #[test]
    fn test_packed_expert_layout_parser() -> std::io::Result<()> {
        let temp_dir = std::env::temp_dir().join("objeta_test_packed");
        std::fs::create_dir_all(&temp_dir)?;

        // Write config.json
        let config_data = serde_json::json!({
            "model_type": "deepseek_v4_flash_packed",
            "num_hidden_layers": 1,
            "hidden_size": 512,
            "intermediate_size": 256,
            "num_experts": 8,
            "top_k": 1,
            "vocab_size": 5000,
            "torch_dtype": "float16"
        });
        std::fs::write(temp_dir.join("config.json"), serde_json::to_string_pretty(&config_data).unwrap())?;

        // Write mock safetensors with 3D packed expert tensors
        let mut tensors = HashMap::new();
        tensors.insert("model.layers.0.mlp.experts.gate_proj.weight".to_string(), ("F16".to_string(), vec![8, 256, 512], (0, 4096)));
        tensors.insert("model.layers.0.mlp.experts.up_proj.weight".to_string(), ("F16".to_string(), vec![8, 256, 512], (4096, 8192)));
        tensors.insert("model.layers.0.mlp.gate.weight".to_string(), ("F16".to_string(), vec![8, 512], (8192, 9216)));

        write_mock_safetensors(&temp_dir.join("model.safetensors"), &tensors)?;

        // Run parser
        let output_dir = temp_dir.join("output");
        parse_deepseek_v4_flash(&temp_dir, &output_dir).unwrap();

        // Verify layout
        let layout_str = std::fs::read_to_string(output_dir.join("deepseek_v4_flash_layout.json"))?;
        let layout: DeepseekLayout = serde_json::from_str(&layout_str).unwrap();
        assert_eq!(layout.model_name, "deepseek_v4_flash_packed");
        assert_eq!(layout.num_experts, 8);

        // Verify expert layout kind
        let expert_str = std::fs::read_to_string(output_dir.join("deepseek_v4_flash_expert_layout.json"))?;
        let expert_layout: ExpertLayout = serde_json::from_str(&expert_str).unwrap();
        assert_eq!(expert_layout.layout_kind, "packed_experts");
        
        for t in &expert_layout.tensors {
            assert!(t.expert_id.is_none());
        }

        // Clean up
        std::fs::remove_dir_all(&temp_dir)?;

        Ok(())
    }

    #[test]
    fn test_deepseek_ffn_fp4_fixture() -> std::io::Result<()> {
        let temp_dir = std::env::temp_dir().join("objeta_test_ds_ffn");
        std::fs::create_dir_all(&temp_dir)?;

        // Write config.json with DeepSeek V4 Flash parameters
        let config_data = serde_json::json!({
            "model_type": "deepseek_v4",
            "num_hidden_layers": 1,
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "vocab_size": 129280,
            "torch_dtype": "bfloat16",
            "expert_dtype": "fp4",
            "num_hash_layers": 3
        });
        std::fs::write(temp_dir.join("config.json"), serde_json::to_string_pretty(&config_data).unwrap())?;

        // Write mock safetensors with DeepSeek ffn naming
        let mut tensors = HashMap::new();
        // Expert 0 weights (I8 = fp4 packed, 2 values per byte)
        tensors.insert("layers.0.ffn.experts.0.w1.weight".to_string(), ("I8".to_string(), vec![2048, 2048], (0, 4194304)));
        tensors.insert("layers.0.ffn.experts.0.w1.scale".to_string(), ("F8_E8M0".to_string(), vec![2048, 128], (4194304, 4456448)));
        tensors.insert("layers.0.ffn.experts.0.w2.weight".to_string(), ("I8".to_string(), vec![4096, 1024], (4456448, 8650752)));
        tensors.insert("layers.0.ffn.experts.0.w2.scale".to_string(), ("F8_E8M0".to_string(), vec![4096, 64], (8650752, 8912896)));
        tensors.insert("layers.0.ffn.experts.0.w3.weight".to_string(), ("I8".to_string(), vec![2048, 2048], (8912896, 13107200)));
        tensors.insert("layers.0.ffn.experts.0.w3.scale".to_string(), ("F8_E8M0".to_string(), vec![2048, 128], (13107200, 13369344)));
        // Router and routing aux tensors
        tensors.insert("layers.0.ffn.gate.weight".to_string(), ("BF16".to_string(), vec![256, 4096], (13369344, 15466496)));
        tensors.insert("layers.0.ffn.gate.tid2eid".to_string(), ("I64".to_string(), vec![129280, 6], (15466496, 21671936)));
        tensors.insert("layers.0.hc_ffn_base".to_string(), ("F32".to_string(), vec![24], (21671936, 21672032)));
        tensors.insert("layers.0.hc_ffn_fn".to_string(), ("F32".to_string(), vec![24, 16384], (21672032, 23244992)));
        tensors.insert("layers.0.hc_ffn_scale".to_string(), ("F32".to_string(), vec![3], (23244992, 23245004)));
        // Attention tensor (should be unknown/other)
        tensors.insert("layers.0.attn.wq_a.weight".to_string(), ("BF16".to_string(), vec![4096, 1024], (23245004, 31637516)));

        write_mock_safetensors(&temp_dir.join("model.safetensors"), &tensors)?;

        // Run parser
        let output_dir = temp_dir.join("output");
        parse_deepseek_v4_flash(&temp_dir, &output_dir).unwrap();

        // Verify expert layout
        let expert_str = std::fs::read_to_string(output_dir.join("deepseek_v4_flash_expert_layout.json"))?;
        let expert_layout: ExpertLayout = serde_json::from_str(&expert_str).unwrap();
        assert_eq!(expert_layout.layout_kind, "explicit_experts");
        assert!(expert_layout.fp4_expert_storage_detected);

        // 3 weights + 3 scales = 6 expert tensors for expert 0
        assert_eq!(expert_layout.tensors.len(), 6);

        // Check gate weight has fp4 metadata
        let gate_weight = expert_layout.tensors.iter().find(|t| t.name.contains("w1.weight")).unwrap();
        assert_eq!(gate_weight.tensor_kind, "gate");
        assert_eq!(gate_weight.dtype, "I8");
        assert_eq!(gate_weight.storage_dtype, Some("I8".to_string()));
        assert_eq!(gate_weight.logical_dtype, Some("FP4".to_string()));
        assert!(gate_weight.scale_tensor_name.as_ref().unwrap().contains("w1.scale"));
        assert_eq!(gate_weight.scale_dtype, Some("F8_E8M0".to_string()));
        assert_eq!(gate_weight.packed_values_per_byte, Some(2));
        // block_size = logical fp4 elements per scale: 2048*2048*2 / (2048*128) = 32
        assert_eq!(gate_weight.block_size, Some(32));
        // logical_shape should double the last dimension
        assert_eq!(gate_weight.logical_shape, Some(vec![2048, 4096]));

        // Check down weight
        let down_weight = expert_layout.tensors.iter().find(|t| t.name.contains("w2.weight")).unwrap();
        assert_eq!(down_weight.tensor_kind, "down");
        assert_eq!(down_weight.storage_dtype, Some("I8".to_string()));

        // Check up weight
        let up_weight = expert_layout.tensors.iter().find(|t| t.name.contains("w3.weight")).unwrap();
        assert_eq!(up_weight.tensor_kind, "up");

        // Check scale tensors are present
        let gate_scale = expert_layout.tensors.iter().find(|t| t.name.contains("w1.scale")).unwrap();
        assert_eq!(gate_scale.tensor_kind, "gate_scale");
        assert_eq!(gate_scale.dtype, "F8_E8M0");
        assert!(gate_scale.storage_dtype.is_none()); // scale is not itself quantized

        let down_scale = expert_layout.tensors.iter().find(|t| t.name.contains("w2.scale")).unwrap();
        assert_eq!(down_scale.tensor_kind, "down_scale");

        let up_scale = expert_layout.tensors.iter().find(|t| t.name.contains("w3.scale")).unwrap();
        assert_eq!(up_scale.tensor_kind, "up_scale");

        // Verify router layout
        let router_str = std::fs::read_to_string(output_dir.join("deepseek_v4_flash_router_layout.json"))?;
        let router_layout: RouterLayout = serde_json::from_str(&router_str).unwrap();
        assert_eq!(router_layout.routers.len(), 5); // gate.weight + tid2eid + hc_ffn_base + hc_ffn_fn + hc_ffn_scale

        let router_names: Vec<&str> = router_layout.routers.iter().map(|r| r.name.as_str()).collect();
        assert!(router_names.contains(&"layers.0.ffn.gate.weight"));
        assert!(router_names.contains(&"layers.0.ffn.gate.tid2eid"));
        assert!(router_names.contains(&"layers.0.hc_ffn_base"));
        assert!(router_names.contains(&"layers.0.hc_ffn_fn"));
        assert!(router_names.contains(&"layers.0.hc_ffn_scale"));

        // Verify layout
        let layout_str = std::fs::read_to_string(output_dir.join("deepseek_v4_flash_layout.json"))?;
        let layout: DeepseekLayout = serde_json::from_str(&layout_str).unwrap();
        assert_eq!(layout.num_layers, 1);
        assert_eq!(layout.hidden_size, 4096);
        assert_eq!(layout.intermediate_size, 2048);
        assert_eq!(layout.num_experts, 256); // from config

        // Verify inventory summary has scale bytes tracked
        let inventory_str = std::fs::read_to_string(output_dir.join("deepseek_v4_flash_inventory_summary.json"))?;
        let inventory: InventorySummary = serde_json::from_str(&inventory_str).unwrap();
        assert!(inventory.bytes_by_tensor_kind.gate > 0);
        assert!(inventory.bytes_by_tensor_kind.gate_scale > 0);
        assert!(inventory.bytes_by_tensor_kind.down_scale > 0);
        assert!(inventory.bytes_by_tensor_kind.up_scale > 0);

        // Clean up
        std::fs::remove_dir_all(&temp_dir)?;

        Ok(())
    }

    #[test]
    fn test_fp4_decode_synthetic_fixture() {
        // 2 rows, 4 logical columns, block_size=4, packed in I8 along last dim
        // physical shape: [2, 2] (4 bytes), logical shape: [2, 4]
        // scale shape: [2, 1] (2 bytes in F8_E8M0)
        let rows = 2;
        let logical_cols = 4;
        let phys_cols = 2;
        let block_size = 4;

        // Row 0 fp4 values: [0.5, 1.0, 1.5, 2.0]
        //   byte 0: low=0b0001(0.5), high=0b0010(1.0) → 0x21
        //   byte 1: low=0b0011(1.5), high=0b0100(2.0) → 0x43
        // Row 1 fp4 values: [3.0, 4.0, 6.0, -0.5]
        //   byte 0: low=0b0101(3.0), high=0b0110(4.0) → 0x65
        //   byte 1: low=0b0111(6.0), high=0b1001(-0.5) → 0x97
        let weight_i8 = vec![0x21u8, 0x43, 0x65, 0x97];

        // Scale (F8_E8M0): 2^(raw-127)
        // scale[0,0] covers cols 0-3 (block_size=4): 2^(130-127)=2^3=8 → raw=130=0x82
        // scale[1,0] covers cols 0-3: 2^(131-127)=2^4=16 → raw=131=0x83
        let scale_f8e8m0 = vec![130u8, 131];

        let result = decode_deepseek_fp4_to_f32(
            &weight_i8,
            &scale_f8e8m0,
            &[rows, phys_cols],
            &[rows, logical_cols],
            block_size,
        );

        // Expected:
        // Row 0: [0.5*8=4.0, 1.0*8=8.0, 1.5*8=12.0, 2.0*8=16.0]
        // Row 1: [3.0*16=48.0, 4.0*16=64.0, 6.0*16=96.0, -0.5*16=-8.0]
        let expected = vec![4.0f32, 8.0, 12.0, 16.0, 48.0, 64.0, 96.0, -8.0];

        assert_eq!(result.len(), expected.len());
        for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "index {}: got {}, expected {}", i, got, exp
            );
        }
    }

    #[test]
    fn test_f8e8m0_to_f32() {
        // 2^(raw - 127)
        assert!((f8e8m0_to_f32(127) - 1.0).abs() < 1e-6);    // 2^0 = 1
        assert!((f8e8m0_to_f32(128) - 2.0).abs() < 1e-6);    // 2^1 = 2
        assert!((f8e8m0_to_f32(130) - 8.0).abs() < 1e-6);    // 2^3 = 8
        assert!((f8e8m0_to_f32(126) - 0.5).abs() < 1e-6);    // 2^-1 = 0.5
        assert!((f8e8m0_to_f32(124) - 0.125).abs() < 1e-6);  // 2^-3 = 0.125
        assert!((f8e8m0_to_f32(0) - 0.0).abs() < 1e-12);     // 2^-127 ≈ tiny, but 0 maps to subnormal
        // 2^-127 in fp32 is ~5.88e-39
        let tiny = f8e8m0_to_f32(0);
        assert!(tiny > 0.0 && tiny < 1e-38);
    }
}
