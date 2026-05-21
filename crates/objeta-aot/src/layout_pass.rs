use std::collections::BTreeMap;
use std::error::Error;

use crate::types::*;
use crate::util::extract_number_after;

/// Build expert layout from model config and tensor index.
pub fn run(
    model_name: &str,
    config: &ModelConfig,
    index: &SafeTensorsIndex,
) -> Result<ExpertLayout, Box<dyn Error>> {
    let mut experts: BTreeMap<ExpertKey, ExpertEntry> = BTreeMap::new();
    let mut packed_expert_layers: BTreeMap<u32, PackedExpertLayerEntry> = BTreeMap::new();
    let mut shared: BTreeMap<u32, SharedExpertEntry> = BTreeMap::new();
    let mut routers: BTreeMap<u32, RouterEntry> = BTreeMap::new();
    let mut unknown_tensors = Vec::new();
    let mut warnings = Vec::new();
    let mut saw_packed = false;

    for (tensor_name, source_file) in &index.weight_map {
        let parsed = parse_qwen_tensor_name(tensor_name, source_file);
        let tref = TensorRef {
            tensor_kind: parsed.tensor_kind,
            tensor_name: parsed.tensor_name.clone(),
            source_file: parsed.source_file.clone(),
            shape: None,
            dtype: None,
            byte_offset: None,
            byte_len: None,
        };

        match parsed.tensor_kind {
            ExpertTensorKind::Router => {
                if let Some(layer) = parsed.layer_idx {
                    routers.insert(layer, RouterEntry { layer, tensor: tref });
                } else {
                    unknown_tensors.push(UnknownTensorEntry {
                        tensor_name: parsed.tensor_name,
                        source_file: parsed.source_file,
                    });
                }
            }
            ExpertTensorKind::Gate
            | ExpertTensorKind::Up
            | ExpertTensorKind::GateUp
            | ExpertTensorKind::Down => {
                if parsed.is_shared {
                    if let Some(layer) = parsed.layer_idx {
                        let entry = shared.entry(layer).or_insert_with(|| SharedExpertEntry {
                            layer,
                            gate: None,
                            up: None,
                            gate_up: None,
                            down: None,
                            shared_gate: None,
                        });
                        assign_shared_tensor(entry, tref);
                    } else {
                        warnings.push(format!(
                            "shared tensor missing layer index: {}",
                            parsed.tensor_name
                        ));
                    }
                } else if let (Some(layer), Some(expert)) = (parsed.layer_idx, parsed.expert_id) {
                    let key = ExpertKey { layer, expert };
                    let entry = experts.entry(key).or_insert_with(|| ExpertEntry {
                        layer,
                        expert,
                        gate: None,
                        up: None,
                        gate_up: None,
                        down: None,
                        source_files: Vec::new(),
                        complete: false,
                    });
                    if !entry.source_files.iter().any(|s| s == &tref.source_file) {
                        entry.source_files.push(tref.source_file.clone());
                    }
                    assign_routed_tensor(entry, tref);
                } else {
                    warnings.push(format!(
                        "expert tensor missing layer or expert index: {}",
                        parsed.tensor_name
                    ));
                }
            }
            ExpertTensorKind::PackedGateUp | ExpertTensorKind::PackedDown => {
                saw_packed = true;
                if let Some(layer) = parsed.layer_idx {
                    let entry = packed_expert_layers
                        .entry(layer)
                        .or_insert_with(|| PackedExpertLayerEntry {
                            layer,
                            num_experts_per_layer: config.effective_num_experts().unwrap_or(0),
                            gate_up: None,
                            down: None,
                            source_files: Vec::new(),
                            complete: false,
                        });
                    if !entry.source_files.iter().any(|s| s == &tref.source_file) {
                        entry.source_files.push(tref.source_file.clone());
                    }
                    match tref.tensor_kind {
                        ExpertTensorKind::PackedGateUp => entry.gate_up = Some(tref),
                        ExpertTensorKind::PackedDown => entry.down = Some(tref),
                        _ => {}
                    }
                } else {
                    warnings.push(format!(
                        "packed expert tensor missing layer index: {}",
                        parsed.tensor_name
                    ));
                }
            }
            ExpertTensorKind::Shared => {
                if parsed.is_shared {
                    if let Some(layer) = parsed.layer_idx {
                        let entry = shared.entry(layer).or_insert_with(|| SharedExpertEntry {
                            layer,
                            gate: None,
                            up: None,
                            gate_up: None,
                            down: None,
                            shared_gate: None,
                        });
                        entry.shared_gate = Some(tref);
                    } else {
                        warnings.push(format!(
                            "shared gate tensor missing layer index: {}",
                            parsed.tensor_name
                        ));
                    }
                } else {
                    unknown_tensors.push(UnknownTensorEntry {
                        tensor_name: parsed.tensor_name,
                        source_file: parsed.source_file,
                    });
                }
            }
            ExpertTensorKind::Unknown => {
                unknown_tensors.push(UnknownTensorEntry {
                    tensor_name: parsed.tensor_name,
                    source_file: parsed.source_file,
                });
            }
        }
    }

    let mut expert_entries: Vec<_> = experts.into_values().collect();
    for entry in &mut expert_entries {
        let complete = if entry.gate_up.is_some() {
            entry.down.is_some()
        } else {
            entry.gate.is_some() && entry.up.is_some() && entry.down.is_some()
        };
        entry.complete = complete;
        if !complete {
            warnings.push(format!(
                "incomplete routed expert layer={} expert={} gate={} up={} gate_up={} down={}",
                entry.layer,
                entry.expert,
                entry.gate.is_some(),
                entry.up.is_some(),
                entry.gate_up.is_some(),
                entry.down.is_some()
            ));
        }
        entry.source_files.sort();
    }

    let mut packed_entries: Vec<_> = packed_expert_layers.into_values().collect();
    for entry in &mut packed_entries {
        entry.complete = entry.gate_up.is_some() && entry.down.is_some();
        if !entry.complete {
            warnings.push(format!(
                "incomplete packed expert layer={} gate_up={} down={}",
                entry.layer,
                entry.gate_up.is_some(),
                entry.down.is_some()
            ));
        }
        entry.source_files.sort();
    }

    for (layer, entry) in &shared {
        let complete = if entry.gate_up.is_some() {
            entry.down.is_some()
        } else {
            entry.gate.is_some() && entry.up.is_some() && entry.down.is_some()
        };
        if !complete {
            warnings.push(format!(
                "incomplete shared expert layer={} gate={} up={} gate_up={} down={}",
                layer,
                entry.gate.is_some(),
                entry.up.is_some(),
                entry.gate_up.is_some(),
                entry.down.is_some()
            ));
        }
    }

    let num_layers = config.effective_num_hidden_layers().unwrap_or(0);
    let num_experts = config.effective_num_experts().unwrap_or(0);
    let logical_routed_expert_count = if saw_packed {
        u64::from(num_layers) * u64::from(num_experts)
    } else if !expert_entries.is_empty() && num_layers > 0 && num_experts > 0 {
        u64::from(num_layers) * u64::from(num_experts)
    } else {
        expert_entries.len() as u64
    };

    Ok(ExpertLayout {
        schema_version: 1,
        model: model_name.to_string(),
        model_type: config.model_type.clone(),
        architectures: config.architectures.clone().unwrap_or_default(),
        num_layers,
        num_experts,
        vocab_size: config.effective_vocab_size(),
        layout_kind: if saw_packed {
            ExpertLayoutKind::PackedExperts
        } else {
            ExpertLayoutKind::PerExpert
        },
        logical_routed_expert_count,
        experts: expert_entries,
        packed_expert_layers: packed_entries,
        shared_experts: shared.into_values().collect(),
        routers: routers.into_values().collect(),
        unknown_tensors,
        warnings,
    })
}

pub fn parse_qwen_tensor_name(tensor_name: &str, source_file: &str) -> ParsedTensorName {
    let layer_idx = extract_number_after(tensor_name, "layers.");
    let expert_id = extract_number_after(tensor_name, "experts.");
    let is_shared = tensor_name.contains(".shared_expert.");
    let is_packed_experts = tensor_name.contains(".mlp.experts.gate_up_proj")
        || tensor_name.contains(".mlp.experts.down_proj");
    let tensor_kind = if tensor_name.contains(".shared_expert_gate.weight") {
        ExpertTensorKind::Shared
    } else if tensor_name.contains(".mlp.gate.weight") && !tensor_name.contains(".experts.") {
        ExpertTensorKind::Router
    } else if tensor_name.contains(".mlp.experts.gate_up_proj") && expert_id.is_none() {
        ExpertTensorKind::PackedGateUp
    } else if tensor_name.contains(".mlp.experts.down_proj") && expert_id.is_none() {
        ExpertTensorKind::PackedDown
    } else if tensor_name.contains("gate_up_proj") {
        ExpertTensorKind::GateUp
    } else if tensor_name.contains("gate_proj") {
        ExpertTensorKind::Gate
    } else if tensor_name.contains("up_proj") {
        ExpertTensorKind::Up
    } else if tensor_name.contains("down_proj") {
        ExpertTensorKind::Down
    } else {
        ExpertTensorKind::Unknown
    };

    ParsedTensorName {
        layer_idx,
        expert_id,
        tensor_kind,
        is_shared,
        is_packed_experts,
        tensor_name: tensor_name.to_string(),
        source_file: source_file.to_string(),
    }
}

fn assign_routed_tensor(entry: &mut ExpertEntry, tref: TensorRef) {
    match tref.tensor_kind {
        ExpertTensorKind::Gate => entry.gate = Some(tref),
        ExpertTensorKind::Up => entry.up = Some(tref),
        ExpertTensorKind::GateUp => entry.gate_up = Some(tref),
        ExpertTensorKind::Down => entry.down = Some(tref),
        _ => {}
    }
}

fn assign_shared_tensor(entry: &mut SharedExpertEntry, tref: TensorRef) {
    match tref.tensor_kind {
        ExpertTensorKind::Gate => entry.gate = Some(tref),
        ExpertTensorKind::Up => entry.up = Some(tref),
        ExpertTensorKind::GateUp => entry.gate_up = Some(tref),
        ExpertTensorKind::Down => entry.down = Some(tref),
        ExpertTensorKind::Shared => entry.shared_gate = Some(tref),
        ExpertTensorKind::Router
        | ExpertTensorKind::Unknown
        | ExpertTensorKind::PackedGateUp
        | ExpertTensorKind::PackedDown => {}
    }
}
