use std::collections::BTreeSet;
use crate::qwen36_forward::Qwen36Runner;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct LayerTrace {
    pub layer: usize,
    pub hidden_norm: f32,
    pub expert_ids: Vec<usize>,
    pub expert_weights: Vec<f32>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct StepTrace {
    pub step: usize,
    pub token_id: usize,
    pub entropy: f32,
    pub logits_topk_ids: Vec<i32>,
    pub logits_topk_values: Vec<f32>,
    pub layers: Vec<LayerTrace>,
}

#[derive(Clone, Default)]
pub struct MoELayerStats {
    pub calls: u64,
    pub shared_calls: u64,
    pub total_executed_experts: u64,
    pub total_executed_mass: f64,
    pub total_dropped_mass: f64,
    pub total_executed_mass_pre_renorm: f64,
    pub total_dropped_mass_pre_renorm: f64,
    pub total_routing_mass_sum_after_renorm: f64,
    pub total_load_count: u64,
    pub total_warm_hit_count: u64,
    pub total_cold_hit_count: u64,
    pub total_compute_sec: f64,
    pub total_bytes_read: u64,
    pub total_logical_bytes_requested: u64,
    pub total_actual_bytes_loaded: u64,
    pub total_resident_cache_bytes_reused: u64,
    pub total_resident_cache_hit_count: u64,
    pub total_resident_cache_miss_count: u64,
    pub total_direct_cold_load_count: u64,
    pub total_dequantized_scratch_bytes: u64,
    pub total_router_sec: f64,
    pub total_select_sec: f64,
    pub total_load_sec: f64,
    pub total_dequant_sec: f64,
    pub total_gemv_sec: f64,
    pub total_accumulate_sec: f64,
    pub total_shared_sec: f64,
    pub total_fused_gate_up_sec: f64,
    pub total_fused_swiglu_sec: f64,
    pub total_fused_down_accum_sec: f64,
    pub total_fused_alloc_sec: f64,
    pub total_fused_stats_sec: f64,
    pub total_router_wall_sec: f64,
    pub total_call_moe_wall_sec: f64,
    pub total_candidate_build_wall_sec: f64,
    pub total_policy_select_wall_sec: f64,
    pub total_cache_lookup_wall_sec: f64,
    pub total_cache_key_build_wall_sec: f64,
    pub total_cache_hit_lookup_wall_sec: f64,
    pub total_cache_miss_load_wall_sec: f64,
    pub total_cache_eviction_wall_sec: f64,
    pub total_cache_insert_wall_sec: f64,
    pub total_cache_page_clone_wall_sec: f64,
    pub total_routed_exec_wall_sec: f64,
    pub total_stats_wall_sec: f64,
    pub total_select_wall_sec: f64,
    pub total_load_wall_sec: f64,
    pub total_exec_wall_sec: f64,
    pub total_accumulate_wall_sec: f64,
    pub unique_expert_ids: BTreeSet<usize>,
    pub last_expert_ids: Vec<usize>,
    pub last_router_top8_ids: Vec<usize>,
    pub last_router_top8_weights: Vec<f32>,
    pub last_candidate_ids: Vec<usize>,
    pub last_candidate_weights: Vec<f32>,
    pub last_selected_ids: Vec<usize>,
    pub last_selected_weights: Vec<f32>,
    pub last_dispatch_ids: Vec<usize>,
    pub last_dispatch_weights: Vec<f32>,
    pub last_selected_count: usize,
    pub last_selected_renormalized: bool,
    pub last_attention_type: String,
}

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug)]
pub struct MoEIoEvent {
    pub step: usize,
    pub token_id: usize,
    pub layer_id: usize,
    pub selected_experts: Vec<usize>,
    pub selected_weights: Vec<f32>,
    pub logical_bytes: u64,
    pub actual_loaded_bytes: u64,
    pub resident_hits: u64,
    pub cold_loads: u64,
    pub resident_bytes: u64,
    pub pinned_resident_bytes: u64,
    pub token_window_peak_resident_bytes: u64,
    pub eviction_count: u64,
    pub eviction_count_during_token: u64,
    pub eviction_count_at_token_end: u64,
    pub dequantized_scratch_bytes: u64,

    // Explicit compliance byte telemetry fields
    pub logical_expert_bytes_requested: u64,
    pub actual_expert_bytes_loaded: u64,
    pub resident_cache_bytes_reused: u64,
    pub resident_cache_resident_bytes: u64,
    pub routing_mass_kept: f32,
    pub routing_mass_dropped: f32,
    pub routing_mass_kept_pre_renorm: f32,
    pub routing_mass_dropped_pre_renorm: f32,
    pub routing_mass_sum_after_renorm: f32,
}

#[derive(Clone, Default)]
pub struct ForwardLayerStats {
    pub calls: u64,
    pub total_layer_wall_sec: f64,
    pub total_deltanet_wall_sec: f64,
    pub total_gqa_wall_sec: f64,
    pub total_shared_wall_sec: f64,
    pub total_moe_wall_sec: f64,
}

impl Qwen36Runner {
    pub fn get_moe_stats_json(&self) -> String {
        let mut layers = Vec::with_capacity(self.moe_stats.len());
        for (layer_idx, s) in self.moe_stats.iter().enumerate() {
            layers.push(serde_json::json!({
                "layer": layer_idx,
                "calls": s.calls,
                "avg_executed_experts": if s.calls > 0 { s.total_executed_experts as f64 / s.calls as f64 } else { 0.0 },
                "avg_selected_experts": if s.calls > 0 { s.total_executed_experts as f64 / s.calls as f64 } else { 0.0 },
                "avg_executed_mass": if s.calls > 0 { s.total_executed_mass / s.calls as f64 } else { 0.0 },
                "avg_routing_mass_kept": if s.calls > 0 { s.total_executed_mass / s.calls as f64 } else { 0.0 },
                "avg_dropped_mass": if s.calls > 0 { s.total_dropped_mass / s.calls as f64 } else { 0.0 },
                "avg_routing_mass_dropped": if s.calls > 0 { s.total_dropped_mass / s.calls as f64 } else { 0.0 },
                "avg_routing_mass_kept_pre_renorm": if s.calls > 0 { s.total_executed_mass_pre_renorm / s.calls as f64 } else { 0.0 },
                "avg_routing_mass_dropped_pre_renorm": if s.calls > 0 { s.total_dropped_mass_pre_renorm / s.calls as f64 } else { 0.0 },
                "avg_routing_mass_sum_after_renorm": if s.calls > 0 { s.total_routing_mass_sum_after_renorm / s.calls as f64 } else { 0.0 },
                "avg_load_count": if s.calls > 0 { s.total_load_count as f64 / s.calls as f64 } else { 0.0 },
                "avg_warm_hit_count": if s.calls > 0 { s.total_warm_hit_count as f64 / s.calls as f64 } else { 0.0 },
                "avg_cold_hit_count": if s.calls > 0 { s.total_cold_hit_count as f64 / s.calls as f64 } else { 0.0 },
                "avg_compute_ms": if s.calls > 0 { (s.total_compute_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_bytes_read": if s.calls > 0 { s.total_bytes_read as f64 / s.calls as f64 } else { 0.0 },
                "avg_logical_expert_bytes_requested": if s.calls > 0 { s.total_logical_bytes_requested as f64 / s.calls as f64 } else { 0.0 },
                "avg_actual_expert_bytes_loaded": if s.calls > 0 { s.total_actual_bytes_loaded as f64 / s.calls as f64 } else { 0.0 },
                "avg_resident_cache_bytes_reused": if s.calls > 0 { s.total_resident_cache_bytes_reused as f64 / s.calls as f64 } else { 0.0 },
                "avg_dequantized_scratch_bytes": if s.calls > 0 { s.total_dequantized_scratch_bytes as f64 / s.calls as f64 } else { 0.0 },
                "logical_expert_bytes_requested": s.total_logical_bytes_requested,
                "actual_expert_bytes_loaded": s.total_actual_bytes_loaded,
                "resident_cache_bytes_reused": s.total_resident_cache_bytes_reused,
                "resident_cache_resident_bytes": self.expert_residency_manager.metadata.iter()
                    .filter(|(k, _)| k.layer_id == layer_idx)
                    .map(|(_, meta)| meta.bytes as u64)
                    .sum::<u64>(),
                "dequantized_scratch_bytes": s.total_dequantized_scratch_bytes,
                "avg_resident_cache_hit_count": if s.calls > 0 { s.total_resident_cache_hit_count as f64 / s.calls as f64 } else { 0.0 },
                "avg_resident_cache_miss_count": if s.calls > 0 { s.total_resident_cache_miss_count as f64 / s.calls as f64 } else { 0.0 },
                "avg_direct_cold_load_count": if s.calls > 0 { s.total_direct_cold_load_count as f64 / s.calls as f64 } else { 0.0 },
                "avg_router_ms": if s.calls > 0 { (s.total_router_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_expert_select_ms": if s.calls > 0 { (s.total_select_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_expert_load_ms": if s.calls > 0 { (s.total_load_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_dequant_ms": if s.calls > 0 { (s.total_dequant_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_gemv_ms": if s.calls > 0 { (s.total_gemv_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_accumulate_ms": if s.calls > 0 { (s.total_accumulate_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_fused_gate_up_ms": if s.calls > 0 { (s.total_fused_gate_up_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_fused_swiglu_ms": if s.calls > 0 { (s.total_fused_swiglu_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_fused_down_accum_ms": if s.calls > 0 { (s.total_fused_down_accum_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_fused_alloc_ms": if s.calls > 0 { (s.total_fused_alloc_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_fused_stats_ms": if s.calls > 0 { (s.total_fused_stats_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_shared_ms": if s.shared_calls > 0 { (s.total_shared_sec * 1000.0) / s.shared_calls as f64 } else { 0.0 },
                "avg_router_wall_ms": if s.calls > 0 { (s.total_router_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_call_moe_wall_ms": if s.calls > 0 { (s.total_call_moe_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_candidate_build_wall_ms": if s.calls > 0 { (s.total_candidate_build_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_policy_select_wall_ms": if s.calls > 0 { (s.total_policy_select_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_cache_lookup_wall_ms": if s.calls > 0 { (s.total_cache_lookup_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_cache_key_build_wall_ms": if s.calls > 0 { (s.total_cache_key_build_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_cache_hit_lookup_wall_ms": if s.calls > 0 { (s.total_cache_hit_lookup_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_cache_miss_load_wall_ms": if s.calls > 0 { (s.total_cache_miss_load_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_cache_eviction_wall_ms": if s.calls > 0 { (s.total_cache_eviction_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_cache_insert_wall_ms": if s.calls > 0 { (s.total_cache_insert_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_cache_page_clone_wall_ms": if s.calls > 0 { (s.total_cache_page_clone_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_select_wall_ms": if s.calls > 0 { (s.total_select_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_load_wall_ms": if s.calls > 0 { (s.total_load_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_routed_exec_wall_ms": if s.calls > 0 { (s.total_routed_exec_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_stats_wall_ms": if s.calls > 0 { (s.total_stats_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_exec_wall_ms": if s.calls > 0 { (s.total_exec_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "avg_accumulate_wall_ms": if s.calls > 0 { (s.total_accumulate_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                "attention_type": s.last_attention_type,
                "unique_expert_ids": s.unique_expert_ids.iter().copied().collect::<Vec<_>>(),
                "last_expert_ids": s.last_expert_ids.clone(),
                "last_router_top8_ids": s.last_router_top8_ids.clone(),
                "last_router_top8_weights": s.last_router_top8_weights.clone(),
                "last_candidate_ids": s.last_candidate_ids.clone(),
                "last_candidate_weights": s.last_candidate_weights.clone(),
                "last_selected_ids": s.last_selected_ids.clone(),
                "last_selected_weights": s.last_selected_weights.clone(),
                "last_dispatch_ids": s.last_dispatch_ids.clone(),
                "last_dispatch_weights": s.last_dispatch_weights.clone(),
                "last_selected_count": s.last_selected_count,
                "selected_expert_count": s.last_selected_count,
                "last_selected_renormalized": s.last_selected_renormalized,
                "routing_mass_kept": if s.calls > 0 { s.total_executed_mass / s.calls as f64 } else { 0.0 },
                "routing_mass_kept_pre_renorm": if s.calls > 0 { s.total_executed_mass_pre_renorm / s.calls as f64 } else { 0.0 },
            }));
        }
        let total_calls: u64 = self.moe_stats.iter().map(|s| s.calls).sum();
        let total_exec: u64 = self.moe_stats.iter().map(|s| s.total_executed_experts).sum();
        let total_mass: f64 = self.moe_stats.iter().map(|s| s.total_executed_mass).sum();
        let total_drop: f64 = self.moe_stats.iter().map(|s| s.total_dropped_mass).sum();
        let total_loads: u64 = self.moe_stats.iter().map(|s| s.total_load_count).sum();
        let total_warm_hits: u64 = self.moe_stats.iter().map(|s| s.total_warm_hit_count).sum();
        let total_cold_hits: u64 = self.moe_stats.iter().map(|s| s.total_cold_hit_count).sum();
        let total_sec: f64 = self.moe_stats.iter().map(|s| s.total_compute_sec).sum();
        let total_bytes: u64 = self.moe_stats.iter().map(|s| s.total_bytes_read).sum();
        let total_logical_bytes: u64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_logical_bytes_requested)
            .sum();
        let total_actual_loaded_bytes: u64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_actual_bytes_loaded)
            .sum();
        let total_dequantized_scratch_bytes: u64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_dequantized_scratch_bytes)
            .sum();
        let total_resident_reused_bytes: u64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_resident_cache_bytes_reused)
            .sum();
        let total_resident_hit_count: u64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_resident_cache_hit_count)
            .sum();
        let total_resident_miss_count: u64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_resident_cache_miss_count)
            .sum();
        let total_direct_cold_load_count: u64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_direct_cold_load_count)
            .sum();
        let total_router_sec: f64 = self.moe_stats.iter().map(|s| s.total_router_sec).sum();
        let total_select_sec: f64 = self.moe_stats.iter().map(|s| s.total_select_sec).sum();
        let total_load_sec: f64 = self.moe_stats.iter().map(|s| s.total_load_sec).sum();
        let total_dequant_sec: f64 = self.moe_stats.iter().map(|s| s.total_dequant_sec).sum();
        let total_gemv_sec: f64 = self.moe_stats.iter().map(|s| s.total_gemv_sec).sum();
        let total_accumulate_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_accumulate_sec).sum();
        let total_fused_gate_up_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_fused_gate_up_sec).sum();
        let total_fused_swiglu_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_fused_swiglu_sec).sum();
        let total_fused_down_accum_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_fused_down_accum_sec)
            .sum();
        let total_fused_alloc_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_fused_alloc_sec).sum();
        let total_fused_stats_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_fused_stats_sec).sum();
        let total_shared_calls: u64 = self.moe_stats.iter().map(|s| s.shared_calls).sum();
        let total_shared_sec: f64 = self.moe_stats.iter().map(|s| s.total_shared_sec).sum();
        let total_router_wall_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_router_wall_sec).sum();
        let total_call_moe_wall_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_call_moe_wall_sec).sum();
        let total_candidate_build_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_candidate_build_wall_sec)
            .sum();
        let total_policy_select_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_policy_select_wall_sec)
            .sum();
        let total_cache_lookup_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_cache_lookup_wall_sec)
            .sum();
        let total_cache_key_build_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_cache_key_build_wall_sec)
            .sum();
        let total_cache_hit_lookup_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_cache_hit_lookup_wall_sec)
            .sum();
        let total_cache_miss_load_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_cache_miss_load_wall_sec)
            .sum();
        let total_cache_eviction_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_cache_eviction_wall_sec)
            .sum();
        let total_cache_insert_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_cache_insert_wall_sec)
            .sum();
        let total_cache_page_clone_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_cache_page_clone_wall_sec)
            .sum();
        let total_select_wall_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_select_wall_sec).sum();
        let total_load_wall_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_load_wall_sec).sum();
        let total_routed_exec_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_routed_exec_wall_sec)
            .sum();
        let total_stats_wall_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_stats_wall_sec).sum();
        let total_exec_wall_sec: f64 =
            self.moe_stats.iter().map(|s| s.total_exec_wall_sec).sum();
        let total_accumulate_wall_sec: f64 = self
            .moe_stats
            .iter()
            .map(|s| s.total_accumulate_wall_sec)
            .sum();
        let forward_layers: Vec<_> = self.forward_stats.iter().enumerate().map(|(layer_idx, s)| serde_json::json!({
            "layer": layer_idx,
            "calls": s.calls,
            "avg_layer_wall_ms": if s.calls > 0 { (s.total_layer_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
            "avg_deltanet_wall_ms": if s.calls > 0 { (s.total_deltanet_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
            "avg_gqa_wall_ms": if s.calls > 0 { (s.total_gqa_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
            "avg_shared_wall_ms": if s.calls > 0 { (s.total_shared_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
            "avg_moe_wall_ms": if s.calls > 0 { (s.total_moe_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
            "selected_experts": self.moe_stats[layer_idx].last_selected_count,
        })).collect();
        let total_layer_calls: u64 = self.forward_stats.iter().map(|s| s.calls).sum();
        let total_layer_wall_sec: f64 =
            self.forward_stats.iter().map(|s| s.total_layer_wall_sec).sum();
        let total_layer_moe_wall_sec: f64 =
            self.forward_stats.iter().map(|s| s.total_moe_wall_sec).sum();
        let gqa_calls: u64 = self
            .moe_stats
            .iter()
            .enumerate()
            .filter(|(idx, _)| self.layers[*idx].is_gqa)
            .map(|(_, s)| s.calls)
            .sum();
        let deltanet_calls: u64 = self
            .moe_stats
            .iter()
            .enumerate()
            .filter(|(idx, _)| !self.layers[*idx].is_gqa)
            .map(|(_, s)| s.calls)
            .sum();
        let gqa_selected_total: u64 = self
            .moe_stats
            .iter()
            .enumerate()
            .filter(|(idx, _)| self.layers[*idx].is_gqa)
            .map(|(_, s)| s.total_executed_experts)
            .sum();
        let deltanet_selected_total: u64 = self
            .moe_stats
            .iter()
            .enumerate()
            .filter(|(idx, _)| !self.layers[*idx].is_gqa)
            .map(|(_, s)| s.total_executed_experts)
            .sum();
        let gqa_moe_wall_sec: f64 = self
            .forward_stats
            .iter()
            .enumerate()
            .filter(|(idx, _)| self.layers[*idx].is_gqa)
            .map(|(_, s)| s.total_moe_wall_sec)
            .sum();
        let deltanet_moe_wall_sec: f64 = self
            .forward_stats
            .iter()
            .enumerate()
            .filter(|(idx, _)| !self.layers[*idx].is_gqa)
            .map(|(_, s)| s.total_moe_wall_sec)
            .sum();
        let resident_cache_enabled = !self.expert_residency_manager.is_bypass();
        let resident_cache_capacity_bytes = self.expert_residency_manager.capacity_bytes;
        let resident_cache_resident_bytes = self.expert_residency_manager.resident_bytes();
        let pinned_resident_bytes = self.expert_residency_manager.pinned_resident_bytes();
        let token_window_peak_resident_bytes =
            self.expert_residency_manager.token_window_peak_resident_bytes;
        let eviction_count_during_token =
            self.expert_residency_manager.eviction_count_during_token;
        let eviction_count_at_token_end =
            self.expert_residency_manager.eviction_count_at_token_end;
        let importance_eviction_enabled =
            self.expert_residency_manager.importance_eviction_enabled;
        let evicted_hot_count = self.expert_residency_manager.evicted_hot_count;
        let evicted_warm_count = self.expert_residency_manager.evicted_warm_count;
        let evicted_cold_count = self.expert_residency_manager.evicted_cold_count;
        let evicted_unknown_count = self.expert_residency_manager.evicted_unknown_count;
        let expert_eviction_policy = self.expert_residency_manager.expert_eviction_policy.clone();
        let json = serde_json::json!({
            "summary": {
                "total_calls": total_calls,
                "avg_executed_experts": if total_calls > 0 { total_exec as f64 / total_calls as f64 } else { 0.0 },
                "avg_selected_experts": if total_calls > 0 { total_exec as f64 / total_calls as f64 } else { 0.0 },
                "avg_executed_mass": if total_calls > 0 { total_mass / total_calls as f64 } else { 0.0 },
                "avg_routing_mass_kept": if total_calls > 0 { total_mass / total_calls as f64 } else { 0.0 },
                "avg_dropped_mass": if total_calls > 0 { total_drop / total_calls as f64 } else { 0.0 },
                "avg_routing_mass_dropped": if total_calls > 0 { total_drop / total_calls as f64 } else { 0.0 },
                "avg_routing_mass_kept_pre_renorm": if total_calls > 0 { self.moe_stats.iter().map(|s| s.total_executed_mass_pre_renorm).sum::<f64>() / total_calls as f64 } else { 0.0 },
                "avg_routing_mass_dropped_pre_renorm": if total_calls > 0 { self.moe_stats.iter().map(|s| s.total_dropped_mass_pre_renorm).sum::<f64>() / total_calls as f64 } else { 0.0 },
                "avg_routing_mass_sum_after_renorm": if total_calls > 0 { self.moe_stats.iter().map(|s| s.total_routing_mass_sum_after_renorm).sum::<f64>() / total_calls as f64 } else { 0.0 },
                "avg_load_count": if total_calls > 0 { total_loads as f64 / total_calls as f64 } else { 0.0 },
                "avg_warm_hit_count": if total_calls > 0 { total_warm_hits as f64 / total_calls as f64 } else { 0.0 },
                "avg_cold_hit_count": if total_calls > 0 { total_cold_hits as f64 / total_calls as f64 } else { 0.0 },
                "avg_compute_ms": if total_calls > 0 { (total_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_bytes_read": if total_calls > 0 { total_bytes as f64 / total_calls as f64 } else { 0.0 },
                "logical_expert_bytes_requested": total_logical_bytes,
                "actual_expert_bytes_loaded": total_actual_loaded_bytes,
                "resident_cache_bytes_reused": total_resident_reused_bytes,
                "dequantized_scratch_bytes": total_dequantized_scratch_bytes,
                "resident_cache_hit_count": total_resident_hit_count,
                "resident_cache_miss_count": total_resident_miss_count,
                "direct_cold_load_count": total_direct_cold_load_count,
                "resident_cache_enabled": resident_cache_enabled,
                "resident_cache_capacity_bytes": resident_cache_capacity_bytes,
                "resident_cache_resident_bytes": resident_cache_resident_bytes,
                "pinned_resident_bytes": pinned_resident_bytes,
                "token_window_peak_resident_bytes": token_window_peak_resident_bytes,
                "eviction_count_during_token": eviction_count_during_token,
                "eviction_count_at_token_end": eviction_count_at_token_end,
                "importance_eviction_enabled": importance_eviction_enabled,
                "evicted_hot_count": evicted_hot_count,
                "evicted_warm_count": evicted_warm_count,
                "evicted_cold_count": evicted_cold_count,
                "evicted_unknown_count": evicted_unknown_count,
                "expert_eviction_policy": expert_eviction_policy,
                "residency_group_size": self.expert_residency_manager.residency_group_size,
                "group_preresolve_wall_ms": self.expert_residency_manager.group_preresolve_wall_ms,
                "group_pinned_bytes": self.expert_residency_manager.group_pinned_bytes,
                "group_preloaded_expert_count": self.expert_residency_manager.group_preloaded_expert_count,
                "group_cache_miss_count": self.expert_residency_manager.group_cache_miss_count,
                "group_preresolve_skipped_by_budget": self.expert_residency_manager.group_preresolve_skipped_by_budget,
                "group_preresolve_requested_bytes": self.expert_residency_manager.group_preresolve_requested_bytes,
                "group_preresolve_loaded_bytes": self.expert_residency_manager.group_preresolve_loaded_bytes,
                "group_preresolve_hit_rate": self.expert_residency_manager.group_preresolve_hit_rate,
                "resident_cache_hit_rate": if (total_resident_hit_count + total_resident_miss_count) > 0 {
                    total_resident_hit_count as f64 / (total_resident_hit_count + total_resident_miss_count) as f64
                } else { 0.0 },
                "avg_logical_expert_bytes_requested": if total_calls > 0 { total_logical_bytes as f64 / total_calls as f64 } else { 0.0 },
                "avg_actual_expert_bytes_loaded": if total_calls > 0 { total_actual_loaded_bytes as f64 / total_calls as f64 } else { 0.0 },
                "avg_resident_cache_bytes_reused": if total_calls > 0 { total_resident_reused_bytes as f64 / total_calls as f64 } else { 0.0 },
                "avg_resident_cache_hit_count": if total_calls > 0 { total_resident_hit_count as f64 / total_calls as f64 } else { 0.0 },
                "avg_resident_cache_miss_count": if total_calls > 0 { total_resident_miss_count as f64 / total_calls as f64 } else { 0.0 },
                "avg_direct_cold_load_count": if total_calls > 0 { total_direct_cold_load_count as f64 / total_calls as f64 } else { 0.0 },
                "avg_router_ms": if total_calls > 0 { (total_router_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_expert_select_ms": if total_calls > 0 { (total_select_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_expert_load_ms": if total_calls > 0 { (total_load_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_dequant_ms": if total_calls > 0 { (total_dequant_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_gemv_ms": if total_calls > 0 { (total_gemv_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_accumulate_ms": if total_calls > 0 { (total_accumulate_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_fused_gate_up_ms": if total_calls > 0 { (total_fused_gate_up_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_fused_swiglu_ms": if total_calls > 0 { (total_fused_swiglu_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_fused_down_accum_ms": if total_calls > 0 { (total_fused_down_accum_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_fused_alloc_ms": if total_calls > 0 { (total_fused_alloc_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_fused_stats_ms": if total_calls > 0 { (total_fused_stats_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_shared_ms": if total_shared_calls > 0 { (total_shared_sec * 1000.0) / total_shared_calls as f64 } else { 0.0 },
                "avg_router_wall_ms": if total_calls > 0 { (total_router_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_call_moe_wall_ms": if total_calls > 0 { (total_call_moe_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_candidate_build_wall_ms": if total_calls > 0 { (total_candidate_build_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_policy_select_wall_ms": if total_calls > 0 { (total_policy_select_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_cache_lookup_wall_ms": if total_calls > 0 { (total_cache_lookup_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_cache_key_build_wall_ms": if total_calls > 0 { (total_cache_key_build_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_cache_hit_lookup_wall_ms": if total_calls > 0 { (total_cache_hit_lookup_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_cache_miss_load_wall_ms": if total_calls > 0 { (total_cache_miss_load_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_cache_eviction_wall_ms": if total_calls > 0 { (total_cache_eviction_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_cache_insert_wall_ms": if total_calls > 0 { (total_cache_insert_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_cache_page_clone_wall_ms": if total_calls > 0 { (total_cache_page_clone_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_select_wall_ms": if total_calls > 0 { (total_select_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_load_wall_ms": if total_calls > 0 { (total_load_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_routed_exec_wall_ms": if total_calls > 0 { (total_routed_exec_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_stats_wall_ms": if total_calls > 0 { (total_stats_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_exec_wall_ms": if total_calls > 0 { (total_exec_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_accumulate_wall_ms": if total_calls > 0 { (total_accumulate_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                "avg_selected_experts_gqa": if gqa_calls > 0 { gqa_selected_total as f64 / gqa_calls as f64 } else { 0.0 },
                "avg_selected_experts_deltanet": if deltanet_calls > 0 { deltanet_selected_total as f64 / deltanet_calls as f64 } else { 0.0 },
                "avg_gqa_moe_wall_ms": if gqa_calls > 0 { (gqa_moe_wall_sec * 1000.0) / gqa_calls as f64 } else { 0.0 },
                "avg_deltanet_moe_wall_ms": if deltanet_calls > 0 { (deltanet_moe_wall_sec * 1000.0) / deltanet_calls as f64 } else { 0.0 },
                "last_repetition_risk": self.last_repetition_risk,
                "last_collapse_risk": self.last_collapse_risk,
                "consecutive_quality_risk": self.runtime_governor.consecutive_quality_risk,
            },
            "forward_summary": {
                "forward_calls": self.forward_calls,
                "layer_calls": total_layer_calls,
                "avg_forward_wall_ms": if self.forward_calls > 0 { (self.forward_wall_sec * 1000.0) / self.forward_calls as f64 } else { 0.0 },
                "avg_lm_head_wall_ms": if self.lm_head_calls > 0 { (self.lm_head_wall_sec * 1000.0) / self.lm_head_calls as f64 } else { 0.0 },
                "avg_layer_wall_ms": if total_layer_calls > 0 { (total_layer_wall_sec * 1000.0) / total_layer_calls as f64 } else { 0.0 },
                "avg_moe_wall_ms_per_layer": if total_layer_calls > 0 { (total_layer_moe_wall_sec * 1000.0) / total_layer_calls as f64 } else { 0.0 },
                "avg_moe_wall_ms_per_token": if self.forward_calls > 0 { (total_layer_moe_wall_sec * 1000.0) / self.forward_calls as f64 } else { 0.0 },
            },
            "layers": layers,
            "forward_layers": forward_layers,
            "moe_io_events": self.moe_io_events,
            "effective_policy": {
                "name": match &self.expert_policy {
                    crate::strategy::ExpertPolicyConfig::Exact => "exact",
                    crate::strategy::ExpertPolicyConfig::LkoAware => "lko_aware",
                    crate::strategy::ExpertPolicyConfig::TopP { .. } => "top_p",
                    crate::strategy::ExpertPolicyConfig::Contribution { .. } => "contribution",
                    crate::strategy::ExpertPolicyConfig::AdaptiveEntropy { .. } => "adaptive_entropy",
                },
                "config": &self.expert_policy,
            },
            "effective_runtime": {
                "runtime_config_source": self.runtime_config_source.as_str(),
                "policy_kind": self.effective_runtime_policy_kind(),
                "effective_moe_top_p": self.effective_moe_top_p(),
                "effective_moe_min_experts": self.effective_moe_min_experts(),
                "effective_moe_max_experts": self.effective_moe_max_experts(),
                "use_fused_moe": self.use_fused_moe,
                "fused_moe_variant": self.fused_down_mode.as_name(),
            },
            "runtime_pack": {
                "runtime_pack_loaded": self.runtime_pack_status.runtime_pack_loaded,
                "runtime_pack_path": self.runtime_pack_status.runtime_pack_path,
                "runtime_profile_loaded": self.runtime_pack_status.runtime_profile_loaded,
                "expert_importance_loaded": self.runtime_pack_status.expert_importance_loaded,
                "residency_plan_loaded": self.runtime_pack_status.residency_plan_loaded,
                "phase_policy_loaded": self.runtime_pack_status.phase_policy_loaded,
                "expert_coresidency_loaded": self.runtime_pack_status.expert_coresidency_loaded,
                "expert_eviction_policy": self.runtime_pack_status.expert_eviction_policy,
                "initial_hot_expert_count": self.runtime_pack_status.initial_hot_expert_count,
                "initial_hot_expert_bytes": self.runtime_pack_status.initial_hot_expert_bytes,
            },
        }).to_string();
        json
    }
}
