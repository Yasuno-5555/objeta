//! objeta-ssd — MoE Expert Storage Layout and Tiering.
//!
//! Generates hot/warm/cold tier assignments for MoE experts
//! based on routing occupancy analysis.

use serde::{Deserialize, Serialize};

/// Expert tier assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageTier {
    Hot,   // Always in RAM
    Warm,  // mmap cached
    Cold,  // SSD, lazy load
}

/// Expert storage entry in the compiled layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertEntry {
    pub expert_id: usize,
    pub layer_idx: usize,
    pub tier: StorageTier,
    pub occupancy: f32,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub prefetch_candidates: Vec<usize>, // next-expert predictions
}

/// Compiled expert storage layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLayout {
    pub model_name: String,
    pub n_layers: usize,
    pub n_experts_per_layer: usize,
    pub total_experts: usize,
    pub entries: Vec<ExpertEntry>,
    pub bridge_layers: Vec<usize>,
}
