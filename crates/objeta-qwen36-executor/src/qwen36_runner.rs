//! Structural definitions and model loading logic for Qwen3.6 runner.

use crate::kv_cache::{KvCache, KvCacheLegacy, KvCacheTokenMajor, KV_LAYOUT};
use crate::strategy::{ExpertPolicyConfig, StrategyConfig};
use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::path::Path;

pub const HDIM: usize = 2048;
pub const HEAD_DIM: usize = 256;
pub const N_KV: usize = 2;
pub const N_Q_ATTN: usize = 16;

#[derive(serde::Deserialize, Debug)]
pub struct WeightManifest {
    pub source_safetensors_hash: String,
    pub quant_format_version: String,
    pub block_size: usize,
    pub block_bytes: usize,
    pub endian: String,
    pub created_at: String,
    pub objeta_commit_hash: String,
}

/// Per-layer weight set. All weights pre-converted to f32.
pub struct LayerWeights {
    // Large weights: f16 (half memory). Total per layer: ~72MB vs ~145MB for f32.
    pub w_qkv: Vec<u16>,
    pub w_o: Vec<u16>,
    pub w_z: Vec<u16>,
    pub se_gate: Vec<u16>,
    pub se_up: Vec<u16>,
    pub se_down: Vec<u16>,
    // Small weights: f32 (negligible memory)
    pub w_b: Vec<f32>,
    pub w_a: Vec<f32>,
    pub w_conv: Vec<f32>,
    pub w_norm: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    pub se_gate_w: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub is_gqa: bool,
    pub has_attn: bool,
    pub qkv_M: usize,
    pub qkv_K: usize,
    pub o_M: usize,
    pub o_K: usize,
}

#[derive(Clone, Default)]
pub struct MoELayerStats {
    pub calls: u64,
    pub shared_calls: u64,
    pub total_executed_experts: u64,
    pub total_executed_mass: f64,
    pub total_dropped_mass: f64,
    pub total_load_count: u64,
    pub total_warm_hit_count: u64,
    pub total_cold_hit_count: u64,
    pub total_compute_sec: f64,
    pub total_bytes_read: u64,
    pub total_router_sec: f64,
    pub total_select_sec: f64,
    pub total_load_sec: f64,
    pub total_dequant_sec: f64,
    pub total_gemv_sec: f64,
    pub total_accumulate_sec: f64,
    pub total_shared_sec: f64,
    pub total_router_wall_sec: f64,
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

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AttnPolicy {
    Full,     // Full GQA or DeltaNet forward
    Collapse, // Koopman identity (skip, J≈I)
    Skip,     // No attention at all
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MoEPolicy {
    Full,     // Dequantize + GEMV all routed experts
    Adaptive, // Entropy-conditioned top-k pruning
    Skip,     // Skip MoE entirely
}

#[derive(Clone, Copy)]
pub struct LayerPolicy {
    pub attn: AttnPolicy,
    pub moe: MoEPolicy,
    pub precision_bits: u8, // target precision for weights (3-16)
    pub is_steering: bool,  // is this a GQA course-correction layer?
}

pub struct Qwen36Runner {
    pub embed: memmap2::Mmap, // mmap'd embed_tokens.bin (2GB, zero-copy)
    pub lm_head: Option<memmap2::Mmap>, // mmap'd lm_head.bin when embeddings are untied
    pub final_norm: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    // KV caches
    pub kv_cache: Box<dyn KvCache>,
    // DeltaNet states
    pub conv_states: Vec<Vec<f32>>, // per layer: (8192 × 4)
    pub conv_ptrs: Vec<usize>,
    pub S_states: Vec<Vec<f32>>, // per layer: (32 × 128 × 128)
    // RoPE
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    // MoE: pre-loaded routers + cached mmaps
    pub routers: Vec<Vec<f32>>,
    pub gu_mmaps: Vec<memmap2::Mmap>,
    pub down_mmaps: Vec<memmap2::Mmap>,
    // Scratch buffers (reused per forward pass)
    pub scratch_qkv: Vec<f32>,
    pub scratch_q: Vec<f32>,
    pub scratch_k: Vec<f32>,
    pub scratch_v: Vec<f32>,
    pub scratch_attn_out: Vec<f32>,
    pub scratch_scores: Vec<f32>,
    pub scratch_attn: Vec<f32>,
    pub scratch_f32: Vec<f32>, // reusable f16→f32 conversion buffer
    pub max_seq: usize,
    /// DeltaNet fusion: fraction of DeltaNet layers to compute (1.0=all, 0.33=1 per GQA block)
    pub fusion_ratio: f64,
    /// Skip MoE+shared expert on non-GQA (DeltaNet) layers
    pub moe_on_deltanet: bool,
    /// Whether routed MoE is globally enabled (used for isolation debugging)
    pub moe_enabled: bool,
    /// Experimental top-p truncation over the routed expert distribution.
    /// 1.0 keeps the exact HF-style top-8 path.
    pub moe_top_p: f32,
    /// Experimental pruning mode: 0=top-p router mass, 1=contribution prior.
    pub moe_prune_mode: i32,
    /// Threshold for contribution-prior pruning.
    pub moe_contrib_threshold: f32,
    pub min_experts: usize,
    pub max_experts: usize,
    /// EMA output norms per layer/expert for contribution-prior pruning.
    pub moe_ema_output_norm: Vec<Vec<f32>>,
    /// Active expert policy used by routed MoE selection.
    pub expert_policy: ExpertPolicyConfig,
    /// Last token entropy observed after the previous lm_head decode step.
    /// This is intentionally previous-token state:
    /// previous token observation -> next token expert policy.
    pub last_decode_entropy: f32,
    /// Last input token seen by forward.
    pub last_input_token_id: Option<usize>,
    /// Previous token's selected experts from the last MoE call.
    /// This is not per-layer history; it is the immediately preceding routed path.
    pub previous_experts: Vec<u16>,
    /// Scheduler: phase-aware execution policy per layer
    pub policy_table: [LayerPolicy; 40],
    /// Whether Metal fused GQA is available (tested at init)
    pub metal_gqa_ok: bool,
    pub metal_gqa_first_fail: bool,
    /// Expert residency cache: (layer, eid) → (gate_f32, up_f32, down_f32)
    pub expert_cache: std::collections::HashMap<(usize, usize), (Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub expert_cache_order: Vec<(usize, usize)>, // LRU order, front = most recent
    pub expert_cache_max: usize,
    /// Expert cache status: number of experts cached per layer (0 = not built)
    pub expert_cache_size: usize,
    /// Per-layer expert frequency data collected during warmup
    pub expert_freq_ready: bool,
    /// Pre-allocated scratch buffers for MoE GEMV (reused, zero allocation)
    pub moe_gate_buf: Vec<f32>, // 512
    pub moe_up_buf: Vec<f32>,     // 512
    pub moe_hidden_buf: Vec<f32>, // 512
    pub moe_down_buf: Vec<f32>,   // 2048
    pub moe_stats: Vec<MoELayerStats>,
    pub forward_stats: Vec<ForwardLayerStats>,
    pub lm_head_wall_sec: f64,
    pub lm_head_calls: u64,
    pub forward_wall_sec: f64,
    pub forward_calls: u64,
}

extern "C" {
    fn lko_metal_gqa_init(rope_cos: *const f32, rope_sin: *const f32, max_seq: i32) -> i32;
}

pub fn apply_contribution_prior_pruning_impl(
    ema: &[f32],
    min_experts: usize,
    max_experts: usize,
    eidx: &[usize],
    ew: &[f32],
    threshold: f32,
) -> (Vec<usize>, Vec<f32>, f32) {
    let threshold = threshold.clamp(0.0, 1.0);
    let mut items: Vec<(usize, f32, f32)> = eidx
        .iter()
        .zip(ew.iter())
        .map(|(&eid, &w)| {
            let norm = ema.get(eid).copied().unwrap_or(1.0).max(1e-6);
            (eid, w, w * norm)
        })
        .collect();
    items.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let total_score: f32 = items.iter().map(|x| x.2).sum::<f32>().max(1e-12);
    let mut kept = Vec::new();
    let mut kept_w = Vec::new();
    let mut score_cum = 0.0f32;
    let mut routing_mass = 0.0f32;
    for (i, (eid, w, score)) in items.into_iter().enumerate() {
        kept.push(eid);
        kept_w.push(w);
        score_cum += score;
        routing_mass += w;
        let count = i + 1;
        if (score_cum / total_score) >= threshold && count >= min_experts {
            break;
        }
        if count >= max_experts {
            break;
        }
    }
    let sum_w: f32 = kept_w.iter().sum::<f32>().max(1e-12);
    for w in &mut kept_w {
        *w /= sum_w;
    }
    (kept, kept_w, routing_mass)
}

impl Qwen36Runner {
    pub fn sync_legacy_policy_fields(&mut self) {
        match &self.expert_policy {
            ExpertPolicyConfig::Exact => {
                self.moe_prune_mode = 0;
                self.moe_top_p = 1.0;
                self.moe_contrib_threshold = 1.0;
            }
            ExpertPolicyConfig::TopP {
                p,
                min_experts,
                max_experts,
            } => {
                self.moe_prune_mode = 0;
                self.moe_top_p = p.clamp(0.0, 1.0);
                self.moe_contrib_threshold = 1.0;
                self.min_experts = *min_experts;
                self.max_experts = *max_experts;
            }
            ExpertPolicyConfig::Contribution {
                threshold,
                min_experts,
                max_experts,
                ..
            } => {
                self.moe_prune_mode = 1;
                self.moe_top_p = 1.0;
                self.moe_contrib_threshold = threshold.clamp(0.0, 1.0);
                self.min_experts = *min_experts;
                self.max_experts = *max_experts;
            }
            ExpertPolicyConfig::AdaptiveEntropy {
                min_experts,
                max_experts,
                ..
            } => {
                self.moe_prune_mode = 0;
                self.moe_top_p = 1.0;
                self.moe_contrib_threshold = 1.0;
                self.min_experts = *min_experts;
                self.max_experts = *max_experts;
            }
        }
    }

    pub fn set_expert_policy(&mut self, policy: ExpertPolicyConfig) {
        self.expert_policy = policy;
        self.sync_legacy_policy_fields();
    }

    pub fn apply_contribution_prior_pruning(
        &self,
        layer_idx: usize,
        eidx: &[usize],
        ew: &[f32],
        threshold: f32,
    ) -> (Vec<usize>, Vec<f32>, f32) {
        apply_contribution_prior_pruning_impl(
            &self.moe_ema_output_norm[layer_idx],
            self.min_experts,
            self.max_experts,
            eidx,
            ew,
            threshold,
        )
    }

    pub fn new(bin_dir: &Path, max_seq: usize) -> Option<Self> {
        // ── Verify Weight Cache Manifest ──
        let manifest_path = bin_dir.join("manifest.json");
        if !manifest_path.exists() {
            eprintln!("[objeta ERROR] Weight cache manifest.json is missing! Please rebuild using --rebuild-cache");
            return None;
        }
        let manifest_file = std::fs::File::open(&manifest_path).ok()?;
        let manifest: WeightManifest = match serde_json::from_reader(manifest_file) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[objeta ERROR] Failed to parse manifest.json: {:?}", e);
                return None;
            }
        };

        // Strict checking
        if manifest.quant_format_version != "v0" {
            eprintln!(
                "[objeta ERROR] Weight cache format version mismatch! Expected \"v0\", got {:?}",
                manifest.quant_format_version
            );
            return None;
        }
        if manifest.block_size != 128 {
            eprintln!(
                "[objeta ERROR] Weight cache block_size mismatch! Expected 128, got {:?}",
                manifest.block_size
            );
            return None;
        }
        if manifest.block_bytes != 68 {
            eprintln!(
                "[objeta ERROR] Weight cache block_bytes mismatch! Expected 68, got {:?}",
                manifest.block_bytes
            );
            return None;
        }
        // mmap embed to save 2GB RAM
        let embed_path = bin_dir.join("embed_tokens.bin");
        let embed_file = std::fs::File::open(&embed_path).ok()?;
        let embed = unsafe { memmap2::Mmap::map(&embed_file).ok()? };
        let _n_vocab = embed.len() / (HDIM * 4); // f32 = 4 bytes

        let lm_head = {
            let path = bin_dir.join("lm_head.bin");
            if path.exists() {
                let file = std::fs::File::open(&path).ok()?;
                eprintln!("[objeta] lm_head loaded from lm_head.bin");
                Some(unsafe { memmap2::Mmap::map(&file).ok()? })
            } else {
                eprintln!("[objeta] lm_head.bin missing, falling back to tied embed weights");
                None
            }
        };

        let norm_bytes = std::fs::read(bin_dir.join("final_norm.bin")).ok()?;
        let final_norm: Vec<f32> = norm_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Load all 40 layers
        let mut layers = Vec::with_capacity(40);
        for l in 0..40 {
            layers.push(load_layer_weights(bin_dir, l)?);
        }

        // Build scheduler policy table
        let runtime_strategy = crate::strategy::load_strategy_config(bin_dir);
        let default_strategy = runtime_strategy.unwrap_or_else(StrategyConfig::default);
        let policy_table = build_policy_table(
            default_strategy.fusion_ratio,
            default_strategy.moe_on_deltanet,
        );

        // Apply strategy.json if present (family-aware precision).
        let disable_strategy = std::env::var("OBJETA_DISABLE_STRATEGY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !disable_strategy {
            if let Some(strategy) = crate::strategy::load_strategy(bin_dir) {
                let ec = &strategy.executor_config;
                for l in 0..40 {
                    let lw = &mut layers[l];
                    let ffn_b = ec.ffn_bits.get(l).copied().unwrap_or(4);
                    let qo_b = ec.attn_qo_bits.get(l).copied().unwrap_or(16);
                    let kv_b = ec.attn_kv_bits.get(l).copied().unwrap_or(16);

                    if ffn_b < 16 {
                        lw.se_gate = crate::strategy::requantize_f16(&lw.se_gate, ffn_b);
                        lw.se_up = crate::strategy::requantize_f16(&lw.se_up, ffn_b);
                        lw.se_down = crate::strategy::requantize_f16(&lw.se_down, ffn_b);
                    }
                    if qo_b < 16 {
                        if lw.is_gqa {
                            let q_size = 16 * HEAD_DIM;
                            let q_part: Vec<u16> = lw.w_qkv[..q_size].to_vec();
                            let q_quant = crate::strategy::requantize_f16(&q_part, qo_b);
                            lw.w_qkv[..q_size].copy_from_slice(&q_quant);
                            lw.w_o = crate::strategy::requantize_f16(&lw.w_o, qo_b);
                        }
                    }
                    if kv_b < 16 && lw.is_gqa {
                        let q_size = 16 * HEAD_DIM;
                        let kv_size = 2 * HEAD_DIM + 2 * HEAD_DIM;
                        let kv_part: Vec<u16> = lw.w_qkv[q_size..q_size + kv_size].to_vec();
                        let kv_quant = crate::strategy::requantize_f16(&kv_part, kv_b);
                        lw.w_qkv[q_size..q_size + kv_size].copy_from_slice(&kv_quant);
                    }
                }
            }
        } else {
            eprintln!("[objeta] strategy disabled via OBJETA_DISABLE_STRATEGY");
        }

        // KV caches (only for full GQA layers, l % 4 == 3)
        let kv_cache: Box<dyn KvCache> = unsafe {
            if KV_LAYOUT == 1 {
                Box::new(KvCacheTokenMajor::new(40, 2, max_seq, HEAD_DIM))
            } else {
                Box::new(KvCacheLegacy::new(40, 2, max_seq, HEAD_DIM))
            }
        };

        // DeltaNet states (for linear attention layers)
        let conv_states: Vec<_> = (0..40)
            .map(|l| {
                if l % 4 != 3 && layers[l].has_attn {
                    vec![0.0f32; 8192 * 4]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let conv_ptrs = vec![0usize; 40];
        let S_states: Vec<_> = (0..40)
            .map(|l| {
                if l % 4 != 3 && layers[l].has_attn {
                    vec![0.0f32; 32 * 128 * 128]
                } else {
                    Vec::new()
                }
            })
            .collect();

        let (rope_cos, rope_sin) = rope_cache(max_seq, HEAD_DIM);

        // Init Metal GQA persistent resources
        let metal_gqa_ok = false;
        let _ = unsafe { lko_metal_gqa_init(rope_cos.as_ptr(), rope_sin.as_ptr(), max_seq as i32) };
        eprintln!("[objeta] Metal GQA: disabled pending kernel parity, using CPU fallback");

        // Pre-load routers + mmap MoE weights
        let mut routers = Vec::with_capacity(40);
        let mut gu_mmaps = Vec::with_capacity(40);
        let mut down_mmaps = Vec::with_capacity(40);
        for l in 0..40 {
            let rpath = bin_dir.join(format!("layer_{}_router.bin", l));
            let rbytes = std::fs::read(&rpath).unwrap_or_default();
            let r: Vec<f32> = rbytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            routers.push(r);

            let gu_f =
                std::fs::File::open(bin_dir.join(format!("layer_{}_gate_up.bin", l))).unwrap();
            let d_f = std::fs::File::open(bin_dir.join(format!("layer_{}_down.bin", l))).unwrap();
            gu_mmaps.push(unsafe { memmap2::Mmap::map(&gu_f).unwrap() });
            down_mmaps.push(unsafe { memmap2::Mmap::map(&d_f).unwrap() });
        }

        // Init per-layer expert caches
        unsafe {
            crate::moe_dispatch::lko_moe_init_caches(40);
        }
        // Init per-layer frequency trackers
        unsafe {
            crate::moe_dispatch::lko_moe_init_freq_tracker(40);
        }

        let expert_policy = default_strategy.effective_expert_policy();

        let mut runner = Qwen36Runner {
            embed,
            lm_head,
            final_norm,
            layers,
            kv_cache,
            conv_states,
            conv_ptrs,
            S_states,
            rope_cos,
            rope_sin,
            routers,
            gu_mmaps,
            down_mmaps,
            policy_table,
            scratch_qkv: vec![0.0f32; 9216],
            scratch_q: vec![0.0f32; 16 * 256],
            scratch_k: vec![0.0f32; 2 * 256],
            scratch_v: vec![0.0f32; 2 * 256],
            scratch_attn_out: vec![0.0f32; 16 * 256],
            scratch_scores: vec![0.0f32; max_seq],
            scratch_attn: vec![0.0f32; max_seq],
            scratch_f32: Vec::with_capacity(20_000_000),
            max_seq,
            fusion_ratio: default_strategy.fusion_ratio,
            moe_on_deltanet: default_strategy.moe_on_deltanet,
            moe_enabled: true,
            moe_top_p: 1.0,
            moe_prune_mode: 0,
            moe_contrib_threshold: 1.0,
            min_experts: default_strategy.min_experts,
            max_experts: default_strategy.max_experts,
            moe_ema_output_norm: vec![vec![1.0f32; 256]; 40],
            expert_policy,
            last_decode_entropy: 0.0,
            last_input_token_id: None,
            previous_experts: Vec::new(),
            metal_gqa_ok,
            metal_gqa_first_fail: true,
            expert_cache: std::collections::HashMap::new(),
            expert_cache_order: Vec::new(),
            expert_cache_max: 50,
            expert_cache_size: 0,
            expert_freq_ready: false,
            moe_gate_buf: vec![0.0f32; 512],
            moe_up_buf: vec![0.0f32; 512],
            moe_hidden_buf: vec![0.0f32; 512],
            moe_down_buf: vec![0.0f32; HDIM],
            moe_stats: vec![MoELayerStats::default(); 40],
            forward_stats: vec![ForwardLayerStats::default(); 40],
            lm_head_wall_sec: 0.0,
            lm_head_calls: 0,
            forward_wall_sec: 0.0,
            forward_calls: 0,
        };
        runner.sync_legacy_policy_fields();
        Some(runner)
    }

    pub fn warmup(&mut self, n_tokens: usize) {
        let mut dummy = vec![0.0f32; HDIM];
        for step in 0..n_tokens {
            let pos = step;
            let seq_len = step + 1;
            let _h = self.forward(1058, pos, seq_len);
        }
        self.expert_freq_ready = true;
    }

    pub fn build_expert_caches(&mut self, cache_size: usize) {
        if !self.expert_freq_ready {
            eprintln!("[objeta] Warning: building expert cache before warmup, frequency stats will be empty");
        }
        let mut total = 0;
        let mut total_layers = 0;
        for l in 0..40 {
            let gu_len = self.gu_mmaps[l].len() as i32;
            let d_len = self.down_mmaps[l].len() as i32;
            let n = unsafe {
                crate::moe_dispatch::lko_moe_build_cache(
                    l as i32,
                    self.gu_mmaps[l].as_ptr(),
                    gu_len,
                    self.down_mmaps[l].as_ptr(),
                    d_len,
                    cache_size as i32,
                )
            };
            if n > 0 {
                total += n;
                total_layers += 1;
            }
        }
        self.expert_cache_size = cache_size;
        eprintln!("[objeta] Built pre-dequantized cache for {total} experts across {total_layers} active layers.");
    }
}

fn rope_cache(max_seq: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let rotary_dim = (hd as f32 * 0.25) as usize;
    let half_rot = rotary_dim / 2;
    let mut cos = vec![0.0f32; max_seq * half_rot];
    let mut sin = vec![0.0f32; max_seq * half_rot];
    for pos in 0..max_seq {
        for i in 0..half_rot {
            let theta = 1.0 / 10000000.0f32.powf(2.0 * i as f32 / rotary_dim as f32);
            cos[pos * half_rot + i] = (pos as f32 * theta).cos();
            sin[pos * half_rot + i] = (pos as f32 * theta).sin();
        }
    }
    (cos, sin)
}

fn load_f16_raw(data: &memmap2::Mmap, off: usize, nelem: usize) -> Vec<u16> {
    let ptr = unsafe { data.as_ptr().add(off) as *const u16 };
    let slice = unsafe { std::slice::from_raw_parts(ptr, nelem) };
    slice.to_vec()
}

fn load_f16_to_f32(data: &memmap2::Mmap, off: usize, nelem: usize) -> Vec<f32> {
    let ptr = unsafe { data.as_ptr().add(off) as *const u16 };
    let slice = unsafe { std::slice::from_raw_parts(ptr, nelem) };
    slice.iter().map(|&h| f16_to_f32(h)).collect()
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = (h >> 10) & 0x1f;
    let m = h as u32 & 0x3ff;
    if e == 0 {
        if m == 0 {
            f32::from_bits(s)
        } else {
            (m as f32) * 2f32.powi(-24) * if s == 0 { 1.0 } else { -1.0 }
        }
    } else if e == 31 {
        if m == 0 {
            f32::from_bits(s | 0x7f80_0000)
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits(s | (((e + 112) as u32) << 23) | (m << 13))
    }
}

fn load_layer_weights(bin_dir: &Path, l: usize) -> Option<LayerWeights> {
    let json_path = bin_dir.join(format!("layer_{}_attn_f16.json", l));
    let mut json_str = String::new();
    std::fs::File::open(&json_path)
        .ok()?
        .read_to_string(&mut json_str)
        .ok()?;
    let meta: serde_json::Value = serde_json::from_str(&json_str).ok()?;

    let bin_path = bin_dir.join(format!("layer_{}_attn_f16.bin", l));
    let file = std::fs::File::open(&bin_path).ok()?;
    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };

    let get_f16 = |name: &str| -> Option<Vec<u16>> {
        let arr = meta.get(name)?.as_array()?;
        let off = arr[1].as_u64()? as usize;
        let nb = arr[2].as_u64()? as usize;
        Some(load_f16_raw(&mmap, off, nb / 2))
    };
    let get_f32 = |name: &str| -> Option<Vec<f32>> {
        let arr = meta.get(name)?.as_array()?;
        let off = arr[1].as_u64()? as usize;
        let nb = arr[2].as_u64()? as usize;
        Some(load_f16_to_f32(&mmap, off, nb / 2))
    };

    let is_gqa = l % 4 == 3;
    let has_attn = get_f16("linear_attn.in_proj_qkv.weight").is_some()
        || get_f16("self_attn.q_proj.weight").is_some();

    let (w_qkv, qkv_M, qkv_K) = if is_gqa {
        let qw = get_f16("self_attn.q_proj.weight")?;
        let kw = get_f16("self_attn.k_proj.weight")?;
        let vw = get_f16("self_attn.v_proj.weight")?;
        let qM = qw.len() / HDIM;
        let kM = kw.len() / HDIM;
        let vM = vw.len() / HDIM;
        let mut cat = Vec::with_capacity(qw.len() + kw.len() + vw.len());
        cat.extend_from_slice(&qw);
        cat.extend_from_slice(&kw);
        cat.extend_from_slice(&vw);
        (cat, qM + kM + vM, HDIM)
    } else if has_attn {
        let w = get_f16("linear_attn.in_proj_qkv.weight")?;
        let M = w.len() / HDIM;
        (w, M, HDIM)
    } else {
        (Vec::new(), 0, 0)
    };

    let (w_o, o_M, o_K) = if is_gqa {
        let w = get_f16("self_attn.o_proj.weight")?;
        let M = w.len() / (16 * 256);
        (w, M, 16 * 256)
    } else if has_attn {
        let w = get_f16("linear_attn.out_proj.weight")?;
        let M = HDIM;
        let K = w.len() / HDIM;
        (w, M, K)
    } else {
        (Vec::new(), 0, 0)
    };

    let w_z = if has_attn && !is_gqa {
        get_f16("linear_attn.in_proj_z.weight")?
    } else {
        Vec::new()
    };
    let w_b = if has_attn && !is_gqa {
        get_f32("linear_attn.in_proj_b.weight")?
    } else {
        Vec::new()
    };
    let w_a = if has_attn && !is_gqa {
        get_f32("linear_attn.in_proj_a.weight")?
    } else {
        Vec::new()
    };
    let w_conv = if has_attn && !is_gqa {
        let w = get_f32("linear_attn.conv1d.weight")?;
        w
    } else {
        Vec::new()
    };
    let w_norm = if has_attn && !is_gqa {
        get_f32("linear_attn.norm.weight")?
    } else {
        Vec::new()
    };
    let dt_bias = if has_attn && !is_gqa {
        get_f32("linear_attn.dt_bias")?
    } else {
        Vec::new()
    };
    let a_log = if has_attn && !is_gqa {
        get_f32("linear_attn.A_log")?
    } else {
        Vec::new()
    };
    let se_gate = get_f16("mlp.shared_expert.gate_proj.weight").unwrap_or_default();
    let se_up = get_f16("mlp.shared_expert.up_proj.weight").unwrap_or_default();
    let se_down = get_f16("mlp.shared_expert.down_proj.weight").unwrap_or_default();
    let se_gate_w = get_f32("mlp.shared_expert_gate.weight").unwrap_or_default();
    let q_norm = get_f32("self_attn.q_norm.weight").unwrap_or_default();
    let k_norm = get_f32("self_attn.k_norm.weight").unwrap_or_default();
    let input_norm = get_f32("input_layernorm.weight").unwrap_or_default();
    let post_norm = get_f32("post_attention_layernorm.weight").unwrap_or_default();

    Some(LayerWeights {
        w_qkv,
        w_o,
        w_z,
        w_b,
        w_a,
        w_conv,
        w_norm,
        dt_bias,
        a_log,
        se_gate,
        se_up,
        se_down,
        se_gate_w,
        q_norm,
        k_norm,
        input_norm,
        post_norm,
        is_gqa,
        has_attn,
        qkv_M,
        qkv_K,
        o_M,
        o_K,
    })
}

pub fn build_policy_table(fusion_ratio: f64, moe_on_deltanet: bool) -> [LayerPolicy; 40] {
    let stride = (1.0 / fusion_ratio.max(0.01)).round() as usize;
    let mut delta_count: usize = 0;
    let mut table = [LayerPolicy {
        attn: AttnPolicy::Full,
        moe: MoEPolicy::Full,
        precision_bits: 16,
        is_steering: false,
    }; 40];

    for l in 0..40 {
        let is_gqa = l % 4 == 3;
        let phase = if l < 3 {
            "unfold"
        } else if l > 35 {
            "divergent"
        } else {
            "isometric"
        };

        let (attn, moe, prec, steering) = if is_gqa {
            delta_count = 0;
            (AttnPolicy::Full, MoEPolicy::Adaptive, 16, true)
        } else {
            delta_count += 1;
            let compute = delta_count % stride.max(1) == 0;
            let attn = if compute {
                AttnPolicy::Full
            } else {
                AttnPolicy::Collapse
            };
            let moe = if moe_on_deltanet {
                MoEPolicy::Adaptive
            } else {
                MoEPolicy::Skip
            };
            let prec = match phase {
                "unfold" => 16,
                "divergent" => 8,
                _ => 4,
            };
            (attn, moe, prec, false)
        };

        table[l] = LayerPolicy {
            attn,
            moe,
            precision_bits: prec,
            is_steering: steering,
        };
    }
    table
}
