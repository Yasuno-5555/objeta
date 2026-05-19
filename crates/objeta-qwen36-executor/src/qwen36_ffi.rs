//! C FFI bindings exposing the Qwen3.6 runner API.

use crate::dispatch_gqa;
use crate::kv_cache::KV_LAYOUT;
use crate::qwen36_forward::{
    delta_net_fused, dot_f32, gemv_f16, gemv_f32, rms_norm, rms_norm_offset,
};
use crate::qwen36_runner::{
    build_policy_table, AttnPolicy, ForwardLayerStats, MoELayerStats, MoEPolicy, Qwen36Runner, HDIM,
};
use crate::strategy::ExpertPolicyConfig;
use std::path::Path;

pub static mut RUNNER: Option<Qwen36Runner> = None;

#[inline]
pub fn objeta_debug_enabled() -> bool {
    std::env::var("OBJETA_DEBUG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[inline]
pub fn objeta_timing_enabled() -> bool {
    std::env::var("OBJETA_TIMING")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[no_mangle]
pub unsafe extern "C" fn lko_runner_reset_kv_cache() -> i32 {
    match &mut RUNNER {
        Some(r) => {
            r.kv_cache.reset();
            1
        }
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_init(bin_dir: *const i8, max_seq: i32) -> i32 {
    let path = unsafe { std::ffi::CStr::from_ptr(bin_dir) }.to_string_lossy();
    let runner = Qwen36Runner::new(Path::new(path.as_ref()), max_seq as usize);
    match runner {
        Some(r) => {
            unsafe {
                RUNNER = Some(r);
            }
            1
        }
        None => 0,
    }
}

/// Dump the fusion schedule (layer policy table) to stdout.
/// Call after setting fusion_ratio and moe_on_deltanet.
#[no_mangle]
pub extern "C" fn lko_runner_dump_fusion_schedule() -> i32 {
    unsafe {
        match &RUNNER {
            Some(r) => {
                println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("FUSION SCHEDULE DUMP (token 0 reference)");
                println!("  fusion_ratio = {:.2}", r.fusion_ratio);
                println!("  moe_on_deltanet = {}", r.moe_on_deltanet);
                println!("  moe_enabled = {}", r.moe_enabled);
                println!(
                    "  effective stride = {} (round(1.0 / fusion_ratio))",
                    (1.0 / r.fusion_ratio.max(0.01)).round() as usize
                );
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                let mut executed_ids: Vec<usize> = vec![];
                let mut skipped_ids: Vec<usize> = vec![];
                let mut gqa_ids: Vec<usize> = vec![];
                let mut deltanet_executed: Vec<usize> = vec![];
                let mut deltanet_skipped: Vec<usize> = vec![];
                let mut moe_executed: Vec<usize> = vec![];
                let mut moe_skipped: Vec<usize> = vec![];
                let mut shared_executed: Vec<usize> = vec![];
                let mut shared_skipped: Vec<usize> = vec![];

                for l in 0..40 {
                    let pol = &r.policy_table[l];
                    let is_gqa = pol.is_steering;

                    if is_gqa {
                        gqa_ids.push(l);
                    }

                    let attn_exec = matches!(pol.attn, AttnPolicy::Full);
                    if attn_exec {
                        executed_ids.push(l);
                        if !is_gqa {
                            deltanet_executed.push(l);
                        }
                    } else {
                        skipped_ids.push(l);
                        if !is_gqa {
                            deltanet_skipped.push(l);
                        }
                    }

                    if matches!(pol.moe, MoEPolicy::Full | MoEPolicy::Adaptive) {
                        moe_executed.push(l);
                    } else {
                        moe_skipped.push(l);
                    }

                    // Shared expert: executed when moe != Skip and layer has shared expert weights
                    let has_se = !r.layers[l].se_gate.is_empty();
                    if matches!(pol.moe, MoEPolicy::Full | MoEPolicy::Adaptive) && has_se {
                        shared_executed.push(l);
                    } else if has_se {
                        shared_skipped.push(l);
                    }

                    println!(
                        "  L{:02} | {:6} | attn={:8} | moe={:8} | prec={:2}bit | {}",
                        l,
                        if is_gqa { "GQA" } else { "ΔNet" },
                        format!("{:?}", pol.attn).to_lowercase(),
                        format!("{:?}", pol.moe).to_lowercase(),
                        pol.precision_bits,
                        if attn_exec && !is_gqa {
                            "Δ state UPDATE"
                        } else if !attn_exec && !is_gqa {
                            "Δ state STALE (no update, S_t not advanced)"
                        } else if attn_exec && is_gqa {
                            "KV cache written"
                        } else {
                            "KV cache STALE"
                        }
                    );
                }

                println!("───");
                println!("  Summary:");
                println!("    Executed layers (attn=Full):  {:?}", executed_ids);
                println!("    Skipped layers (attn!=Full):  {:?}", skipped_ids);
                println!("    GQA layers:                   {:?}", gqa_ids);
                println!(
                    "    GQA executed: {:?}",
                    gqa_ids
                        .iter()
                        .filter(|&&l| executed_ids.contains(&l))
                        .copied()
                        .collect::<Vec<_>>()
                );
                println!(
                    "    ΔNet executed: {:?}",
                    deltanet_executed
                );
                println!(
                    "    ΔNet skipped:  {:?}",
                    deltanet_skipped
                );
                println!("    MoE executed:  {:?}", moe_executed);
                println!("    MoE skipped:   {:?}", moe_skipped);
                println!("    Shared expert executed: {:?}", shared_executed);
                println!("    Shared expert skipped:  {:?}", shared_skipped);
                println!();
                println!("  Skip semantics:");
                println!("    Attn=Collapse → ao=zeros(2048), h += 0 (no-op)");
                println!("    Attn=Skip     → ao=zeros(2048), h += 0 (no-op)");
                println!("    MoE=Skip      → moe_out not computed, h += 0 for expert path");
                println!();
                println!("  ΔNet state handling on skip:");
                println!("    When attn=Collapse/Skip on a DeltaNet layer:");
                println!("      - conv_state is NOT updated (stale from last execution)");
                println!("      - S_state is NOT updated (stale recurrent matrix)");
                println!("      - conv_ptr is NOT advanced");
                println!("    On next ΔNet execution: stale state resumes from last update point.");
                println!();
                println!("  MoE residual on skip:");
                println!("    When moe=Skip on a DeltaNet layer:");
                println!("      - routed expert output is NOT added to h");
                println!("      - shared expert is NOT added to h");
                println!("      - MoE residual is neither double-added nor leaked");
                println!("    When moe!=Skip on a GQA layer:");
                println!("      - MoE path always uses post_attention_layernorm input");
                println!("      - routed expert + shared expert both added to h");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
                1
            }
            None => 0,
        }
    }
}

/// Set DeltaNet fusion ratio: 1.0 = all layers (default), 0.33 = 1 per GQA block.
#[no_mangle]
pub extern "C" fn lko_runner_set_fusion_ratio(ratio: f64) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                let r = r;
                r.fusion_ratio = ratio.clamp(0.0, 1.0);
                r.policy_table = build_policy_table(r.fusion_ratio, r.moe_on_deltanet);
                1
            }
            None => 0,
        }
    }
}

/// Run warmup to collect expert routing frequencies.
#[no_mangle]
pub extern "C" fn lko_runner_warmup(n_tokens: i32) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    runner.warmup(n_tokens as usize);
    1
}

/// Build per-layer expert caches from warmup data.
#[no_mangle]
pub extern "C" fn lko_runner_build_caches(cache_size: i32) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    runner.build_expert_caches(cache_size as usize);
    runner.expert_cache_size as i32
}

/// Skip MoE dispatch + shared expert on non-GQA (DeltaNet) layers.
#[no_mangle]
pub extern "C" fn lko_runner_set_moe_on_deltanet(enabled: i32) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                let r = r;
                r.moe_on_deltanet = enabled != 0;
                r.policy_table = build_policy_table(r.fusion_ratio, r.moe_on_deltanet);
                1
            }
            None => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_forward(
    token_id: i32,
    pos: i32,
    seq_len: i32,
    h_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let h = runner.forward(token_id as usize, pos as usize, seq_len as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(h.as_ptr(), h_out, HDIM);
    }
    HDIM as i32
}

/// Forward pass through only the first N layers. Returns hidden state after N layers.
#[no_mangle]
pub extern "C" fn lko_runner_forward_n(
    token_id: i32,
    pos: i32,
    seq_len: i32,
    n_layers: i32,
    h_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let mut h = {
        let ptr = unsafe { runner.embed.as_ptr().add(token_id as usize * HDIM * 4) as *const f32 };
        (0..HDIM)
            .map(|i| unsafe { *ptr.add(i) })
            .collect::<Vec<f32>>()
    };
    let n = n_layers.min(40) as usize;
    for l in 0..n {
        let policy = runner.policy_table[l];
        let lw = &runner.layers[l];
        let h_norm = if !lw.input_norm.is_empty() {
            rms_norm(&h, &lw.input_norm)
        } else {
            h.clone()
        };
        let ao = match policy.attn {
            AttnPolicy::Full => {
                if policy.is_steering {
                    dispatch_gqa!(runner, l, lw, &h_norm, pos as usize, seq_len as usize)
                } else {
                    let mut ao = vec![0.0f32; HDIM];
                    delta_net_fused(
                        &lw.w_qkv,
                        &lw.w_z,
                        &lw.w_b,
                        &lw.w_a,
                        &lw.w_o,
                        &lw.w_conv,
                        &lw.w_norm,
                        &lw.dt_bias,
                        &lw.a_log,
                        &h_norm,
                        &mut runner.conv_states[l],
                        &mut runner.conv_ptrs[l],
                        &mut runner.S_states[l],
                        &mut ao,
                        &mut runner.scratch_f32,
                        l,
                        pos as usize,
                    );
                    ao
                }
            }
            AttnPolicy::Collapse | AttnPolicy::Skip => {
                vec![0.0f32; HDIM]
            }
        };
        for i in 0..HDIM {
            h[i] += ao[i];
        }
        let h_norm2 = if !lw.post_norm.is_empty() {
            rms_norm(&h, &lw.post_norm)
        } else {
            h.clone()
        };
        if policy.moe != MoEPolicy::Skip && !lw.se_gate.is_empty() {
            let gate = gemv_f16(&lw.se_gate, &h_norm2, 512, HDIM);
            let up = gemv_f16(&lw.se_up, &h_norm2, 512, HDIM);
            let mut hidden = gate.clone();
            for i in 0..512 {
                hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i];
            }
            let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
            let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h_norm2)).exp());
            for i in 0..HDIM {
                h[i] += se_out[i] * se_gate;
            }
        }
        if runner.moe_enabled && policy.moe != MoEPolicy::Skip {
            let moe_out = runner.call_moe(&h_norm2, l, pos as usize, token_id as usize);
            for i in 0..HDIM {
                h[i] += moe_out[i];
            }
        }
    }
    unsafe {
        std::ptr::copy_nonoverlapping(h.as_ptr(), h_out, HDIM);
    }
    HDIM as i32
}

/// Compute lm_head + top-k in Rust. Returns top_k indices/values via output buffers.
#[no_mangle]
pub extern "C" fn lko_runner_lm_head(
    hn: *const f32,
    top_k: i32,
    indices_out: *mut i32,
    values_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_ref() }.expect("runner not initialized");
    let h_slice = unsafe { std::slice::from_raw_parts(hn, HDIM) };
    let (indices, values) = runner.lm_head_topk(h_slice, top_k as usize);
    let k = indices.len().min(top_k as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(indices.as_ptr(), indices_out, k);
        std::ptr::copy_nonoverlapping(values.as_ptr(), values_out, k);
    }
    k as i32
}

/// Profiled forward pass. Returns timing breakdown in `timing_out` (5 f64: delta, gqa, shared, moe, norm).
#[no_mangle]
pub extern "C" fn lko_runner_forward_timed(
    token_id: i32,
    pos: i32,
    seq_len: i32,
    h_out: *mut f32,
    timing_out: *mut f64,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let (h, timing) = runner.forward_timed(token_id as usize, pos as usize, seq_len as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(h.as_ptr(), h_out, HDIM);
        std::ptr::copy_nonoverlapping(timing.as_ptr(), timing_out, 5);
    }
    HDIM as i32
}

/// Full generation step: forward(hidden) + RMSNorm + lm_head.
#[no_mangle]
pub extern "C" fn lko_runner_step(
    token_id: i32,
    pos: i32,
    seq_len: i32,
    hn_out: *mut f32,
    top_k: i32,
    indices_out: *mut i32,
    values_out: *mut f32,
) -> i32 {
    use std::time::Instant;
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let (h, _timing) = runner.forward_timed(token_id as usize, pos as usize, seq_len as usize);

    // RMSNorm
    let hn = rms_norm(&h, &runner.final_norm);
    unsafe {
        std::ptr::copy_nonoverlapping(hn.as_ptr(), hn_out, HDIM);
    }

    // lm_head top-k
    let t_lm = Instant::now();
    let (indices, values) = runner.lm_head_topk(&hn, top_k as usize);
    runner.lm_head_wall_sec += t_lm.elapsed().as_secs_f64();
    runner.lm_head_calls += 1;
    let k = indices.len().min(top_k as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(indices.as_ptr(), indices_out, k);
        std::ptr::copy_nonoverlapping(values.as_ptr(), values_out, k);
    }
    k as i32
}

/// Full generation step: forward(hidden) + RMSNorm + lm_head + entropy.
#[no_mangle]
pub extern "C" fn lko_runner_step_with_entropy(
    token_id: i32,
    pos: i32,
    seq_len: i32,
    hn_out: *mut f32,
    top_k: i32,
    indices_out: *mut i32,
    values_out: *mut f32,
    entropy_out: *mut f32,
) -> i32 {
    use std::time::Instant;
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let (h, _timing) = runner.forward_timed(token_id as usize, pos as usize, seq_len as usize);

    // RMSNorm
    let hn = rms_norm(&h, &runner.final_norm);
    unsafe {
        std::ptr::copy_nonoverlapping(hn.as_ptr(), hn_out, HDIM);
    }

    // lm_head top-k with entropy
    let t_lm = Instant::now();
    let (indices, values, entropy) = runner.lm_head_topk_with_entropy(&hn, top_k as usize);
    runner.lm_head_wall_sec += t_lm.elapsed().as_secs_f64();
    runner.lm_head_calls += 1;
    let k = indices.len().min(top_k as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(indices.as_ptr(), indices_out, k);
        std::ptr::copy_nonoverlapping(values.as_ptr(), values_out, k);
        *entropy_out = entropy;
    }
    k as i32
}

/// Set MoE enabled state globally for isolation testing/debugging.
#[no_mangle]
pub extern "C" fn lko_runner_set_moe_enabled(enabled: i32) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                r.moe_enabled = enabled != 0;
                1
            }
            None => 0,
        }
    }
}

/// Set experimental routed expert top-p truncation. 1.0 = disabled (exact top-8).
#[no_mangle]
pub extern "C" fn lko_runner_set_moe_top_p(top_p: f32) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                let p = top_p.clamp(0.0, 1.0);
                r.set_expert_policy(ExpertPolicyConfig::TopP {
                    p,
                    min_experts: r.min_experts.max(1),
                    max_experts: r.max_experts.max(r.min_experts.max(1)),
                });
                1
            }
            None => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_moe_prune_mode(mode: i32) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                if mode == 1 {
                    r.set_expert_policy(ExpertPolicyConfig::Contribution {
                        threshold: r.moe_contrib_threshold.clamp(0.0, 1.0),
                        min_experts: r.min_experts.max(1),
                        max_experts: r.max_experts.max(r.min_experts.max(1)),
                        ema_beta: 0.95,
                    });
                } else {
                    r.set_expert_policy(ExpertPolicyConfig::TopP {
                        p: r.moe_top_p.clamp(0.0, 1.0),
                        min_experts: r.min_experts.max(1),
                        max_experts: r.max_experts.max(r.min_experts.max(1)),
                    });
                }
                1
            }
            None => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_moe_contrib_threshold(threshold: f32) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                let threshold = threshold.clamp(0.0, 1.0);
                r.set_expert_policy(ExpertPolicyConfig::Contribution {
                    threshold,
                    min_experts: r.min_experts.max(1),
                    max_experts: r.max_experts.max(r.min_experts.max(1)),
                    ema_beta: 0.95,
                });
                1
            }
            None => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_moe_min_experts(min_e: i32) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                let min_experts = min_e.max(1) as usize;
                let max_experts = r.max_experts.max(min_experts);
                let new_policy = match &r.expert_policy {
                    ExpertPolicyConfig::Exact => ExpertPolicyConfig::Exact,
                    ExpertPolicyConfig::TopP { p, .. } => ExpertPolicyConfig::TopP {
                        p: *p,
                        min_experts,
                        max_experts,
                    },
                    ExpertPolicyConfig::Contribution {
                        threshold,
                        ema_beta,
                        ..
                    } => ExpertPolicyConfig::Contribution {
                        threshold: *threshold,
                        min_experts,
                        max_experts,
                        ema_beta: *ema_beta,
                    },
                    ExpertPolicyConfig::AdaptiveEntropy {
                        low_entropy_p,
                        mid_entropy_p,
                        high_entropy_p,
                        repetition_p,
                        low_entropy_threshold,
                        mid_entropy_threshold,
                        ..
                    } => ExpertPolicyConfig::AdaptiveEntropy {
                        low_entropy_p: *low_entropy_p,
                        mid_entropy_p: *mid_entropy_p,
                        high_entropy_p: *high_entropy_p,
                        repetition_p: *repetition_p,
                        low_entropy_threshold: *low_entropy_threshold,
                        mid_entropy_threshold: *mid_entropy_threshold,
                        min_experts,
                        max_experts,
                    },
                };
                r.set_expert_policy(new_policy);
                1
            }
            None => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_moe_max_experts(max_e: i32) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                let min_experts = r.min_experts.max(1);
                let max_experts = (max_e.max(1) as usize).max(min_experts);
                let new_policy = match &r.expert_policy {
                    ExpertPolicyConfig::Exact => ExpertPolicyConfig::Exact,
                    ExpertPolicyConfig::TopP { p, .. } => ExpertPolicyConfig::TopP {
                        p: *p,
                        min_experts,
                        max_experts,
                    },
                    ExpertPolicyConfig::Contribution {
                        threshold,
                        ema_beta,
                        ..
                    } => ExpertPolicyConfig::Contribution {
                        threshold: *threshold,
                        min_experts,
                        max_experts,
                        ema_beta: *ema_beta,
                    },
                    ExpertPolicyConfig::AdaptiveEntropy {
                        low_entropy_p,
                        mid_entropy_p,
                        high_entropy_p,
                        repetition_p,
                        low_entropy_threshold,
                        mid_entropy_threshold,
                        ..
                    } => ExpertPolicyConfig::AdaptiveEntropy {
                        low_entropy_p: *low_entropy_p,
                        mid_entropy_p: *mid_entropy_p,
                        high_entropy_p: *high_entropy_p,
                        repetition_p: *repetition_p,
                        low_entropy_threshold: *low_entropy_threshold,
                        mid_entropy_threshold: *mid_entropy_threshold,
                        min_experts,
                        max_experts,
                    },
                };
                r.set_expert_policy(new_policy);
                1
            }
            None => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_expert_policy_json(policy_json: *const i8) -> i32 {
    if policy_json.is_null() {
        return 0;
    }
    let json = unsafe { std::ffi::CStr::from_ptr(policy_json) }.to_string_lossy();
    let policy: ExpertPolicyConfig = match crate::strategy::parse_expert_policy_json(&json) {
        Ok(p) => p,
        Err(_) => return 0,
    };
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                r.set_expert_policy(policy);
                1
            }
            None => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_reset_moe_stats() -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                r.moe_stats = vec![MoELayerStats::default(); 40];
                r.forward_stats = vec![ForwardLayerStats::default(); 40];
                r.lm_head_wall_sec = 0.0;
                r.lm_head_calls = 0;
                r.forward_wall_sec = 0.0;
                r.forward_calls = 0;
                1
            }
            None => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_get_moe_stats_json() -> *mut std::os::raw::c_char {
    unsafe {
        match &RUNNER {
            Some(r) => {
                let mut layers = Vec::with_capacity(r.moe_stats.len());
                for (layer_idx, s) in r.moe_stats.iter().enumerate() {
                    layers.push(serde_json::json!({
                        "layer": layer_idx,
                        "calls": s.calls,
                        "avg_executed_experts": if s.calls > 0 { s.total_executed_experts as f64 / s.calls as f64 } else { 0.0 },
                        "avg_executed_mass": if s.calls > 0 { s.total_executed_mass / s.calls as f64 } else { 0.0 },
                        "avg_dropped_mass": if s.calls > 0 { s.total_dropped_mass / s.calls as f64 } else { 0.0 },
                        "avg_load_count": if s.calls > 0 { s.total_load_count as f64 / s.calls as f64 } else { 0.0 },
                        "avg_warm_hit_count": if s.calls > 0 { s.total_warm_hit_count as f64 / s.calls as f64 } else { 0.0 },
                        "avg_cold_hit_count": if s.calls > 0 { s.total_cold_hit_count as f64 / s.calls as f64 } else { 0.0 },
                        "avg_compute_ms": if s.calls > 0 { (s.total_compute_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_bytes_read": if s.calls > 0 { s.total_bytes_read as f64 / s.calls as f64 } else { 0.0 },
                        "avg_router_ms": if s.calls > 0 { (s.total_router_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_expert_select_ms": if s.calls > 0 { (s.total_select_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_expert_load_ms": if s.calls > 0 { (s.total_load_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_dequant_ms": if s.calls > 0 { (s.total_dequant_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_gemv_ms": if s.calls > 0 { (s.total_gemv_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_accumulate_ms": if s.calls > 0 { (s.total_accumulate_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_shared_ms": if s.shared_calls > 0 { (s.total_shared_sec * 1000.0) / s.shared_calls as f64 } else { 0.0 },
                        "avg_router_wall_ms": if s.calls > 0 { (s.total_router_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_select_wall_ms": if s.calls > 0 { (s.total_select_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_load_wall_ms": if s.calls > 0 { (s.total_load_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_exec_wall_ms": if s.calls > 0 { (s.total_exec_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "avg_accumulate_wall_ms": if s.calls > 0 { (s.total_accumulate_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                        "unique_expert_ids": s.unique_expert_ids.iter().copied().collect::<Vec<_>>(),
                        "last_expert_ids": s.last_expert_ids,
                        "last_router_top8_ids": s.last_router_top8_ids,
                        "last_router_top8_weights": s.last_router_top8_weights,
                        "last_candidate_ids": s.last_candidate_ids,
                        "last_candidate_weights": s.last_candidate_weights,
                        "last_selected_ids": s.last_selected_ids,
                        "last_selected_weights": s.last_selected_weights,
                        "last_dispatch_ids": s.last_dispatch_ids,
                        "last_dispatch_weights": s.last_dispatch_weights,
                        "last_selected_count": s.last_selected_count,
                        "last_selected_renormalized": s.last_selected_renormalized,
                    }));
                }
                let total_calls: u64 = r.moe_stats.iter().map(|s| s.calls).sum();
                let total_exec: u64 = r.moe_stats.iter().map(|s| s.total_executed_experts).sum();
                let total_mass: f64 = r.moe_stats.iter().map(|s| s.total_executed_mass).sum();
                let total_drop: f64 = r.moe_stats.iter().map(|s| s.total_dropped_mass).sum();
                let total_loads: u64 = r.moe_stats.iter().map(|s| s.total_load_count).sum();
                let total_warm_hits: u64 = r.moe_stats.iter().map(|s| s.total_warm_hit_count).sum();
                let total_cold_hits: u64 = r.moe_stats.iter().map(|s| s.total_cold_hit_count).sum();
                let total_sec: f64 = r.moe_stats.iter().map(|s| s.total_compute_sec).sum();
                let total_bytes: u64 = r.moe_stats.iter().map(|s| s.total_bytes_read).sum();
                let total_router_sec: f64 = r.moe_stats.iter().map(|s| s.total_router_sec).sum();
                let total_select_sec: f64 = r.moe_stats.iter().map(|s| s.total_select_sec).sum();
                let total_load_sec: f64 = r.moe_stats.iter().map(|s| s.total_load_sec).sum();
                let total_dequant_sec: f64 = r.moe_stats.iter().map(|s| s.total_dequant_sec).sum();
                let total_gemv_sec: f64 = r.moe_stats.iter().map(|s| s.total_gemv_sec).sum();
                let total_accumulate_sec: f64 =
                    r.moe_stats.iter().map(|s| s.total_accumulate_sec).sum();
                let total_shared_calls: u64 = r.moe_stats.iter().map(|s| s.shared_calls).sum();
                let total_shared_sec: f64 = r.moe_stats.iter().map(|s| s.total_shared_sec).sum();
                let total_router_wall_sec: f64 =
                    r.moe_stats.iter().map(|s| s.total_router_wall_sec).sum();
                let total_select_wall_sec: f64 =
                    r.moe_stats.iter().map(|s| s.total_select_wall_sec).sum();
                let total_load_wall_sec: f64 =
                    r.moe_stats.iter().map(|s| s.total_load_wall_sec).sum();
                let total_exec_wall_sec: f64 =
                    r.moe_stats.iter().map(|s| s.total_exec_wall_sec).sum();
                let total_accumulate_wall_sec: f64 = r
                    .moe_stats
                    .iter()
                    .map(|s| s.total_accumulate_wall_sec)
                    .sum();
                let forward_layers: Vec<_> = r.forward_stats.iter().enumerate().map(|(layer_idx, s)| serde_json::json!({
                    "layer": layer_idx,
                    "calls": s.calls,
                    "avg_layer_wall_ms": if s.calls > 0 { (s.total_layer_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                    "avg_deltanet_wall_ms": if s.calls > 0 { (s.total_deltanet_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                    "avg_gqa_wall_ms": if s.calls > 0 { (s.total_gqa_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                    "avg_shared_wall_ms": if s.calls > 0 { (s.total_shared_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                    "avg_moe_wall_ms": if s.calls > 0 { (s.total_moe_wall_sec * 1000.0) / s.calls as f64 } else { 0.0 },
                })).collect();
                let total_layer_calls: u64 = r.forward_stats.iter().map(|s| s.calls).sum();
                let total_layer_wall_sec: f64 =
                    r.forward_stats.iter().map(|s| s.total_layer_wall_sec).sum();
                let total_layer_moe_wall_sec: f64 =
                    r.forward_stats.iter().map(|s| s.total_moe_wall_sec).sum();
                let json = serde_json::json!({
                    "summary": {
                        "total_calls": total_calls,
                        "avg_executed_experts": if total_calls > 0 { total_exec as f64 / total_calls as f64 } else { 0.0 },
                        "avg_executed_mass": if total_calls > 0 { total_mass / total_calls as f64 } else { 0.0 },
                        "avg_dropped_mass": if total_calls > 0 { total_drop / total_calls as f64 } else { 0.0 },
                        "avg_load_count": if total_calls > 0 { total_loads as f64 / total_calls as f64 } else { 0.0 },
                        "avg_warm_hit_count": if total_calls > 0 { total_warm_hits as f64 / total_calls as f64 } else { 0.0 },
                        "avg_cold_hit_count": if total_calls > 0 { total_cold_hits as f64 / total_calls as f64 } else { 0.0 },
                        "avg_compute_ms": if total_calls > 0 { (total_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_bytes_read": if total_calls > 0 { total_bytes as f64 / total_calls as f64 } else { 0.0 },
                        "avg_router_ms": if total_calls > 0 { (total_router_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_expert_select_ms": if total_calls > 0 { (total_select_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_expert_load_ms": if total_calls > 0 { (total_load_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_dequant_ms": if total_calls > 0 { (total_dequant_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_gemv_ms": if total_calls > 0 { (total_gemv_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_accumulate_ms": if total_calls > 0 { (total_accumulate_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_shared_ms": if total_shared_calls > 0 { (total_shared_sec * 1000.0) / total_shared_calls as f64 } else { 0.0 },
                        "avg_router_wall_ms": if total_calls > 0 { (total_router_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_select_wall_ms": if total_calls > 0 { (total_select_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_load_wall_ms": if total_calls > 0 { (total_load_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_exec_wall_ms": if total_calls > 0 { (total_exec_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                        "avg_accumulate_wall_ms": if total_calls > 0 { (total_accumulate_wall_sec * 1000.0) / total_calls as f64 } else { 0.0 },
                    },
                    "forward_summary": {
                        "forward_calls": r.forward_calls,
                        "layer_calls": total_layer_calls,
                        "avg_forward_wall_ms": if r.forward_calls > 0 { (r.forward_wall_sec * 1000.0) / r.forward_calls as f64 } else { 0.0 },
                        "avg_lm_head_wall_ms": if r.lm_head_calls > 0 { (r.lm_head_wall_sec * 1000.0) / r.lm_head_calls as f64 } else { 0.0 },
                        "avg_layer_wall_ms": if total_layer_calls > 0 { (total_layer_wall_sec * 1000.0) / total_layer_calls as f64 } else { 0.0 },
                        "avg_moe_wall_ms_per_layer": if total_layer_calls > 0 { (total_layer_moe_wall_sec * 1000.0) / total_layer_calls as f64 } else { 0.0 },
                        "avg_moe_wall_ms_per_token": if r.forward_calls > 0 { (total_layer_moe_wall_sec * 1000.0) / r.forward_calls as f64 } else { 0.0 },
                    },
                    "layers": layers,
                    "forward_layers": forward_layers,
                    "effective_policy": {
                        "name": match &r.expert_policy {
                            ExpertPolicyConfig::Exact => "exact",
                            ExpertPolicyConfig::TopP { .. } => "top_p",
                            ExpertPolicyConfig::Contribution { .. } => "contribution",
                            ExpertPolicyConfig::AdaptiveEntropy { .. } => "adaptive_entropy",
                        },
                        "config": &r.expert_policy,
                    },
                }).to_string();
                std::ffi::CString::new(json).unwrap().into_raw()
            }
            None => std::ptr::null_mut(),
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_free_moe_stats_json(ptr: *mut std::os::raw::c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = std::ffi::CString::from_raw(ptr);
        }
    }
}

/// Forward pass through only the first N layers, tracing intermediate layer hidden states.
/// `h_trace_out`: output buffer of size `n_layers * HDIM` floats.
#[no_mangle]
pub extern "C" fn lko_runner_trace_layers(
    token_id: i32,
    pos: i32,
    seq_len: i32,
    n_layers: i32,
    h_trace_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let mut h = {
        let ptr = unsafe { runner.embed.as_ptr().add(token_id as usize * HDIM * 4) as *const f32 };
        (0..HDIM)
            .map(|i| unsafe { *ptr.add(i) })
            .collect::<Vec<f32>>()
    };
    let n = n_layers.min(40) as usize;
    for l in 0..n {
        let policy = runner.policy_table[l];
        let lw = &runner.layers[l];
        let h_norm = if !lw.input_norm.is_empty() {
            rms_norm(&h, &lw.input_norm)
        } else {
            h.clone()
        };
        if objeta_debug_enabled() && l == 0 && pos == 0 {
            let h_orig_norm = h.iter().map(|v| v * v).sum::<f32>().sqrt();
            let h_norm_norm = h_norm.iter().map(|v| v * v).sum::<f32>().sqrt();
            println!(
                "[RUST DEBUG L0] h_orig norm: {:.6}, first 5: {:?}",
                h_orig_norm,
                &h[..5]
            );
            println!(
                "[RUST DEBUG L0] h_norm norm: {:.6}, first 5: {:?}",
                h_norm_norm,
                &h_norm[..5]
            );
        }
        let ao = match policy.attn {
            AttnPolicy::Full => {
                if policy.is_steering {
                    dispatch_gqa!(runner, l, lw, &h_norm, pos as usize, seq_len as usize)
                } else {
                    let mut ao = vec![0.0f32; HDIM];
                    delta_net_fused(
                        &lw.w_qkv,
                        &lw.w_z,
                        &lw.w_b,
                        &lw.w_a,
                        &lw.w_o,
                        &lw.w_conv,
                        &lw.w_norm,
                        &lw.dt_bias,
                        &lw.a_log,
                        &h_norm,
                        &mut runner.conv_states[l],
                        &mut runner.conv_ptrs[l],
                        &mut runner.S_states[l],
                        &mut ao,
                        &mut runner.scratch_f32,
                        l,
                        pos as usize,
                    );
                    ao
                }
            }
            AttnPolicy::Collapse | AttnPolicy::Skip => {
                vec![0.0f32; HDIM]
            }
        };
        for i in 0..HDIM {
            h[i] += ao[i];
        }
        let h_norm2 = if !lw.post_norm.is_empty() {
            rms_norm(&h, &lw.post_norm)
        } else {
            h.clone()
        };
        if policy.moe != MoEPolicy::Skip && !lw.se_gate.is_empty() {
            let gate = gemv_f16(&lw.se_gate, &h_norm2, 512, HDIM);
            let up = gemv_f16(&lw.se_up, &h_norm2, 512, HDIM);
            let mut hidden = gate.clone();
            for i in 0..512 {
                hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i];
            }
            let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
            let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h_norm2)).exp());
            for i in 0..HDIM {
                h[i] += se_out[i] * se_gate;
            }
        }
        if runner.moe_enabled && policy.moe != MoEPolicy::Skip {
            let moe_out = runner.call_moe(&h_norm2, l, pos as usize, token_id as usize);
            for i in 0..HDIM {
                h[i] += moe_out[i];
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(h.as_ptr(), h_trace_out.add(l * HDIM), HDIM);
        }
    }
    HDIM as i32
}

/// Trace one layer's internal components for a single token.
#[no_mangle]
pub extern "C" fn lko_runner_trace_layer_components(
    token_id: i32,
    pos: i32,
    seq_len: i32,
    target_layer: i32,
    h_after_attn_out: *mut f32,
    h_norm2_out: *mut f32,
    shared_out: *mut f32,
    moe_out: *mut f32,
    h_after_mlp_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let mut h = {
        let ptr = unsafe { runner.embed.as_ptr().add(token_id as usize * HDIM * 4) as *const f32 };
        (0..HDIM)
            .map(|i| unsafe { *ptr.add(i) })
            .collect::<Vec<f32>>()
    };
    let target = target_layer.clamp(0, 39) as usize;

    for l in 0..=target {
        let policy = runner.policy_table[l];
        let lw = &runner.layers[l];
        let h_norm = if !lw.input_norm.is_empty() {
            rms_norm(&h, &lw.input_norm)
        } else {
            h.clone()
        };
        let ao = match policy.attn {
            AttnPolicy::Full => {
                if policy.is_steering {
                    dispatch_gqa!(runner, l, lw, &h_norm, pos as usize, seq_len as usize)
                } else {
                    let mut ao = vec![0.0f32; HDIM];
                    delta_net_fused(
                        &lw.w_qkv,
                        &lw.w_z,
                        &lw.w_b,
                        &lw.w_a,
                        &lw.w_o,
                        &lw.w_conv,
                        &lw.w_norm,
                        &lw.dt_bias,
                        &lw.a_log,
                        &h_norm,
                        &mut runner.conv_states[l],
                        &mut runner.conv_ptrs[l],
                        &mut runner.S_states[l],
                        &mut ao,
                        &mut runner.scratch_f32,
                        l,
                        pos as usize,
                    );
                    ao
                }
            }
            AttnPolicy::Collapse | AttnPolicy::Skip => vec![0.0f32; HDIM],
        };
        for i in 0..HDIM {
            h[i] += ao[i];
        }

        let h_after_attn = h.clone();
        let h_norm2 = if !lw.post_norm.is_empty() {
            rms_norm(&h, &lw.post_norm)
        } else {
            h.clone()
        };

        let mut shared = vec![0.0f32; HDIM];
        if policy.moe != MoEPolicy::Skip && !lw.se_gate.is_empty() {
            let gate = gemv_f16(&lw.se_gate, &h_norm2, 512, HDIM);
            let up = gemv_f16(&lw.se_up, &h_norm2, 512, HDIM);
            let mut hidden = gate.clone();
            for i in 0..512 {
                hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i];
            }
            let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
            let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h_norm2)).exp());
            for i in 0..HDIM {
                shared[i] = se_out[i] * se_gate;
            }
        }

        let moe = if runner.moe_enabled && policy.moe != MoEPolicy::Skip {
            runner.call_moe(&h_norm2, l, pos as usize, token_id as usize)
        } else {
            vec![0.0f32; HDIM]
        };

        for i in 0..HDIM {
            h[i] += shared[i] + moe[i];
        }

        if l == target {
            unsafe {
                if !h_after_attn_out.is_null() {
                    std::ptr::copy_nonoverlapping(h_after_attn.as_ptr(), h_after_attn_out, HDIM);
                }
                if !h_norm2_out.is_null() {
                    std::ptr::copy_nonoverlapping(h_norm2.as_ptr(), h_norm2_out, HDIM);
                }
                if !shared_out.is_null() {
                    std::ptr::copy_nonoverlapping(shared.as_ptr(), shared_out, HDIM);
                }
                if !moe_out.is_null() {
                    std::ptr::copy_nonoverlapping(moe.as_ptr(), moe_out, HDIM);
                }
                if !h_after_mlp_out.is_null() {
                    std::ptr::copy_nonoverlapping(h.as_ptr(), h_after_mlp_out, HDIM);
                }
            }
            return HDIM as i32;
        }
    }

    -1
}

/// Trace routed MoE router stats for one layer on one token.
#[no_mangle]
pub extern "C" fn lko_runner_trace_router(
    token_id: i32,
    pos: i32,
    seq_len: i32,
    target_layer: i32,
    top_k: i32,
    router_logits_out: *mut f32,
    topk_idx_out: *mut i32,
    topk_weight_out: *mut f32,
    entropy_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let mut h = {
        let ptr = unsafe { runner.embed.as_ptr().add(token_id as usize * HDIM * 4) as *const f32 };
        (0..HDIM)
            .map(|i| unsafe { *ptr.add(i) })
            .collect::<Vec<f32>>()
    };
    let target = target_layer.clamp(0, 39) as usize;
    let top_k = top_k.max(1) as usize;

    for l in 0..=target {
        let policy = runner.policy_table[l];
        let lw = &runner.layers[l];
        let h_norm = if !lw.input_norm.is_empty() {
            rms_norm(&h, &lw.input_norm)
        } else {
            h.clone()
        };
        let ao = match policy.attn {
            AttnPolicy::Full => {
                if policy.is_steering {
                    dispatch_gqa!(runner, l, lw, &h_norm, pos as usize, seq_len as usize)
                } else {
                    let mut ao = vec![0.0f32; HDIM];
                    delta_net_fused(
                        &lw.w_qkv,
                        &lw.w_z,
                        &lw.w_b,
                        &lw.w_a,
                        &lw.w_o,
                        &lw.w_conv,
                        &lw.w_norm,
                        &lw.dt_bias,
                        &lw.a_log,
                        &h_norm,
                        &mut runner.conv_states[l],
                        &mut runner.conv_ptrs[l],
                        &mut runner.S_states[l],
                        &mut ao,
                        &mut runner.scratch_f32,
                        l,
                        pos as usize,
                    );
                    ao
                }
            }
            AttnPolicy::Collapse | AttnPolicy::Skip => vec![0.0f32; HDIM],
        };
        for i in 0..HDIM {
            h[i] += ao[i];
        }

        let h_norm2 = if !lw.post_norm.is_empty() {
            rms_norm(&h, &lw.post_norm)
        } else {
            h.clone()
        };

        if l == target {
            let router_w = &runner.routers[l];
            let n_experts = router_w.len() / HDIM;
            let mut logits = vec![0.0f32; n_experts];
            for (eid, row) in router_w.chunks(HDIM).enumerate() {
                logits[eid] = dot_f32(row, &h_norm2);
            }

            let mut indexed: Vec<(f32, usize)> = logits
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();
            indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let k = top_k.min(indexed.len());
            let top = &indexed[..k];
            let max_logit = top
                .iter()
                .map(|(v, _)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = top.iter().map(|(v, _)| (*v - max_logit).exp()).sum();
            let weights: Vec<f32> = top
                .iter()
                .map(|(v, _)| (*v - max_logit).exp() / exp_sum.max(1e-12))
                .collect();
            let entropy: f32 = -weights
                .iter()
                .map(|&w| if w > 1e-10 { w * w.ln() } else { 0.0 })
                .sum::<f32>();

            unsafe {
                if !router_logits_out.is_null() {
                    std::ptr::copy_nonoverlapping(logits.as_ptr(), router_logits_out, n_experts);
                }
                if !topk_idx_out.is_null() {
                    for (i, (_, eid)) in top.iter().enumerate() {
                        *topk_idx_out.add(i) = *eid as i32;
                    }
                }
                if !topk_weight_out.is_null() {
                    std::ptr::copy_nonoverlapping(weights.as_ptr(), topk_weight_out, k);
                }
                if !entropy_out.is_null() {
                    *entropy_out = entropy;
                }
            }
            return n_experts as i32;
        }

        let mut shared = vec![0.0f32; HDIM];
        if policy.moe != MoEPolicy::Skip && !lw.se_gate.is_empty() {
            let gate = gemv_f16(&lw.se_gate, &h_norm2, 512, HDIM);
            let up = gemv_f16(&lw.se_up, &h_norm2, 512, HDIM);
            let mut hidden = gate.clone();
            for i in 0..512 {
                hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i];
            }
            let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
            let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h_norm2)).exp());
            for i in 0..HDIM {
                shared[i] = se_out[i] * se_gate;
            }
        }
        let moe = if runner.moe_enabled && policy.moe != MoEPolicy::Skip {
            runner.call_moe(&h_norm2, l, pos as usize, token_id as usize)
        } else {
            vec![0.0f32; HDIM]
        };
        for i in 0..HDIM {
            h[i] += shared[i] + moe[i];
        }
    }

    -1
}

/// Dense selected-expert MoE path for implementation-parity checks.
#[no_mangle]
pub extern "C" fn lko_moe_dense_selected(
    x: *const f32,
    gate_w: *const f32,
    up_w: *const f32,
    down_w: *const f32,
    routing_weights: *const f32,
    n_selected: i32,
    gate_out: *mut f32,
    up_out: *mut f32,
    hidden_out: *mut f32,
    expert_out: *mut f32,
    weighted_out: *mut f32,
    routed_sum_out: *mut f32,
) -> i32 {
    if x.is_null()
        || gate_w.is_null()
        || up_w.is_null()
        || down_w.is_null()
        || routing_weights.is_null()
    {
        return -1;
    }
    let n_selected = n_selected.max(0) as usize;
    let x = unsafe { std::slice::from_raw_parts(x, HDIM) };
    let gate_w = unsafe { std::slice::from_raw_parts(gate_w, n_selected * 512 * HDIM) };
    let up_w = unsafe { std::slice::from_raw_parts(up_w, n_selected * 512 * HDIM) };
    let down_w = unsafe { std::slice::from_raw_parts(down_w, n_selected * HDIM * 512) };
    let routing_weights = unsafe { std::slice::from_raw_parts(routing_weights, n_selected) };

    let mut routed_sum = vec![0.0f32; HDIM];

    for i in 0..n_selected {
        let gate_w_i = &gate_w[i * 512 * HDIM..(i + 1) * 512 * HDIM];
        let up_w_i = &up_w[i * 512 * HDIM..(i + 1) * 512 * HDIM];
        let down_w_i = &down_w[i * HDIM * 512..(i + 1) * HDIM * 512];

        let gate = gemv_f32(gate_w_i, x, 512, HDIM);
        let up = gemv_f32(up_w_i, x, 512, HDIM);
        let mut hidden = vec![0.0f32; 512];
        for j in 0..512 {
            hidden[j] = gate[j] / (1.0 + (-gate[j]).exp()) * up[j];
        }
        let expert = gemv_f32(down_w_i, &hidden, HDIM, 512);
        let rw = routing_weights[i];
        let mut weighted = vec![0.0f32; HDIM];
        for j in 0..HDIM {
            weighted[j] = expert[j] * rw;
            routed_sum[j] += weighted[j];
        }

        unsafe {
            if !gate_out.is_null() {
                std::ptr::copy_nonoverlapping(gate.as_ptr(), gate_out.add(i * 512), 512);
            }
            if !up_out.is_null() {
                std::ptr::copy_nonoverlapping(up.as_ptr(), up_out.add(i * 512), 512);
            }
            if !hidden_out.is_null() {
                std::ptr::copy_nonoverlapping(hidden.as_ptr(), hidden_out.add(i * 512), 512);
            }
            if !expert_out.is_null() {
                std::ptr::copy_nonoverlapping(expert.as_ptr(), expert_out.add(i * HDIM), HDIM);
            }
            if !weighted_out.is_null() {
                std::ptr::copy_nonoverlapping(weighted.as_ptr(), weighted_out.add(i * HDIM), HDIM);
            }
        }
    }

    unsafe {
        if !routed_sum_out.is_null() {
            std::ptr::copy_nonoverlapping(routed_sum.as_ptr(), routed_sum_out, HDIM);
        }
    }

    n_selected as i32
}

/// Evaluate a single layer from a provided hidden input using the runner's current caches/state.
#[no_mangle]
pub extern "C" fn lko_runner_eval_layer_from_hidden(
    target_layer: i32,
    pos: i32,
    seq_len: i32,
    h_in: *const f32,
    h_after_attn_out: *mut f32,
    h_norm2_out: *mut f32,
    shared_out: *mut f32,
    moe_out: *mut f32,
    h_after_mlp_out: *mut f32,
) -> i32 {
    if h_in.is_null() {
        return -1;
    }
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let l = target_layer.clamp(0, 39) as usize;
    let policy = runner.policy_table[l];
    let lw = &runner.layers[l];
    let mut h = unsafe { std::slice::from_raw_parts(h_in, HDIM) }.to_vec();

    let h_norm = if !lw.input_norm.is_empty() {
        rms_norm(&h, &lw.input_norm)
    } else {
        h.clone()
    };
    let ao = match policy.attn {
        AttnPolicy::Full => {
            if policy.is_steering {
                dispatch_gqa!(runner, l, lw, &h_norm, pos as usize, seq_len as usize)
            } else {
                let mut ao = vec![0.0f32; HDIM];
                delta_net_fused(
                    &lw.w_qkv,
                    &lw.w_z,
                    &lw.w_b,
                    &lw.w_a,
                    &lw.w_o,
                    &lw.w_conv,
                    &lw.w_norm,
                    &lw.dt_bias,
                    &lw.a_log,
                    &h_norm,
                    &mut runner.conv_states[l],
                    &mut runner.conv_ptrs[l],
                    &mut runner.S_states[l],
                    &mut ao,
                    &mut runner.scratch_f32,
                    l,
                    pos as usize,
                );
                ao
            }
        }
        AttnPolicy::Collapse | AttnPolicy::Skip => vec![0.0f32; HDIM],
    };
    for i in 0..HDIM {
        h[i] += ao[i];
    }

    let h_after_attn = h.clone();
    let h_norm2 = if !lw.post_norm.is_empty() {
        rms_norm(&h, &lw.post_norm)
    } else {
        h.clone()
    };

    let mut shared = vec![0.0f32; HDIM];
    if policy.moe != MoEPolicy::Skip && !lw.se_gate.is_empty() {
        let gate = gemv_f16(&lw.se_gate, &h_norm2, 512, HDIM);
        let up = gemv_f16(&lw.se_up, &h_norm2, 512, HDIM);
        let mut hidden = gate.clone();
        for i in 0..512 {
            hidden[i] = hidden[i] / (1.0 + (-hidden[i]).exp()) * up[i];
        }
        let se_out = gemv_f16(&lw.se_down, &hidden, HDIM, 512);
        let se_gate = 1.0 / (1.0 + (-dot_f32(&lw.se_gate_w, &h_norm2)).exp());
        for i in 0..HDIM {
            shared[i] = se_out[i] * se_gate;
        }
    }
    let moe = if runner.moe_enabled && policy.moe != MoEPolicy::Skip {
        runner.call_moe(&h_norm2, l, pos as usize, 0)
    } else {
        vec![0.0f32; HDIM]
    };
    for i in 0..HDIM {
        h[i] += shared[i] + moe[i];
    }

    unsafe {
        if !h_after_attn_out.is_null() {
            std::ptr::copy_nonoverlapping(h_after_attn.as_ptr(), h_after_attn_out, HDIM);
        }
        if !h_norm2_out.is_null() {
            std::ptr::copy_nonoverlapping(h_norm2.as_ptr(), h_norm2_out, HDIM);
        }
        if !shared_out.is_null() {
            std::ptr::copy_nonoverlapping(shared.as_ptr(), shared_out, HDIM);
        }
        if !moe_out.is_null() {
            std::ptr::copy_nonoverlapping(moe.as_ptr(), moe_out, HDIM);
        }
        if !h_after_mlp_out.is_null() {
            std::ptr::copy_nonoverlapping(h.as_ptr(), h_after_mlp_out, HDIM);
        }
    }
    HDIM as i32
}

/// Run selected experts through the runner's q4 expert path for the given layer.
#[no_mangle]
pub extern "C" fn lko_runner_selected_expert_q4(
    layer_idx: i32,
    x: *const f32,
    expert_ids: *const i32,
    routing_weights: *const f32,
    n_selected: i32,
    expert_out: *mut f32,
    weighted_out: *mut f32,
    routed_sum_out: *mut f32,
) -> i32 {
    if x.is_null() || expert_ids.is_null() || routing_weights.is_null() {
        return -1;
    }
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let l = layer_idx.clamp(0, 39) as usize;
    let n_selected = n_selected.max(0) as usize;
    let x = unsafe { std::slice::from_raw_parts(x, HDIM) };
    let expert_ids = unsafe { std::slice::from_raw_parts(expert_ids, n_selected) };
    let routing_weights = unsafe { std::slice::from_raw_parts(routing_weights, n_selected) };

    let gu_addr = runner.gu_mmaps[l].as_ptr() as usize;
    let d_addr = runner.down_mmaps[l].as_ptr() as usize;
    let mut routed_sum = vec![0.0f32; HDIM];

    for i in 0..n_selected {
        let eid = expert_ids[i].max(0) as usize;
        let rw = routing_weights[i];
        let gu_off = eid * 1_310_720;
        let d_off = eid * 655_360;
        let gu_ptr = unsafe { (gu_addr as *const u8).add(gu_off) };
        let d_ptr = unsafe { (d_addr as *const u8).add(d_off) };
        let (gate, up, down) =
            crate::moe_dispatch::dequantize_expert_f32(gu_ptr, 1_310_720, d_ptr, 655_360);

        let gate_out_v = gemv_f32(&gate, x, 512, HDIM);
        let up_out_v = gemv_f32(&up, x, 512, HDIM);
        let mut hidden = vec![0.0f32; 512];
        for j in 0..512 {
            hidden[j] = gate_out_v[j] / (1.0 + (-gate_out_v[j]).exp()) * up_out_v[j];
        }
        let expert_v = gemv_f32(&down, &hidden, HDIM, 512);
        let mut weighted_v = vec![0.0f32; HDIM];
        for j in 0..HDIM {
            weighted_v[j] = expert_v[j] * rw;
            routed_sum[j] += weighted_v[j];
        }

        unsafe {
            if !expert_out.is_null() {
                std::ptr::copy_nonoverlapping(expert_v.as_ptr(), expert_out.add(i * HDIM), HDIM);
            }
            if !weighted_out.is_null() {
                std::ptr::copy_nonoverlapping(
                    weighted_v.as_ptr(),
                    weighted_out.add(i * HDIM),
                    HDIM,
                );
            }
        }
    }

    unsafe {
        if !routed_sum_out.is_null() {
            std::ptr::copy_nonoverlapping(routed_sum.as_ptr(), routed_sum_out, HDIM);
        }
    }
    n_selected as i32
}
