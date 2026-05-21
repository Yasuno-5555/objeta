use std::path::Path;
use std::ffi::CStr;
use std::os::raw::{c_char, c_float};
use std::io::BufRead;

use crate::qwen36_forward::{
    Qwen36Runner, HDIM, HEAD_DIM, EXPERT_TOTAL_BYTES, RuntimeConfigSource,
    build_policy_table, AttnPolicy, MoEPolicy, LayerPolicy,
    rms_norm, rms_norm_offset, gqa_metal_try, gqa_attention_fused, delta_net_fused,
    dot_f32, gemv_f32, gemv_f16, silu_inplace, softmax_inplace, l2_norm_rows,
    delta_state_update, rms_norm_gated, sigmoid_inplace,
};
use crate::moe_stats::{MoELayerStats, ForwardLayerStats, StepTrace};

pub static mut RUNNER: Option<Qwen36Runner> = None;

#[inline]
fn objeta_debug_enabled() -> bool {
    std::env::var("OBJETA_DEBUG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// FP32 GEMV: y = W @ x. W is (M, K) row-major f32.
#[no_mangle]
pub extern "C" fn lko_q36_f32_gemv(
    w: *const f32,
    m: i32,
    k: i32,
    x: *const f32,
    y: *mut f32,
) -> i32 {
    let w_slice = unsafe { std::slice::from_raw_parts(w, (m * k) as usize) };
    let x_slice = unsafe { std::slice::from_raw_parts(x, k as usize) };
    let r = gemv_f32(w_slice, x_slice, m as usize, k as usize);
    unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), y, m as usize) };
    m
}

/// FP16 GEMV: y = W @ x. W is (M, K) row-major f16.
#[no_mangle]
pub extern "C" fn lko_q36_f16_gemv(
    w: *const u16,
    m: i32,
    k: i32,
    x: *const f32,
    y: *mut f32,
) -> i32 {
    let w_slice = unsafe { std::slice::from_raw_parts(w, (m * k) as usize) };
    let x_slice = unsafe { std::slice::from_raw_parts(x, k as usize) };
    let r = gemv_f16(w_slice, x_slice, m as usize, k as usize);
    unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), y, m as usize) };
    m
}

/// RMSNorm: x = RMSNorm(x, weight), in-place.
#[no_mangle]
pub extern "C" fn lko_q36_rms_norm(x: *mut f32, weight: *const f32, n: i32) -> i32 {
    let x_slice = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let w_slice = unsafe { std::slice::from_raw_parts(weight, n as usize) };
    let r = rms_norm(x_slice, w_slice);
    unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), x, n as usize) };
    n
}

#[no_mangle]
pub extern "C" fn lko_q36_softmax(x: *mut f32, rows: i32, dim: i32) -> i32 {
    let n = (rows * dim) as usize;
    let x_slice = unsafe { std::slice::from_raw_parts_mut(x, n) };
    softmax_inplace(x_slice, dim as usize);
    n as i32
}

#[no_mangle]
pub extern "C" fn lko_q36_silu(x: *mut f32, n: i32) -> i32 {
    let x_slice = unsafe { std::slice::from_raw_parts_mut(x, n as usize) };
    silu_inplace(x_slice);
    n
}

#[no_mangle]
pub extern "C" fn lko_q36_l2_norm(x: *mut f32, rows: i32, dim: i32, with_scale: i32) -> i32 {
    let x_slice = unsafe { std::slice::from_raw_parts_mut(x, (rows * dim) as usize) };
    let scale = 1.0 / (dim as f32).sqrt();
    for row in x_slice.chunks_mut(dim as usize) {
        let sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sq + 1e-6).sqrt();
        let s = if with_scale != 0 { inv * scale } else { inv };
        for v in row.iter_mut() {
            *v *= s;
        }
    }
    rows * dim
}

#[no_mangle]
pub extern "C" fn lko_q36_delta_update(
    s_ptr: *mut f32,
    k: *const f32,
    q: *const f32,
    v: *const f32,
    beta: *const f32,
    exp_g: *const f32,
    n_heads: i32,
    kv_dim: i32,
    v_dim: i32,
    output: *mut f32,
) -> i32 {
    let nh = n_heads as usize;
    let kd = kv_dim as usize;
    let vd = v_dim as usize;
    let s_mut = unsafe { std::slice::from_raw_parts_mut(s_ptr, nh * kd * vd) };
    let k_slice = unsafe { std::slice::from_raw_parts(k, nh * kd) };
    let q_slice = unsafe { std::slice::from_raw_parts(q, nh * kd) };
    let v_slice = unsafe { std::slice::from_raw_parts(v, nh * vd) };
    let beta_slice = unsafe { std::slice::from_raw_parts(beta, nh) };
    let g_slice = unsafe { std::slice::from_raw_parts(exp_g, nh) };
    let out = unsafe { std::slice::from_raw_parts_mut(output, nh * vd) };

    delta_state_update(
        s_mut, k_slice, q_slice, v_slice, beta_slice, g_slice, nh, kd, vd, out,
    );
    (nh * vd) as i32
}

#[no_mangle]
pub extern "C" fn lko_q36_rms_norm_gated(
    output: *const f32,
    z: *const f32,
    w_norm: *const f32,
    n_heads: i32,
    v_dim: i32,
    gated_out: *mut f32,
) -> i32 {
    let nh = n_heads as usize;
    let vd = v_dim as usize;
    let o = unsafe { std::slice::from_raw_parts(output, nh * vd) };
    let zs = unsafe { std::slice::from_raw_parts(z, nh * vd) };
    let wn = unsafe { std::slice::from_raw_parts(w_norm, vd) };
    let r = rms_norm_gated(o, zs, wn, nh, vd);
    unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), gated_out, r.len()) };
    r.len() as i32
}

#[no_mangle]
pub extern "C" fn lko_q36_sigmoid(x: *mut f32, n: i32) -> i32 {
    let x_slice = unsafe { std::slice::from_raw_parts_mut(x, n as usize) };
    sigmoid_inplace(x_slice);
    n
}

/// One fused layer forward call.
/// Returns attention output in `ao_out` (HDIM f32).
/// Updates kv_cache or delta_state in place.
#[no_mangle]
pub extern "C" fn lko_q36_fused_layer(
    _w_ptr: *const f32,
    _w_sizes: *const i32,
    _n_mats: i32,
    _h: *const f32,
    _conv_state: *mut f32,
    _conv_ptr: *mut i32,
    _s_state: *mut f32,
    _k_cache: *mut f32,
    _v_cache: *mut f32,
    _rope_cos: *const f32,
    _rope_sin: *const f32,
    _pos: i32,
    _seq_len: i32,
    _max_seq: i32,
    _layer_type: i32,
    _ao_out: *mut f32,
) -> i32 {
    -1
}

/// Temporary compatibility FFI. Avoid using in multi-runner or daemon setups.
#[no_mangle]
pub extern "C" fn lko_runner_get_instance() -> *mut Qwen36Runner {
    unsafe {
        match &mut RUNNER {
            Some(r) => r as *mut Qwen36Runner,
            None => std::ptr::null_mut(),
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_init_page_cache(
    runner: *mut Qwen36Runner,
    capacity_bytes: i64,
) -> i32 {
    if runner.is_null() {
        return 0;
    }
    unsafe {
        (&mut *runner).init_page_cache(capacity_bytes as u64);
    }
    1
}

#[no_mangle]
pub extern "C" fn lko_moe_init_page_cache(capacity_bytes: i64) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            r.init_page_cache(capacity_bytes as u64);
            1
        } else {
            0
        }
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

#[no_mangle]
pub extern "C" fn lko_runner_load_runtime_profile(path_ptr: *const std::os::raw::c_char) -> i32 {
    if path_ptr.is_null() {
        return 0;
    }
    let runner = unsafe { RUNNER.as_mut() };
    match runner {
        Some(runner) => {
            let c_str = unsafe { std::ffi::CStr::from_ptr(path_ptr) };
            match c_str.to_str() {
                Ok(path) => match crate::runtime_profile::load_runtime_profile(Path::new(path)) {
                    Ok(profile) => {
                        crate::runtime_profile::apply_runtime_profile(runner, &profile);
                        1
                    }
                    Err(_) => 0,
                },
                Err(_) => 0,
            }
        }
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_load_runtime_pack(
    runner: *mut Qwen36Runner,
    path_ptr: *const std::os::raw::c_char,
) -> i32 {
    if runner.is_null() || path_ptr.is_null() {
        return 0;
    }
    let runner = unsafe { &mut *runner };
    let c_str = unsafe { std::ffi::CStr::from_ptr(path_ptr) };
    match c_str.to_str() {
        Ok(path) => runner.load_runtime_pack(Path::new(path)).map(|_| 1).unwrap_or(0),
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_auto_tune_runtime(
    output_dir_ptr: *const std::os::raw::c_char,
    max_candidates: i32,
) -> i32 {
    if output_dir_ptr.is_null() {
        return 0;
    }
    let c_str = unsafe { std::ffi::CStr::from_ptr(output_dir_ptr) };
    match c_str.to_str() {
        Ok(path) => {
            crate::runtime_tuner::auto_tune_default(Path::new(path), max_candidates.max(1) as usize)
                .map(|_| 1)
                .unwrap_or(0)
        }
        Err(_) => 0,
    }
}

/// Set DeltaNet fusion ratio: 1.0 = all layers (default), 0.33 = 1 per GQA block.
#[no_mangle]
pub extern "C" fn lko_runner_set_fusion_ratio(ratio: f64) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                r.fusion_ratio = ratio.clamp(0.0, 1.0);
                r.policy_table = build_policy_table(r.fusion_ratio, r.moe_on_deltanet);
                r.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
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
                r.moe_on_deltanet = enabled != 0;
                r.policy_table = build_policy_table(r.fusion_ratio, r.moe_on_deltanet);
                r.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
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
            rms_norm_offset(&h, &lw.input_norm)
        } else {
            h.clone()
        };
        let ao = match policy.attn {
            AttnPolicy::Full => {
                if policy.is_steering {
                    if runner.metal_gqa_ok {
                        if let Some(ao) = gqa_metal_try(
                            l,
                            &lw.w_qkv,
                            &lw.w_o,
                            &h_norm,
                            pos as u32,
                            seq_len as u32,
                            runner.max_seq as u32,
                            &mut runner.kv_k[l],
                            &mut runner.kv_v[l],
                            &mut runner.metal_gqa_first_fail,
                        ) {
                            ao
                        } else {
                            let mut ao = vec![0.0f32; HDIM];
                            gqa_attention_fused(
                                &lw.w_qkv,
                                &lw.w_o,
                                &lw.q_norm,
                                &lw.k_norm,
                                &h_norm,
                                &mut runner.kv_k[l],
                                &mut runner.kv_v[l],
                                &runner.rope_cos,
                                &runner.rope_sin,
                                16,
                                2,
                                HEAD_DIM,
                                pos as usize,
                                seq_len as usize,
                                runner.max_seq,
                                &mut ao,
                                &mut runner.scratch_qkv,
                                &mut runner.scratch_q,
                                &mut runner.scratch_k,
                                &mut runner.scratch_v,
                                &mut runner.scratch_attn_out,
                                &mut runner.scratch_scores,
                                &mut runner.scratch_attn,
                            );
                            ao
                        }
                    } else {
                        let mut ao = vec![0.0f32; HDIM];
                        gqa_attention_fused(
                            &lw.w_qkv,
                            &lw.w_o,
                            &lw.q_norm,
                            &lw.k_norm,
                            &h_norm,
                            &mut runner.kv_k[l],
                            &mut runner.kv_v[l],
                            &runner.rope_cos,
                            &runner.rope_sin,
                            16,
                            2,
                            HEAD_DIM,
                            pos as usize,
                            seq_len as usize,
                            runner.max_seq,
                            &mut ao,
                            &mut runner.scratch_qkv,
                            &mut runner.scratch_q,
                            &mut runner.scratch_k,
                            &mut runner.scratch_v,
                            &mut runner.scratch_attn_out,
                            &mut runner.scratch_scores,
                            &mut runner.scratch_attn,
                        );
                        ao
                    }
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
            rms_norm_offset(&h, &lw.post_norm)
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
            let moe_out = runner.call_moe(&h_norm2, l);
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

#[no_mangle]
pub extern "C" fn lko_runner_lm_head(
    hn: *const f32,
    top_k: i32,
    indices_out: *mut i32,
    values_out: *mut f32,
) -> i32 {
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
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
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    runner.begin_token_residency(token_id as usize);
    let (h, _timing) = runner.forward_timed(token_id as usize, pos as usize, seq_len as usize);

    // RMSNorm
    let hn = rms_norm(&h, &runner.final_norm);
    unsafe {
        std::ptr::copy_nonoverlapping(hn.as_ptr(), hn_out, HDIM);
    }

    // lm_head top-k
    let (indices, values) = runner.lm_head_topk(&hn, top_k as usize);
    let k = indices.len().min(top_k as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(indices.as_ptr(), indices_out, k);
        std::ptr::copy_nonoverlapping(values.as_ptr(), values_out, k);
    }
    runner.finish_step();
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
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    runner.begin_token_residency(token_id as usize);
    let (h, _timing) = runner.forward_timed(token_id as usize, pos as usize, seq_len as usize);

    // RMSNorm
    let hn = rms_norm(&h, &runner.final_norm);
    unsafe {
        std::ptr::copy_nonoverlapping(hn.as_ptr(), hn_out, HDIM);
    }

    // lm_head top-k with entropy
    let (indices, values, entropy) = runner.lm_head_topk_with_entropy(&hn, top_k as usize);
    let k = indices.len().min(top_k as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(indices.as_ptr(), indices_out, k);
        std::ptr::copy_nonoverlapping(values.as_ptr(), values_out, k);
        *entropy_out = entropy;
    }
    runner.finish_step();
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

/// Forward pass through only the first N layers, tracing intermediate layer hidden states.
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
            rms_norm_offset(&h, &lw.input_norm)
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
                    if runner.metal_gqa_ok {
                        if let Some(ao) = gqa_metal_try(
                            l,
                            &lw.w_qkv,
                            &lw.w_o,
                            &h_norm,
                            pos as u32,
                            seq_len as u32,
                            runner.max_seq as u32,
                            &mut runner.kv_k[l],
                            &mut runner.kv_v[l],
                            &mut runner.metal_gqa_first_fail,
                        ) {
                            ao
                        } else {
                            let mut ao = vec![0.0f32; HDIM];
                            gqa_attention_fused(
                                &lw.w_qkv,
                                &lw.w_o,
                                &lw.q_norm,
                                &lw.k_norm,
                                &h_norm,
                                &mut runner.kv_k[l],
                                &mut runner.kv_v[l],
                                &runner.rope_cos,
                                &runner.rope_sin,
                                16,
                                2,
                                HEAD_DIM,
                                pos as usize,
                                seq_len as usize,
                                runner.max_seq,
                                &mut ao,
                                &mut runner.scratch_qkv,
                                &mut runner.scratch_q,
                                &mut runner.scratch_k,
                                &mut runner.scratch_v,
                                &mut runner.scratch_attn_out,
                                &mut runner.scratch_scores,
                                &mut runner.scratch_attn,
                            );
                            ao
                        }
                    } else {
                        let mut ao = vec![0.0f32; HDIM];
                        gqa_attention_fused(
                            &lw.w_qkv,
                            &lw.w_o,
                            &lw.q_norm,
                            &lw.k_norm,
                            &h_norm,
                            &mut runner.kv_k[l],
                            &mut runner.kv_v[l],
                            &runner.rope_cos,
                            &runner.rope_sin,
                            16,
                            2,
                            HEAD_DIM,
                            pos as usize,
                            seq_len as usize,
                            runner.max_seq,
                            &mut ao,
                            &mut runner.scratch_qkv,
                            &mut runner.scratch_q,
                            &mut runner.scratch_k,
                            &mut runner.scratch_v,
                            &mut runner.scratch_attn_out,
                            &mut runner.scratch_scores,
                            &mut runner.scratch_attn,
                        );
                        ao
                    }
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
            rms_norm_offset(&h, &lw.post_norm)
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
            let moe_out = runner.call_moe(&h_norm2, l);
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
            rms_norm_offset(&h, &lw.input_norm)
        } else {
            h.clone()
        };
        let ao = match policy.attn {
            AttnPolicy::Full => {
                if policy.is_steering {
                    if runner.metal_gqa_ok {
                        if let Some(ao) = gqa_metal_try(
                            l,
                            &lw.w_qkv,
                            &lw.w_o,
                            &h_norm,
                            pos as u32,
                            seq_len as u32,
                            runner.max_seq as u32,
                            &mut runner.kv_k[l],
                            &mut runner.kv_v[l],
                            &mut runner.metal_gqa_first_fail,
                        ) {
                            ao
                        } else {
                            let mut ao = vec![0.0f32; HDIM];
                            gqa_attention_fused(
                                &lw.w_qkv,
                                &lw.w_o,
                                &lw.q_norm,
                                &lw.k_norm,
                                &h_norm,
                                &mut runner.kv_k[l],
                                &mut runner.kv_v[l],
                                &runner.rope_cos,
                                &runner.rope_sin,
                                16,
                                2,
                                HEAD_DIM,
                                pos as usize,
                                seq_len as usize,
                                runner.max_seq,
                                &mut ao,
                                &mut runner.scratch_qkv,
                                &mut runner.scratch_q,
                                &mut runner.scratch_k,
                                &mut runner.scratch_v,
                                &mut runner.scratch_attn_out,
                                &mut runner.scratch_scores,
                                &mut runner.scratch_attn,
                            );
                            ao
                        }
                    } else {
                        let mut ao = vec![0.0f32; HDIM];
                        gqa_attention_fused(
                            &lw.w_qkv,
                            &lw.w_o,
                            &lw.q_norm,
                            &lw.k_norm,
                            &h_norm,
                            &mut runner.kv_k[l],
                            &mut runner.kv_v[l],
                            &runner.rope_cos,
                            &runner.rope_sin,
                            16,
                            2,
                            HEAD_DIM,
                            pos as usize,
                            seq_len as usize,
                            runner.max_seq,
                            &mut ao,
                            &mut runner.scratch_qkv,
                            &mut runner.scratch_q,
                            &mut runner.scratch_k,
                            &mut runner.scratch_v,
                            &mut runner.scratch_attn_out,
                            &mut runner.scratch_scores,
                            &mut runner.scratch_attn,
                        );
                        ao
                    }
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
            rms_norm_offset(&h, &lw.post_norm)
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
            runner.call_moe(&h_norm2, l)
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

#[no_mangle]
pub extern "C" fn lko_runner_set_force_attn_full(enabled: i32) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            r.debug_force_attn_full = enabled != 0;
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_force_moe_skip(enabled: i32) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            r.debug_force_moe_skip = enabled != 0;
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_trace_record(path: *const std::os::raw::c_char) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            if path.is_null() {
                r.record_trace_path = None;
            } else {
                let c_str = std::ffi::CStr::from_ptr(path);
                if let Ok(s) = c_str.to_str() {
                    r.record_trace_path = Some(s.to_string());
                } else {
                    return 0;
                }
            }
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_trace_replay(path: *const std::os::raw::c_char) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            if path.is_null() {
                r.replay_traces = None;
            } else {
                let c_str = std::ffi::CStr::from_ptr(path);
                if let Ok(s) = c_str.to_str() {
                    if let Ok(file) = std::fs::File::open(s) {
                        let reader = std::io::BufReader::new(file);
                        let mut traces = Vec::new();
                        for line in reader.lines() {
                            if let Ok(line_str) = line {
                                if let Ok(trace) = serde_json::from_str::<StepTrace>(&line_str) {
                                    traces.push(trace);
                                }
                            }
                        }
                        r.replay_traces = Some(traces);
                    } else {
                        return 0;
                    }
                } else {
                    return 0;
                }
            }
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_moe_top_p(p: f32) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            r.moe_top_p = p;
            let p_val = p.clamp(0.0, 1.0);
            r.set_expert_policy(crate::strategy::ExpertPolicyConfig::TopP {
                p: p_val,
                min_experts: r.min_experts.max(1),
                max_experts: r.max_experts.max(r.min_experts.max(1)),
            });
            r.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_moe_prune_mode(mode: i32) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            r.moe_prune_mode = mode;
            if mode == 1 {
                r.set_expert_policy(crate::strategy::ExpertPolicyConfig::Contribution {
                    threshold: r.moe_contrib_threshold.clamp(0.0, 1.0),
                    min_experts: r.min_experts.max(1),
                    max_experts: r.max_experts.max(r.min_experts.max(1)),
                    ema_beta: 0.95,
                });
            } else {
                r.set_expert_policy(crate::strategy::ExpertPolicyConfig::TopP {
                    p: r.moe_top_p.clamp(0.0, 1.0),
                    min_experts: r.min_experts.max(1),
                    max_experts: r.max_experts.max(r.min_experts.max(1)),
                });
            }
            r.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_moe_contrib_threshold(threshold: f32) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            r.moe_contrib_threshold = threshold;
            let t_val = threshold.clamp(0.0, 1.0);
            r.set_expert_policy(crate::strategy::ExpertPolicyConfig::Contribution {
                threshold: t_val,
                min_experts: r.min_experts.max(1),
                max_experts: r.max_experts.max(r.min_experts.max(1)),
                ema_beta: 0.95,
            });
            r.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_moe_min_experts(min_experts: i32) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            r.min_experts = min_experts as usize;
            let min_e = min_experts.max(1) as usize;
            let max_e = r.max_experts.max(min_e);
            let new_policy = match &r.expert_policy {
                crate::strategy::ExpertPolicyConfig::Exact => {
                    crate::strategy::ExpertPolicyConfig::Exact
                }
                crate::strategy::ExpertPolicyConfig::LkoAware => {
                    crate::strategy::ExpertPolicyConfig::LkoAware
                }
                crate::strategy::ExpertPolicyConfig::TopP { p, .. } => {
                    crate::strategy::ExpertPolicyConfig::TopP {
                        p: *p,
                        min_experts: min_e,
                        max_experts: max_e,
                    }
                }
                crate::strategy::ExpertPolicyConfig::Contribution {
                    threshold,
                    ema_beta,
                    ..
                } => crate::strategy::ExpertPolicyConfig::Contribution {
                    threshold: *threshold,
                    min_experts: min_e,
                    max_experts: max_e,
                    ema_beta: *ema_beta,
                },
                crate::strategy::ExpertPolicyConfig::AdaptiveEntropy {
                    low_entropy_p,
                    mid_entropy_p,
                    high_entropy_p,
                    repetition_p,
                    low_entropy_threshold,
                    mid_entropy_threshold,
                    ..
                } => crate::strategy::ExpertPolicyConfig::AdaptiveEntropy {
                    low_entropy_p: *low_entropy_p,
                    mid_entropy_p: *mid_entropy_p,
                    high_entropy_p: *high_entropy_p,
                    repetition_p: *repetition_p,
                    low_entropy_threshold: *low_entropy_threshold,
                    mid_entropy_threshold: *mid_entropy_threshold,
                    min_experts: min_e,
                    max_experts: max_e,
                },
            };
            r.set_expert_policy(new_policy);
            r.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_moe_max_experts(max_experts: i32) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            r.max_experts = max_experts as usize;
            let min_e = r.min_experts.max(1);
            let max_e = (max_experts.max(1) as usize).max(min_e);
            let new_policy = match &r.expert_policy {
                crate::strategy::ExpertPolicyConfig::Exact => {
                    crate::strategy::ExpertPolicyConfig::Exact
                }
                crate::strategy::ExpertPolicyConfig::LkoAware => {
                    crate::strategy::ExpertPolicyConfig::LkoAware
                }
                crate::strategy::ExpertPolicyConfig::TopP { p, .. } => {
                    crate::strategy::ExpertPolicyConfig::TopP {
                        p: *p,
                        min_experts: min_e,
                        max_experts: max_e,
                    }
                }
                crate::strategy::ExpertPolicyConfig::Contribution {
                    threshold,
                    ema_beta,
                    ..
                } => crate::strategy::ExpertPolicyConfig::Contribution {
                    threshold: *threshold,
                    min_experts: min_e,
                    max_experts: max_e,
                    ema_beta: *ema_beta,
                },
                crate::strategy::ExpertPolicyConfig::AdaptiveEntropy {
                    low_entropy_p,
                    mid_entropy_p,
                    high_entropy_p,
                    repetition_p,
                    low_entropy_threshold,
                    mid_entropy_threshold,
                    ..
                } => crate::strategy::ExpertPolicyConfig::AdaptiveEntropy {
                    low_entropy_p: *low_entropy_p,
                    mid_entropy_p: *mid_entropy_p,
                    high_entropy_p: *high_entropy_p,
                    repetition_p: *repetition_p,
                    low_entropy_threshold: *low_entropy_threshold,
                    mid_entropy_threshold: *mid_entropy_threshold,
                    min_experts: min_e,
                    max_experts: max_e,
                },
            };
            r.set_expert_policy(new_policy);
            r.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_set_expert_policy_json(json_ptr: *const std::os::raw::c_char) -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            if json_ptr.is_null() {
                r.set_expert_policy(crate::strategy::ExpertPolicyConfig::Exact);
                r.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
                1
            } else {
                let c_str = std::ffi::CStr::from_ptr(json_ptr);
                if let Ok(s) = c_str.to_str() {
                    if let Ok(policy) = crate::strategy::parse_expert_policy_json(s) {
                        r.set_expert_policy(policy);
                        r.note_runtime_config_source(RuntimeConfigSource::StrategyConfig);
                        1
                    } else {
                        0
                    }
                } else {
                    0
                }
            }
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_reset_moe_stats() -> i32 {
    unsafe {
        if let Some(r) = RUNNER.as_mut() {
            for s in &mut r.moe_stats {
                *s = MoELayerStats::default();
            }
            for s in &mut r.forward_stats {
                *s = ForwardLayerStats::default();
            }
            r.lm_head_calls = 0;
            r.lm_head_wall_sec = 0.0;
            r.forward_calls = 0;
            r.forward_wall_sec = 0.0;
            r.moe_io_events.clear();
            r.runtime_governor.reset_counters();
            r.os_telemetry.reset();
            r.decode_started = false;
            r.decode_token_count = 0;
            r.current_governor_phase = crate::runtime_governor::GovernorPhase::Prefill;
            r.last_repetition_kind = None;
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn lko_runner_get_moe_stats_json() -> *mut std::os::raw::c_char {
    unsafe {
        match &RUNNER {
            Some(r) => {
                let json = r.get_moe_stats_json();
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

#[no_mangle]
pub extern "C" fn lko_runner_set_use_fused_moe(enabled: i32) -> i32 {
    unsafe {
        match &mut RUNNER {
            Some(r) => {
                r.use_fused_moe = enabled != 0;
                1
            }
            None => 0,
        }
    }
}

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

#[no_mangle]
pub extern "C" fn lko_runner_selected_expert_q4_fused(
    layer_idx: i32,
    x: *const f32,
    expert_ids: *const i32,
    routing_weights: *const f32,
    n_selected: i32,
    routed_sum_out: *mut f32,
) -> i32 {
    if x.is_null() || expert_ids.is_null() || routing_weights.is_null() || routed_sum_out.is_null()
    {
        return -1;
    }
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let l = layer_idx.clamp(0, 39) as usize;
    let n_selected = n_selected.max(0) as usize;
    let x_slice = unsafe { std::slice::from_raw_parts(x, HDIM) };
    let expert_ids_slice = unsafe { std::slice::from_raw_parts(expert_ids, n_selected) };
    let routing_weights_slice = unsafe { std::slice::from_raw_parts(routing_weights, n_selected) };

    let eidx: Vec<usize> = expert_ids_slice.iter().map(|&id| id as usize).collect();

    let out = crate::moe_dispatch::fused_moe_q4_selected_v0(
        &runner.gu_mmaps[l],
        &runner.down_mmaps[l],
        x_slice,
        &eidx,
        routing_weights_slice,
    );

    unsafe {
        std::ptr::copy_nonoverlapping(out.as_ptr(), routed_sum_out, HDIM);
    }
    n_selected as i32
}

#[no_mangle]
pub extern "C" fn lko_runner_selected_expert_q4_path(
    layer_idx: i32,
    x: *const f32,
    expert_ids: *const i32,
    routing_weights: *const f32,
    n_selected: i32,
    use_fused: i32,
    down_mode_kind: i32,
    chunk_rows: i32,
    routed_sum_out: *mut f32,
) -> i32 {
    if x.is_null() || expert_ids.is_null() || routing_weights.is_null() || routed_sum_out.is_null()
    {
        return -1;
    }
    let runner = unsafe { RUNNER.as_mut() }.expect("runner not initialized");
    let l = layer_idx.clamp(0, 39) as usize;
    let n_selected = n_selected.max(0) as usize;
    let x_slice = unsafe { std::slice::from_raw_parts(x, HDIM) };
    let expert_ids_slice = unsafe { std::slice::from_raw_parts(expert_ids, n_selected) };
    let routing_weights_slice = unsafe { std::slice::from_raw_parts(routing_weights, n_selected) };
    let eidx: Vec<usize> = expert_ids_slice.iter().map(|&id| id as usize).collect();
    let mode = match down_mode_kind {
        0 => crate::moe_dispatch::FusedDownMode::Serial,
        2 => crate::moe_dispatch::FusedDownMode::Chunked(chunk_rows.max(1) as usize),
        _ => crate::moe_dispatch::FusedDownMode::RowParallel,
    };
    let res = runner.execute_selected_moe(
        x_slice,
        l,
        &eidx,
        routing_weights_slice,
        use_fused != 0,
        mode,
    );
    let out = res.output;
    unsafe {
        std::ptr::copy_nonoverlapping(out.as_ptr(), routed_sum_out, HDIM);
    }
    n_selected as i32
}
