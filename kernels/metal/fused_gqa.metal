// objeta — Fused GQA Attention Metal Kernel v2
//
// Single-pass: QKV projection (f16 weights) + RoPE + KV write +
//              online softmax + V sum + Q-gate
// Separate pass: O-proj (f16 weights) → final 2048-dim output

#include <metal_stdlib>
using namespace metal;

constant uint N_Q   = 16;
constant uint N_KV  = 2;
constant uint HD    = 256;
constant uint REP   = 8;
constant uint D     = 2048;
constant uint Q_SZ  = 8192;  // Q(4096) + Q-gate(4096)
constant uint K_SZ  = 512;

// ── Stage 1: QKV + RoPE + online softmax + V sum + Q-gate ─────────

kernel void fused_gqa(
    device const half*  W_qkv   [[buffer(0)]],  // f16: [9216, 2048]
    device const float* h       [[buffer(1)]],
    device float*       k_cache [[buffer(2)]],
    device float*       v_cache [[buffer(3)]],
    device const float* cos_tab [[buffer(4)]],
    device const float* sin_tab [[buffer(5)]],
    constant uint&     pos      [[buffer(6)]],
    constant uint&     seq_len  [[buffer(7)]],
    constant uint&     max_seq  [[buffer(8)]],
    device float*       attn_out [[buffer(9)]],
    uint               q_head   [[threadgroup_position_in_grid]],
    uint               tid      [[thread_position_in_threadgroup]],
    threadgroup float* tg       [[threadgroup(0)]]
)
{
    uint kv_head = q_head / REP;
    uint hd2     = HD / 2;

    // ── QKV projection (f16 weights → f32 accum) ──
    uint q_row  = q_head * HD + tid;
    uint qg_row = N_Q * HD + q_head * HD + tid;
    float q_val = 0.0, q_gate_val = 0.0;
    for (uint j = 0; j < D; j++) {
        float hj = h[j];
        q_val      += float(W_qkv[q_row  * D + j]) * hj;
        q_gate_val += float(W_qkv[qg_row * D + j]) * hj;
    }

    uint k_row = Q_SZ + kv_head * HD + tid;
    float k_val = 0.0;
    for (uint j = 0; j < D; j++) { k_val += float(W_qkv[k_row * D + j]) * h[j]; }

    uint v_row = Q_SZ + K_SZ + kv_head * HD + tid;
    float v_val = 0.0;
    for (uint j = 0; j < D; j++) { v_val += float(W_qkv[v_row * D + j]) * h[j]; }

    // ── RoPE: share Q values via tg, compute rotation ──
    tg[tid] = q_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float cos_v = cos_tab[pos * hd2 + (tid % hd2)];
    float sin_v = sin_tab[pos * hd2 + (tid % hd2)];
    float q_rot;

    if (tid < hd2) {
        q_rot = q_val * cos_v - tg[tid + hd2] * sin_v;
    } else {
        q_rot = tg[tid - hd2] * sin_v + q_val * cos_v;
    }

    // ── RoPE for K ──
    tg[tid] = k_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_rot;
    if (tid < hd2) {
        k_rot = k_val * cos_v - tg[tid + hd2] * sin_v;
    } else {
        k_rot = tg[tid - hd2] * sin_v + k_val * cos_v;
    }

    // ── Write KV cache: [kv_head][pos][dim] ──
    uint kv_off = kv_head * max_seq * HD + pos * HD + tid;
    k_cache[kv_off] = k_rot;
    v_cache[kv_off] = v_val;

    // ── Online softmax attention ──
    float scale = 1.0f / sqrt(float(HD));
    float max_score = -INFINITY;
    float sum_exp   = 0.0f;
    float weighted_v = 0.0f;

    for (uint t = 0; t < seq_len; t++) {
        float k_td = k_cache[kv_head * max_seq * HD + t * HD + tid];
        tg[tid] = q_rot * k_td;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float score = 0.0f;
            for (uint d = 0; d < HD; d++) score += tg[d];
            tg[HD] = score * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = tg[HD];
        float new_max = max(max_score, score);
        float old_scale = exp(max_score - new_max);
        float exp_val   = exp(score - new_max);
        sum_exp = sum_exp * old_scale + exp_val;
        max_score = new_max;
        weighted_v = weighted_v * old_scale + exp_val * v_cache[kv_head * max_seq * HD + t * HD + tid];
    }

    float attn_dim = weighted_v / max(sum_exp, 1e-12f);

    // ── Q-gate ──
    float gate = 1.0f / (1.0f + exp(-q_gate_val));
    attn_out[q_head * HD + tid] = attn_dim * gate;
}

// ── Stage 2: O-proj (f16 weights) ──────────────────────────────────

kernel void gqa_oproj_f16(
    device const half*  W_o     [[buffer(0)]],  // f16: [2048, 4096]
    device const float* attn    [[buffer(1)]],  // f32: [4096]
    device float*       output  [[buffer(2)]],  // f32: [2048]
    uint                row     [[thread_position_in_grid]]
)
{
    float sum = 0.0;
    for (uint j = 0; j < 4096; j++) {
        sum += float(W_o[row * 4096 + j]) * attn[j];
    }
    output[row] = sum;
}
