// objeta — FP16 GEMV + Attention Metal Kernels
//
// Replaces MLX matmul + softmax with native Metal dispatch.
// Weights are fp16 (stored in mmap), inputs/outputs are fp32.

#include <metal_stdlib>
using namespace metal;

// ── FP16 GEMV ─────────────────────────────────────────────────────────────
// y = W @ x  where W is fp16 (M×K), x is fp32 (K), y is fp32 (M)
// Each thread computes one output element.

kernel void fp16_gemv(
    device const half*  w       [[buffer(0)]],  // (M, K) row-major fp16
    device const float* x       [[buffer(1)]],  // (K,) fp32
    device float*       y       [[buffer(2)]],  // (M,) fp32
    constant uint2&    dims    [[buffer(3)]],  // (M, K)
    uint               tid     [[thread_position_in_grid]])
{
    uint M = dims.x;
    uint K = dims.y;
    if (tid >= M) return;

    float sum = 0.0;
    uint row_off = tid * K;
    for (uint j = 0; j < K; j++) {
        sum += float(w[row_off + j]) * x[j];
    }
    y[tid] = sum;
}

// ── Fused QKV Projection ──────────────────────────────────────────────────
// Projects x to Q, K, V in one kernel using a combined weight matrix.
// w_qkv: (Q_dim+K_dim+V_dim, K) row-major fp16
// Output: qkv = (Q_dim+K_dim+V_dim,) fp32

kernel void qkv_projection(
    device const half*  w_qkv   [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       qkv     [[buffer(2)]],
    constant uint2&    dims    [[buffer(3)]],  // (total_out, K)
    uint               tid     [[thread_position_in_grid]])
{
    uint M = dims.x;
    uint K = dims.y;
    if (tid >= M) return;

    float sum = 0.0;
    uint row_off = tid * K;
    for (uint j = 0; j < K; j++) {
        sum += float(w_qkv[row_off + j]) * x[j];
    }
    qkv[tid] = sum;
}

// ── GQA Attention ─────────────────────────────────────────────────────────
// Computes attention output for one token.
// Q: (n_heads, head_dim) fp32
// K_cache: (n_kv_heads, seq_len, head_dim) fp32
// V_cache: (n_kv_heads, seq_len, head_dim) fp32
// Output: (n_heads * head_dim,) fp32
//
// Thread grid: (n_heads, head_dim) — each thread computes one output element.
// Threadgroup memory: used for softmax reduction per head.

kernel void gqa_attention(
    device const float* q         [[buffer(0)]],  // (n_heads, head_dim)
    device const float* k_cache   [[buffer(1)]],  // (n_kv_heads, seq_len, head_dim)
    device const float* v_cache   [[buffer(2)]],  // (n_kv_heads, seq_len, head_dim)
    device float*       output    [[buffer(3)]],  // (n_heads * head_dim)
    constant uint&     n_heads   [[buffer(4)]],
    constant uint&     n_kv      [[buffer(5)]],
    constant uint&     head_dim  [[buffer(6)]],
    constant uint&     seq_len   [[buffer(7)]],
    constant float&    scale     [[buffer(8)]],
    uint               tid       [[thread_position_in_grid]],
    uint               head_idx  [[threadgroup_position_in_grid]],
    threadgroup float* shared    [[threadgroup(0)]],
    uint               thread_in_tg [[thread_position_in_threadgroup]])
{
    // Each threadgroup processes one head
    uint tg_size = head_dim;  // threads per threadgroup = head_dim
    uint kv_head = head_idx / (n_heads / n_kv);  // GQA: map Q head to KV head
    uint local_tid = thread_in_tg;

    // Load Q element for this head+dim
    float q_val = q[head_idx * head_dim + local_tid];

    // ── Step 1: Compute attention scores ──
    // score[t] = Σ_d q[d] * k_cache[kv_head, t, d]
    // Each thread computes one score per seq position

    // For efficiency with small seq_len: each thread handles one position
    // With larger seq_len: use loop
    float max_score = -INFINITY;

    // Compute scores and find max (sequential per thread)
    for (uint t = local_tid; t < seq_len; t += tg_size) {
        float score = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            // Need full q vector — broadcast via threadgroup
            // For now, each thread computes partial dot product
            score += q[head_idx * head_dim + d] * k_cache[kv_head * seq_len * head_dim + t * head_dim + d];
        }
        // Store in shared for softmax
        shared[t] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find max across all positions (thread 0 does the scan)
    if (local_tid == 0) {
        for (uint t = 0; t < seq_len; t++) {
            max_score = max(max_score, shared[t]);
        }
        shared[seq_len] = max_score;  // store max in extra slot
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = shared[seq_len];

    // ── Step 2: Softmax ──
    float sum_exp = 0.0;
    for (uint t = local_tid; t < seq_len; t += tg_size) {
        float exp_val = exp(shared[t] - max_score);
        shared[t] = exp_val;
        sum_exp += exp_val;
    }
    // Reduction for sum_exp
    shared[seq_len + 1 + local_tid] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_tid == 0) {
        float total = 0.0;
        for (uint i = 0; i < tg_size; i++) {
            total += shared[seq_len + 1 + i];
        }
        shared[seq_len] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0 / max(shared[seq_len], 1e-12f);

    // Normalize
    for (uint t = local_tid; t < seq_len; t += tg_size) {
        shared[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 3: Weighted sum ──
    // output[head_idx, d] = Σ_t attn[t] * v_cache[kv_head, t, d]
    float out_val = 0.0;
    for (uint t = 0; t < seq_len; t++) {
        float attn = shared[t];
        out_val += attn * v_cache[kv_head * seq_len * head_dim + t * head_dim + local_tid];
    }
    output[head_idx * head_dim + local_tid] = out_val;
}
