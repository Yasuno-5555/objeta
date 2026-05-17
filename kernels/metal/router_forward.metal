// objeta — Router Forward + Softmax + Top-K Metal Kernel
//
// Single-pass router: W @ h → softmax → top-k selection
// Fuses three operations that would otherwise be separate kernel launches.
//
// Buffers:
//   [[buffer(0)]] router_w: router weight matrix (fp32, N_EXPERTS × HIDDEN_DIM)
//   [[buffer(1)]] h: input hidden state (fp32, HIDDEN_DIM)
//   [[buffer(2)]] logits: intermediate logits (fp32, N_EXPERTS)
//   [[buffer(3)]] probs: softmax probabilities (fp32, N_EXPERTS)
//   [[buffer(4)]] top_k_indices: output (int32, TOP_K)
//   [[buffer(5)]] top_k_probs: output (fp32, TOP_K)
//   [[buffer(6)]] dims: uint4(N_EXPERTS, HIDDEN_DIM, TOP_K, 0)

#include <metal_stdlib>
using namespace metal;

/// Stage 1: Compute logits = W @ h. One thread per expert.
kernel void router_logits(
    device const float* router_w  [[buffer(0)]],
    device const float* h         [[buffer(1)]],
    device float*       logits    [[buffer(2)]],
    constant uint2&    dims      [[buffer(3)]],  // (N_EXPERTS, HIDDEN_DIM)
    uint               tid       [[thread_position_in_grid]])
{
    uint N = dims.x;
    uint D = dims.y;
    if (tid >= N) return;

    float sum = 0.0;
    device const float* row = router_w + tid * D;
    for (uint j = 0; j < D; j++) {
        sum += row[j] * h[j];
    }
    logits[tid] = sum;
}

/// Stage 2: Softmax in shared memory.
/// Single threadgroup, all N_EXPERTS (256) elements fit in threadgroup memory.
kernel void router_softmax_topk(
    device float*       logits       [[buffer(0)]],
    device float*       probs        [[buffer(1)]],
    device int*         topk_idx     [[buffer(2)]],
    device float*       topk_prob    [[buffer(3)]],
    constant uint&      N            [[buffer(4)]],
    constant uint&      top_k        [[buffer(5)]],
    threadgroup float*  shared       [[threadgroup(0)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                tg_size      [[threads_per_threadgroup]])
{
    // ── Find max ──
    float max_val = -INFINITY;
    for (uint i = tid; i < N; i += tg_size) {
        max_val = max(max_val, logits[i]);
    }
    shared[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction for max (256 elements, 256 threads → log2(256)=8 steps)
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Compute exp and sum ──
    float sum_exp = 0.0;
    for (uint i = tid; i < N; i += tg_size) {
        float p = exp(logits[i] - max_val);
        probs[i] = p;
        sum_exp += p;
    }
    shared[tid] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum_exp = shared[0];

    // ── Normalize ──
    float inv_sum = 1.0 / max(sum_exp, 1e-12f);
    for (uint i = tid; i < N; i += tg_size) {
        probs[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Top-K selection via bitonic sort ──
    // Each thread handles one expert. We store (prob, index) pairs.
    // For simplicity with 256 experts: parallel compare-and-swap.

    // Initialize index array
    shared[tid] = probs[tid];
    shared[tg_size + tid] = float(tid);  // second half for indices
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simple selection: each thread scans for its "rank"
    // Given 256 elements and top_k=8, this is efficient enough.
    uint my_idx = tid;
    float my_prob = probs[tid];

    if (tid < N) {
        uint rank = 0;
        for (uint i = 0; i < N; i++) {
            if (probs[i] > my_prob || (probs[i] == my_prob && i < my_idx)) {
                rank++;
            }
        }
        if (rank < top_k) {
            topk_idx[rank] = int(my_idx);
            topk_prob[rank] = my_prob;
        }
    }
}
