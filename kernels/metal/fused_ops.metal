// objeta — Fused Residual + RMSNorm + Shared Expert Metal Kernels
//
// Fuses multiple operations to eliminate intermediate buffers and kernel launches.
// Critical for MoE where each layer has: attn_out + expert_out + residual → norm

#include <metal_stdlib>
using namespace metal;

// ── Fused Residual Accumulate + RMSNorm ───────────────────────────────────

/// Fuses:  h = h + attn_out + moe_out
///         h = RMSNorm(h, weight)
///
/// Eliminates 2 intermediate buffers and 1 kernel launch per layer.
///
/// Buffers:
///   [[buffer(0)]] h: input/output hidden state (fp32, D)
///   [[buffer(1)]] attn_out: attention output (fp32, D)
///   [[buffer(2)]] moe_out: MoE output (fp32, D)
///   [[buffer(3)]] norm_weight: RMSNorm weight (fp32, D)
///   [[buffer(4)]] dim: uint (hidden_dim)
kernel void fused_residual_norm(
    device float*       h           [[buffer(0)]],
    device const float* attn_out    [[buffer(1)]],
    device const float* moe_out     [[buffer(2)]],
    device const float* norm_w      [[buffer(3)]],
    constant uint&     D           [[buffer(4)]],
    uint               tid         [[thread_position_in_grid]],
    threadgroup float* shared      [[threadgroup(0)]])
{
    if (tid >= D) return;

    // Accumulate
    h[tid] = h[tid] + attn_out[tid] + moe_out[tid];

    // Partial sum of squares for RMS
    float sq = h[tid] * h[tid];
    shared[tid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction
    uint tg_size = min(D, 1024u);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared[0] / float(D) + 1e-6f);
    h[tid] = (h[tid] / rms) * norm_w[tid];
}

// ── Shared Expert Forward ─────────────────────────────────────────────────

/// Qwen3.6 shared expert: SiLU-gated FFN applied to all tokens.
/// gate_out = sigmoid(gate_w @ h) · (up_w @ h)
/// out = down_w @ gate_out
///
/// Buffers:
///   [[buffer(0)]] h: input (fp32, D)
///   [[buffer(1)]] gate_w: gate projection (fp32, F × D)
///   [[buffer(2)]] up_w: up projection (fp32, F × D)
///   [[buffer(3)]] down_w: down projection (fp32, D × F)
///   [[buffer(4)]] out: output (fp32, D)
///   [[buffer(5)]] dims: uint4(D, F, 0, 0)
///   [[buffer(6)]] intermediate: scratch (fp32, F) — optional, for debugging
kernel void shared_expert_forward(
    device const float* h         [[buffer(0)]],
    device const float* gate_w    [[buffer(1)]],
    device const float* up_w      [[buffer(2)]],
    device const float* down_w    [[buffer(3)]],
    device float*       out       [[buffer(4)]],
    constant uint2&    dims      [[buffer(5)]],  // (D, F)
    device float*       intermediate [[buffer(6)]],
    uint               tid       [[thread_position_in_grid]])
{
    uint D = dims.x;
    uint F = dims.y;

    // Shared expert is small (F=512 for Qwen3.6), each thread computes one output element.
    // For efficiency with F=512 and D=2048:
    //   Best approach: one threadgroup, each thread computes partial sums.

    if (tid >= D) return;

    float gate_dot = 0.0;
    float up_dot = 0.0;
    for (uint j = 0; j < F; j++) {
        gate_dot += gate_w[j * D + tid] * h[tid];
        up_dot += up_w[j * D + tid] * h[tid];
    }
    // Actually the above is wrong — gate_w is (F × D), so gate_w[j*D + i] is weight[j,i]
    // gate_out = gate_w @ h → gate_out[j] = Σ_i gate_w[j,i] * h[i]

    // For the output: out[i] = Σ_j down_w[i,j] * gate_out[j]
    // Each thread computes one output element.

    // We need the full gate_out first. Use threadgroup collaboration.
    // Simplified for F=512: each thread participates in computing all gate_out values.

    float out_i = 0.0;
    for (uint j = 0; j < F; j++) {
        // Compute gate_out[j] = sigmoid(Σ_k gate_w[j,k] * h[k]) * (Σ_k up_w[j,k] * h[k])
        float gj = 0.0;
        float uj = 0.0;
        for (uint k = 0; k < D; k++) {
            gj += gate_w[j * D + k] * h[k];
            uj += up_w[j * D + k] * h[k];
        }
        float act = gj / (1.0 + exp(-gj)) * uj;  // SiLU(gj) * uj
        out_i += down_w[tid * F + j] * act;
    }

    out[tid] = out_i;
}
