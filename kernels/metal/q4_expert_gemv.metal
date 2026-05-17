// objeta — Q4_K_APPL Expert GEMV Metal Kernel
//
// Q4_K_APPL format (160 bytes per 256-element block):
//   Bytes 0-15:   8 fp16 scales (one per 32-element sub-block)
//   Bytes 16-31:  8 fp16 mins (one per sub-block)
//   Bytes 32-159: 128 bytes of packed 4-bit quants
//
// Packed layout for 4 groups of 64 elements (g ∈ [0,3]):
//   Byte at offset 32 + g*32 + l (l ∈ [0,31]):
//     low 4 bits  → L[g*64 + l]      (elements g*64 + 0..31)
//     high 4 bits → L[g*64 + 32 + l]  (elements g*64 + 32..63)
//
// Sub-block j (j ∈ [0,7]) = element range [j*32, (j+1)*32)
//   scale[j], min[j] apply to all elements in sub-block j
//   w[pos] = L[pos] * scale[pos/32] + min[pos/32]

#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;
constant uint N_SUB = 8;
constant uint SUB_SIZE = 32;
constant uint BLOCK_BYTES = 160;

kernel void q4_expert_gemv(
    device const uchar* q4_data [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float*       y        [[buffer(2)]],
    constant uint4&    dims     [[buffer(3)]],
    uint               tid      [[thread_position_in_grid]])
{
    uint M = dims.x;
    uint K = dims.y;
    uint num_blocks = dims.z;

    if (tid >= M) return;

    uint row_off = tid * num_blocks * BLOCK_BYTES;
    float sum = 0.0;

    for (uint b = 0; b < num_blocks; b++) {
        uint blk = row_off + b * BLOCK_BYTES;

        // Decode all 8 sub-blocks' scales and mins
        half scales[N_SUB], mins[N_SUB];
        for (uint j = 0; j < N_SUB; j++) {
            uint sr = ((uint)q4_data[blk + j*2 + 1] << 8) | q4_data[blk + j*2];
            uint mr = ((uint)q4_data[blk + 16 + j*2 + 1] << 8) | q4_data[blk + 16 + j*2];
            scales[j] = *(thread half*)&sr;
            mins[j] = *(thread half*)&mr;
        }

        // Decode all 256 quants
        uchar L[QK_K];
        for (uint g = 0; g < 4; g++) {
            for (uint l = 0; l < 32; l++) {
                uchar byte = q4_data[blk + 32 + g*32 + l];
                L[g*64 + l] = byte & 0xF;
                L[g*64 + 32 + l] = byte >> 4;
            }
        }

        // Compute dot product
        float block_sum = 0.0;
        uint k_start = b * QK_K;
        for (uint pos = 0; pos < QK_K && (k_start + pos) < K; pos++) {
            uint j = pos / SUB_SIZE;
            float w = float(mins[j] + half(L[pos]) * scales[j]);
            block_sum += w * x[k_start + pos];
        }
        sum += block_sum;
    }

    y[tid] = sum;
}

// ── Multi-Expert Parallel Dispatch ────────────────────────────────────────

/// Process multiple experts in parallel. One threadgroup per expert.
///
/// Buffers:
///   [[buffer(0)]] all_q4_data — concatenated q4 weights for all experts
///   [[buffer(1)]] expert_offsets — [M, K, num_blocks, q4_offset] × n_experts (u32)
///   [[buffer(2)]] x — input vector (fp32, K)
///   [[buffer(3)]] y — output buffer (fp32, sum of all M_i)
///   [[buffer(4)]] output_offsets — per-expert output offset in y (u32)
///   [[buffer(5)]] n_experts — u32
kernel void multi_expert_gemv(
    device const uchar*  all_q4_data    [[buffer(0)]],
    device const uint*   expert_offsets [[buffer(1)]],
    device const float*  x              [[buffer(2)]],
    device float*        y              [[buffer(3)]],
    device const uint*   output_offsets [[buffer(4)]],
    constant uint&       n_experts      [[buffer(5)]],
    uint                 tid            [[thread_position_in_grid]])
{
    // Find which expert this thread belongs to.
    // expert_offsets layout: [M, K, n_blocks, q4_off] repeated per expert
    // We scan to find which expert's rows contain this global tid.
    uint row_tid = tid;
    for (uint e = 0; e < n_experts; e++) {
        uint exp_M = expert_offsets[e * 4];
        if (row_tid < exp_M) {
            uint exp_K    = expert_offsets[e * 4 + 1];
            uint n_blocks = expert_offsets[e * 4 + 2];
            uint q4_off   = expert_offsets[e * 4 + 3];
            uint out_off  = output_offsets[e];

            device const uchar* q4 = all_q4_data + q4_off;
            device float* y_out = y + out_off;
            uint row_off = row_tid * n_blocks * BLOCK_BYTES;
            float sum = 0.0;

            for (uint b = 0; b < n_blocks; b++) {
                uint blk = row_off + b * BLOCK_BYTES;

                half scales[N_SUB], mins[N_SUB];
                for (uint j = 0; j < N_SUB; j++) {
                    uint sr = ((uint)q4[blk + j*2 + 1] << 8) | q4[blk + j*2];
                    uint mr = ((uint)q4[blk + 16 + j*2 + 1] << 8) | q4[blk + 16 + j*2];
                    scales[j] = *(thread half*)&sr;
                    mins[j] = *(thread half*)&mr;
                }

                uchar L[QK_K];
                for (uint g = 0; g < 4; g++) {
                    for (uint l = 0; l < 32; l++) {
                        uchar byte = q4[blk + 32 + g*32 + l];
                        L[g*64 + l] = byte & 0xF;
                        L[g*64 + 32 + l] = byte >> 4;
                    }
                }

                float block_sum = 0.0;
                uint k_start = b * QK_K;
                for (uint pos = 0; pos < QK_K && (k_start + pos) < exp_K; pos++) {
                    uint j = pos / SUB_SIZE;
                    float w = float(mins[j] + half(L[pos]) * scales[j]);
                    block_sum += w * x[k_start + pos];
                }
                sum += block_sum;
            }

            y_out[row_tid] = sum;
            return;
        }
        row_tid -= exp_M;
    }
}
