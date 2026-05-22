// DeepSeek V4 Flash — activation quantization kernel.
//
// Matches inference/kernel.py `act_quant` with `round_scale=true` (E8M0 power-of-2).
//   block_size = 128
//   Input:  FP32 activation [rows, cols]
//   Output: F8_E4M3 values [rows, cols] + F8_E8M0 scales [rows, cols/128]
//
// Accumulation users later consume values + scales for fp8_gemm / fp4_gemm.

// -- F8_E4M3 encode: f32 -> u8 -------------------------------------------------
__device__ __forceinline__ unsigned char f32_to_f8e4m3(float v) {
    if (v < -448.0f) return 0xFE;
    if (v >  448.0f) return 0x7E;
    // NaN check
    if (v != v) return 0x7F;

    unsigned int bits = __float_as_uint(v);
    unsigned int sign = (bits >> 31) & 1;
    unsigned int exp32 = (bits >> 23) & 0xFF;
    unsigned int mant32 = bits & 0x7FFFFF;

    int exp = (int)exp32 - 127;

    // F8_E4M3: bias = 7, mantissa bits = 3
    // Normal range: exp in [-6, 7]
    int exp_f8 = exp + 7;

    if (exp_f8 <= 0) {
        // Subnormal or zero
        // Minimum positive normal in F8_E4M3 is 2^-6 ≈ 0.015625
        float abs = fabsf(v);
        if (abs < 0.0078125f) { // 2^-7
            return (unsigned char)(sign << 7); // zero
        }
        // Map to subnormal: mantissa encodes fraction
        // Scale up: v * 2^6 gives fraction bits
        float scaled = abs * 64.0f; // 2^6
        int mant = (int)(scaled + 0.5f);
        if (mant > 7) mant = 7;
        return (unsigned char)((sign << 7) | mant);
    } else if (exp_f8 >= 15) {
        // Overflow to max representable
        return (unsigned char)((sign << 7) | (14 << 3) | 7); // sign + max exp + max mantissa
    }

    // Normal: round mantissa to 3 bits
    unsigned int mant_rounded = (mant32 + (1 << 19)) >> 20; // round to nearest 3 bits
    if (mant_rounded >= 8) {
        mant_rounded = 0;
        exp_f8 += 1;
    }
    if (exp_f8 >= 15) {
        return (unsigned char)((sign << 7) | (14 << 3) | 7);
    }
    return (unsigned char)((sign << 7) | (exp_f8 << 3) | mant_rounded);
}

// -- Main kernel ----------------------------------------------------------------
extern "C" __global__ void act_quant_fp8_e4m3_e8m0(
    const float* __restrict__ input,
    unsigned char* __restrict__ values,
    unsigned char* __restrict__ scales,
    int rows,
    int cols
) {
    const int BLOCK = 128;
    int row = blockIdx.y;
    int block_col = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;
    int block_start = block_col * BLOCK;
    if (block_start >= cols) return;

    __shared__ float sdata[BLOCK];
    __shared__ float s_scale;

    int idx = block_start + tid;
    float val = 0.0f;
    if (idx < cols) {
        val = input[row * cols + idx];
    }
    sdata[tid] = fabsf(val);
    __syncthreads();

    // Reduction to find amax
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = sdata[tid];
            float b = sdata[tid + stride];
            sdata[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float amax = sdata[0];
        amax = (amax < 1e-4f) ? 1e-4f : amax;

        // fast_round_scale: scale = 2^ceil(log2(amax / 448))
        float raw = amax * (1.0f / 448.0f);
        float log2_raw = log2f(raw);
        int log2_ceil = (int)ceilf(log2_raw);
        float scale = exp2f((float)log2_ceil);
        s_scale = scale;

        // F8_E8M0: value = 2^(e - 127), so e = log2(scale) + 127
        int e = log2_ceil + 127;
        if (e < 0) e = 0;
        if (e > 255) e = 255;
        int num_blocks = cols / BLOCK;
        scales[row * num_blocks + block_col] = (unsigned char)e;
    }
    __syncthreads();

    // Quantize values
    float inv_scale = 1.0f / s_scale;
    if (idx < cols) {
        float x = input[row * cols + idx];
        float q = x * inv_scale;
        q = fminf(448.0f, fmaxf(-448.0f, q));
        values[row * cols + idx] = f32_to_f8e4m3(q);
    }
}
