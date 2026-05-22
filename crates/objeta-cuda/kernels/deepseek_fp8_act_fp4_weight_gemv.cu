// DeepSeek V4 Flash — FP8 activation × FP4 weight GEMV (official routed expert).
//
// Matches inference/kernel.py `fp4_gemm` called with act_quant-ed activation.
//   A: F8_E4M3 activation values [1, K]
//   A_scale: F8_E8M0 activation scales [1, K/128]  (one per 128 K elements)
//   B: DeepSeek FP4 E2M1FN packed I8 weight [rows, K/2]
//   B_scale: F8_E8M0 weight scales [rows, K/32]     (one per 32 K elements)
//   Output: fp32 [rows]

// -- F8_E8M0 decode -----------------------------------------------------------
__device__ __forceinline__ float decode_f8_e8m0(unsigned char raw) {
    if (raw == 0) {
        return __uint_as_float(1u << 22);  // 2^-127 as subnormal expressible in fp32
    }
    return __uint_as_float(((unsigned int)raw) << 23);
}

// -- F8_E4M3 decode -----------------------------------------------------------
__device__ __forceinline__ float decode_f8_e4m3(unsigned char raw) {
    unsigned int sign = (raw >> 7) & 1;
    unsigned int exp  = (raw >> 3) & 0xF;
    unsigned int mant = raw & 0x7;

    if (exp == 0) {
        // Subnormal: mantissa * 2^-6
        float val = ((float)(int)mant) * 0.015625f;  // 2^-6
        return sign ? -val : val;
    }
    if (exp == 15) {
        // NaN or Inf — clamp to max
        return sign ? -448.0f : 448.0f;
    }
    // Normal: (1 + mant/8) * 2^(exp - 7)
    float val = (1.0f + (float)(int)mant * 0.125f) * exp2f((float)((int)exp - 7));
    return sign ? -val : val;
}

// -- FP4 E2M1FN table ---------------------------------------------------------
__device__ const float fp4_table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// -- Main kernel ---------------------------------------------------------------
extern "C" __global__ void fp8_act_fp4_weight_gemv(
    const unsigned char* __restrict__ act_values,    // [K] F8_E4M3
    const unsigned char* __restrict__ act_scales,    // [K/128] F8_E8M0
    const unsigned char* __restrict__ weight_packed, // [rows, K/2] packed F8_I8
    const unsigned char* __restrict__ weight_scales, // [rows, K/32] F8_E8M0
    float* __restrict__ y,                           // [rows]
    unsigned int rows,                               // output rows
    unsigned int K                                   // reduction dimension (logical cols)
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int ACT_BLOCK = 128;
    const unsigned int WT_BLOCK = 32;
    const unsigned int K_half = K / 2;               // physical weight cols

    const unsigned char* act_v = act_values;
    const unsigned char* act_s = act_scales;
    const unsigned char* wt = weight_packed + ((size_t)row * K_half);
    const unsigned char* wt_s = weight_scales + ((size_t)row * (K / WT_BLOCK));

    float sum = 0.0f;

    // Each thread handles every 128-th element of K, with stride = block_size
    for (unsigned int k = tid; k < K; k += ACT_BLOCK) {
        unsigned int k_phys = k / 2;  // physical index in packed weight
        int nibble_pos = k & 1;       // 0 = low nibble, 1 = high nibble

        // Decode activation
        float act_v_f = decode_f8_e4m3(act_v[k]);
        float act_s_f = decode_f8_e8m0(act_s[k / ACT_BLOCK]);

        // Unpack FP4 weight nibble
        unsigned char byte_v = wt[k_phys];
        unsigned char nibble = (nibble_pos == 0) ? (byte_v & 0x0F) : (byte_v >> 4);

        float wt_v_f = fp4_table[nibble];
        float wt_s_f = decode_f8_e8m0(wt_s[k / WT_BLOCK]);

        sum += act_v_f * act_s_f * wt_v_f * wt_s_f;
    }

    // Shared memory reduction across 128 threads
    __shared__ float partial[128];
    partial[tid] = sum;
    __syncthreads();

    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[row] = partial[0];
    }
}
