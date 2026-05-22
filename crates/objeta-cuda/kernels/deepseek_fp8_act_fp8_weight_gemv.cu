// DeepSeek V4 Flash — FP8 activation × FP8 weight GEMV (official shared expert).
//
// A: F8_E4M3 activation values [K]
// A_scale: F8_E8M0 activation scales [K/128] (one per 128 K elements)
// B: F8_E4M3 weight values [rows, K]
// B_scale: F8_E8M0 weight scales [rows/128, K/128] (128x128 tile grid)
// Output: fp32 [rows]

// -- F8_E8M0 decode -----------------------------------------------------------
__device__ __forceinline__ float decode_f8_e8m0(unsigned char raw) {
    if (raw == 0) return __uint_as_float(1u << 22);
    return __uint_as_float(((unsigned int)raw) << 23);
}

// -- F8_E4M3 decode -----------------------------------------------------------
__device__ __forceinline__ float decode_f8_e4m3(unsigned char raw) {
    unsigned int sign = (raw >> 7) & 1;
    unsigned int exp  = (raw >> 3) & 0xF;
    unsigned int mant = raw & 0x7;
    if (exp == 0) { float v = ((float)(int)mant) * 0.015625f; return sign ? -v : v; }
    if (exp == 15) return sign ? -448.0f : 448.0f;
    float v = (1.0f + (float)(int)mant * 0.125f) * exp2f((float)((int)exp - 7));
    return sign ? -v : v;
}

extern "C" __global__ void fp8_act_fp8_weight_gemv(
    const unsigned char* __restrict__ act_values,    // [K]
    const unsigned char* __restrict__ act_scales,    // [K/128]
    const unsigned char* __restrict__ weight,        // [rows, K]
    const unsigned char* __restrict__ weight_scales, // [rows/128, K/128]
    float* __restrict__ y,
    unsigned int rows,
    unsigned int K
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int TILE = 128;

    const unsigned char* wt_row = weight + ((size_t)row * K);
    unsigned int wt_scale_row = row / TILE;
    unsigned int wt_scale_stride = K / TILE;
    const unsigned char* wt_s_base = weight_scales + ((size_t)wt_scale_row * wt_scale_stride);

    float sum = 0.0f;

    for (unsigned int k = tid; k < K; k += TILE) {
        float act_v = decode_f8_e4m3(act_values[k]);
        float act_scale = decode_f8_e8m0(act_scales[k / TILE]);

        float wt_v = decode_f8_e4m3(wt_row[k]);
        float wt_scale = decode_f8_e8m0(wt_s_base[k / TILE]);

        sum += act_v * act_scale * wt_v * wt_scale;
    }

    __shared__ float partial[128];
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = partial[0];
}
