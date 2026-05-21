extern "C" __global__ void silu_mul(
    const float* gate,
    const float* up,
    float* act,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float u = up[idx];
        float silu = g / (1.0f + expf(-g));
        act[idx] = silu * u;
    }
}

extern "C" __global__ void weighted_accum(
    const float* down,
    float* out_vec,
    float weight,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_vec[idx] += weight * down[idx];
    }
}
