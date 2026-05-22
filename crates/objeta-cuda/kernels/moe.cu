/// DeepSeek V4 Flash — MoE kernels (SwiGLU + weighted accumulate).
extern "C" __global__ void silu_mul(
    const float* gate,
    const float* up,
    float* act,
    unsigned int n,
    float swiglu_limit
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float u = up[idx];
        // Official clamp: up = clamp(up, -limit, limit), gate = clamp(gate, max=limit)
        if (swiglu_limit > 0.0f) {
            u = fminf(swiglu_limit, fmaxf(-swiglu_limit, u));
            g = fminf(swiglu_limit, g);
        }
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
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
