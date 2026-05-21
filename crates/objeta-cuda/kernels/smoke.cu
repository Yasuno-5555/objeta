extern "C" __global__ void smoke_vector_add_f32(
    const float* a,
    const float* b,
    float* out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void smoke_gemv_f32(
    const float* matrix,
    const float* vector,
    float* out,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float sum = 0.0f;
    for (unsigned int col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += matrix[row * cols + col] * vector[col];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
