__device__ __forceinline__ float decode_f8_e8m0(unsigned char raw) {
    if (raw == 0) {
        unsigned int bits = 1 << 22;
        return __uint_as_float(bits);
    } else {
        unsigned int bits = (unsigned int)raw << 23;
        return __uint_as_float(bits);
    }
}

__device__ const float fp4_table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void fp4_e2m1_gemv(
    const unsigned char* qweights,
    const float* x,
    float* y,
    unsigned int rows,
    unsigned int cols,
    unsigned int blocks_per_row,
    unsigned int row_bytes
) {
    unsigned int row = blockIdx.x;
    unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= 32) {
        return;
    }

    const unsigned char* row_q = qweights + ((size_t)row * row_bytes);
    float sum = 0.0f;

    for (unsigned int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const unsigned char* block_q = row_q + block_idx * 17;
        unsigned char scale_raw = block_q[0];
        float scale = decode_f8_e8m0(scale_raw);

        unsigned int col = block_idx * 32 + lane;
        if (col < cols) {
            unsigned char packed = block_q[1 + (lane >> 1)];
            unsigned char q = (lane & 1) == 0 ? (packed & 0x0F) : (packed >> 4);
            float w = fp4_table[q] * scale;
            sum += w * x[col];
        }
    }

    __shared__ float partial[32];
    partial[lane] = sum;
    __syncthreads();

    for (unsigned int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        y[row] = partial[0];
    }
}

extern "C" __global__ void fp4_e2m1_gemv_split(
    const unsigned char* qweights,
    const unsigned char* scales,
    const float* x,
    float* y,
    unsigned int rows,
    unsigned int cols,
    unsigned int blocks_per_row
) {
    unsigned int row = blockIdx.x;
    unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= 32) {
        return;
    }

    const unsigned char* row_w = qweights + ((size_t)row * (cols / 2));
    const unsigned char* row_s = scales + ((size_t)row * (cols / 32));
    float sum = 0.0f;

    for (unsigned int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        unsigned char scale_raw = row_s[block_idx];
        float scale = decode_f8_e8m0(scale_raw);

        unsigned int col = block_idx * 32 + lane;
        if (col < cols) {
            unsigned char packed = row_w[block_idx * 16 + (lane >> 1)];
            unsigned char q = (lane & 1) == 0 ? (packed & 0x0F) : (packed >> 4);
            float w = fp4_table[q] * scale;
            sum += w * x[col];
        }
    }

    __shared__ float partial[32];
    partial[lane] = sum;
    __syncthreads();

    for (unsigned int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        y[row] = partial[0];
    }
}

