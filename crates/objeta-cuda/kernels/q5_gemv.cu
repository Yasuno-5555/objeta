extern "C" __device__ __forceinline__ float f16_bits_to_f32_q5(unsigned short h) {
    unsigned int sign = (unsigned int)(h & 0x8000) << 16;
    unsigned int exp = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x03FF;

    unsigned int out_bits;
    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;
        } else {
            int exponent = -14;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exponent -= 1;
            }
            mant &= 0x03FF;
            out_bits = sign | (unsigned int)(exponent + 127) << 23 | (mant << 13);
        }
    } else if (exp == 0x1F) {
        out_bits = sign | 0x7F800000 | (mant << 13);
    } else {
        out_bits = sign | (exp + (127 - 15)) << 23 | (mant << 13);
    }
    return __uint_as_float(out_bits);
}

extern "C" __global__ void q5_gemv_f32_accum(
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
        const unsigned char* block_q = row_q + block_idx * 22;
        unsigned short scale_bits = (unsigned short)block_q[0] | ((unsigned short)block_q[1] << 8);
        float scale = f16_bits_to_f32_q5(scale_bits);

        unsigned int col = block_idx * 32 + lane;
        if (col < cols) {
            unsigned char packed_low = block_q[6 + (lane >> 1)];
            unsigned char low = (lane & 1) == 0 ? (packed_low & 0x0F) : (packed_low >> 4);
            unsigned char high = (block_q[2 + (lane >> 3)] >> (lane & 7)) & 0x01;
            unsigned char q = low | (high << 4);
            float w = ((float)((int)q - 16)) * scale;
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
