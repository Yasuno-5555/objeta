#!/usr/bin/env python3
"""Custom Metal GEMV kernel via MLX fast.metal_kernel API.

Replaces MLX's built-in matmul with a custom Metal shader.
Weights stay in MLX arrays (zero-copy), kernel dispatched directly.

This is the LKO pattern: MLX as a GPU buffer manager + shader compiler,
NOT as a computation library.
"""

import mlx.core as mx
import numpy as np
import time

# ── Metal GEMV Shader ─────────────────────────────────────────────────────

GEMV_SOURCE = """
    uint row = thread_position_in_grid.x;
    if (row >= M) return;

    float acc = 0.0;
    for (uint k = 0; k < K; k += 4) {
        float4 w = *(device const float4*)(W + row * K + k);
        float4 xv = *(device const float4*)(x + k);
        acc += dot(w, xv);
    }
    y[row] = acc;
"""

# Compiled kernel cache: (M, K) → callable
_gemv_cache = {}

def compiled_gemv(M, K):
    """Get or compile a GEMV kernel for dimensions (M, K)."""
    # Use a size-bucket key to limit cache entries
    bucket = (M, K) if M * K < 1_000_000 else (0, 0)  # generic for large
    if bucket in _gemv_cache:
        return _gemv_cache[bucket]

    # Handle K not divisible by 4: pad with a bounds check
    source = GEMV_SOURCE
    if K % 4 != 0:
        source = """
            uint row = thread_position_in_grid.x;
            if (row >= M) return;

            float acc = 0.0;
            for (uint k = 0; k + 3 < K; k += 4) {
                float4 w = *(device const float4*)(W + row * K + k);
                float4 xv = *(device const float4*)(x + k);
                acc += dot(w, xv);
            }
            for (uint k = (K / 4) * 4; k < K; k++) {
                acc += W[row * K + k] * x[k];
            }
            y[row] = acc;
        """

    kernel = mx.fast.metal_kernel(
        name=f"gemv_{M}_{K}",
        input_names=["W", "x", "M", "K"],
        output_names=["y"],
        source=source,
        header="#include <metal_stdlib>\nusing namespace metal;",
    )
    _gemv_cache[bucket] = kernel
    return kernel


def metal_gemv(W: mx.array, x: mx.array) -> mx.array:
    """Compute y = W @ x using custom Metal GEMV kernel.

    Args:
        W: weight matrix (M, K) — MLX array, fp32
        x: input vector (K,) — MLX array, fp32

    Returns:
        y: output vector (M,) — MLX array, fp32
    """
    M, K = W.shape
    kernel = compiled_gemv(M, K)

    (y,) = kernel(
        inputs=[W, x, M, K],
        output_shapes=[(M,)],
        output_dtypes=[mx.float32],
        grid=(M, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return y


# ── Benchmark ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test correctness vs MLX matmul
    for M, K in [(4096, 2048), (8192, 2048), (2048, 4096), (512, 2048)]:
        W_np = np.random.randn(M, K).astype(np.float32)
        x_np = np.random.randn(K).astype(np.float32)

        W_mx = mx.array(W_np)
        x_mx = mx.array(x_np)

        # MLX built-in
        y_mlx = W_mx @ x_mx
        mx.eval(y_mlx)

        # Custom Metal
        y_metal = metal_gemv(W_mx, x_mx)
        mx.eval(y_metal)

        cos = float(np.dot(np.array(y_mlx), np.array(y_metal)) /
                    (np.linalg.norm(np.array(y_mlx)) * np.linalg.norm(np.array(y_metal)) + 1e-12))

        # Benchmark
        for _ in range(10):
            _ = metal_gemv(W_mx, x_mx)
        mx.eval(y_metal)

        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            _ = metal_gemv(W_mx, x_mx)
        mx.eval(_)
        metal_t = (time.perf_counter() - t0) / n * 1000

        t0 = time.perf_counter()
        for _ in range(n):
            _ = W_mx @ x_mx
        mx.eval(_)
        mlx_t = (time.perf_counter() - t0) / n * 1000

        print(f"M={M:>6} K={K:>6}  cos={cos:.6f}  "
              f"metal={metal_t:.3f}ms  mlx={mlx_t:.3f}ms  "
              f"speedup={mlx_t/metal_t:.1f}x")
