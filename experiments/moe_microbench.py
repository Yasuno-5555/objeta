#!/usr/bin/env python3
import sys
import time
import ctypes
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.qwen36_executor import get_lib

BIN = PROJECT / "models" / "qwen36_bin"
HDIM = 2048

def run_benchmarks(iters: int):
    lib = get_lib()
    assert lib is not None, "Rust library not found"

    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32

    lib.lko_runner_selected_expert_q4.argtypes = [
        ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_selected_expert_q4.restype = ctypes.c_int32

    lib.lko_runner_selected_expert_q4_fused.argtypes = [
        ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.lko_runner_selected_expert_q4_fused.restype = ctypes.c_int32

    print(f"Initializing runner with bin directory: {BIN}")
    assert lib.lko_runner_init(str(BIN).encode(), 256), "runner init failed"

    layers = [0, 7, 31]
    
    print("\n# MoE Microbenchmark: Legacy vs Fused Dispatch")
    print(f"Iterations per run: {iters}")
    
    # Print the markdown table header
    print("\n| Layer | N | Legacy (ms/iter) | Fused (ms/iter) | Speedup |")
    print("|---|---|---|---|---|")

    # Fix seed for reproducible benchmarks
    np.random.seed(42)

    for layer_idx in layers:
        for n in range(1, 9):
            x = np.random.normal(scale=0.02, size=HDIM).astype(np.float32)
            expert_ids = np.random.choice(256, size=n, replace=False).astype(np.int32)
            routing_weights = np.random.uniform(0.1, 1.0, size=n).astype(np.float32)
            routing_weights /= routing_weights.sum()

            # Warmup
            expert_out = np.zeros((n, HDIM), dtype=np.float32)
            weighted_out = np.zeros((n, HDIM), dtype=np.float32)
            routed_sum_out = np.zeros(HDIM, dtype=np.float32)
            fused_routed_sum_out = np.zeros(HDIM, dtype=np.float32)

            for _ in range(5):
                lib.lko_runner_selected_expert_q4(
                    layer_idx, x.ctypes.data, expert_ids.ctypes.data, routing_weights.ctypes.data, n,
                    expert_out.ctypes.data, weighted_out.ctypes.data, routed_sum_out.ctypes.data
                )
                lib.lko_runner_selected_expert_q4_fused(
                    layer_idx, x.ctypes.data, expert_ids.ctypes.data, routing_weights.ctypes.data, n,
                    fused_routed_sum_out.ctypes.data
                )

            # Measure Legacy
            t0 = time.perf_counter()
            for _ in range(iters):
                lib.lko_runner_selected_expert_q4(
                    layer_idx, x.ctypes.data, expert_ids.ctypes.data, routing_weights.ctypes.data, n,
                    expert_out.ctypes.data, weighted_out.ctypes.data, routed_sum_out.ctypes.data
                )
            dt_legacy = (time.perf_counter() - t0) * 1000.0 / iters

            # Measure Fused
            t0 = time.perf_counter()
            for _ in range(iters):
                lib.lko_runner_selected_expert_q4_fused(
                    layer_idx, x.ctypes.data, expert_ids.ctypes.data, routing_weights.ctypes.data, n,
                    fused_routed_sum_out.ctypes.data
                )
            dt_fused = (time.perf_counter() - t0) * 1000.0 / iters

            speedup = dt_legacy / dt_fused
            print(f"| {layer_idx} | {n} | {dt_legacy:.3f} | {dt_fused:.3f} | {speedup:.2f}x |")

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_benchmarks(iters)

