#!/usr/bin/env python3
"""Correctness test for Fused MoE Dispatch v0.

Compares legacy top-k selected expert execution with the fused dispatcher.
Asserts that cosine similarity >= 0.9999.
"""

import ctypes
import os
import sys
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.qwen36_executor import get_lib

BIN = PROJECT / "models" / "qwen36_bin"
HDIM = 2048

def cos(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def norm_ratio(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return na / max(nb, 1e-12)

def main():
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

    # Fix seed for reproducible inputs
    np.random.seed(42)

    all_passed = True
    print("\n--- Correctness Swell Verification ---")
    print(f"{'Layer':<6} | {'N':<3} | {'Cosine Sim':<12} | {'Max Abs Diff':<14} | {'Norm Ratio':<10} | {'Status':<6}")
    print("-" * 65)

    for layer in [0, 7, 31]:
        for n in range(1, 9):
            # Input activation
            x = np.random.normal(scale=0.02, size=HDIM).astype(np.float32)

            # Selected expert IDs and weights
            expert_ids = np.random.choice(256, size=n, replace=False).astype(np.int32)
            routing_weights = np.random.uniform(0.1, 1.0, size=n).astype(np.float32)
            routing_weights /= routing_weights.sum()

            # Outputs
            expert_out = np.zeros((n, HDIM), dtype=np.float32)
            weighted_out = np.zeros((n, HDIM), dtype=np.float32)
            routed_sum_out = np.zeros(HDIM, dtype=np.float32)

            # Run legacy sequential q4selected expert gemv
            ret = lib.lko_runner_selected_expert_q4(
                layer,
                x.ctypes.data,
                expert_ids.ctypes.data,
                routing_weights.ctypes.data,
                n,
                expert_out.ctypes.data,
                weighted_out.ctypes.data,
                routed_sum_out.ctypes.data,
            )
            assert ret == n, f"lko_runner_selected_expert_q4 failed: {ret}"

            # Run new fused selected expert
            fused_routed_sum_out = np.zeros(HDIM, dtype=np.float32)
            ret_f = lib.lko_runner_selected_expert_q4_fused(
                layer,
                x.ctypes.data,
                expert_ids.ctypes.data,
                routing_weights.ctypes.data,
                n,
                fused_routed_sum_out.ctypes.data,
            )
            assert ret_f == n, f"lko_runner_selected_expert_q4_fused failed: {ret_f}"

            # Compare
            c = cos(routed_sum_out, fused_routed_sum_out)
            max_abs = np.max(np.abs(routed_sum_out - fused_routed_sum_out))
            nr = norm_ratio(fused_routed_sum_out, routed_sum_out)

            passed = c >= 0.9999
            if not passed:
                all_passed = False

            status = "PASS" if passed else "FAIL"
            print(f"{layer:<6} | {n:<3} | {c:<12.6f} | {max_abs:<14.6e} | {nr:<10.6f} | {status:<6}")

    print("-" * 65)
    if all_passed:
        print("ALL TESTS PASSED! Cosine similarity >= 0.9999 for all sweeps.")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! Cosine similarity below threshold.")
        sys.exit(1)

if __name__ == "__main__":
    main()
