#!/usr/bin/env python3
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.qwen36_executor import get_lib

BIN = PROJECT / "models" / "qwen36_bin"
HDIM = 2048


def load_stats(lib):
    stats_ptr = lib.lko_runner_get_moe_stats_json()
    if not stats_ptr:
        return {}
    raw = ctypes.cast(stats_ptr, ctypes.c_char_p).value.decode("utf-8")
    return json.loads(raw)


def bench_case(lib, layer, n_selected, use_fused, down_mode_kind=1, chunk_rows=64, iters=100):
    if hasattr(lib, "lko_runner_reset_moe_stats"):
        lib.lko_runner_reset_moe_stats()
    if hasattr(lib, "lko_runner_build_caches"):
        lib.lko_runner_build_caches(0)

    np.random.seed(1000 + layer * 10 + n_selected)
    x = np.random.normal(scale=0.02, size=HDIM).astype(np.float32)
    expert_ids = np.random.choice(256, size=n_selected, replace=False).astype(np.int32)
    routing_weights = np.random.uniform(0.1, 1.0, size=n_selected).astype(np.float32)
    routing_weights /= routing_weights.sum()
    out = np.zeros(HDIM, dtype=np.float32)

    for _ in range(5):
        ret = lib.lko_runner_selected_expert_q4_path(
            layer,
            x.ctypes.data,
            expert_ids.ctypes.data,
            routing_weights.ctypes.data,
            n_selected,
            int(use_fused),
            down_mode_kind,
            chunk_rows,
            out.ctypes.data,
        )
        assert ret == n_selected, f"path call failed: {ret}"

    t0 = time.perf_counter()
    for _ in range(iters):
        ret = lib.lko_runner_selected_expert_q4_path(
            layer,
            x.ctypes.data,
            expert_ids.ctypes.data,
            routing_weights.ctypes.data,
            n_selected,
            int(use_fused),
            down_mode_kind,
            chunk_rows,
            out.ctypes.data,
        )
        assert ret == n_selected, f"path call failed: {ret}"
    external_ms = (time.perf_counter() - t0) * 1000.0 / iters

    stats = load_stats(lib)
    summary = stats.get("summary", {})
    layers = stats.get("layers", [])
    layer_stats = next((x for x in layers if x.get("layer") == layer), {})
    kernel_ms = float(layer_stats.get("avg_compute_ms", 0.0))
    fused_stats_ms = float(layer_stats.get("avg_fused_stats_ms", 0.0))
    total_internal_ms = kernel_ms + fused_stats_ms
    return {
        "external_ms": external_ms,
        "kernel_ms": kernel_ms,
        "total_internal_ms": total_internal_ms,
        "overhead_ms": external_ms - total_internal_ms,
        "layer_stats": layer_stats,
        "summary": summary,
    }


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    lib = get_lib()
    assert lib is not None, "Rust library not found"

    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    lib.lko_runner_reset_moe_stats.argtypes = []
    lib.lko_runner_reset_moe_stats.restype = ctypes.c_int32
    lib.lko_runner_build_caches.argtypes = [ctypes.c_int32]
    lib.lko_runner_build_caches.restype = ctypes.c_int32
    lib.lko_runner_get_moe_stats_json.argtypes = []
    lib.lko_runner_get_moe_stats_json.restype = ctypes.c_void_p
    lib.lko_runner_selected_expert_q4_path.argtypes = [
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.lko_runner_selected_expert_q4_path.restype = ctypes.c_int32

    assert lib.lko_runner_init(str(BIN).encode(), 256), "runner init failed"

    print("# call_moe microbench (actual runner execution path)")
    print(f"iters={iters}, cache=off, fixed selected experts")
    print()
    print("| layer | N | path | external_ms | kernel_ms | total_call_moe_ms | overhead_ms |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for layer in (0, 7, 31):
        for n in (8, 6, 4):
            for label, use_fused in (("legacy", False), ("fused", True)):
                r = bench_case(lib, layer, n, use_fused, down_mode_kind=1, chunk_rows=64, iters=iters)
                print(
                    f"| {layer} | {n} | {label} | "
                    f"{r['external_ms']:.3f} | {r['kernel_ms']:.3f} | "
                    f"{r['total_internal_ms']:.3f} | {r['overhead_ms']:.3f} |"
                )

    print()
    print("# layer31 N=8 fused down-mode variants")
    print("| mode | external_ms | kernel_ms | total_call_moe_ms | overhead_ms | gate_up_ms | swiglu_ms | down_accum_ms | alloc_ms | stats_ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, mode_kind, chunk_rows in (
        ("serial", 0, 1),
        ("row_parallel", 1, 1),
        ("chunked32", 2, 32),
        ("chunked64", 2, 64),
        ("chunked128", 2, 128),
    ):
        r = bench_case(lib, 31, 8, True, down_mode_kind=mode_kind, chunk_rows=chunk_rows, iters=iters)
        ls = r["layer_stats"]
        print(
            f"| {label} | {r['external_ms']:.3f} | {r['kernel_ms']:.3f} | "
            f"{r['total_internal_ms']:.3f} | {r['overhead_ms']:.3f} | "
            f"{float(ls.get('avg_fused_gate_up_ms', 0.0)):.3f} | "
            f"{float(ls.get('avg_fused_swiglu_ms', 0.0)):.3f} | "
            f"{float(ls.get('avg_fused_down_accum_ms', 0.0)):.3f} | "
            f"{float(ls.get('avg_fused_alloc_ms', 0.0)):.3f} | "
            f"{float(ls.get('avg_fused_stats_ms', 0.0)):.3f} |"
        )


if __name__ == "__main__":
    main()
