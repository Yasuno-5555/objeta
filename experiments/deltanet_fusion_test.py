#!/usr/bin/env python3
"""DeltaNet fusion timing test: compare tok/s at different fusion ratios."""
import ctypes, numpy as np, time, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
lib = get_lib()

HDIM = 2048

# Init Metal
lib.lko_metal_init.argtypes = [ctypes.c_char_p]
lib.lko_metal_init.restype = ctypes.c_int32
METALLIB = str(Path(__file__).parent.parent / "target" / "objeta.metallib")
lib.lko_metal_init(METALLIB.encode())

# Set up step API
lib.lko_runner_step.argtypes = [
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
]
lib.lko_runner_step.restype = ctypes.c_int32

# Set fusion ratio API
lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32

BIN_DIR = str(Path(__file__).parent.parent / "models" / "qwen36_bin")

def step(token_id, pos, seq_len):
    hn = np.zeros(HDIM, dtype=np.float32)
    idx = np.zeros(50, dtype=np.int32)
    val = np.zeros(50, dtype=np.float32)
    lib.lko_runner_step(token_id, pos, seq_len, hn.ctypes.data, 50, idx.ctypes.data, val.ctypes.data)
    return hn

ratios = [1.0, 0.5, 0.33, 0.25]

# Init once
print("Loading Qwen3.6-35B (one-time)...")
assert lib.lko_runner_init(BIN_DIR.encode(), 32), "Init failed"
print("Warmup...")
step(1058, 0, 1)
step(1058, 1, 2)

for ratio in ratios:
    print(f"\n{'='*50}")
    print(f"  Fusion ratio: {ratio:.0%}")
    lib.lko_runner_set_fusion_ratio(ratio)

    # Timed: 5 forward passes
    t0 = time.perf_counter()
    for pos in range(2, 7):
        step(1058, pos, pos+1)
    elapsed = time.perf_counter() - t0

    delta_computed = int(30 * ratio)
    delta_skipped = 30 - delta_computed
    strat = "all"
    if ratio <= 0.25: strat = "1 per 2 GQA blocks (5 ΔN)"
    elif ratio <= 0.33: strat = "1 per GQA block (10 ΔN)"
    elif ratio <= 0.5: strat = "every other (15 ΔN)"

    print(f"  {elapsed:.2f}s → {5/elapsed:.2f} tok/s ({delta_computed}/{30} ΔN, {delta_skipped} skipped)")
    print(f"  Strategy: {strat}")
