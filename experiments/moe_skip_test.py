#!/usr/bin/env python3
"""MoE skip on DeltaNet layers — timing test."""
import ctypes, numpy as np, time, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
lib = get_lib()

HDIM = 2048

lib.lko_metal_init.argtypes = [ctypes.c_char_p]
lib.lko_metal_init.restype = ctypes.c_int32
METALLIB = str(Path(__file__).parent.parent / "target" / "objeta.metallib")
lib.lko_metal_init(METALLIB.encode())

lib.lko_runner_step.argtypes = [
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
]
lib.lko_runner_step.restype = ctypes.c_int32

lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32

BIN_DIR = str(Path(__file__).parent.parent / "models" / "qwen36_bin")

def step(token_id, pos, seq_len):
    hn = np.zeros(HDIM, dtype=np.float32)
    idx = np.zeros(50, dtype=np.int32)
    val = np.zeros(50, dtype=np.float32)
    lib.lko_runner_step(token_id, pos, seq_len, hn.ctypes.data, 50, idx.ctypes.data, val.ctypes.data)
    return hn, idx, val

print("Loading Qwen3.6-35B (one-time)...")
assert lib.lko_runner_init(BIN_DIR.encode(), 32), "Init failed"

for mode in ["MoE on all layers", "MoE only on GQA"]:
    print(f"\n{'='*50}")
    print(f"  {mode}")
    lib.lko_runner_set_moe_on_deltanet(1 if "all" in mode else 0)

    # Warmup
    step(1058, 0, 1)
    step(1058, 1, 2)

    t0 = time.perf_counter()
    for pos in range(2, 7):
        _, indices, _ = step(1058, pos, pos+1)
    elapsed = time.perf_counter() - t0

    print(f"  {elapsed:.2f}s → {5/elapsed:.2f} tok/s")
