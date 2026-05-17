#!/usr/bin/env python3
"""Profile Rust forward pass: where does the time go?"""
import ctypes, numpy as np, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
lib = get_lib()

BIN = str(Path(__file__).parent.parent / "models" / "qwen36_bin")
lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
lib.lko_runner_init.restype = ctypes.c_int32
assert lib.lko_runner_init(BIN.encode(), 256)

lib.lko_runner_forward_timed.argtypes = [
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p,
]
lib.lko_runner_forward_timed.restype = ctypes.c_int32

# Run 3 tokens, measure breakdown
tokens = [1058, 304, 1374]
totals = np.zeros(5)

for i, tid in enumerate(tokens):
    h = np.zeros(2048, dtype=np.float32)
    timing = np.zeros(5, dtype=np.float64)
    t0 = time.perf_counter()
    lib.lko_runner_forward_timed(tid, i, i+1, h.ctypes.data, timing.ctypes.data)
    wall = time.perf_counter() - t0
    totals += timing * 1000  # convert to ms
    print(f"Token {i}: wall={wall:.1f}s delta={timing[0]*1000:.0f}ms gqa={timing[1]*1000:.0f}ms shared={timing[2]*1000:.0f}ms moe={timing[3]*1000:.0f}ms norm={timing[4]*1000:.0f}ms norm_h={np.linalg.norm(h):.1f}")

avg = totals / len(tokens)
total_ms = avg.sum()
print(f"\nAverage (ms): delta={avg[0]:.0f} gqa={avg[1]:.0f} shared={avg[2]:.0f} moe={avg[3]:.0f} norm={avg[4]:.0f}")
print(f"Total: {total_ms:.0f}ms = {total_ms/1000:.1f}s")
print(f"  delta_net: {avg[0]/total_ms*100:.0f}%")
print(f"  gqa_attn:  {avg[1]/total_ms*100:.0f}%")
print(f"  shared:    {avg[2]/total_ms*100:.0f}%")
print(f"  moe:       {avg[3]/total_ms*100:.0f}%")
