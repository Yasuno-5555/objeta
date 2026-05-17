#!/usr/bin/env python3
"""Quick timing probe — measure exact time per component using forward_timed."""
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

# forward_timed API
lib.lko_runner_forward_timed.argtypes = [
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p,
]
lib.lko_runner_forward_timed.restype = ctypes.c_int32

BIN_DIR = str(Path(__file__).parent.parent / "models" / "qwen36_bin")

print("Loading...")
assert lib.lko_runner_init(BIN_DIR.encode(), 32), "Init failed"

# Warmup
hn = np.zeros(HDIM, dtype=np.float32)
timing = np.zeros(5, dtype=np.float64)
lib.lko_runner_forward_timed(1058, 0, 1, hn.ctypes.data, timing.ctypes.data)

# Timed measurement (5 tokens)
t_delta = np.zeros(5)
t_gqa = np.zeros(5)
t_shared = np.zeros(5)
t_moe = np.zeros(5)
t_norm = np.zeros(5)

for pos in range(5):
    lib.lko_runner_forward_timed(1058, pos, pos+1, hn.ctypes.data, timing.ctypes.data)
    t_delta[pos] = timing[0]
    t_gqa[pos] = timing[1]
    t_shared[pos] = timing[2]
    t_moe[pos] = timing[3]
    t_norm[pos] = timing[4]

print(f"\n  Component         Time/token    %")
print(f"  ───────────────── ────────  ─────")
total = np.mean([sum(t) for t in zip(t_delta, t_gqa, t_shared, t_moe, t_norm)])
for name, vals in [("DeltaNet", t_delta), ("GQA", t_gqa), ("SharedExpert", t_shared),
                    ("MoE dispatch", t_moe), ("RMSNorm", t_norm)]:
    mean_s = np.mean(vals)
    pct = mean_s / total * 100
    print(f"  {name:<17} {mean_s:>6.0f}ms  {pct:>5.0f}%")
print(f"  {'─'*17} {'─'*8}")
print(f"  {'Total':<17} {total*1000:>6.0f}ms")
print(f"  tok/s: {1/total:.2f}")
