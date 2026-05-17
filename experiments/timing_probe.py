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

# Apply fusion ratio + MoE skip (must happen BEFORE warmup/build_caches)
lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32

FUSION = 0.33
MOE_ON_DN = 0
lib.lko_runner_set_fusion_ratio(FUSION)
lib.lko_runner_set_moe_on_deltanet(MOE_ON_DN)

# Warmup + build expert caches
lib.lko_runner_warmup.argtypes = [ctypes.c_int32]
lib.lko_runner_warmup.restype = ctypes.c_int32
lib.lko_runner_build_caches.argtypes = [ctypes.c_int32]
lib.lko_runner_build_caches.restype = ctypes.c_int32

# Warmup: touch q4 pages to bring them into OS page cache
print("Warming OS page cache...")
lib.lko_runner_warmup.argtypes = [ctypes.c_int32]
lib.lko_runner_warmup.restype = ctypes.c_int32
lib.lko_runner_warmup(100)

# First forwards to prime caches (positions 0-2)
print("Priming caches...")
hn = np.zeros(HDIM, dtype=np.float32)
timing = np.zeros(5, dtype=np.float64)
for pos in range(3):
    lib.lko_runner_forward_timed(1058, pos, pos+1, hn.ctypes.data, timing.ctypes.data)
    t = timing.sum()
    print(f"  Pos {pos}: {t*1000:.0f}ms ({1/t:.2f} tok/s)")

# Timed measurement (positions 3-10)
n_measure = 8
t_delta = np.zeros(n_measure)
t_gqa = np.zeros(n_measure)
t_shared = np.zeros(n_measure)
t_moe = np.zeros(n_measure)
t_norm = np.zeros(n_measure)

for i in range(n_measure):
    pos = 3 + i
    lib.lko_runner_forward_timed(1058, pos, pos+1, hn.ctypes.data, timing.ctypes.data)
    t_delta[i] = timing[0]
    t_gqa[i] = timing[1]
    t_shared[i] = timing[2]
    t_moe[i] = timing[3]
    t_norm[i] = timing[4]

steady = slice(2, n_measure)  # skip first 2 warmup tokens in measured set
print(f"\n  ── ΔN={FUSION:.0%} ({int(30*FUSION)}/30), MoE on ΔN={'yes' if MOE_ON_DN else 'no'} (OS page cache) ──")
print(f"  Component         Time/token    %")
print(f"  ───────────────── ────────  ─────")
total = np.mean([sum(t) for t in zip(t_delta[steady], t_gqa[steady], t_shared[steady], t_moe[steady], t_norm[steady])])
for name, vals in [("DeltaNet", t_delta[steady]), ("GQA", t_gqa[steady]), ("SharedExpert", t_shared[steady]),
                    ("MoE dispatch", t_moe[steady]), ("RMSNorm", t_norm[steady])]:
    mean_s = np.mean(vals)
    pct = mean_s / total * 100
    print(f"  {name:<17} {mean_s:>6.0f}ms  {pct:>5.0f}%")
print(f"  {'─'*17} {'─'*8}")
print(f"  {'Total':<17} {total*1000:>6.0f}ms")
print(f"  tok/s: {1/total:.2f}")
