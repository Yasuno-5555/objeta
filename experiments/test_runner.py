#!/usr/bin/env python3
"""Quick test of the Rust Qwen3.6 executor."""
import ctypes, numpy as np, time, sys
sys.path.insert(0, 'experiments')
from qwen36_executor import get_lib
lib = get_lib()

print("Initializing runner...", end=" ", flush=True)
t0 = time.perf_counter()
lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
lib.lko_runner_init.restype = ctypes.c_int32
r = lib.lko_runner_init(b'models/qwen36_bin', 128)
print(f"{'OK' if r else 'FAIL'} ({time.perf_counter()-t0:.1f}s)")

lib.lko_runner_forward.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
lib.lko_runner_forward.restype = ctypes.c_int32

h = np.zeros(2048, dtype=np.float32)
tokens = [1058, 304, 1374, 374, 279]
times = []
for i, tid in enumerate(tokens):
    t0 = time.perf_counter()
    lib.lko_runner_forward(tid, i, i+1, h.ctypes.data)
    t = time.perf_counter()-t0
    times.append(t)
    print(f'Token {i}: {t:.2f}s norm={np.linalg.norm(h):.2f}')

avg = sum(times[1:])/len(times[1:]) if len(times)>1 else times[0]
print(f'\nAvg (excluding cold start): {avg:.2f}s = {1/avg:.2f} tok/s')
