#!/usr/bin/env python3
"""Check if Rust loads the same weight values as Python."""
import ctypes, numpy as np, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
lib = get_lib()

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"

# Python: load weight
with open(BIN/"layer_0_attn_f16.json") as f: meta = json.load(f)
mm = np.memmap(BIN/"layer_0_attn_f16.bin", dtype=np.float16, mode='r')

# Get in_proj_qkv (first DeltaNet weight)
shape, off, nbytes = meta['linear_attn.in_proj_qkv.weight']
nelem = nbytes // 2
py_w = mm[off//2 : off//2 + nelem].reshape(shape).astype(np.float32)

print(f"linear_attn.in_proj_qkv.weight: shape={shape}, offset={off}, nbytes={nbytes}")
print(f"Python: first 10 values = {py_w.flatten()[:10]}")
print(f"Python: last 10 values = {py_w.flatten()[-10:]}")
print(f"Python: norm={np.linalg.norm(py_w):.4f}, mean={py_w.mean():.6f}")

# Rust: call the GEMV with known input to infer weight values
# lko_q36_f16_gemv(W, M, K, x, y) computes y = W @ x
# We'll pass a one-hot vector to extract each row/column

# Test: compute GEMV with identity-like input to check first few weights
lib.lko_q36_f16_gemv.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
lib.lko_q36_f16_gemv.restype = ctypes.c_int32

M, K = shape
# Pass the weight bytes directly via Rust FFI
# The .bin file data starts at the beginning of the mmap
# But we need to pass the right slice: from `off` for `nbytes` bytes
raw = mm[off//2 : off//2 + nelem]  # f16 numpy array
w_bytes = raw.tobytes()
print(f"\nWeight bytes: {len(w_bytes)} (expected {nbytes})")

# Test: compute y = W @ [1,0,0,...] → should give first column
x = np.zeros(K, dtype=np.float32); x[0] = 1.0
y_rust = np.zeros(M, dtype=np.float32)
lib.lko_q36_f16_gemv(w_bytes, M, K, x.ctypes.data, y_rust.ctypes.data)
y_py = (py_w @ x)
print(f"\nW @ [1,0,0,...]:")
print(f"Python first 5: {y_py[:5]}")
print(f"Rust   first 5: {y_rust[:5]}")

# Test: y = W @ [0,1,0,...] → second column
x2 = np.zeros(K, dtype=np.float32); x2[1] = 1.0
y2_rust = np.zeros(M, dtype=np.float32)
lib.lko_q36_f16_gemv(w_bytes, M, K, x2.ctypes.data, y2_rust.ctypes.data)
y2_py = (py_w @ x2)
print(f"\nW @ [0,1,0,...]:")
print(f"Python first 5: {y2_py[:5]}")
print(f"Rust   first 5: {y2_rust[:5]}")

# Check: should match py_w[i,0] and py_w[i,1]
print(f"\npy_w[0,0]={py_w[0,0]:.6f} py_w[0,1]={py_w[0,1]:.6f}")
print(f"rust[0]={y_rust[0]:.6f} rust[1]={y2_rust[0]:.6f}")

cos = np.dot(y_rust, y_py)/(np.linalg.norm(y_rust)*np.linalg.norm(y_py)+1e-12)
print(f"\ncos(rust, py) for first column: {cos:.6f}")
