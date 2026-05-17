#!/usr/bin/env python3
"""Diagnose where the model collapses: test 0..N layers, check token recovery."""
import ctypes, numpy as np, json, mmap, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib

lib = get_lib()
BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
HDIM = 2048

# ── Python reference: embedding + RMSNorm + lm_head ──

embed_data = np.memmap(BIN/"embed_tokens.bin", dtype=np.float32, mode='r').reshape(-1, HDIM)
fn_bytes = (BIN/"final_norm.bin").read_bytes()
final_norm = np.frombuffer(fn_bytes, dtype=np.float32)

def rms_norm(x, w):
    return (x / np.sqrt(np.mean(x**2) + 1e-6)) * w

def py_embed_lm_head(token_id):
    """Embedding + RMSNorm + lm_head, no layers."""
    h = embed_data[token_id].copy()
    hn = rms_norm(h, final_norm)
    logits = embed_data @ hn  # tied weights: embed @ hn = vocab scores
    return hn, logits

# ── Init Rust ──

lib.lko_metal_init.argtypes = [ctypes.c_char_p]
lib.lko_metal_init.restype = ctypes.c_int32
METALLIB = str(Path(__file__).parent.parent / "target" / "objeta.metallib")
lib.lko_metal_init(METALLIB.encode())

lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
lib.lko_runner_init.restype = ctypes.c_int32
assert lib.lko_runner_init(str(BIN).encode(), 128), "Runner init failed"

# Set fusion=1.0 (all DeltaNet layers), MoE on all layers (full compute, diagnostics)
lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
lib.lko_runner_set_fusion_ratio(1.0)     # all DeltaNet
lib.lko_runner_set_moe_on_deltanet(1)     # MoE on all layers — full compute

# N-layer forward API
lib.lko_runner_forward_n.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
lib.lko_runner_forward_n.restype = ctypes.c_int32

def rust_forward_n(token_id, pos, seq_len, n_layers):
    """Run only first N layers, return hidden state."""
    h = np.zeros(HDIM, dtype=np.float32)
    lib.lko_runner_forward_n(token_id, pos, seq_len, n_layers, h.ctypes.data)
    return h

def lm_head_score(h, target_token):
    """Apply RMSNorm + lm_head, return logit for target token."""
    hn = rms_norm(h, final_norm)
    logits = embed_data @ hn
    return float(logits[target_token]), np.argsort(logits)[-5:][::-1]

# ── Step-based forward: run N steps and collect per-step hidden state ──

lib.lko_runner_step.argtypes = [
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
]
lib.lko_runner_step.restype = ctypes.c_int32

def rust_step(token_id, pos, seq_len, top_k=10):
    hn = np.zeros(HDIM, dtype=np.float32)
    idx = np.zeros(top_k, dtype=np.int32)
    val = np.zeros(top_k, dtype=np.float32)
    k = lib.lko_runner_step(token_id, pos, seq_len, hn.ctypes.data, top_k, idx.ctypes.data, val.ctypes.data)
    return hn, idx[:k], val[:k]

# ══════════════════════════════════════════════════════════════════

print("=== Embedding + lm_head (no layers) ===")
for tid in [1058, 1, 100, 1000]:  # "The", ".", etc.
    hn, logits = py_embed_lm_head(tid)
    top5 = np.argsort(logits)[-5:][::-1]
    recovered = tid == top5[0]
    print(f"  token {tid}: top-5={top5.tolist()}, recovered={'✓' if recovered else '✗ (got '+str(top5[0])+')'}")
    top_score = logits[tid]
    print(f"    score for correct token: {top_score:.2f} (max: {logits.max():.2f})")

print("\n=== Layer-by-layer collapse: 'The' (1058) at pos=0 ===")
print(f"  {'Layers':<8} {'Norm':>8} {'Score(1058)':>12} {'Top-1':>8} {'Top-5'}")
token_id = 1058
for n in [0, 1, 2, 3, 4, 5, 8, 12, 16, 20, 30, 40]:
    h = rust_forward_n(token_id, 0, 1, n)
    hn = rms_norm(h, final_norm)
    logits = embed_data @ hn
    score = float(logits[token_id])
    top5 = np.argsort(logits)[-5:][::-1]
    norm = np.linalg.norm(h)
    print(f"  {n:<8} {norm:>8.2f} {score:>12.2f} {top5[0]:>8} {top5[:5].tolist()}")

print("\n=== Single layer delta check: L0 (DeltaNet) vs Python ===")
# Python: L0 DeltaNet
import json
class AttnWeights:
    def __init__(self, layer_idx):
        with open(BIN / f"layer_{layer_idx}_attn_f16.json") as f:
            self.meta = json.load(f)
        self._mmap = np.memmap(BIN / f"layer_{layer_idx}_attn_f16.bin", dtype=np.float16, mode='r')
    def get(self, name):
        shape, offset, nbytes = self.meta[name]
        nelem = nbytes // 2
        return self._mmap[offset//2 : offset//2 + nelem].reshape(shape).astype(np.float32)
    def has(self, name): return name in self.meta

a0 = AttnWeights(0)
h_in = embed_data[token_id].copy()
in_w = a0.get('input_layernorm.weight')
h_norm = rms_norm(h_in, in_w)

# Rust L0 output
rust_h0 = rust_forward_n(token_id, 0, 1, 1)

# Python L0
w_qkv = a0.get('linear_attn.in_proj_qkv.weight')
w_z = a0.get('linear_attn.in_proj_z.weight')
w_out = a0.get('linear_attn.out_proj.weight')

py_mqkv = w_qkv @ h_norm
py_z = w_z @ h_norm
py_qkv_conv = py_mqkv  # pos=0, conv state is all zeros, so weighted sum = mqkv[:,0]*w[:,0]
py_qkv_act = py_qkv_conv / (1.0 + np.exp(-py_qkv_conv))

N_KH, N_VH, HK, HV = 16, 32, 128, 128
q_raw = py_qkv_act[:2048].reshape(N_KH, HK)
k_raw = py_qkv_act[2048:4096].reshape(N_KH, HK)
v_raw = py_qkv_act[4096:].reshape(N_VH, HV)
z_rs = py_z.reshape(N_VH, HV)

rep = N_VH // N_KH
q = np.repeat(q_raw, rep, axis=0)
k = np.repeat(k_raw, rep, axis=0)

w_b = a0.get('linear_attn.in_proj_b.weight')
w_a = a0.get('linear_attn.in_proj_a.weight')
dt_bias = a0.get('linear_attn.dt_bias')
A_log = a0.get('linear_attn.A_log')
b = w_b @ h_norm
a_vec = w_a @ h_norm

beta = 1.0/(1.0+np.exp(-b))
g = -np.exp(A_log) * np.log(1.0+np.exp(a_vec+dt_bias))
q = q/(np.sqrt(np.sum(q**2,axis=1,keepdims=True))+1e-6) / math.sqrt(HK)
k = k/(np.sqrt(np.sum(k**2,axis=1,keepdims=True))+1e-6)

S = np.zeros((N_VH, HK, HV), dtype=np.float32)
S = S * np.exp(g).reshape(N_VH, 1, 1)
kv_mem = np.sum(S * k[:,:,None], axis=1)
delta = (v_raw - kv_mem) * beta.reshape(N_VH, 1)
S = S + k[:,:,None] * delta[:,None,:]
output = np.sum(S * q[...,None], axis=1)

w_norm = a0.get('linear_attn.norm.weight')
rms = np.sqrt(np.mean(output**2, axis=1, keepdims=True) + 1e-6)
on_n = (output/rms) * w_norm.reshape(1, HV)
gated = on_n * z_rs / (1.0+np.exp(-z_rs))
py_h = w_out @ gated.reshape(-1)

# Also: residual connection
py_h_out = h_norm + py_h  # L0 has no MoE/shared (pre-L3)

cos = np.dot(py_h_out, rust_h0) / (np.linalg.norm(py_h_out) * np.linalg.norm(rust_h0) + 1e-12)
print(f"  Python L0 norm: {np.linalg.norm(py_h_out):.4f}")
print(f"  Rust L0 norm:   {np.linalg.norm(rust_h0):.4f}")
print(f"  cos(py, rust):  {cos:.6f}")
print(f"  Python L0 top-1 after lm_head: {np.argsort(embed_data @ rms_norm(py_h_out, final_norm))[-1]}")
print(f"  Rust L0 top-1 after lm_head:   {np.argsort(embed_data @ rms_norm(rust_h0, final_norm))[-1]}")
print(f"  Correct token (1058): score_py={float((embed_data @ rms_norm(py_h_out, final_norm))[1058]):.2f}, score_rust={float((embed_data @ rms_norm(rust_h0, final_norm))[1058]):.2f}")

