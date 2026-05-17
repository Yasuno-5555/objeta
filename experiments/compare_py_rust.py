#!/usr/bin/env python3
"""Compare Python numpy DeltaNet output vs Rust executor for the same input."""
import ctypes, numpy as np, json, mmap, math, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
lib = get_lib()

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
HDIM = 2048

# ── Python DeltaNet (from differential_trace) ────────────────────────────

class AttnWeights:
    def __init__(self, layer_idx):
        with open(BIN / f"layer_{layer_idx}_attn_f16.json") as f:
            self.meta = json.load(f)
        self._mmap = np.memmap(BIN / f"layer_{layer_idx}_attn_f16.bin", dtype=np.float16, mode='r')
    def get(self, name):
        shape, offset, nbytes = self.meta[name]
        nelem = nbytes // 2
        return self._mmap[offset//2 : offset//2 + nelem].reshape(shape).astype(np.float32)

def rms_norm(x, w):
    return (x / np.sqrt(np.mean(x**2) + 1e-6)) * w

N_KH, N_VH, HK, HV = 16, 32, 128, 128

def py_deltanet(h, a, state):
    w = {k: a.get(k) for k in [
        'linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight',
        'linear_attn.in_proj_b.weight','linear_attn.in_proj_a.weight',
        'linear_attn.out_proj.weight','linear_attn.norm.weight',
        'linear_attn.dt_bias','linear_attn.A_log',
    ]}
    w['conv'] = a.get('linear_attn.conv1d.weight').reshape(8192, 4)

    mqkv = w['linear_attn.in_proj_qkv.weight'] @ h
    z = w['linear_attn.in_proj_z.weight'] @ h
    b = w['linear_attn.in_proj_b.weight'] @ h
    a_vec = w['linear_attn.in_proj_a.weight'] @ h

    cs, ptr = state['conv_state'], state['conv_ptr']
    cs[:, ptr] = mqkv
    np_ptr = (ptr + 1) % 4; state['conv_ptr'] = np_ptr
    order = [(np_ptr - i + 4) % 4 for i in range(4)]
    qkv_conv = np.sum(w['conv'] * cs[:, order], axis=1)
    qkv_act = qkv_conv / (1.0 + np.exp(-qkv_conv))

    q = qkv_act[:2048].reshape(N_KH, HK); q_pre = np.linalg.norm(q, axis=1).mean()
    k = qkv_act[2048:4096].reshape(N_KH, HK)
    v = qkv_act[4096:].reshape(N_VH, HV)
    z_rs = z.reshape(N_VH, HV)

    rep = N_VH // N_KH
    q = np.repeat(q, rep, axis=0); k = np.repeat(k, rep, axis=0)
    beta = 1.0/(1.0+np.exp(-b))
    g = -np.exp(w['linear_attn.A_log']) * np.log(1.0+np.exp(a_vec+w['linear_attn.dt_bias']))
    q = q/(np.sqrt(np.sum(q**2,axis=1,keepdims=True))+1e-6) / math.sqrt(HK)
    k = k/(np.sqrt(np.sum(k**2,axis=1,keepdims=True))+1e-6)

    S = state['S']
    S = S * np.exp(g).reshape(N_VH, 1, 1)
    kv_mem = np.sum(S * k[:,:,None], axis=1)
    delta = (v - kv_mem) * beta.reshape(N_VH, 1)
    S = S + k[:,:,None] * delta[:,None,:]
    output = np.sum(S * q[...,None], axis=1)
    state['S'] = S

    rms = np.sqrt(np.mean(output**2, axis=1, keepdims=True) + 1e-6)
    on_n = (output/rms) * w['linear_attn.norm.weight'].reshape(1, HV)
    gated = on_n * z_rs / (1.0+np.exp(-z_rs))
    return w['linear_attn.out_proj.weight'] @ gated.reshape(-1), state, q_pre

# ── Init Rust runner ──
lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
lib.lko_runner_init.restype = ctypes.c_int32
lib.lko_runner_init(str(BIN).encode(), 128)

lib.lko_runner_forward.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
lib.lko_runner_forward.restype = ctypes.c_int32

# ── Compare ──
embed = np.memmap(BIN/"embed_tokens.bin", dtype=np.float32, mode='r').reshape(248320, HDIM)
a0 = AttnWeights(0)  # L0 is DeltaNet

# Prefill token 0 through all 40 layers
token_id = 1058  # "The"
print(f"Token {token_id} — comparing Python vs Rust hidden states\n")

# Python: just L0 DeltaNet
h_in = embed[token_id].copy()
in_w = a0.get('input_layernorm.weight')
h_norm = rms_norm(h_in, in_w)
py_state = {'conv_state': np.zeros((8192,4),dtype=np.float32), 'conv_ptr':0, 'S': np.zeros((N_VH,HK,HV),dtype=np.float32)}
py_h, _, q_pre = py_deltanet(h_norm, a0, py_state)
print(f"Python L0 DeltaNet output: norm={np.linalg.norm(py_h):.4f}, q_pre_l2={q_pre:.4f}")

# Rust: full 40 layers (forward pass)
rust_h = np.zeros(HDIM, dtype=np.float32)
lib.lko_runner_forward(token_id, 0, 1, rust_h.ctypes.data)
print(f"Rust full forward (40L):    norm={np.linalg.norm(rust_h):.4f}")

# Compare norms after just 1 layer would need the Rust executor to expose per-layer output
# For now, check if the Rust norm is reasonable
cos = np.dot(py_h, rust_h) / (np.linalg.norm(py_h) * np.linalg.norm(rust_h) + 1e-12)
print(f"cos(py_L0, rust_L39) = {cos:.6f} (different layers, expected low)")

# Also check: what's the input norm weight for L0?
in_w_norm = np.linalg.norm(in_w)
print(f"L0 input_norm weight: norm={in_w_norm:.4f} (expected ~1.0)")
print(f"  first 5: {in_w[:5]}")
print(f"  min={in_w.min():.4f} max={in_w.max():.4f} mean={in_w.mean():.4f}")
