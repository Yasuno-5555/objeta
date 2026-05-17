#!/usr/bin/env python3
"""Test fused GQA Metal kernel: cos vs Python reference + tok/s."""
import ctypes, numpy as np, json, math, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
lib = get_lib()

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
L = 3  # first Full GQA layer
MAX_SEQ = 64

# ── Load weights ──
with open(BIN/f"layer_{L}_attn_f16.json") as f: meta = json.load(f)
mm = np.memmap(BIN/f"layer_{L}_attn_f16.bin", dtype=np.float16, mode='r')
def gw(n):
    s,o,nb = meta[n]; ne=nb//2
    return mm[o//2:o//2+ne].reshape(s).astype(np.float32)

qw = gw('self_attn.q_proj.weight')  # (8192, 2048)
kw = gw('self_attn.k_proj.weight')  # (512, 2048)
vw = gw('self_attn.v_proj.weight')  # (512, 2048)
ow = gw('self_attn.o_proj.weight')  # (2048, 4096)
in_w = gw('input_layernorm.weight')

# Concatenate QKV weights for fused kernel: Q(8192) + K(512) + V(512) = 9216
W_qkv = np.concatenate([qw, kw, vw], axis=0).astype(np.float32)
print(f"W_qkv shape: {W_qkv.shape}")

# ── RoPE tables ──
def make_rope(max_seq, hd):
    theta = 1.0 / (10000.0 ** (np.arange(0, hd, 2) / hd))
    freqs = np.arange(max_seq)[:, None] * theta[None, :]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

rope_cos, rope_sin = make_rope(MAX_SEQ, 256)

# ── Init Metal ──
lib.lko_metal_init.argtypes = [ctypes.c_char_p]
lib.lko_metal_init.restype = ctypes.c_int
r = lib.lko_metal_init(b'/Users/yasuno/projects/objeta/target/objeta.metallib')
assert r, "Metal init failed"

lib.lko_metal_fused_gqa.argtypes = [
    ctypes.c_void_p, ctypes.c_int32,  # W_qkv, bytes
    ctypes.c_void_p,                   # h
    ctypes.c_void_p, ctypes.c_void_p,  # rope_cos, rope_sin
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # pos, seq_len, max_seq
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,  # k_cache, v_cache, kv_bytes
    ctypes.c_void_p,                   # attn_out
]
lib.lko_metal_fused_gqa.restype = ctypes.c_int32

# ── Python reference GQA ──
def py_gqa(h, pos, seq_len, k_cache, v_cache):
    N_H, N_KV, HD = 16, 2, 256
    q_full = qw @ h
    n_q = N_H * HD  # 4096
    q = q_full[:n_q].reshape(N_H, HD)
    q_gate = 1.0 / (1.0 + np.exp(-q_full[n_q:]))  # (4096,)
    k = (kw @ h).reshape(N_KV, HD)
    v = (vw @ h).reshape(N_KV, HD)

    # RoPE
    d2 = HD // 2
    for hi in range(N_H):
        q_e, q_o = q[hi, :d2], q[hi, d2:]
        c = rope_cos[pos, :d2]; s = rope_sin[pos, :d2]
        q[hi] = np.concatenate([q_e*c - q_o*s, q_e*s + q_o*c])
    for hi in range(N_KV):
        k_e, k_o = k[hi, :d2], k[hi, d2:]
        c = rope_cos[pos, :d2]; s = rope_sin[pos, :d2]
        k[hi] = np.concatenate([k_e*c - k_o*s, k_e*s + k_o*c])

    # Write KV cache
    k_cache[:, pos, :] = k
    v_cache[:, pos, :] = v

    # Attention
    n_rep = N_H // N_KV
    k_rep = np.repeat(k_cache[:, :seq_len, :], n_rep, axis=0)
    v_rep = np.repeat(v_cache[:, :seq_len, :], n_rep, axis=0)
    scale = 1.0 / math.sqrt(HD)
    scores = np.sum(q[:, None, :] * k_rep, axis=-1) * scale
    scores -= np.max(scores, axis=-1, keepdims=True)
    attn_w = np.exp(scores); attn_w /= np.sum(attn_w, axis=-1, keepdims=True)
    attn_out = np.sum(attn_w[:, :, None] * v_rep, axis=1).flatten()

    # Q-gate
    gated = attn_out * q_gate
    return ow @ gated, (k_cache, v_cache)

# ── Test: 5 tokens, compare at each position ──
embed = np.memmap(BIN/"embed_tokens.bin", dtype=np.float32, mode='r').reshape(248320, 2048)
tokens = [1058, 304, 1374, 374, 279]

py_kv = (np.zeros((2, MAX_SEQ, 256), dtype=np.float32),
         np.zeros((2, MAX_SEQ, 256), dtype=np.float32))
metal_kv = (np.zeros((2, MAX_SEQ, 256), dtype=np.float32),
            np.zeros((2, MAX_SEQ, 256), dtype=np.float32))

print(f"\n{'='*60}")
print(f"Token  pos  seq_len  cos(py,metal)  py_norm  metal_norm")
print(f"{'='*60}")

for i, tid in enumerate(tokens):
    h_in = embed[tid].copy()
    h = h_in / np.sqrt(np.mean(h_in**2) + 1e-6) * in_w

    # Python reference
    py_ao, py_kv = py_gqa(h, i, i+1, py_kv[0], py_kv[1])

    # Metal fused GQA
    h_metal = h.astype(np.float32)
    kc = metal_kv[0].flatten()
    vc = metal_kv[1].flatten()
    attn_out = np.zeros(4096, dtype=np.float32)

    lib.lko_metal_fused_gqa(
        W_qkv.ctypes.data_as(ctypes.c_void_p), W_qkv.nbytes,
        h_metal.ctypes.data_as(ctypes.c_void_p),
        rope_cos.ctypes.data_as(ctypes.c_void_p),
        rope_sin.ctypes.data_as(ctypes.c_void_p),
        i, i+1, MAX_SEQ,
        kc.ctypes.data_as(ctypes.c_void_p),
        vc.ctypes.data_as(ctypes.c_void_p),
        kc.nbytes,
        attn_out.ctypes.data_as(ctypes.c_void_p),
    )

    # Reconstruct KV cache from flat buffer
    metal_kv = (kc.reshape(2, MAX_SEQ, 256).copy(),
                vc.reshape(2, MAX_SEQ, 256).copy())

    # Apply O-proj to Metal output (same as Python)
    metal_gated = attn_out  # Q-gate already applied in kernel
    metal_ao = ow @ metal_gated

    cos = np.dot(py_ao, metal_ao) / (np.linalg.norm(py_ao) * np.linalg.norm(metal_ao) + 1e-12)
    flag = "✓" if cos > 0.99 else ("△" if cos > 0.9 else "✗")
    print(f"  {i:>3d}   {i:>3d}     {i+1:>3d}     {cos:>12.6f} {flag}  {np.linalg.norm(py_ao):>8.4f}  {np.linalg.norm(metal_ao):>10.4f}")

# ── Check KV cache match ──
kv_cos_k = np.dot(py_kv[0].flatten(), metal_kv[0].flatten()) / (np.linalg.norm(py_kv[0]) * np.linalg.norm(metal_kv[0]) + 1e-12)
kv_cos_v = np.dot(py_kv[1].flatten(), metal_kv[1].flatten()) / (np.linalg.norm(py_kv[1]) * np.linalg.norm(metal_kv[1]) + 1e-12)
print(f"\nKV cache cos: K={kv_cos_k:.6f} V={kv_cos_v:.6f}")

# ── Benchmark ──
print(f"\n{'='*60}")
print(f"Benchmark: 10-layer GQA simulation (20 tokens)")
print(f"{'='*60}")
n_warmup, n_iters = 3, 10
for _ in range(n_warmup):
    kc_test = np.zeros(2*MAX_SEQ*256, dtype=np.float32)
    vc_test = np.zeros(2*MAX_SEQ*256, dtype=np.float32)
    ao_test = np.zeros(4096, dtype=np.float32)
    lib.lko_metal_fused_gqa(W_qkv.ctypes.data, W_qkv.nbytes, h_metal.ctypes.data, rope_cos.ctypes.data, rope_sin.ctypes.data, 0, 10, MAX_SEQ, kc_test.ctypes.data, vc_test.ctypes.data, kc_test.nbytes, ao_test.ctypes.data)

t0 = time.perf_counter()
for _ in range(n_iters):
    lib.lko_metal_fused_gqa(W_qkv.ctypes.data, W_qkv.nbytes, h_metal.ctypes.data, rope_cos.ctypes.data, rope_sin.ctypes.data, 0, 10, MAX_SEQ, kc_test.ctypes.data, vc_test.ctypes.data, kc_test.nbytes, ao_test.ctypes.data)
metal_ms = (time.perf_counter() - t0) / n_iters * 1000

print(f"Metal fused GQA: {metal_ms:.2f}ms (1 layer, seq_len=10)")
print(f"Estimated 10 GQA layers: {metal_ms*10:.1f}ms")
print(f"Estimated tok/s (40 layers total, ~20s): {1000/(metal_ms*10 + 7000):.2f} tok/s")
