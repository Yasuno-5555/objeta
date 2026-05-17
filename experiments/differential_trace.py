#!/usr/bin/env python3
"""A1: Differential Trace — find where MLX Qwen3.6 diverges.

Traces hidden state at every intermediate point through one forward pass.
Identifies the first layer where norm/cosine diverges from expected behavior.

Usage:
  python experiments/differential_trace.py
"""

import ctypes, json, math, os, sys, struct, mmap, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
_lib = get_lib()

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
HIDDEN_DIM = 2048
N_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256
N_K_HEADS = 16
N_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128

# ── Weight Loading (same mmap approach) ──────────────────────────────────

class AttnWeights:
    def __init__(self, layer_idx):
        with open(BIN / f"layer_{layer_idx}_attn_f16.json") as f:
            self.meta = json.load(f)
        self._mmap = np.memmap(
            BIN / f"layer_{layer_idx}_attn_f16.bin", dtype=np.float16, mode='r')

    def get(self, name):
        shape, offset, nbytes = self.meta[name]
        nelem = nbytes // 2
        return self._mmap[offset // 2 : offset // 2 + nelem].reshape(shape).astype(np.float32)

    def has(self, name):
        return name in self.meta

# ── RMSNorm ───────────────────────────────────────────────────────────────

def rms_norm(x, w):
    rms = np.sqrt(np.mean(x**2) + 1e-6)
    return (x / rms) * w

# ── RoPE ──────────────────────────────────────────────────────────────────

def precompute_rope(max_seq, head_dim):
    theta = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = np.arange(max_seq)[:, None] * theta[None, :]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

def apply_rope(x, cos, sin, pos):
    d2 = x.shape[-1] // 2
    c = cos[pos, :d2][None, :]
    s = sin[pos, :d2][None, :]
    return np.concatenate([
        x[:, :d2] * c - x[:, d2:] * s,
        x[:, :d2] * s + x[:, d2:] * c,
    ], axis=-1)

# ── Full GQA Attention (numpy, no MLX) ────────────────────────────────────

def full_attn_np(h, a, kv_cache, pos, seq_len, rope_cos, rope_sin):
    q_w = a.get('self_attn.q_proj.weight')
    k_w = a.get('self_attn.k_proj.weight')
    v_w = a.get('self_attn.v_proj.weight')
    o_w = a.get('self_attn.o_proj.weight')

    q_full = q_w @ h
    q = q_full[:N_HEADS * HEAD_DIM].reshape(N_HEADS, HEAD_DIM)
    q_gate = 1.0 / (1.0 + np.exp(-q_full[N_HEADS * HEAD_DIM:]))
    k = (k_w @ h).reshape(N_KV_HEADS, HEAD_DIM)
    v = (v_w @ h).reshape(N_KV_HEADS, HEAD_DIM)

    q = apply_rope(q, rope_cos, rope_sin, pos)
    k = apply_rope(k, rope_cos, rope_sin, pos)

    Kc, Vc = kv_cache
    Kc[:, pos, :] = k
    Vc[:, pos, :] = v

    n_rep = N_HEADS // N_KV_HEADS
    k_rep = np.repeat(Kc[:, :seq_len, :], n_rep, axis=0)
    v_rep = np.repeat(Vc[:, :seq_len, :], n_rep, axis=0)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = np.sum(q[:, None, :] * k_rep, axis=-1) * scale
    scores -= np.max(scores, axis=-1, keepdims=True)
    attn_w = np.exp(scores)
    attn_w /= np.sum(attn_w, axis=-1, keepdims=True)
    attn_out = np.sum(attn_w[:, :, None] * v_rep, axis=1).flatten()

    return o_w @ (attn_out * q_gate), (Kc, Vc)

# ── GatedDeltaNet (numpy) ─────────────────────────────────────────────────

def delta_net_np(h, a, state):
    w_qkv = a.get('linear_attn.in_proj_qkv.weight')
    w_z = a.get('linear_attn.in_proj_z.weight')
    w_b = a.get('linear_attn.in_proj_b.weight')
    w_a = a.get('linear_attn.in_proj_a.weight')
    w_out = a.get('linear_attn.out_proj.weight')
    w_conv = a.get('linear_attn.conv1d.weight').reshape(8192, 4)
    w_norm = a.get('linear_attn.norm.weight')
    dt_bias = a.get('linear_attn.dt_bias')
    A_log = a.get('linear_attn.A_log')

    mixed_qkv = w_qkv @ h  # (8192,)
    z = w_z @ h            # (4096,)
    b = w_b @ h            # (32,)
    a_vec = w_a @ h        # (32,)

    # Causal conv1d
    conv_state = state['conv_state']
    ptr = state['conv_ptr']
    conv_state[:, ptr] = mixed_qkv
    new_ptr = (ptr + 1) % 4
    state['conv_ptr'] = new_ptr
    order = [(ptr+1)%4, (ptr+2)%4, (ptr+3)%4, ptr]  # weight[3]=newest (cross-corr)
    qkv_conv = np.sum(w_conv * conv_state[:, order], axis=1)
    qkv_act = qkv_conv / (1.0 + np.exp(-qkv_conv))  # SiLU

    q = qkv_act[:2048].reshape(N_K_HEADS, HEAD_K_DIM)
    k = qkv_act[2048:4096].reshape(N_K_HEADS, HEAD_K_DIM)
    v = qkv_act[4096:].reshape(N_V_HEADS, HEAD_V_DIM)
    z = z.reshape(N_V_HEADS, HEAD_V_DIM)

    rep = N_V_HEADS // N_K_HEADS
    q = np.repeat(q, rep, axis=0)
    k = np.repeat(k, rep, axis=0)

    beta = 1.0 / (1.0 + np.exp(-b))
    # softplus: log(1 + exp(x))
    sp = np.log(1.0 + np.exp(a_vec + dt_bias))
    g = -np.exp(A_log) * sp

    # L2 normalize q and k
    q_n = np.sqrt(np.sum(q**2, axis=1, keepdims=True) + 1e-6)
    k_n = np.sqrt(np.sum(k**2, axis=1, keepdims=True) + 1e-6)
    q = q / q_n
    k = k / k_n
    q = q / math.sqrt(HEAD_K_DIM)

    S = state['S']  # (32, 128, 128)
    g_t = np.exp(g).reshape(N_V_HEADS, 1, 1)
    S = S * g_t

    kv_mem = np.sum(S * k[:, :, None], axis=1)  # (32, 128)
    delta = (v - kv_mem) * beta.reshape(N_V_HEADS, 1)
    S = S + k[:, :, None] * delta[:, None, :]
    output = np.sum(S * q[..., None], axis=1)  # (32, 128)
    state['S'] = S

    # RMSNormGated
    on = output
    rms = np.sqrt(np.mean(on**2, axis=1, keepdims=True) + 1e-6)
    on_normed = (on / rms) * w_norm.reshape(1, HEAD_V_DIM)
    gated = on_normed * z / (1.0 + np.exp(-z))  # z * sigmoid(z)
    return w_out @ gated.reshape(-1)

# ── MoE dispatch via Rust ─────────────────────────────────────────────────

def rust_moe(x_np, router, gu_mmap, d_mmap, layer_idx):
    out = np.zeros(HIDDEN_DIM, dtype=np.float32)
    eidx = np.zeros(8, dtype=np.int32)
    ew = np.zeros(8, dtype=np.float32)
    _lib.lko_moe_forward_layer(
        router.ctypes.data_as(ctypes.c_void_p),
        gu_mmap.ctypes.data_as(ctypes.c_void_p), gu_mmap.nbytes,
        d_mmap.ctypes.data_as(ctypes.c_void_p), d_mmap.nbytes,
        x_np.ctypes.data_as(ctypes.c_void_p), 8, layer_idx,
        eidx.ctypes.data_as(ctypes.c_void_p),
        ew.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    return out

# ── Trace Points ──────────────────────────────────────────────────────────

class TracePoint:
    """Snapshot of hidden state at one point in the pipeline."""
    def __init__(self, name, h, meta=None):
        self.name = name
        self.h = h.copy()
        self.norm = float(np.linalg.norm(h))
        self.mean = float(np.mean(h))
        self.std = float(np.std(h))
        self.min = float(np.min(h))
        self.max = float(np.max(h))
        self.meta = meta or {}

    def __repr__(self):
        return (f"  {self.name:<40s} norm={self.norm:>10.4f} "
                f"mean={self.mean:>8.4f} std={self.std:>8.4f} "
                f"range=[{self.min:.4f}, {self.max:.4f}]")

# ── Main Differential Trace ───────────────────────────────────────────────

def trace_layer(l, h, pos, seq_len, kv_cache, delta_state, attn_weights, rope_cos, rope_sin, routers, gu_mmaps, d_mmaps):
    """Run one layer and collect trace points."""
    trace = []
    a = attn_weights[l]
    trace.append(TracePoint(f"L{l} input", h))

    # ── Input RMSNorm ──
    if a.has('input_layernorm.weight'):
        in_norm_w = a.get('input_layernorm.weight')
        hn = rms_norm(h, in_norm_w)
    else:
        hn = h.copy()
    trace.append(TracePoint(f"L{l} after input_norm", hn))

    # ── Attention ──
    if l % 4 == 3:  # Full GQA
        ao, kv_cache = full_attn_np(hn, a, kv_cache, pos, seq_len, rope_cos, rope_sin)
        attn_type = "full_gqa"
    elif a.has('linear_attn.in_proj_qkv.weight'):
        ao = delta_net_np(hn, a, delta_state)
        attn_type = "delta_net"
    else:
        ao = np.zeros(HIDDEN_DIM, dtype=np.float32)
        attn_type = "none"
    trace.append(TracePoint(f"L{l} attn_out ({attn_type})", ao,
                            {"attn_type": attn_type}))

    # ── Residual after attention ──
    h_after_attn = h + ao
    trace.append(TracePoint(f"L{l} after attn_residual", h_after_attn))

    # ── Post-attention RMSNorm ──
    if a.has('post_attention_layernorm.weight'):
        post_norm_w = a.get('post_attention_layernorm.weight')
        hn2 = rms_norm(h_after_attn, post_norm_w)
    else:
        hn2 = h_after_attn.copy()
    trace.append(TracePoint(f"L{l} after post_norm", hn2))

    # ── Shared Expert (if present) ──
    shared_delta = np.zeros(HIDDEN_DIM, dtype=np.float32)
    if a.has('mlp.shared_expert.gate_proj.weight'):
        gate_w = a.get('mlp.shared_expert.gate_proj.weight')
        up_w = a.get('mlp.shared_expert.up_proj.weight')
        down_w = a.get('mlp.shared_expert.down_proj.weight')
        gate_w_gate = a.get('mlp.shared_expert_gate.weight').flatten()

        gate_out = gate_w @ hn2
        up_out = up_w @ hn2
        intermediate = gate_out / (1.0 + np.exp(-gate_out)) * up_out  # SiLU
        shared_raw = down_w @ intermediate
        shared_gate = 1.0 / (1.0 + np.exp(-(gate_w_gate @ hn2)))  # sigmoid gate
        shared_delta = shared_raw * shared_gate
        trace.append(TracePoint(f"L{l} shared_expert_out", shared_delta,
                                {"has_shared": True}))
    else:
        trace.append(TracePoint(f"L{l} shared_expert_out (NONE)", shared_delta,
                                {"has_shared": False}))

    # ── Routed MoE ──
    router = routers[l]
    moe_out = rust_moe(hn2.astype(np.float32), router, gu_mmaps[l], d_mmaps[l], l)
    trace.append(TracePoint(f"L{l} moe_out", moe_out))

    # ── Final residual ──
    h_new = h_after_attn + shared_delta + moe_out
    trace.append(TracePoint(f"L{l} output", h_new))

    return h_new, kv_cache, trace

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Differential Trace — Finding where Qwen3.6 diverges")
    print("=" * 70)

    # Load embedding + final norm
    embed = np.memmap(BIN / "embed_tokens.bin", dtype=np.float32, mode='r')
    embed = embed.reshape(248320, HIDDEN_DIM)
    final_norm_w = np.fromfile(BIN / "final_norm.bin", dtype=np.float32)
    print(f"Embedding: {embed.shape}, final_norm: {final_norm_w.shape}")

    # Load attention weights (lazy)
    print("Loading attention weights...")
    attn_weights = [AttnWeights(l) for l in range(40)]

    # Load routers and MoE mmaps
    print("Loading routers and MoE weights...")
    routers = []
    gu_mmaps = []
    d_mmaps = []
    for l in range(40):
        routers.append(np.fromfile(BIN / f"layer_{l}_router.bin", dtype=np.float32).reshape(256, HIDDEN_DIM))
        gu_mmaps.append(np.memmap(BIN / f"layer_{l}_gate_up.bin", dtype=np.uint8, mode='r'))
        d_mmaps.append(np.memmap(BIN / f"layer_{l}_down.bin", dtype=np.uint8, mode='r'))

    # Precompute RoPE
    rope_cos, rope_sin = precompute_rope(128, HEAD_DIM)

    # ── Run trace ──
    token_id = 1058  # "The" in Qwen tokenizer
    print(f"\nTracing token {token_id}...")

    h = embed[token_id].astype(np.float32).copy()
    print(f"Token embedding: norm={np.linalg.norm(h):.4f}")

    kv_caches = [(np.zeros((N_KV_HEADS, 128, HEAD_DIM), dtype=np.float32),
                  np.zeros((N_KV_HEADS, 128, HEAD_DIM), dtype=np.float32))
                 for _ in range(40)]
    delta_states = [{
        'conv_state': np.zeros((8192, 4), dtype=np.float32),
        'conv_ptr': 0,
        'S': np.zeros((N_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float32),
    } for _ in range(40)]

    # ── TRACE ALL LAYERS ──
    print(f"\n{'='*70}")
    print(f"{'Layer Trace':<42s} {'norm':>10s} {'mean':>8s} {'std':>8s} {'range':>20s}")
    print(f"{'='*70}")

    prev_norm = np.linalg.norm(h)
    divergence_layer = None
    all_traces = []

    for l in range(40):
        h, kv_caches[l], trace = trace_layer(
            l, h, 0, 1, kv_caches[l], delta_states[l],
            attn_weights, rope_cos, rope_sin, routers, gu_mmaps, d_mmaps
        )
        all_traces.append(trace)

        # Only print key trace points
        for tp in trace:
            is_key = any(x in tp.name for x in ['input', 'output', 'attn_out', 'moe_out', 'shared_expert'])
            if is_key:
                print(tp)

        # Divergence detection: sudden norm change
        curr_norm = np.linalg.norm(h)
        norm_ratio = curr_norm / max(prev_norm, 1e-12)
        if norm_ratio > 5.0 or norm_ratio < 0.2:
            if divergence_layer is None:
                divergence_layer = l
                print(f"  ⚠ DIVERGENCE at L{l}: norm ratio = {norm_ratio:.2f}")
        prev_norm = curr_norm
        print()

    # Final norm
    hn = rms_norm(h, final_norm_w)
    print(f"\nFinal: norm after RMSNorm = {np.linalg.norm(hn):.4f}")

    if divergence_layer is not None:
        print(f"\n⚠ First divergence at L{divergence_layer}")
    else:
        print(f"\n✓ No norm divergence detected in 40 layers")

    # ── Per-layer delta analysis ──
    print(f"\n{'='*70}")
    print("Per-layer Δ analysis (where does the trajectory go?)")
    print(f"{'='*70}")
    print(f"{'L':<4} {'||Δ_attn||':>12} {'||Δ_shared||':>12} {'||Δ_moe||':>12} {'cos(h,Δ)':>10}")
    print("-" * 52)

    for l in range(40):
        trace = all_traces[l]
        h_in = trace[0].h  # input
        h_out = trace[-1].h  # output
        delta = h_out - h_in

        # Find attn, shared, moe contributions from trace
        attn_out = next((t.h for t in trace if 'attn_out' in t.name), np.zeros(HIDDEN_DIM))
        shared_out = next((t.h for t in trace if 'shared_expert' in t.name), np.zeros(HIDDEN_DIM))
        moe_out = next((t.h for t in trace if 'moe_out' in t.name), np.zeros(HIDDEN_DIM))

        dn = np.linalg.norm(delta)
        cos_hd = np.dot(h_in, delta) / (np.linalg.norm(h_in) * dn + 1e-12)

        flag = ""
        if abs(cos_hd) < 0.01: flag = " ⟂"
        elif cos_hd < 0: flag = " ⇅"

        print(f"L{l:<3} {np.linalg.norm(attn_out):>12.4f} {np.linalg.norm(shared_out):>12.4f} "
              f"{np.linalg.norm(moe_out):>12.4f} {cos_hd:>10.4f}{flag}")


if __name__ == "__main__":
    main()
