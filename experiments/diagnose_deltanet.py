#!/usr/bin/env python3
"""Diagnose DeltaNet: conv1d state, q/k/v norms, S trace."""

import numpy as np, json, struct, mmap, math, sys, time
from pathlib import Path

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
HIDDEN = 2048
N_K_HEADS, N_V_HEADS = 16, 32
HK, HV = 128, 128
REP = N_V_HEADS // N_K_HEADS  # 2

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
    def has(self, name): return name in self.meta

def rms_norm(x, w):
    return (x / np.sqrt(np.mean(x**2) + 1e-6)) * w

def softplus_np(x):
    return np.log(1.0 + np.exp(x))

def diagnose_layer(l, h, state, a):
    """Run one DeltaNet layer and print ALL intermediate values."""
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

    # Conv1d
    conv_state = state['conv_state']  # (8192, 4)
    ptr = state['conv_ptr']
    conv_state[:, ptr] = mixed_qkv
    new_ptr = (ptr + 1) % 4
    # PyTorch Conv1d: weight[c,k] applied to input[t-k]
    # k=0 → newest, k=3 → oldest
    order = [(new_ptr - i + 4) % 4 for i in range(4)]  # newest to oldest
    qkv_conv = np.sum(w_conv * conv_state[:, order], axis=1)
    qkv_act = qkv_conv / (1.0 + np.exp(-qkv_conv))  # SiLU

    # Split
    q_raw = qkv_act[:2048].reshape(N_K_HEADS, HK)
    k_raw = qkv_act[2048:4096].reshape(N_K_HEADS, HK)
    v_raw = qkv_act[4096:].reshape(N_V_HEADS, HV)
    z_rs = z.reshape(N_V_HEADS, HV)

    q = np.repeat(q_raw, REP, axis=0)
    k = np.repeat(k_raw, REP, axis=0)

    beta = 1.0 / (1.0 + np.exp(-b))
    g = -np.exp(A_log) * softplus_np(a_vec + dt_bias)

    # L2 norm
    q_pre_norm = np.linalg.norm(q, axis=1)
    k_pre_norm = np.linalg.norm(k, axis=1)
    q = q / (np.sqrt(np.sum(q**2, axis=1, keepdims=True) + 1e-6))
    k = k / (np.sqrt(np.sum(k**2, axis=1, keepdims=True) + 1e-6))
    q_post_norm = np.linalg.norm(q, axis=1)
    k_post_norm = np.linalg.norm(k, axis=1)
    q = q / math.sqrt(HK)

    S = state['S']  # (32, 128, 128)
    S_norm_before = np.linalg.norm(S.reshape(N_V_HEADS, -1), axis=1)

    g_t = np.exp(g).reshape(N_V_HEADS, 1, 1)
    S = S * g_t
    kv_mem = np.sum(S * k[:, :, None], axis=1)
    delta = (v_raw - kv_mem) * beta.reshape(N_V_HEADS, 1)
    S = S + k[:, :, None] * delta[:, None, :]

    S_norm_after = np.linalg.norm(S.reshape(N_V_HEADS, -1), axis=1)
    output = np.sum(S * q[..., None], axis=1)  # (32, 128)

    # RMSNormGated
    rms = np.sqrt(np.mean(output**2, axis=1, keepdims=True) + 1e-6)
    on_normed = (output / rms) * w_norm.reshape(1, HV)
    gated = on_normed * z_rs / (1.0 + np.exp(-z_rs))
    final = w_out @ gated.reshape(-1)

    state['S'] = S
    state['conv_ptr'] = new_ptr

    # ── Report ──
    print(f"  mixed_qkv: norm={np.linalg.norm(mixed_qkv):.4f} "
          f"mean={np.mean(mixed_qkv):.4f} std={np.std(mixed_qkv):.4f}")
    print(f"  conv_state (col {ptr}): norm={np.linalg.norm(conv_state[:,ptr]):.4f} "
          f"mean={np.mean(conv_state[:,ptr]):.4f}")
    print(f"  w_conv: shape={w_conv.shape} norm={np.linalg.norm(w_conv):.4f} "
          f"mean={np.mean(w_conv):.6f}")
    print(f"  qkv_conv (after causal conv1d): norm={np.linalg.norm(qkv_conv):.4f} "
          f"mean={np.mean(qkv_conv):.4f} std={np.std(qkv_conv):.4f}")
    print(f"  qkv_act (after SiLU): norm={np.linalg.norm(qkv_act):.4f} "
          f"mean={np.mean(qkv_act):.4f} min={np.min(qkv_act):.4f} max={np.max(qkv_act):.4f}")
    print(f"  q: pre-L2norm={q_pre_norm.mean():.4f} post-L2norm={q_post_norm.mean():.4f} "
          f"(expected 1.0)")
    print(f"  k: pre-L2norm={k_pre_norm.mean():.4f} post-L2norm={k_post_norm.mean():.4f}")
    print(f"  v_raw: norm={np.linalg.norm(v_raw):.4f} mean={np.mean(np.abs(v_raw)):.4f}")
    print(f"  beta: mean={np.mean(beta):.4f} range=[{np.min(beta):.4f},{np.max(beta):.4f}]")
    print(f"  g: mean={np.mean(g):.4f} min={np.min(g):.4f} max={np.max(g):.4f}")
    print(f"  exp(g): mean={np.mean(np.exp(g)):.4f} max={np.max(np.exp(g)):.4f}")
    print(f"  A_log: mean={np.mean(A_log):.4f} range=[{np.min(A_log):.4f},{np.max(A_log):.4f}]")
    print(f"  dt_bias: mean={np.mean(dt_bias):.4f} range=[{np.min(dt_bias):.4f},{np.max(dt_bias):.4f}]")
    print(f"  S_norm: before={S_norm_before.mean():.4f} after={S_norm_after.mean():.4f}")
    print(f"  delta: norm={np.linalg.norm(delta):.4f} mean={np.mean(np.abs(delta)):.4f}")
    print(f"  output: norm={np.linalg.norm(output):.4f} mean={np.mean(np.abs(output)):.4f}")
    print(f"  RMSNormGated output: norm={np.linalg.norm(gated):.4f}")
    print(f"  FINAL out: norm={np.linalg.norm(final):.4f}")

    return final, state


def main():
    l = 1  # DeltaNet layer (not full GQA)
    print(f"=== DeltaNet Diagnosis — Layer {l} ===\n")

    a = AttnWeights(l)
    print(f"Layer {l} attention type: "
          f"{'Full GQA' if l % 4 == 3 else 'DeltaNet' if a.has('linear_attn.in_proj_qkv.weight') else 'NONE'}")

    # Simulate the first 5 tokens to fill conv state
    embed = np.memmap(BIN / "embed_tokens.bin", dtype=np.float32, mode='r')
    embed = embed.reshape(248320, HIDDEN)

    state = {
        'conv_state': np.zeros((8192, 4), dtype=np.float32),
        'conv_ptr': 0,
        'S': np.zeros((N_V_HEADS, HK, HV), dtype=np.float32),
    }

    # Use a real token ID
    token_ids = [1058, 304, 1374, 374, 279, 1313]  # "The meaning of life is" approximate

    for step, tid in enumerate(token_ids):
        h = embed[tid].astype(np.float32).copy()
        in_norm_w = a.get('input_layernorm.weight')
        h = rms_norm(h, in_norm_w)
        print(f"\n── Token {step} (id={tid}), h norm after input_norm: {np.linalg.norm(h):.4f} ──")
        final, state = diagnose_layer(l, h, state, a)

    # After 5 tokens, the conv state should be fully filled
    print(f"\n=== After {len(token_ids)} tokens ===")
    print(f"conv_state (all cols): norm={np.linalg.norm(state['conv_state']):.4f}")
    for c in range(4):
        print(f"  col {c}: norm={np.linalg.norm(state['conv_state'][:,c]):.4f} "
              f"mean={np.mean(state['conv_state'][:,c]):.4f}")
    print(f"State S: norm={np.linalg.norm(state['S']):.4f}")


if __name__ == "__main__":
    main()
