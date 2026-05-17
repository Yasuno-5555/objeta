#!/usr/bin/env python3
"""A1: Compare our DeltaNet vs HuggingFace reference, single layer, single token.

Traces every intermediate value to find where pre-L2 norm=0.04 originates.
"""

import numpy as np
import json, struct, mmap, math, sys, time
from pathlib import Path
import torch

BIN = Path(__file__).parent.parent / "models" / "qwen36_bin"
HDIM = 2048
N_K_HEADS, N_V_HEADS = 16, 32
HK, HV = 128, 128

# ── Our numpy DeltaNet (exact copy from generator) ────────────────────────

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

def rms_norm(x, w):
    return (x / np.sqrt(np.mean(x**2) + 1e-6)) * w

def our_deltanet(h, a, state):
    """Our numpy implementation — returns (output, trace_dict)."""
    t = {}
    w_qkv = a.get('linear_attn.in_proj_qkv.weight')
    w_z = a.get('linear_attn.in_proj_z.weight')
    w_b = a.get('linear_attn.in_proj_b.weight')
    w_a = a.get('linear_attn.in_proj_a.weight')
    w_out = a.get('linear_attn.out_proj.weight')
    w_conv = a.get('linear_attn.conv1d.weight').reshape(8192, 4)
    w_norm = a.get('linear_attn.norm.weight')
    dt_bias = a.get('linear_attn.dt_bias')
    A_log = a.get('linear_attn.A_log')

    mixed_qkv = w_qkv @ h; t['mixed_qkv_norm'] = np.linalg.norm(mixed_qkv)
    z = w_z @ h; t['z_norm'] = np.linalg.norm(z)
    b = w_b @ h; t['b_mean'] = np.mean(b)
    a_vec = w_a @ h

    cs, ptr = state['conv_state'], state['conv_ptr']
    cs[:, ptr] = mixed_qkv
    new_ptr = (ptr + 1) % 4; state['conv_ptr'] = new_ptr
    order = [(new_ptr - i + 4) % 4 for i in range(4)]
    qkv_conv = np.sum(w_conv * cs[:, order], axis=1)
    t['qkv_conv_norm'] = np.linalg.norm(qkv_conv)
    qkv_act = qkv_conv / (1.0 + np.exp(-qkv_conv))
    t['qkv_act_norm'] = np.linalg.norm(qkv_act)

    q = qkv_act[:2048].reshape(N_K_HEADS, HK); t['q_pre_l2norm'] = np.linalg.norm(q, axis=1).mean()
    k = qkv_act[2048:4096].reshape(N_K_HEADS, HK); t['k_pre_l2norm'] = np.linalg.norm(k, axis=1).mean()
    v = qkv_act[4096:].reshape(N_V_HEADS, HV); t['v_norm'] = np.linalg.norm(v)
    z_rs = z.reshape(N_V_HEADS, HV)

    rep = N_V_HEADS // N_K_HEADS
    q = np.repeat(q, rep, axis=0); k = np.repeat(k, rep, axis=0)

    beta = 1.0 / (1.0 + np.exp(-b))
    g = -np.exp(A_log) * np.log(1.0 + np.exp(a_vec + dt_bias))
    t['beta_mean'] = np.mean(beta); t['g_mean'] = np.mean(g)
    t['exp_g_mean'] = np.mean(np.exp(g))

    q = q / (np.sqrt(np.sum(q**2, axis=1, keepdims=True)) + 1e-6)
    k = k / (np.sqrt(np.sum(k**2, axis=1, keepdims=True)) + 1e-6)
    q = q / math.sqrt(HK)
    t['q_post_l2norm'] = np.linalg.norm(q, axis=1).mean()
    t['k_post_l2norm'] = np.linalg.norm(k, axis=1).mean()

    S = state['S']
    t['S_norm_before'] = np.linalg.norm(S)
    S = S * np.exp(g).reshape(N_V_HEADS, 1, 1)
    kv_mem = np.sum(S * k[:, :, None], axis=1)
    delta = (v - kv_mem) * beta.reshape(N_V_HEADS, 1)
    S = S + k[:, :, None] * delta[:, None, :]
    output = np.sum(S * q[..., None], axis=1)
    t['S_norm_after'] = np.linalg.norm(S)
    t['delta_mean_abs'] = np.mean(np.abs(delta))
    t['output_pre_norm'] = np.linalg.norm(output)
    state['S'] = S

    rms = np.sqrt(np.mean(output**2, axis=1, keepdims=True) + 1e-6)
    on_normed = (output / rms) * w_norm.reshape(1, HV)
    gated = on_normed * z_rs / (1.0 + np.exp(-z_rs))
    t['gated_norm'] = np.linalg.norm(gated)

    final = w_out @ gated.reshape(-1)
    t['final_norm'] = np.linalg.norm(final)
    return final, state, t

# ── HuggingFace reference DeltaNet ────────────────────────────────────────

def load_hf_deltanet(layer_idx):
    """Load a single Qwen3.6 layer from HuggingFace."""
    from transformers import AutoConfig
    import torch.nn as nn

    # Load config + model on CPU
    print(f"Loading HF Qwen3.6-35B-A3B (layer {layer_idx} only)...", end=" ", flush=True)
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)

    # Load full model but only keep one layer to save memory
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-35B-A3B", trust_remote_code=True,
        torch_dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print("done")

    # Get the specific layer
    layer = model.model.layers[layer_idx]
    return model, layer, cfg

def hf_deltanet_forward(layer, hidden_states, attention_mask=None, position_ids=None):
    """Run the HF layer forward and capture intermediate values."""
    # The HF model processes chunks, not single tokens.
    # For a single token, we need to handle the attention properly.
    with torch.no_grad():
        # Access the DeltaNet sublayer
        # Qwen3.6 uses Qwen3_5MoeGatedDeltaNet for linear attention
        linear_attn = layer.self_attn  # This is the attention module

        # The forward method signature varies. Let's just run the full layer.
        output = layer(
            hidden_states.unsqueeze(0).unsqueeze(0),  # (1, 1, hidden)
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    return output[0].squeeze()


if __name__ == "__main__":
    L = 1  # Test layer 1 (DeltaNet)

    # ── Load our weights ──
    a = AttnWeights(L)
    print(f"Layer {L}: {'DeltaNet' if a.has('linear_attn.in_proj_qkv.weight') else 'Full GQA'}")

    # ── Generate test input (same as what the model sees) ──
    # Use the actual embedding for a real token
    embed = np.memmap(BIN / "embed_tokens.bin", dtype=np.float32, mode='r').reshape(248320, HDIM)

    # Prefill: build up conv state and S matrix with first 5 tokens
    tokens = [1058, 304, 1374, 374, 279]  # "The meaning of life is" (approximate)
    state = {
        'conv_state': np.zeros((8192, 4), dtype=np.float32),
        'conv_ptr': 0,
        'S': np.zeros((N_V_HEADS, HK, HV), dtype=np.float32),
    }

    print("\n=== OUR DeltaNet trace (5 prefill tokens) ===")
    h = None
    for i, tid in enumerate(tokens):
        h_in = embed[tid].astype(np.float32).copy()
        in_norm_w = a.get('input_layernorm.weight')
        h_norm = rms_norm(h_in, in_norm_w)
        print(f"\n── Token {i} (id={tid}), h post input_norm norm={np.linalg.norm(h_norm):.4f} ──")
        h, state, trace = our_deltanet(h_norm, a, state)
        for k, v in trace.items():
            print(f"  {k:<25s} = {v:.6f}")

    print(f"\n=== Load HF reference... ===")
    try:
        model, layer, cfg = load_hf_deltanet(L)
        print(f"HF model loaded. Config: hidden={cfg.hidden_size}")

        # Run the same input through HF
        h_in_torch = torch.from_numpy(embed[tokens[-1]].copy()).float()
        in_norm_torch = torch.from_numpy(a.get('input_layernorm.weight')).float()

        # Apply RMSNorm
        h_norm_torch = h_in_torch / torch.sqrt(torch.mean(h_in_torch**2) + 1e-6) * in_norm_torch

        print(f"HF input norm: {torch.norm(h_norm_torch):.4f}")

        # Run HF layer (full layer, not just DeltaNet)
        hf_out = hf_deltanet_forward(layer, h_norm_torch, position_ids=torch.tensor([len(tokens)-1]))
        print(f"HF output norm: {torch.norm(hf_out):.4f}")
        print(f"OUR output norm: {np.linalg.norm(h):.4f}")

        cos = torch.dot(hf_out.flatten(), torch.from_numpy(h).float()) / (
            torch.norm(hf_out) * torch.norm(torch.from_numpy(h).float()) + 1e-12)
        print(f"Cosine similarity (HF vs Ours): {cos:.6f}")

    except Exception as e:
        print(f"HF comparison failed: {e}")
        print("(Model may be too large to load on this machine)")
