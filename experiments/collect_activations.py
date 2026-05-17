#!/usr/bin/env python3
"""Step 1: Activation Dataset Collector.

Runs real prompts through TinyLlama and collects:
  - per-layer hidden states
  - FFN activations (gate_out, up_out, intermediate)
  - attention outputs
  - token position
  - entropy
  - top-k logits

This dataset becomes the ground truth for:
  - Latent expert extraction (Step 2)
  - Trajectory clustering (Step 3)
  - Runtime IR generation (Step 4)

Usage:
  python experiments/collect_activations.py --n-prompts 20 --max-tokens 50
"""

import numpy as np
import json, struct, mmap, math, sys, time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────

MODEL_PATH = ("/Users/yasuno/.cache/huggingface/hub/"
              "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
              "snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/"
              "model.safetensors")

N_LAYERS = 22
HIDDEN_DIM = 2048
FFN_DIM = 5632
N_HEADS = 32
N_KV_HEADS = 4
HEAD_DIM = 64
MAX_SEQ = 256

# Diverse prompts spanning different domains
PROMPTS = [
    "The meaning of life is",
    "Quantum mechanics describes",
    "The capital of France is",
    "Machine learning is a field of",
    "The history of the Roman Empire",
    "Photosynthesis is the process by which",
    "The theory of relativity explains",
    "In mathematics, a prime number is",
    "The French Revolution began in",
    "Artificial intelligence can be defined as",
    "The structure of DNA was discovered by",
    "Climate change is caused by",
    "Shakespeare's most famous play is",
    "The speed of light in vacuum is",
    "Democracy is a form of government where",
    "The first law of thermodynamics states that",
    "Neural networks are composed of",
    "The Pacific Ocean is the",
    "Human rights are",
    "The Fibonacci sequence is defined as",
]

# ── Weight Loading ────────────────────────────────────────────────────────

class LazyWeights:
    """mmap-based lazy weight loader with bf16→f32 conversion."""
    def __init__(self, path):
        with open(path, 'rb') as fh:
            header_len = struct.unpack('<Q', fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        self._tensors = {}
        for k, v in header.items():
            if k == '__metadata__':
                continue
            self._tensors[k] = {
                'dtype': v['dtype'],
                'shape': v['shape'],
                'start': v['data_offsets'][0] + 8 + header_len,
                'end': v['data_offsets'][1] + 8 + header_len,
            }
        self._fd = open(path, 'rb')
        self._mmap = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._cache:
            info = self._tensors[name]
            raw = self._mmap[info['start']:info['end']]
            dtype, shape = info['dtype'], info['shape']
            if dtype == 'BF16':
                arr = np.frombuffer(raw, dtype=np.uint16)
                arr = (arr.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()
            elif dtype == 'F16':
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
            elif dtype == 'F32':
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
            else:
                arr = np.frombuffer(raw, dtype=np.uint8)
            self._cache[name] = arr
        return self._cache[name]

# ── RMSNorm ───────────────────────────────────────────────────────────────

def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return (x / rms) * weight

# ── RoPE ──────────────────────────────────────────────────────────────────

def precompute_rope(max_seq, head_dim):
    theta = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = np.arange(max_seq)[:, None] * theta[None, :]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

def apply_rope(x, cos, sin, pos):
    d2 = x.shape[-1] // 2
    c = cos[pos, :d2][None, :]
    s = sin[pos, :d2][None, :]
    rot_even = x[:, :d2] * c - x[:, d2:] * s
    rot_odd = x[:, :d2] * s + x[:, d2:] * c
    return np.concatenate([rot_even, rot_odd], axis=-1)

# ── Forward Pass ──────────────────────────────────────────────────────────

def attention_forward(h, layer_idx, pos, seq_len, w, kv_cache, rope_cos, rope_sin):
    pfx = f"model.layers.{layer_idx}.self_attn"
    q_full = w[f"{pfx}.q_proj.weight"] @ h
    k_full = w[f"{pfx}.k_proj.weight"] @ h
    v_full = w[f"{pfx}.v_proj.weight"] @ h
    o_w = w[f"{pfx}.o_proj.weight"]

    n_q = N_HEADS
    n_kv = N_KV_HEADS
    hd = HEAD_DIM

    q = q_full.reshape(n_q, hd)
    k = k_full.reshape(n_kv, hd)
    v = v_full.reshape(n_kv, hd)

    q = apply_rope(q, rope_cos, rope_sin, pos)
    k = apply_rope(k, rope_cos, rope_sin, pos)

    Kc, Vc = kv_cache
    Kc[:, pos, :] = k
    Vc[:, pos, :] = v

    n_rep = n_q // n_kv
    k_rep = np.repeat(Kc[:, :seq_len, :], n_rep, axis=0)
    v_rep = np.repeat(Vc[:, :seq_len, :], n_rep, axis=0)

    scale = 1.0 / math.sqrt(hd)
    scores = np.sum(q[:, None, :] * k_rep, axis=-1) * scale
    attn_w = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_w = attn_w / np.sum(attn_w, axis=-1, keepdims=True)

    attn_out = np.sum(attn_w[:, :, None] * v_rep, axis=1).flatten()
    ao = o_w @ attn_out
    return ao, (Kc, Vc), attn_w

def ffn_forward_detailed(h, layer_idx, w):
    """FFN forward returning intermediate activations for analysis."""
    pfx = f"model.layers.{layer_idx}.mlp"
    gate_w = w[f"{pfx}.gate_proj.weight"]
    up_w = w[f"{pfx}.up_proj.weight"]
    down_w = w[f"{pfx}.down_proj.weight"]

    gate_out = gate_w @ h  # (ffn_dim,)
    up_out = up_w @ h      # (ffn_dim,)
    intermediate = gate_out / (1.0 + np.exp(-gate_out)) * up_out  # SiLU(gate) * up
    delta = down_w @ intermediate  # (hidden_dim,)
    return delta, gate_out, up_out, intermediate

def forward_layer(h, layer_idx, pos, seq_len, w, kv_cache, rope_cos, rope_sin):
    """Full transformer layer. Returns (h_new, kv_cache, layer_record)."""
    pfx = f"model.layers.{layer_idx}"

    # Input norm
    in_norm_w = w[f"{pfx}.input_layernorm.weight"]
    hn = rms_norm(h, in_norm_w)

    # Attention
    attn_out, kv_cache, attn_weights = attention_forward(
        hn, layer_idx, pos, seq_len, w, kv_cache, rope_cos, rope_sin)
    h_after_attn = h + attn_out

    # Post-attention norm
    post_norm_w = w[f"{pfx}.post_attention_layernorm.weight"]
    hn2 = rms_norm(h_after_attn, post_norm_w)

    # FFN with detailed activations
    ffn_delta, gate_out, up_out, intermediate = ffn_forward_detailed(hn2, layer_idx, w)
    h_new = h_after_attn + ffn_delta

    record = {
        'h_before': h.copy(),
        'h_after_attn': h_after_attn.copy(),
        'attn_out': attn_out.copy(),
        'ffn_delta': ffn_delta.copy(),
        'ffn_gate_out': gate_out.copy(),
        'ffn_up_out': up_out.copy(),
        'ffn_intermediate': intermediate.copy(),
        'attn_entropy': float(-np.sum(attn_weights * np.log(attn_weights + 1e-10)) / N_HEADS),
    }
    return h_new, kv_cache, record

# ── Generate + Collect ────────────────────────────────────────────────────

def collect_activations(w, tokenizer, prompt_ids, max_tokens, rope_cos, rope_sin):
    """Run autoregressive generation and collect ALL activations."""
    embed_w = w["model.embed_tokens.weight"]
    final_norm_w = w["model.norm.weight"]
    lm_head_w = w["lm_head.weight"]

    kv_caches = [(np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32),
                  np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32))
                 for _ in range(N_LAYERS)]

    tokens = list(prompt_ids)
    dataset = {
        'prompt_tokens': tokens,
        'generated_tokens': [],
        'layers': [{
            'hidden_states': [],
            'ffn_deltas': [],
            'ffn_gate_outs': [],
            'ffn_up_outs': [],
            'ffn_intermediates': [],
            'attn_outs': [],
            'attn_entropies': [],
        } for _ in range(N_LAYERS)],
        'entropies': [],
        'logits_top5': [],
    }

    # Prefill
    for pos, tid in enumerate(tokens):
        h = embed_w[tid].astype(np.float32)
        for l in range(N_LAYERS):
            h, kv_caches[l], record = forward_layer(h, l, pos, pos+1, w, kv_caches[l], rope_cos, rope_sin)
        hn = rms_norm(h, final_norm_w)
        logits = lm_head_w @ hn

    # Generate + collect
    for step in range(max_tokens):
        # Greedy
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
        pos = len(tokens) - 1

        # Entropy
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        dataset['entropies'].append(entropy)

        # Top-5
        top5 = np.argsort(logits)[-5:][::-1]
        top5_probs = probs[top5]
        dataset['logits_top5'].append({
            'tokens': top5.tolist(),
            'probs': top5_probs.tolist(),
        })

        if next_token == 2:
            break

        h = embed_w[next_token].astype(np.float32)
        for l in range(N_LAYERS):
            h, kv_caches[l], record = forward_layer(h, l, pos, pos+1, w, kv_caches[l], rope_cos, rope_sin)
            # Store
            ds = dataset['layers'][l]
            ds['hidden_states'].append(record['h_before'])
            ds['ffn_deltas'].append(record['ffn_delta'])
            ds['ffn_gate_outs'].append(record['ffn_gate_out'])
            ds['ffn_up_outs'].append(record['ffn_up_out'])
            ds['ffn_intermediates'].append(record['ffn_intermediate'])
            ds['attn_outs'].append(record['attn_out'])
            ds['attn_entropies'].append(record['attn_entropy'])

        hn = rms_norm(h, final_norm_w)
        logits = lm_head_w @ hn

    dataset['generated_tokens'] = tokens[len(dataset['prompt_tokens']):]
    return dataset

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--output", type=str, default="experiments/activations")
    args = parser.parse_args()

    print("=" * 70)
    print("Activation Dataset Collector")
    print("=" * 70)

    # Load
    print(f"Loading model...", end=" ", flush=True)
    w = LazyWeights(MODEL_PATH)
    print(f"{len(w._tensors)} tensors indexed")

    from transformers import AutoTokenizer
    model_dir = str(Path(MODEL_PATH).parent)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    rope_cos, rope_sin = precompute_rope(MAX_SEQ, HEAD_DIM)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS[:args.n_prompts]
    all_datasets = []

    total_start = time.perf_counter()
    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt)
        print(f"\n[{i+1}/{len(prompts)}] \"{prompt}\" ({len(prompt_ids)} tokens)")
        t0 = time.perf_counter()
        ds = collect_activations(w, tokenizer, prompt_ids, args.max_tokens, rope_cos, rope_sin)
        elapsed = time.perf_counter() - t0

        n_gen = len(ds['generated_tokens'])
        text = tokenizer.decode(ds['generated_tokens'])
        print(f"  Generated {n_gen} tokens in {elapsed:.0f}s: \"{text[:100]}\"")

        # Per-layer activation statistics
        for l in [0, 2, 7, 14, 21]:
            ds_l = ds['layers'][l]
            if ds_l['ffn_deltas']:
                deltas = np.array(ds_l['ffn_deltas'])
                mean_norm = np.mean(np.linalg.norm(deltas, axis=1))
                print(f"  L{l}: {len(ds_l['ffn_deltas'])} samples, "
                      f"||Δ||={mean_norm:.4f}, "
                      f"attn_entropy={np.mean(ds_l['attn_entropies']):.3f}")

        all_datasets.append(ds)

    total_time = time.perf_counter() - total_start

    # ── Save ──
    # Convert to serializable format (arrays → lists for JSON, or use npz)
    print(f"\n{'='*70}")
    print(f"Saving dataset ({len(all_datasets)} prompts, {total_time:.0f}s)...")

    # Save as compressed npz (one file per prompt)
    for i, ds in enumerate(all_datasets):
        save_dict = {
            'prompt_tokens': np.array(ds['prompt_tokens'], dtype=np.int32),
            'generated_tokens': np.array(ds['generated_tokens'], dtype=np.int32),
            'entropies': np.array(ds['entropies'], dtype=np.float32),
        }
        for l in range(N_LAYERS):
            pfx = f"layer{l}"
            ld = ds['layers'][l]
            if ld['ffn_deltas']:
                save_dict[f"{pfx}_hidden"] = np.array(ld['hidden_states'], dtype=np.float16)
                save_dict[f"{pfx}_ffn_delta"] = np.array(ld['ffn_deltas'], dtype=np.float16)
                save_dict[f"{pfx}_ffn_intermediate"] = np.array(ld['ffn_intermediates'], dtype=np.float16)
                save_dict[f"{pfx}_attn_out"] = np.array(ld['attn_outs'], dtype=np.float16)
                save_dict[f"{pfx}_attn_entropy"] = np.array(ld['attn_entropies'], dtype=np.float32)

        np.savez_compressed(output_dir / f"prompt_{i:03d}.npz", **save_dict)

    # Save metadata
    meta = {
        'model': 'TinyLlama-1.1B-Chat-v1.0',
        'n_layers': N_LAYERS,
        'hidden_dim': HIDDEN_DIM,
        'ffn_dim': FFN_DIM,
        'n_prompts': len(all_datasets),
        'prompts': prompts[:len(all_datasets)],
        'total_samples': sum(len(ds['entropies']) for ds in all_datasets),
        'collection_time_s': total_time,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_samples = sum(len(ds['entropies']) for ds in all_datasets)
    print(f"Total: {total_samples} token samples across {len(all_datasets)} prompts")
    print(f"Output: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
