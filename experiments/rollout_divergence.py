#!/usr/bin/env python3
"""Autoregressive rollout divergence: baseline vs rotation kernel.

Measures whether 22-layer rotation kernel accumulation causes collapse.

Metrics:
  - Token agreement (per-step exact match rate)
  - KL divergence (distribution fidelity)
  - Entropy drift (collapse detection)
  - cos(hidden_baseline, hidden_rotation) per layer per step

Usage:
  python experiments/rollout_divergence.py
"""

import numpy as np
import sys, time, json, math
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
VOCAB_SIZE = 32000
MAX_SEQ = 128

# Rotation rank sweep
RANKS = [64, 96, 128, 160, 192]

# Prompt
PROMPT = "The meaning of life is"

# ── Weight Loading ────────────────────────────────────────────────────────

class LazyWeights:
    """Lazy-loading weight accessor with mmap and bf16→f32 conversion. Supports w[key] syntax."""
    def __init__(self, path):
        import safetensors, json, struct, mmap
        # Parse header to get tensor offsets
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
        # mmap the file
        self._fd = open(path, 'rb')
        self._mmap = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._cache:
            info = self._tensors[name]
            raw = self._mmap[info['start']:info['end']]
            dtype = info['dtype']
            shape = info['shape']

            if dtype == 'BF16':
                arr = np.frombuffer(raw, dtype=np.uint16)
                arr = (arr.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()
            elif dtype == 'F16':
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
            elif dtype == 'F32':
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
            elif dtype == 'I64':
                arr = np.frombuffer(raw, dtype=np.int64).reshape(shape)
            else:
                raise ValueError(f"Unknown dtype: {dtype}")
            self._cache[name] = arr
        return self._cache[name]

    def preload_all(self):
        """Preload embed, final norm, lm_head."""
        print("Preloading core weights...", end=" ", flush=True)
        t0 = time.perf_counter()
        _ = self["model.embed_tokens.weight"]
        _ = self["model.norm.weight"]
        _ = self["lm_head.weight"]
        print(f"done ({time.perf_counter()-t0:.1f}s)")

    def preload_layer(self, l):
        """Preload all weights for one layer."""
        pfx = f"model.layers.{l}"
        for suffix in [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        ]:
            _ = self[pfx + suffix]

# ── RMSNorm ───────────────────────────────────────────────────────────────

def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return (x / rms) * weight

# ── RoPE ──────────────────────────────────────────────────────────────────

def precompute_rope(max_seq, head_dim):
    theta = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    positions = np.arange(max_seq)[:, None]
    freqs = positions * theta[None, :]
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    return cos, sin

def apply_rope(x, cos, sin, pos):
    """x: (n_heads, head_dim)"""
    d2 = x.shape[-1] // 2
    x_even = x[:, :d2]
    x_odd = x[:, d2:]
    c = cos[pos, :d2][None, :]
    s = sin[pos, :d2][None, :]
    rot_even = x_even * c - x_odd * s
    rot_odd = x_even * s + x_odd * c
    return np.concatenate([rot_even, rot_odd], axis=-1)

# ── Attention ─────────────────────────────────────────────────────────────

def attention_forward(h, layer_idx, pos, seq_len, w, kv_cache, rope_cos, rope_sin):
    """GQA attention for one layer. Returns (output, updated_kv_cache)."""
    pfx = f"model.layers.{layer_idx}.self_attn"
    q_w = w[f"{pfx}.q_proj.weight"]
    k_w = w[f"{pfx}.k_proj.weight"]
    v_w = w[f"{pfx}.v_proj.weight"]
    o_w = w[f"{pfx}.o_proj.weight"]

    # Project
    q_full = q_w @ h  # (n_heads * head_dim,)
    k_full = k_w @ h  # (n_kv_heads * head_dim,)
    v_full = v_w @ h

    n_q_heads = N_HEADS
    n_kv = N_KV_HEADS
    hd = HEAD_DIM

    q = q_full.reshape(n_q_heads, hd)
    k = k_full.reshape(n_kv, hd)
    v = v_full.reshape(n_kv, hd)

    # RoPE
    q = apply_rope(q, rope_cos, rope_sin, pos)
    k = apply_rope(k, rope_cos, rope_sin, pos)

    # Update KV cache
    Kc, Vc = kv_cache
    Kc[:, pos, :] = k
    Vc[:, pos, :] = v

    # Repeat KV for GQA
    n_rep = n_q_heads // n_kv
    k_rep = np.repeat(Kc[:, :seq_len, :], n_rep, axis=0)  # (n_heads, seq_len, hd)
    v_rep = np.repeat(Vc[:, :seq_len, :], n_rep, axis=0)

    # Attention
    scale = 1.0 / math.sqrt(hd)
    scores = np.sum(q[:, None, :] * k_rep, axis=-1) * scale  # (n_heads, seq_len)
    attn_w = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_w = attn_w / np.sum(attn_w, axis=-1, keepdims=True)

    attn_out = np.sum(attn_w[:, :, None] * v_rep, axis=1).flatten()  # (n_heads * hd,)
    return o_w @ attn_out, (Kc, Vc)

# ── FFN (full, baseline) ─────────────────────────────────────────────────

def ffn_forward(h, layer_idx, w):
    pfx = f"model.layers.{layer_idx}.mlp"
    gate_w = w[f"{pfx}.gate_proj.weight"]
    up_w = w[f"{pfx}.up_proj.weight"]
    down_w = w[f"{pfx}.down_proj.weight"]

    gate = gate_w @ h
    up = up_w @ h
    hidden = gate / (1.0 + np.exp(-gate)) * up  # SiLU
    return down_w @ hidden

# ── Rotation bases: compute from layer weights ────────────────────────────

def compute_rotation_bases(w, n_samples=200, max_k=256):
    """Compute per-layer rotation bases (U, Σ, V) from empirical FFN deltas."""
    bases = {}
    rng = np.random.RandomState(42)

    print(f"Computing rotation bases ({n_samples} samples, max_k={max_k})...")
    t0 = time.perf_counter()

    for l in range(N_LAYERS):
        pfx = f"model.layers.{l}.mlp"
        gate_w = w[f"{pfx}.gate_proj.weight"]
        up_w = w[f"{pfx}.up_proj.weight"]
        down_w = w[f"{pfx}.down_proj.weight"]

        # Generate random inputs
        inputs = rng.randn(n_samples, HIDDEN_DIM).astype(np.float32)
        inputs = inputs / np.linalg.norm(inputs, axis=1, keepdims=True)

        # Compute deltas
        deltas = np.zeros((n_samples, HIDDEN_DIM), dtype=np.float32)
        for i in range(n_samples):
            g = gate_w @ inputs[i]
            u = up_w @ inputs[i]
            h = g / (1.0 + np.exp(-g)) * u
            deltas[i] = down_w @ h

        # SVD of delta matrix
        U, S, Vt = np.linalg.svd(deltas.T, full_matrices=False)
        k = min(max_k, len(S))
        bases[l] = {
            'U': U[:, :k].copy(),
            'S': S[:k].copy(),
            'eff_rank': float(np.sum(S)**2 / np.sum(S**2)),
            'sv_ratio': float(S[0] / S[1]) if len(S) > 1 else 0,
        }

        if l % 5 == 0:
            print(f"  L{l}: eff_rank={bases[l]['eff_rank']:.1f} σ₁/σ₂={bases[l]['sv_ratio']:.1f}")

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    return bases

# ── Rotation kernel FFN ──────────────────────────────────────────────────

def rotation_ffn(h, layer_idx, w, bases, k):
    """Low-rank FFN: Δ ≈ U_k @ Σ_k @ V_k^T @ x, oracle projection."""
    base = bases[layer_idx]
    Uk = base['U'][:, :k]
    Sk = base['S'][:k]

    # Project: z = Uk^T @ (unknown Δ). We need to learn V such that
    # V^T @ x ≈ Σ^{-1} @ Uk^T @ Δ_full(x).
    # For now, use the empirical projection: compute Δ_full and project.
    # In production, V would be pre-computed via least squares.
    #
    # Actually, for this experiment we need a practical approach.
    # Since we can't learn V without the full FFN, we use a hybrid:
    # compute the full FFN delta and project it onto Uk.
    # This measures the ceiling of what rotation kernel can achieve.
    # The V matrix would be learned offline to approximate this projection.

    # For the ROLLOUT experiment, we need a fully online method.
    # Use the empirical projection: Δ_rot = U_k @ U_k^T @ Δ_full
    # This is the "oracle" ceiling — if V is perfectly learned, this is
    # the best possible rotation kernel output.
    pfx = f"model.layers.{layer_idx}.mlp"
    gate_w = w[f"{pfx}.gate_proj.weight"]
    up_w = w[f"{pfx}.up_proj.weight"]
    down_w = w[f"{pfx}.down_proj.weight"]

    # Full FFN (needed for oracle projection)
    gate = gate_w @ h
    up = up_w @ h
    hidden = gate / (1.0 + np.exp(-gate)) * up
    delta_full = down_w @ hidden

    # Project onto top-k subspace
    z = Uk.T @ delta_full
    return Uk @ z

def rotation_ffn_learned(h, layer_idx, w, bases, k):
    """Low-rank FFN using pre-learned V matrix.

    Δ ≈ U_k @ Σ_k @ V_k^T @ h

    Where V_k is learned via least squares from training data.
    For this experiment, V_k is baked into the bases dict during
    the compute_rotation_bases_with_v() call.
    """
    base = bases[layer_idx]
    Uk = base['U'][:, :k]
    Vk = base.get('V')

    if Vk is None:
        # Fallback to oracle projection
        return rotation_ffn(h, layer_idx, w, bases, k)

    Vk = Vk[:, :k]
    Sk = base['S'][:k]

    # Δ = U_k @ Σ_k @ V_k^T @ h
    z = Vk.T @ h
    z = Sk * z
    return Uk @ z

# ── Full forward pass ─────────────────────────────────────────────────────

def forward_layer(h, layer_idx, pos, seq_len, w, kv_cache, rope_cos, rope_sin):
    """One full transformer layer (baseline)."""
    pfx = f"model.layers.{layer_idx}"

    # Input norm
    in_norm_w = w[f"{pfx}.input_layernorm.weight"]
    hn = rms_norm(h, in_norm_w)

    # Attention
    attn_out, kv_cache = attention_forward(hn, layer_idx, pos, seq_len, w, kv_cache, rope_cos, rope_sin)
    h = h + attn_out

    # Post-attention norm
    post_norm_w = w[f"{pfx}.post_attention_layernorm.weight"]
    hn2 = rms_norm(h, post_norm_w)

    # FFN
    ffn_out = ffn_forward(hn2, layer_idx, w)
    h = h + ffn_out

    return h, kv_cache

def forward_layer_rotation(h, layer_idx, pos, seq_len, w, kv_cache, rope_cos, rope_sin, bases, k, use_oracle=True):
    """One layer with rotation kernel replacing FFN."""
    pfx = f"model.layers.{layer_idx}"

    # Input norm
    in_norm_w = w[f"{pfx}.input_layernorm.weight"]
    hn = rms_norm(h, in_norm_w)

    # Attention (same as baseline)
    attn_out, kv_cache = attention_forward(hn, layer_idx, pos, seq_len, w, kv_cache, rope_cos, rope_sin)
    h = h + attn_out

    # Post-attention norm
    post_norm_w = w[f"{pfx}.post_attention_layernorm.weight"]
    hn2 = rms_norm(h, post_norm_w)

    # Rotation kernel instead of full FFN
    if use_oracle:
        ffn_out = rotation_ffn(hn2, layer_idx, w, bases, k)
    else:
        ffn_out = rotation_ffn_learned(hn2, layer_idx, w, bases, k)
    h = h + ffn_out

    return h, kv_cache

# ── Tokenizer ─────────────────────────────────────────────────────────────

class SimpleTokenizer:
    def __init__(self, w):
        from transformers import AutoTokenizer
        model_dir = str(Path(MODEL_PATH).parent)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def encode(self, text):
        return self.tok.encode(text)

    def decode(self, ids):
        return self.tok.decode(ids)

# ── Generate ───────────────────────────────────────────────────────────────

def generate(model_type, w, tokenizer, bases, prompt_ids, max_tokens, k, use_oracle):
    """Autoregressive generation. Returns (tokens, trace)."""
    rope_cos, rope_sin = precompute_rope(MAX_SEQ, HEAD_DIM)
    embed_w = w["model.embed_tokens.weight"]
    final_norm_w = w["model.norm.weight"]
    lm_head_w = w["lm_head.weight"]

    # Initialize KV caches
    kv_caches = [(np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32),
                  np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32))
                 for _ in range(N_LAYERS)]

    tokens = list(prompt_ids)
    trace = {
        'hidden_states': [],  # per layer, per step
        'entropies': [],
        'tokens': [],
    }

    # Prefill
    for pos, tid in enumerate(tokens):
        h = embed_w[tid].astype(np.float32)
        for l in range(N_LAYERS):
            if model_type == 'baseline':
                h, kv_caches[l] = forward_layer(h, l, pos, pos+1, w, kv_caches[l], rope_cos, rope_sin)
            else:
                h, kv_caches[l] = forward_layer_rotation(h, l, pos, pos+1, w, kv_caches[l], rope_cos, rope_sin, bases, k, use_oracle)
        hn = rms_norm(h, final_norm_w)
        logits = lm_head_w @ hn

    # Generate
    for step in range(max_tokens):
        # Greedy
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
        pos = len(tokens) - 1

        # Compute entropy
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        trace['entropies'].append(float(entropy))
        trace['tokens'].append(next_token)

        if next_token == 2:  # EOS
            break

        h = embed_w[next_token].astype(np.float32)
        layer_hs = []
        for l in range(N_LAYERS):
            if model_type == 'baseline':
                h, kv_caches[l] = forward_layer(h, l, pos, pos+1, w, kv_caches[l], rope_cos, rope_sin)
            else:
                h, kv_caches[l] = forward_layer_rotation(h, l, pos, pos+1, w, kv_caches[l], rope_cos, rope_sin, bases, k, use_oracle)
            layer_hs.append(h.copy())
        trace['hidden_states'].append(layer_hs)

        hn = rms_norm(h, final_norm_w)
        logits = lm_head_w @ hn

    return tokens, trace

# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(baseline_tokens, rot_tokens, baseline_trace, rot_trace):
    """Compare two autoregressive rollouts."""
    metrics = {}

    # Token agreement
    min_len = min(len(baseline_tokens), len(rot_tokens))
    agreement = sum(1 for i in range(min_len) if baseline_tokens[i] == rot_tokens[i])
    metrics['token_agreement'] = agreement / max(min_len, 1)

    # Exact match
    metrics['exact_match'] = baseline_tokens[:min_len] == rot_tokens[:min_len]

    # Per-step KL divergence (if traces available)
    if baseline_trace['entropies'] and rot_trace['entropies']:
        bl_ent = baseline_trace['entropies']
        rt_ent = rot_trace['entropies']
        min_e = min(len(bl_ent), len(rt_ent))
        metrics['entropy_diff_mean'] = np.mean([
            abs(bl_ent[i] - rt_ent[i]) for i in range(min_e)
        ])
        metrics['entropy_drift'] = bl_ent[-1] - rt_ent[-1] if min_e > 0 else 0

    # Hidden state cosine per layer (first generated token)
    if (baseline_trace['hidden_states'] and rot_trace['hidden_states'] and
        baseline_trace['hidden_states'][0] and rot_trace['hidden_states'][0]):
        bl_hs = baseline_trace['hidden_states'][0]
        rt_hs = rot_trace['hidden_states'][0]
        layer_cos = []
        for l in range(N_LAYERS):
            a = bl_hs[l]
            b = rt_hs[l]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            layer_cos.append(cos)
        metrics['hidden_cos_first_token'] = layer_cos
        metrics['hidden_cos_mean'] = float(np.mean(layer_cos))
        metrics['hidden_cos_min'] = float(np.min(layer_cos))

    # Repetition
    if len(rot_tokens) > 1:
        repeats = sum(1 for i in range(1, len(rot_tokens)) if rot_tokens[i] == rot_tokens[i-1])
        metrics['rep_rate_rot'] = repeats / (len(rot_tokens) - 1)
    if len(baseline_tokens) > 1:
        repeats = sum(1 for i in range(1, len(baseline_tokens)) if baseline_tokens[i] == baseline_tokens[i-1])
        metrics['rep_rate_baseline'] = repeats / (len(baseline_tokens) - 1)

    return metrics

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Autoregressive Rollout Divergence: Baseline vs Rotation Kernel")
    print("=" * 70)

    # Load
    print(f"Opening weights from {MODEL_PATH}...", end=" ", flush=True)
    w = LazyWeights(MODEL_PATH)
    print(f"{len(w._tensors)} tensors indexed")
    tokenizer = SimpleTokenizer(w)
    prompt_ids = tokenizer.encode(PROMPT)
    print(f"Prompt: \"{PROMPT}\" → {len(prompt_ids)} tokens")

    # Compute rotation bases
    bases = compute_rotation_bases(w, n_samples=200, max_k=256)

    # Baseline generation
    print(f"\n{'='*70}")
    print("BASELINE (full FFN)")
    print(f"{'='*70}")
    t0 = time.perf_counter()
    baseline_tokens, baseline_trace = generate(
        'baseline', w, tokenizer, bases, prompt_ids, max_tokens=30, k=0, use_oracle=False
    )
    baseline_time = time.perf_counter() - t0
    baseline_text = tokenizer.decode(baseline_tokens[len(prompt_ids):])
    print(f"Generated: \"{baseline_text[:200]}\"")
    print(f"Time: {baseline_time:.1f}s ({len(baseline_tokens)-len(prompt_ids)} tokens)")
    print(f"Entropy: mean={np.mean(baseline_trace['entropies']):.3f}, "
          f"final={baseline_trace['entropies'][-1]:.3f}" if baseline_trace['entropies'] else "")

    # Rotation kernel sweep
    results = {}
    print(f"\n{'='*70}")
    print(f"ROTATION KERNEL SWEEP (k = {RANKS})")
    print(f"{'='*70}")
    print()
    print(f"{'k':<8} {'agreement':>10} {'entropy_Δ':>10} {'hidden_cos':>10} "
          f"{'time':>8} {'rep_rate':>10} {'text'}")
    print("-" * 90)

    for k in RANKS:
        t0 = time.perf_counter()
        rot_tokens, rot_trace = generate(
            'rotation', w, tokenizer, bases, prompt_ids, max_tokens=30, k=k, use_oracle=True
        )
        rot_time = time.perf_counter() - t0

        metrics = compute_metrics(baseline_tokens, rot_tokens, baseline_trace, rot_trace)
        results[k] = metrics

        rot_text = tokenizer.decode(rot_tokens[len(prompt_ids):])
        print(f"{k:<8} {metrics['token_agreement']:>10.3f} "
              f"{metrics.get('entropy_diff_mean', 0):>10.4f} "
              f"{metrics.get('hidden_cos_mean', 0):>10.4f} "
              f"{rot_time:>7.1f}s "
              f"{metrics.get('rep_rate_rot', 0):>10.3f} "
              f"\"{rot_text[:50]}\"")

    # Detailed per-layer hidden cos for best k
    print(f"\n{'='*70}")
    print("PER-LAYER HIDDEN STATE COSINE (first generated token)")
    print(f"{'='*70}")
    print(f"{'Layer':<8}", end="")
    for k in RANKS:
        print(f"k={k:<6}", end="")
    print()
    print("-" * (8 + 8 * len(RANKS)))

    for l in range(N_LAYERS):
        print(f"L{l:<7}", end="")
        for k in RANKS:
            cos_val = results[k].get('hidden_cos_first_token', [0]*N_LAYERS)
            if l < len(cos_val):
                c = cos_val[l]
                marker = "!" if c < 0.9 else ("." if c < 0.97 else " ")
                print(f"{c:.4f}{marker:<2}", end="")
            else:
                print(f"  N/A  ", end="")
        print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline: \"{baseline_text[:100]}\"")
    print(f"Baseline entropy: {np.mean(baseline_trace['entropies']):.3f} "
          f"(final: {baseline_trace['entropies'][-1]:.3f})" if baseline_trace['entropies'] else "")
    print()
    for k in RANKS:
        r = results[k]
        rot_text = tokenizer.decode(
            [t for t in rot_tokens[len(prompt_ids):] if t not in (0, 2)][:20]
        ) if k in [RANKS[0], RANKS[-1]] else ""
        status = "✓" if r['token_agreement'] > 0.8 else ("△" if r['token_agreement'] > 0.5 else "✗")
        print(f"  k={k:<4} {status} agree={r['token_agreement']:.3f} "
              f"hidden_cos={r.get('hidden_cos_mean', 0):.4f} "
              f"min_cos={r.get('hidden_cos_min', 0):.4f} "
              f"entropy_Δ={r.get('entropy_diff_mean', 0):.4f} "
              f"rep={r.get('rep_rate_rot', 0):.3f}")

    # Save results
    output = {
        'baseline_text': baseline_text,
        'baseline_entropy_mean': float(np.mean(baseline_trace['entropies'])) if baseline_trace['entropies'] else 0,
        'results': {str(k): {kk: vv for kk, vv in r.items() if kk != 'hidden_cos_first_token'}
                    for k, r in results.items()}
    }
    out_path = Path(__file__).parent / "rollout_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

if __name__ == "__main__":
    main()
