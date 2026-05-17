#!/usr/bin/env python3
"""M5: Trajectory Archetype Extraction — the go/no-go for trajectory VM.

Core question:
  Can we replace FFN(h) with archetype_lookup(archetype_id, layer)?
  var(Δ_l | archetype) << var(Δ_l) must hold.

Method:
  1. Run diverse prompts through TinyLlama
  2. Collect per-token per-layer (h_l, Δ_l) for the full 22-layer trajectory
  3. Cluster trajectories into archetypes based on their shape
  4. Measure intra-archetype Δ variance vs total Δ variance
  5. Test Markov predictability of archetype transitions

Usage:
  python experiments/trajectory_archetypes.py --n-prompts 10 --max-tokens 40
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

# ── LazyWeights (same as before) ──────────────────────────────────────────

class LazyWeights:
    def __init__(self, path):
        with open(path, 'rb') as fh:
            header_len = struct.unpack('<Q', fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        self._tensors = {}
        for k, v in header.items():
            if k == '__metadata__': continue
            self._tensors[k] = {
                'dtype': v['dtype'], 'shape': v['shape'],
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
            else:
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
            self._cache[name] = arr
        return self._cache[name]

# ── RMSNorm, RoPE, Attention, FFN ─────────────────────────────────────────

def rms_norm(x, weight, eps=1e-6):
    return (x / np.sqrt(np.mean(x**2) + eps)) * weight

def precompute_rope(max_seq, head_dim):
    theta = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = np.arange(max_seq)[:, None] * theta[None, :]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

def apply_rope(x, cos, sin, pos):
    d2 = x.shape[-1] // 2
    c, s = cos[pos, :d2][None, :], sin[pos, :d2][None, :]
    return np.concatenate([x[:, :d2] * c - x[:, d2:] * s,
                           x[:, :d2] * s + x[:, d2:] * c], axis=-1)

def forward_layer_full(h, layer_idx, pos, seq_len, w, kv_cache, rope_cos, rope_sin):
    """Full forward pass. Returns (h_new, kv_cache, delta, h_before_ffn)."""
    pfx = f"model.layers.{layer_idx}"
    in_norm_w = w[f"{pfx}.input_layernorm.weight"]
    post_norm_w = w[f"{pfx}.post_attention_layernorm.weight"]
    q_w = w[f"{pfx}.self_attn.q_proj.weight"]
    k_w = w[f"{pfx}.self_attn.k_proj.weight"]
    v_w = w[f"{pfx}.self_attn.v_proj.weight"]
    o_w = w[f"{pfx}.self_attn.o_proj.weight"]
    gate_w = w[f"{pfx}.mlp.gate_proj.weight"]
    up_w = w[f"{pfx}.mlp.up_proj.weight"]
    down_w = w[f"{pfx}.mlp.down_proj.weight"]

    # Input norm
    hn = rms_norm(h, in_norm_w)

    # Attention
    q_full = q_w @ hn
    k_full = k_w @ hn
    v_full = v_w @ hn
    q = q_full.reshape(N_HEADS, HEAD_DIM)
    k = k_full.reshape(N_KV_HEADS, HEAD_DIM)
    v = v_full.reshape(N_KV_HEADS, HEAD_DIM)
    q = apply_rope(q, rope_cos, rope_sin, pos)
    k = apply_rope(k, rope_cos, rope_sin, pos)
    Kc, Vc = kv_cache
    Kc[:, pos, :] = k; Vc[:, pos, :] = v
    n_rep = N_HEADS // N_KV_HEADS
    k_rep = np.repeat(Kc[:, :seq_len, :], n_rep, axis=0)
    v_rep = np.repeat(Vc[:, :seq_len, :], n_rep, axis=0)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = np.sum(q[:, None, :] * k_rep, axis=-1) * scale
    attn_w = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_w = attn_w / np.sum(attn_w, axis=-1, keepdims=True)
    attn_out = np.sum(attn_w[:, :, None] * v_rep, axis=1).flatten()
    h_attn = h + o_w @ attn_out

    # FFN
    hn2 = rms_norm(h_attn, post_norm_w)
    gate = gate_w @ hn2
    up = up_w @ hn2
    hidden = gate / (1.0 + np.exp(-gate)) * up
    delta = down_w @ hidden

    h_new = h_attn + delta
    return h_new, (Kc, Vc), delta, h_attn

# ── Trajectory Collection ─────────────────────────────────────────────────

def collect_trajectories(w, tokenizer, prompt_ids, max_tokens, rope_cos, rope_sin):
    """Run autoregressive generation and collect per-token trajectories."""
    embed_w = w["model.embed_tokens.weight"]
    final_norm_w = w["model.norm.weight"]
    lm_head_w = w["lm_head.weight"]

    kv_caches = [(np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32),
                  np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32))
                 for _ in range(N_LAYERS)]

    tokens = list(prompt_ids)
    trajectories = []  # list of {h_seq, delta_seq, token, entropy}

    # Prefill
    for pos, tid in enumerate(tokens):
        h = embed_w[tid].astype(np.float32)
        for l in range(N_LAYERS):
            h, kv_caches[l], _, _ = forward_layer_full(
                h, l, pos, pos+1, w, kv_caches[l], rope_cos, rope_sin)
        hn = rms_norm(h, final_norm_w)
        logits = lm_head_w @ hn

    # Generate + collect
    for step in range(max_tokens):
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
        pos = len(tokens) - 1

        if next_token == 2:  # EOS
            break

        h = embed_w[next_token].astype(np.float32)
        h_seq = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.float32)
        delta_seq = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.float32)

        for l in range(N_LAYERS):
            h, kv_caches[l], delta, _ = forward_layer_full(
                h, l, pos, pos+1, w, kv_caches[l], rope_cos, rope_sin)
            h_seq[l] = h
            delta_seq[l] = delta

        trajectories.append({
            'h_seq': h_seq,        # (22, 2048)
            'delta_seq': delta_seq, # (22, 2048)
            'token': next_token,
            'position': pos,
        })

        hn = rms_norm(h, final_norm_w)
        logits = lm_head_w @ hn

    return trajectories

# ── Trajectory Feature Extraction ─────────────────────────────────────────

def trajectory_features(trajectories):
    """Extract compact features for clustering.

    Instead of using raw 22×2048 = 45K-dimensional vectors,
    we use:
    - Per-layer norm: ||h_l||, ||Δ_l|| (22 + 22 = 44 dims)
    - Per-layer cos(h_l, h_{l+1}) (21 dims)
    - Per-layer cos(Δ_l, Δ_{l+1}) (21 dims)
    - Entropy proxy: std of h across layers (1 dim)
    Total: ~87 dims — compact and geometrically meaningful.
    """
    features = []
    for traj in trajectories:
        h = traj['h_seq']
        d = traj['delta_seq']

        # Norms
        h_norms = np.linalg.norm(h, axis=1)  # (22,)
        d_norms = np.linalg.norm(d, axis=1)  # (22,)

        # Cosines between adjacent layers
        h_cos = []
        d_cos = []
        for l in range(N_LAYERS - 1):
            hc = np.dot(h[l], h[l+1]) / (h_norms[l] * h_norms[l+1] + 1e-12)
            dc = np.dot(d[l], d[l+1]) / (d_norms[l] * d_norms[l+1] + 1e-12)
            h_cos.append(hc)
            d_cos.append(dc)

        feat = np.concatenate([
            h_norms / (np.mean(h_norms) + 1e-12),  # normalized
            d_norms / (np.mean(d_norms) + 1e-12),
            np.array(h_cos),
            np.array(d_cos),
            [np.std(h_norms)],
        ])
        features.append(feat)

    return np.array(features, dtype=np.float32)

# ── K-Means (pure numpy) ──────────────────────────────────────────────────

def kmeans(X, n_clusters, seed=42, max_iters=50):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    centroids = np.zeros((n_clusters, d))
    centroids[0] = X[rng.randint(n)]
    for c in range(1, n_clusters):
        dists = np.min([np.sum((X - centroids[i])**2, axis=1) for i in range(c)], axis=0)
        probs = dists / (dists.sum() + 1e-12)
        centroids[c] = X[rng.choice(n, p=probs)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        dists = np.array([np.sum((X - centroids[i])**2, axis=1) for i in range(n_clusters)])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels): break
        labels = new_labels
        for c in range(n_clusters):
            mask = labels == c
            if mask.any(): centroids[c] = X[mask].mean(axis=0)
    return labels, centroids

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--n-archetypes", type=int, default=8)
    args = parser.parse_args()

    print("=" * 70)
    print("Trajectory Archetype Extraction")
    print("=" * 70)
    print(f"Prompts: {args.n_prompts}, Max tokens: {args.max_tokens}, "
          f"Archetypes: {args.n_archetypes}")
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    w = LazyWeights(MODEL_PATH)
    print(f"done")

    from transformers import AutoTokenizer
    model_dir = str(Path(MODEL_PATH).parent)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rope_cos, rope_sin = precompute_rope(MAX_SEQ, HEAD_DIM)

    # Collect trajectories
    all_trajectories = []
    prompts_used = PROMPTS[:args.n_prompts]
    total_start = time.perf_counter()

    for i, prompt in enumerate(prompts_used):
        prompt_ids = tokenizer.encode(prompt)
        t0 = time.perf_counter()
        trajs = collect_trajectories(w, tokenizer, prompt_ids, args.max_tokens, rope_cos, rope_sin)
        elapsed = time.perf_counter() - t0
        text = tokenizer.decode([t['token'] for t in trajs])
        print(f"[{i+1}/{len(prompts_used)}] {len(trajs)} tokens in {elapsed:.0f}s: "
              f"\"{text[:60]}\"")
        all_trajectories.extend(trajs)

    total_time = time.perf_counter() - total_start
    n_total = len(all_trajectories)
    print(f"\nTotal: {n_total} trajectories in {total_time:.0f}s")

    # Save trajectories to disk for later analysis
    save_path = Path("experiments/trajectories.npz")
    print(f"Saving trajectories to {save_path}...", end=" ", flush=True)
    save_dict = {}
    for i, t in enumerate(all_trajectories):
        save_dict[f"h_seq_{i}"] = t['h_seq'].astype(np.float16)
        save_dict[f"delta_seq_{i}"] = t['delta_seq'].astype(np.float16)
        save_dict[f"token_{i}"] = np.int32(t['token'])
        save_dict[f"position_{i}"] = np.int32(t['position'])
    save_dict['n_trajectories'] = np.int32(n_total)
    np.savez_compressed(save_path, **save_dict)
    print(f"done")

    if n_total < args.n_archetypes * 3:
        print(f"Too few trajectories ({n_total}) for {args.n_archetypes} archetypes. "
              f"Need at least {args.n_archetypes * 3}.")
        return

    # Extract features and cluster
    print(f"\nClustering into {args.n_archetypes} archetypes...")
    features = trajectory_features(all_trajectories)
    labels, centroids = kmeans(features, args.n_archetypes)

    # Archetype sizes
    sizes = [int(np.sum(labels == c)) for c in range(args.n_archetypes)]
    print(f"Archetype sizes: {sizes}")

    # ── THE KEY MEASUREMENT: Intra-archetype Δ variance ──
    print(f"\n{'='*70}")
    print("CRITICAL: var(Δ_l | archetype) / var(Δ_l)")
    print("  If this ratio << 1.0 for archetypes with reasonable size,")
    print("  then archetype lookup can replace FFN computation.")
    print(f"{'='*70}")

    # For each layer, compute total Δ variance and per-archetype variance
    all_deltas = np.array([t['delta_seq'] for t in all_trajectories])  # (N, 22, 2048)
    total_var = np.var(all_deltas, axis=0).mean(axis=1)  # (22,) mean over hidden_dim

    print(f"\n{'Layer':<6} {'total_var':>12}", end="")
    for c in range(args.n_archetypes):
        if sizes[c] >= 3:
            print(f" {'A'+str(c):>10}", end="")
    print(f" {'best_ratio':>10} {'viable?':>8}")
    print("-" * (18 + 11 * (args.n_archetypes + 1)))

    viable_archetypes = 0
    for l in range(N_LAYERS):
        tv = total_var[l]
        print(f"L{l:<5} {tv:>12.6f}", end="")
        best_ratio = 1.0

        for c in range(args.n_archetypes):
            if sizes[c] < 3:
                continue
            mask = labels == c
            archetype_deltas = all_deltas[mask, l, :]  # (size_c, 2048)
            intra_var = np.var(archetype_deltas, axis=0).mean()  # mean over hidden_dim
            ratio = intra_var / (tv + 1e-12)
            best_ratio = min(best_ratio, ratio)
            marker = "✓" if ratio < 0.3 else ("△" if ratio < 0.6 else "✗")
            print(f" {ratio:>9.3f}{marker}", end="")

        viable = "✓" if best_ratio < 0.3 else ("△" if best_ratio < 0.6 else "✗")
        if best_ratio < 0.3:
            viable_archetypes += 1
        print(f" {best_ratio:>9.3f} {viable:>8}")

    print(f"\nViable layers (best_ratio < 0.3): {viable_archetypes}/{N_LAYERS}")

    # ── Markov Transition Test ──
    print(f"\n{'='*70}")
    print("MARKOV: Does layer l archetype predict layer l+1 archetype?")
    print(f"{'='*70}")

    # Per-layer archetype assignment
    layer_archetypes = np.zeros((n_total, N_LAYERS), dtype=int)
    for i in range(n_total):
        for l in range(N_LAYERS):
            # Assign this layer to the nearest archetype centroid
            h_l = all_trajectories[i, l, :]
            h_norm = np.linalg.norm(h_l)
            # Simplified: just use the trajectory-level label
            layer_archetypes[i, l] = labels[i]

    # If trajectory labels are constant across layers (same archetype for all layers),
    # then Markov predictability is trivial. Check if archetype changes within a trajectory.

    # Better: cluster PER-LAYER hidden states and measure transitions
    print("\nPer-layer hidden state clustering...")
    per_layer_labels = np.zeros((n_total, N_LAYERS), dtype=int)
    for l in range(N_LAYERS):
        h_l = np.array([t['h_seq'][l] for t in all_trajectories])  # (N, 2048)
        # Normalize
        h_l = h_l / (np.linalg.norm(h_l, axis=1, keepdims=True) + 1e-12)
        lbl, _ = kmeans(h_l, min(args.n_archetypes, n_total // 3), seed=42)
        per_layer_labels[:, l] = lbl

    # Transition matrix
    n_states = args.n_archetypes
    transitions = np.zeros((n_states, n_states))
    for i in range(n_total):
        for l in range(N_LAYERS - 1):
            src = per_layer_labels[i, l]
            dst = per_layer_labels[i, l + 1]
            transitions[src, dst] += 1

    # Normalize
    row_sums = transitions.sum(axis=1, keepdims=True) + 1e-12
    trans_prob = transitions / row_sums

    # Diagonal dominance = how predictable transitions are
    diag_mean = np.mean(np.diag(trans_prob))
    print(f"Mean diagonal transition probability: {diag_mean:.3f}")
    print(f"  (>0.5 = highly predictable, <0.2 = nearly random)")
    print(f"\nTransition matrix (row = layer l, col = layer l+1):")
    print("     ", end="")
    for c in range(n_states):
        print(f"  A{c:<3}", end="")
    print()
    for src in range(n_states):
        print(f"  A{src}:", end="")
        for dst in range(n_states):
            print(f" {trans_prob[src, dst]:.3f}", end="")
        print()

    # ── Archetype Separation ──
    print(f"\n{'='*70}")
    print("ARCHETYPE SEPARATION: cos between archetype mean Δ")
    print(f"{'='*70}")
    for c1 in range(args.n_archetypes):
        if sizes[c1] < 3: continue
        mask1 = labels == c1
        mean_delta1 = all_deltas[mask1].mean(axis=0)  # (22, 2048)
        print(f"  A{c1} (n={sizes[c1]}):", end="")
        for c2 in range(c1 + 1, args.n_archetypes):
            if sizes[c2] < 3: continue
            mask2 = labels == c2
            mean_delta2 = all_deltas[mask2].mean(axis=0)
            # Average cosine across layers
            cos_vals = []
            for l in range(N_LAYERS):
                a = mean_delta1[l]; b = mean_delta2[l]
                cos_vals.append(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            mean_cos = np.mean(cos_vals)
            print(f" cos(A{c1},A{c2})={mean_cos:.3f}", end="")
        print()

    # ── Summary ──
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    viable = viable_archetypes / N_LAYERS
    if viable > 0.7 and diag_mean > 0.5:
        print("✓ Trajectory VM is VIABLE. Archetypes explain Δ variance, transitions are predictable.")
    elif viable > 0.4:
        print("△ Trajectory VM is MARGINAL. Some layers benefit, but global lookup insufficient.")
    else:
        print("✗ Trajectory VM is NOT viable. Δ variance is not explained by trajectory archetypes.")
    print(f"  Viable layers: {viable_archetypes}/{N_LAYERS} ({viable:.0%})")
    print(f"  Markov predictability: {diag_mean:.3f}")

if __name__ == "__main__":
    main()
