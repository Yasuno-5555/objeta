#!/usr/bin/env python3
"""Step 2: FFN Neuron Graph → Latent Expert Extraction.

Loads activation dataset, builds per-layer neuron co-activation graph,
applies spectral clustering to discover latent experts.

Each expert = a group of FFN neurons that fire together across tokens.
These replace the dense FFN at runtime: instead of computing all 5632 neurons,
only the active expert group(s) for the current trajectory state execute.

Theory:
  FFN(x) = Σ_i gate_i(x) · E_i(x)
  where E_i is a low-rank expert extracted from co-activation clusters.

Usage:
  python experiments/build_neuron_graph.py
"""

import numpy as np
from pathlib import Path
import json

# ── Config ────────────────────────────────────────────────────────────────

DATA_DIR = Path("experiments/activations")
N_LAYERS = 22
HIDDEN_DIM = 2048
FFN_DIM = 5632
N_EXPERTS_PER_LAYER = 8  # target expert count

# ── Load Dataset ──────────────────────────────────────────────────────────

def load_dataset():
    """Load all collected activation npz files."""
    datasets = []
    for npz_path in sorted(DATA_DIR.glob("prompt_*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        datasets.append(data)
    print(f"Loaded {len(datasets)} activation files")
    return datasets

# ── Neuron Co-activation Matrix ────────────────────────────────────────────

def load_layer_activations(datasets, layer_idx):
    """Load intermediate FFN activations for one layer across all prompts."""
    all_activations = []
    for ds in datasets:
        key = f"layer{layer_idx}_ffn_intermediate"
        if key in ds:
            all_activations.append(ds[key])
    if not all_activations:
        return None

    activations = np.concatenate(all_activations, axis=0).astype(np.float32)
    n_samples = activations.shape[0]

    # Sparsity: fraction of near-zero activations (inactive neurons)
    sparsity = float(np.mean(np.abs(activations) < 1e-6))
    # Mean activation per neuron
    mean_act = np.mean(np.abs(activations), axis=0)
    active_fraction = float(np.mean(mean_act > 1e-6))

    print(f"  Layer {layer_idx}: {n_samples} samples, "
          f"sparsity={sparsity:.3f}, active_neurons={active_fraction:.1%}")

    return activations

# ── Pure NumPy K-Means ────────────────────────────────────────────────────

def kmeans_numpy(X, n_clusters, seed=42, max_iters=30):
    """Simple k-means clustering (pure numpy)."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    # Initialize with k-means++
    centroids = np.zeros((n_clusters, d))
    centroids[0] = X[rng.randint(n)]
    for c in range(1, n_clusters):
        dists = np.min([np.sum((X - centroids[i])**2, axis=1) for i in range(c)], axis=0)
        probs = dists / dists.sum()
        centroids[c] = X[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        # Assign
        dists = np.array([np.sum((X - centroids[i])**2, axis=1) for i in range(n_clusters)])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                centroids[c] = X[mask].mean(axis=0)
    return labels

# ── Spectral Clustering → Experts ─────────────────────────────────────────

def extract_experts(activations, n_experts):
    """Cluster neurons into latent experts via PCA + k-means.

    Strategy:
    1. PCA on activation matrix → reduce (n_samples, ffn_dim) to compact space
    2. K-means on the per-neuron component loadings
    3. Each cluster = one latent expert

    Returns:
      experts: list of dicts with {neuron_ids, centroid, rank, coverage}
    """
    if activations is None:
        return []

    n_samples, n_neurons = activations.shape

    # PCA: reduce neurons to a compact representation
    # SVD of activations: (n_samples × n_neurons) = U Σ V^T
    # V^T rows are the principal components of neuron space
    # We cluster neurons based on their V^T loadings
    #
    # For memory efficiency, use the Gram matrix trick:
    # If n_samples < n_neurons, compute eigenvectors of X^T X via X X^T
    if n_samples < n_neurons:
        # Compute X X^T (n_samples × n_samples) — much smaller
        act_c = activations - np.mean(activations, axis=0, keepdims=True)
        gram = act_c @ act_c.T  # (n_samples, n_samples)
        eigvals, eigvecs_u = np.linalg.eigh(gram.astype(np.float64))
        # Top components (largest eigenvalues)
        n_comp = min(64, n_samples - 1)
        top_eigvecs = eigvecs_u[:, -n_comp:]  # (n_samples, n_comp)
        top_eigvals = eigvals[-n_comp:]

        # Transform to neuron space: V = X^T @ U @ Σ^{-1/2}
        sigma_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(top_eigvals, 1e-12)))
        neuron_components = act_c.T @ top_eigvecs @ sigma_inv_sqrt  # (n_neurons, n_comp)
    else:
        # Direct SVD (expensive, fallback)
        act_c = activations - np.mean(activations, axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(act_c.astype(np.float32).T, full_matrices=False)
        n_comp = min(64, Vt.shape[0])
        neuron_components = Vt[:n_comp].T  # (n_neurons, n_comp)

    # Normalize
    norms = np.linalg.norm(neuron_components, axis=1, keepdims=True) + 1e-12
    neuron_components = neuron_components / norms

    # K-means on the component space
    labels = kmeans_numpy(neuron_components, n_experts, seed=42)

    # Build experts
    experts = []
    for c in range(n_experts):
        neuron_ids = np.where(labels == c)[0]
        if len(neuron_ids) == 0:
            continue

        # Expert centroid = mean activation pattern
        centroid = np.mean(activations[:, neuron_ids], axis=0)

        # Effective rank of this expert's activation subspace
        if len(neuron_ids) > 4:
            _, S, _ = np.linalg.svd(activations[:, neuron_ids].T, full_matrices=False)
            eff_rank = float(np.sum(S)**2 / (np.sum(S**2) + 1e-12))
        else:
            eff_rank = float(len(neuron_ids))

        coverage = len(neuron_ids) / FFN_DIM

        # Input signature: mean of the expert's dominant activation direction
        # (first principal component of the neuron activations within this cluster)
        if len(neuron_ids) > 1:
            # First principal component via power iteration
            sub = activations[:, neuron_ids]  # (n_samples, n_neurons)
            sub_c = sub - np.mean(sub, axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(sub_c.T, full_matrices=False)
            input_sig = Vt[0] if Vt.shape[0] > 0 else np.zeros(len(neuron_ids))
        else:
            input_sig = np.zeros(len(neuron_ids))

        experts.append({
            'id': c,
            'neuron_ids': neuron_ids.tolist(),
            'n_neurons': len(neuron_ids),
            'centroid_norm': float(np.linalg.norm(centroid)),
            'effective_rank': eff_rank,
            'coverage': coverage,
            'input_signature_norm': float(np.linalg.norm(input_sig)),
        })

    # Sort by coverage (largest experts first)
    experts.sort(key=lambda e: e['n_neurons'], reverse=True)

    return experts

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FFN Neuron Graph → Latent Expert Extraction")
    print("=" * 70)

    datasets = load_dataset()
    if not datasets:
        print("No activation data found. Run collect_activations.py first.")
        return

    all_experts = {}
    layer_stats = []

    for l in range(N_LAYERS):
        print(f"\nLayer {l}:")
        activations = load_layer_activations(datasets, l)
        if activations is None:
            print(f"  No data")
            continue

        experts = extract_experts(activations, N_EXPERTS_PER_LAYER)
        all_experts[l] = experts

        # Layer statistics
        n_total = sum(e['n_neurons'] for e in experts)
        max_cov = max(e['coverage'] for e in experts) if experts else 0
        mean_rank = np.mean([e['effective_rank'] for e in experts]) if experts else 0
        layer_stats.append({
            'layer': l,
            'n_experts': len(experts),
            'neurons_covered': n_total,
            'coverage_ratio': n_total / FFN_DIM,
            'max_expert_coverage': max_cov,
            'mean_effective_rank': mean_rank,
        })

        for e in experts[:3]:
            print(f"  Expert {e['id']}: {e['n_neurons']} neurons "
                  f"(cov={e['coverage']:.3f}, eff_rank={e['effective_rank']:.1f})")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'L':<4} {'n_exp':>6} {'covered':>8} {'max_cov':>8} {'mean_rank':>10}")
    print("-" * 40)
    for s in layer_stats:
        print(f"L{s['layer']:<3} {s['n_experts']:>6} {s['coverage_ratio']:>8.3f} "
              f"{s['max_expert_coverage']:>8.3f} {s['mean_effective_rank']:>10.1f}")

    # ── Save ──
    output = {
        'n_layers': N_LAYERS,
        'ffn_dim': FFN_DIM,
        'n_experts_per_layer': N_EXPERTS_PER_LAYER,
        'layers': {str(l): experts for l, experts in all_experts.items()},
        'layer_stats': layer_stats,
    }
    out_path = Path("experiments/neuron_graph.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
