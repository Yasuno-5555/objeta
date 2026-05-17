#!/usr/bin/env python3
"""Objeta v1.0: MoE Routing Compiler — Qwen3.6 Expert Analysis.

Analyzes Qwen3.6-35B-A3B routing structure:
  1. Expert occupancy per layer
  2. Expert transition matrix
  3. Routing entropy map
  4. Bridge layer detection
  5. Position-conditioned routing

Output: execution_plan.json

Uses existing Qwen3.6 binary weights from LKO project.
"""

import numpy as np
import json, struct, math, time
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────

BIN_DIR = Path("/Users/yasuno/projects/LKO/runtime/moe/converted/qwen36_bin")
N_LAYERS = 40
HIDDEN_DIM = 2048
N_EXPERTS = 256
TOP_K = 8

# ── Router Loading ────────────────────────────────────────────────────────

def load_routers():
    """Load all 40 router weight matrices (256 × 2048 fp32)."""
    routers = []
    for l in range(N_LAYERS):
        path = BIN_DIR / f"layer_{l}_router.bin"
        if path.exists():
            w = np.fromfile(path, dtype=np.float32).reshape(N_EXPERTS, HIDDEN_DIM)
            routers.append(w)
        else:
            routers.append(None)
    print(f"Loaded {sum(1 for r in routers if r is not None)}/{N_LAYERS} router matrices")
    return routers

# ── Synthetic Forward Pass ────────────────────────────────────────────────

def run_routing_analysis(routers, n_prompts=10, max_tokens=30):
    """Run forward pass through routers only and collect expert selections.

    We don't need the full FFN — just router logits to determine
    which experts would be selected. For a real measurement we'd need
    actual hidden states; here we use the router weights themselves
    to simulate realistic routing distributions.

    Method:
    - Generate synthetic hidden states (Gaussian, normalized)
    - Pass through router: logits = router @ h
    - Select top-8 experts
    - Track occupancy, transitions, entropy
    """
    rng = np.random.RandomState(42)
    n_inputs = n_prompts * max_tokens

    # Per-layer router outputs
    all_topk = np.zeros((N_LAYERS, n_inputs, TOP_K), dtype=np.int32)
    all_probs = np.zeros((N_LAYERS, n_inputs, TOP_K), dtype=np.float32)
    all_entropy = np.zeros((N_LAYERS, n_inputs), dtype=np.float32)

    for i in range(n_inputs):
        # Generate hidden state with position-dependent structure
        pos_frac = (i % max_tokens) / max_tokens
        h = rng.randn(HIDDEN_DIM).astype(np.float32)
        # Add position-dependent bias
        h += pos_frac * 0.1
        h /= np.linalg.norm(h)

        for l in range(N_LAYERS):
            if routers[l] is None:
                continue

            # Router forward: logits = W_router @ h (256,)
            logits = routers[l] @ h

            # Softmax
            logits_max = np.max(logits)
            probs = np.exp(logits - logits_max)
            probs /= np.sum(probs)

            # Top-k selection
            topk_idx = np.argpartition(-probs, TOP_K)[:TOP_K]
            topk_idx = topk_idx[np.argsort(-probs[topk_idx])]
            topk_probs = probs[topk_idx]

            all_topk[l, i] = topk_idx
            all_probs[l, i] = topk_probs

            # Routing entropy: H = -Σ p_i log p_i
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            all_entropy[l, i] = entropy

        # Simulate trajectory: hidden state evolves based on previous routing
        # (current approximation: just random walk)
        h = h + 0.01 * rng.randn(HIDDEN_DIM).astype(np.float32)

    return all_topk, all_probs, all_entropy

# ── Occupancy Analysis ────────────────────────────────────────────────────

def compute_occupancy(all_topk, all_probs):
    """Per-layer expert occupancy histogram."""
    occupancy = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.float32)

    for l in range(N_LAYERS):
        for i in range(all_topk.shape[1]):
            for k in range(TOP_K):
                eid = all_topk[l, i, k]
                occupancy[l, eid] += all_probs[l, i, k]

    # Normalize per layer
    for l in range(N_LAYERS):
        total = occupancy[l].sum()
        if total > 0:
            occupancy[l] /= total

    return occupancy

# ── Transition Matrix ─────────────────────────────────────────────────────

def compute_transitions(all_topk):
    """Expert transition matrix: P(expert_j at layer l+1 | expert_i at layer l).

    For each token, we track which expert was top-1 at each layer.
    """
    n_tokens = all_topk.shape[1]
    transitions = np.zeros((N_LAYERS - 1, N_EXPERTS, N_EXPERTS), dtype=np.float32)

    for i in range(n_tokens):
        for l in range(N_LAYERS - 1):
            src = all_topk[l, i, 0]  # top-1 expert at layer l
            dst = all_topk[l + 1, i, 0]  # top-1 expert at layer l+1
            transitions[l, src, dst] += 1

    # Normalize per layer
    for l in range(N_LAYERS - 1):
        row_sums = transitions[l].sum(axis=1, keepdims=True) + 1e-12
        transitions[l] /= row_sums

    return transitions

# ── Routing Entropy Map ────────────────────────────────────────────────────

def routing_entropy_map(all_entropy):
    """Per-layer mean routing entropy and variance."""
    mean_h = all_entropy.mean(axis=1)
    std_h = all_entropy.std(axis=1)
    return mean_h, std_h

# ── Bridge Layer Detection ─────────────────────────────────────────────────

def detect_bridge_layers(occupancy, transitions, mean_entropy):
    """Detect bridge layers where routing structure changes significantly.

    Signals:
    1. High routing entropy → expert uncertainty
    2. High transition entropy → expert switching
    3. Occupancy redistribution → expert rank changes
    """
    bridges = []

    for l in range(1, N_LAYERS - 1):
        score = 0.0
        reasons = []

        # 1. Entropy spike
        if mean_entropy[l] > mean_entropy[l-1] * 1.2:
            score += 1.0
            reasons.append(f"entropy_spike({mean_entropy[l-1]:.2f}→{mean_entropy[l]:.2f})")

        # 2. Transition entropy: -Σ P(dst|src) log P(dst|src)
        trans_entropy = -np.sum(
            transitions[l-1] * np.log(transitions[l-1] + 1e-12), axis=1
        ).mean()
        prev_trans_entropy = -np.sum(
            transitions[l-2] * np.log(transitions[l-2] + 1e-12), axis=1
        ).mean() if l >= 2 else trans_entropy
        if trans_entropy > prev_trans_entropy * 1.3:
            score += 1.0
            reasons.append(f"trans_entropy_spike({prev_trans_entropy:.2f}→{trans_entropy:.2f})")

        # 3. Occupancy correlation with previous layer
        occ_corr = np.corrcoef(occupancy[l], occupancy[l-1])[0, 1]
        if occ_corr < 0.7:
            score += 1.0
            reasons.append(f"occ_corr_drop({occ_corr:.3f})")

        if score >= 2.0:
            bridges.append({
                'layer': l,
                'score': score,
                'reasons': reasons,
                'entropy': float(mean_entropy[l]),
                'trans_entropy': float(trans_entropy),
            })

    return bridges

# ── Execution Plan Generation ─────────────────────────────────────────────

def generate_execution_plan(occupancy, transitions, mean_entropy, bridges):
    """Generate the ExecutionPlan for MoE runtime."""

    # Hot experts: top-8 per layer by occupancy (always in RAM)
    # Warm experts: next 16 per layer (mmap cached)
    # Cold experts: rest (SSD, lazy)

    hot_experts = {}
    warm_experts = {}
    cold_experts = {}

    for l in range(N_LAYERS):
        sorted_experts = np.argsort(-occupancy[l])
        hot_experts[str(l)] = sorted_experts[:8].tolist()
        warm_experts[str(l)] = sorted_experts[8:24].tolist()
        cold_experts[str(l)] = sorted_experts[24:].tolist()

    # Prefetch schedule: for each layer and top expert, predict next-layer experts
    prefetch_schedule = {}
    for l in range(N_LAYERS - 1):
        layer_schedule = {}
        for e_src in range(N_EXPERTS):
            # Top-3 most likely next experts
            next_experts = np.argsort(-transitions[l, e_src])[:3]
            layer_schedule[str(e_src)] = next_experts.tolist()
        prefetch_schedule[str(l)] = layer_schedule

    # Bridge layer policy
    bridge_policy = {}
    for b in bridges:
        l = b['layer']
        bridge_policy[str(l)] = {
            'wider_prefetch': True,  # prefetch top-8 instead of top-3
            'dual_residency': True,  # keep both src and dst experts loaded
            'delayed_eviction': True,
            'score': b['score'],
        }

    return {
        'model': 'Qwen3.6-35B-A3B',
        'n_layers': N_LAYERS,
        'n_experts': N_EXPERTS,
        'top_k': TOP_K,
        'layers_with_routers': sum(1 for _ in range(N_LAYERS)),
        'hot_experts': hot_experts,
        'warm_experts': warm_experts,
        'cold_experts': cold_experts,
        'prefetch_schedule': prefetch_schedule,
        'bridge_layers': [b['layer'] for b in bridges],
        'bridge_details': bridges,
        'bridge_policy': bridge_policy,
        'routing_entropy_mean': {str(l): float(mean_entropy[l]) for l in range(N_LAYERS)},
        'occupancy_skew': {str(l): float(occupancy[l].max() / (occupancy[l].mean() + 1e-12))
                          for l in range(N_LAYERS)},
    }

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Objeta v1.0 — MoE Routing Compiler")
    print("=" * 70)

    # Load router weights
    print("\n[1/5] Loading router weights...")
    routers = load_routers()
    if not any(r is not None for r in routers):
        print("ERROR: No router weights found. Run LKO's convert_q4_to_bin.py first.")
        return

    # Run routing analysis
    print("\n[2/5] Running routing analysis (300 synthetic tokens)...")
    t0 = time.perf_counter()
    all_topk, all_probs, all_entropy = run_routing_analysis(routers, n_prompts=10, max_tokens=30)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Occupancy
    print("\n[3/5] Computing expert occupancy...")
    occupancy = compute_occupancy(all_topk, all_probs)

    # Transitions
    print("[4/5] Computing transition matrices...")
    transitions = compute_transitions(all_topk)
    mean_entropy, std_entropy = routing_entropy_map(all_entropy)

    # Bridge detection
    print("[5/5] Detecting bridge layers and generating execution plan...")
    bridges = detect_bridge_layers(occupancy, transitions, mean_entropy)
    plan = generate_execution_plan(occupancy, transitions, mean_entropy, bridges)

    # ── Report ──
    print(f"\n{'='*70}")
    print("ROUTING ANALYSIS RESULTS")
    print(f"{'='*70}")

    # Top experts per layer
    print(f"\n  Top-8 experts (by occupancy) for key layers:")
    for l in [0, 2, 5, 10, 20, 30, 39]:
        top8 = np.argsort(-occupancy[l])[:8]
        print(f"  L{l:>2}: {top8.tolist()}")

    # Occupancy skew
    print(f"\n  Occupancy skew (max/mean) per zone:")
    for l in [0, 2, 5, 10, 20, 30, 39]:
        skew = occupancy[l].max() / (occupancy[l].mean() + 1e-12)
        print(f"  L{l:>2}: {skew:.1f}x")

    # Routing entropy
    print(f"\n  Routing entropy:")
    for l in [0, 2, 5, 10, 20, 30, 39]:
        print(f"  L{l:>2}: H={mean_entropy[l]:.3f} ± {std_entropy[l]:.3f}")

    # Bridge layers
    print(f"\n  Bridge layers detected: {len(bridges)}")
    for b in bridges[:5]:
        print(f"  L{b['layer']}: score={b['score']:.1f} "
              f"H={b['entropy']:.3f} reasons={b['reasons']}")

    # Prefetch accuracy estimate
    print(f"\n  Prefetch accuracy estimate:")
    for l in [0, 5, 10, 20]:
        diag_sum = 0.0
        for e in range(N_EXPERTS):
            diag_sum += transitions[l, e, e]  # P(same expert)
        diag_mean = diag_sum / N_EXPERTS
        print(f"  L{l}→L{l+1}: P(same_expert)={diag_mean:.3f}")

    # ── Save ──
    output_path = Path("experiments/execution_plan.json")
    with open(output_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\n  Execution plan saved: {output_path}")

if __name__ == "__main__":
    main()
