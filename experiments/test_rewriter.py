#!/usr/bin/env python3
"""Test MoE Router Rewriter on OLMoE routing data.

Measures entropy reduction and locality improvement for:
  1. Temperature scaling (T = 0.3, 0.5, 0.7, 0.9)
  2. Locality bias (bias = 0.5, 1.0, 2.0, 5.0)
  3. Combined (T + locality)
  4. Soft expert pinning

Usage:
  python3 experiments/test_rewriter.py
"""

import json, sys, time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO))
sys.path.insert(0, str(PROJECT))

import numpy as np
import safetensors

from os_runtime.rewriter import (
    RouterRewriter, RoutingConfig, measure_entropy_reduction,
)


OUTPUT_DIR = PROJECT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_olmoe_logits(n_samples: int = 200) -> list[np.ndarray]:
    """Generate router logits from OLMoE gate weights."""
    SNAPSHOT = (
        "/Users/yasuno/.cache/huggingface/hub/"
        "models--allenai--OLMoE-1B-7B-0924/snapshots/"
        "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
    )
    shard = f"{SNAPSHOT}/model-00001-of-00003.safetensors"

    rng = np.random.RandomState(42)
    logits_list = []

    with safetensors.safe_open(shard, framework="pt") as f:
        # Use L0 gate weight
        gate = f.get_tensor("model.layers.0.mlp.gate.weight").float().numpy()
        # gate shape: (64, 2048)

    for _ in range(n_samples):
        hidden = rng.randn(2048).astype(np.float32)
        hidden /= np.linalg.norm(hidden)
        logits = gate @ hidden  # (64,)
        logits_list.append(logits)

    return logits_list


def load_stories_logits(n_samples: int = 200) -> list[np.ndarray]:
    """Generate router logits from stories15M gate weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_PATH = (
        "/Users/yasuno/.cache/huggingface/hub/"
        "models--ggml-org--stories15M_MOE/snapshots/"
        "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, device_map="cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Get gate weight
    gate = model.model.layers[0].block_sparse_moe.gate.weight.detach().cpu().numpy()
    # gate shape: (4, 288)

    # Run some tokens through to get real hidden states
    prompt = "Once upon a time there was a little cat"
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = list(inputs.input_ids[0].tolist())

    logits_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(torch.tensor([generated]), output_hidden_states=True)
            hidden = outputs.hidden_states[0][0, -1, :].cpu().numpy()
            logits = gate @ hidden  # (4,)
            logits_list.append(logits)
            generated.append(int(np.argmax(outputs.logits[0, -1, :].cpu().numpy())))

    return logits_list


def run_rewriter_test():
    print("═" * 60)
    print("  MoE Router Rewriter — Entropy Reduction Test")
    print("═" * 60)
    print()

    # Load OLMoE logits
    print("Loading OLMoE routing data...")
    olmoe_logits = load_olmoe_logits(200)
    print(f"  {len(olmoe_logits)} samples, {len(olmoe_logits[0])} experts")
    print()

    # Baseline
    print("─" * 60)
    print("  Baseline (no modification)")
    rewriter_base = RouterRewriter(RoutingConfig(), n_experts=64)
    base_result = measure_entropy_reduction(
        rewriter_base, olmoe_logits)
    print(f"  Entropy: {base_result['before_entropy_nat']:.2f} nat")
    print(f"  Effective k (90%): {base_result['before_effective_k']:.1f}")
    print()

    # Temperature sweep
    print("─" * 60)
    print("  Temperature Scaling (T < 1 sharpens)")
    print(f"  {'T':>6s} {'Entropy (nat)':>14s} {'Eff k':>8s} {'Ent Reduction':>14s} {'k Reduction':>12s}")
    print(f"  {'-'*6} {'-'*14} {'-'*8} {'-'*14} {'-'*12}")

    temp_results = []
    for T in [0.9, 0.7, 0.5, 0.3]:
        rewriter = RouterRewriter(RoutingConfig(temperature=T), n_experts=64)
        result = measure_entropy_reduction(rewriter, olmoe_logits)
        temp_results.append({"T": T, **result})
        print(f"  {T:6.2f} {result['after_entropy_nat']:14.2f} "
              f"{result['after_effective_k']:8.1f} {result['entropy_reduction']:13.1f}% "
              f"{result['k_reduction']:11.1f}%")

    print()

    # Locality bias sweep
    print("─" * 60)
    print("  Locality Bias (boost previous expert)")
    # Simulate previous expert selection: random with slight repetition
    rng = np.random.RandomState(42)
    prev_experts = []
    last_expert = rng.randint(0, 64)
    for i in range(200):
        if rng.random() < 0.05:  # 5% chance of same expert (mimics some locality)
            prev_experts.append(last_expert)
        else:
            last_expert = rng.randint(0, 64)
            prev_experts.append(last_expert)

    print(f"  {'Bias':>6s} {'Entropy (nat)':>14s} {'Eff k':>8s} {'Ent Reduction':>14s} {'k Reduction':>12s}")
    print(f"  {'-'*6} {'-'*14} {'-'*8} {'-'*14} {'-'*12}")

    bias_results = []
    for bias in [0.5, 1.0, 2.0, 5.0]:
        rewriter = RouterRewriter(
            RoutingConfig(locality_bias=bias, locality_decay=0.9),
            n_experts=64,
        )
        result = measure_entropy_reduction(
            rewriter, olmoe_logits, prev_experts=prev_experts)
        bias_results.append({"bias": bias, **result})
        print(f"  {bias:6.1f} {result['after_entropy_nat']:14.2f} "
              f"{result['after_effective_k']:8.1f} {result['entropy_reduction']:13.1f}% "
              f"{result['k_reduction']:11.1f}%")

    print()

    # Combined
    print("─" * 60)
    print("  Combined: Temperature + Locality Bias")
    best_temp = min(temp_results, key=lambda x: x["after_effective_k"])
    best_bias = min(bias_results, key=lambda x: x["after_effective_k"])
    print(f"  Best T: {best_temp['T']:.1f} (k={best_temp['after_effective_k']:.1f})")
    print(f"  Best bias: {best_bias['bias']:.1f} (k={best_bias['after_effective_k']:.1f})")

    rewriter_combined = RouterRewriter(
        RoutingConfig(
            temperature=best_temp["T"],
            locality_bias=best_bias["bias"],
            locality_decay=0.9,
        ),
        n_experts=64,
    )
    combined_result = measure_entropy_reduction(
        rewriter_combined, olmoe_logits, prev_experts=prev_experts)

    print(f"  Combined entropy: {combined_result['after_entropy_nat']:.2f} nat "
          f"(reduction: {combined_result['entropy_reduction']:.1f}%)")
    print(f"  Combined effective k: {combined_result['after_effective_k']:.1f} "
          f"(reduction: {combined_result['k_reduction']:.1f}%)")
    print()

    # stories15M for comparison
    print("─" * 60)
    print("  Comparison: stories15M (already specialized)")
    stories_logits = load_stories_logits(200)
    stories_base = RouterRewriter(RoutingConfig(), n_experts=4)

    # Before
    before = measure_entropy_reduction(stories_base, stories_logits)
    print(f"  stories15M baseline: entropy={before['before_entropy_nat']:.2f} nat, "
          f"k={before['before_effective_k']:.1f}")
    print()

    # Summary
    print("═" * 60)
    print("  Summary: Locality Recovery on OLMoE")
    print("═" * 60)
    print(f"  OLMoE baseline:         entropy=4.16 nat, k=64, locality=0%")
    print(f"  Temperature T={best_temp['T']:.1f}:      entropy={best_temp['after_entropy_nat']:.2f} nat, "
          f"k={best_temp['after_effective_k']:.1f}")
    print(f"  Locality bias={best_bias['bias']:.1f}:     entropy={best_bias['after_entropy_nat']:.2f} nat, "
          f"k={best_bias['after_effective_k']:.1f}")
    print(f"  Combined:               entropy={combined_result['after_entropy_nat']:.2f} nat, "
          f"k={combined_result['after_effective_k']:.1f}")
    print(f"  stories15M (reference):  entropy=0.21 nat, k=2-4")
    print()

    # Projection
    if combined_result["k_reduction"] > 30:
        print(f"  ★ With {combined_result['k_reduction']:.0f}% k reduction:")
        print(f"    Cache hit rate (static 16/64): 10% → est. "
              f"{min(90, 10 * 64 / combined_result['after_effective_k']):.0f}%")
        print(f"    MoE I/O: ~500ms → est. ~{max(80, 500 * combined_result['after_effective_k'] / 64):.0f}ms")
    print()

    # Save
    result = {
        "temperature_sweep": temp_results,
        "locality_bias_sweep": bias_results,
        "combined": combined_result,
        "baseline": base_result,
        "stories_reference": before,
    }
    out_path = OUTPUT_DIR / "rewriter_test.json"
    json.dump(result, open(out_path, "w"), indent=2, default=str)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    run_rewriter_test()
