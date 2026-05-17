#!/usr/bin/env python3
"""
Phase A: Trajectory-Centric Perturbation Analysis

Measures what actually matters for rollout stability — not weight fidelity, not
layerwise cos, but TOKEN TRAJECTORY DIVERGENCE under quantization perturbation.

Metrics:
  1. Branch Survival Length (BSL) — tokens until trajectory diverges from baseline
  2. Basin Flip Probability — fraction of stochastic seeds causing bifurcation
  3. Per-Layer Perturbation Sensitivity — which layers trigger earliest divergence
  4. Attention-Inclusive Jacobian spectral radius — finite-difference ||J_l||

Core hypothesis (LKO):
  - UNFOLD (L2) = trajectory basin compiler → highest perturbation sensitivity
  - ISOMETRIC (L3-L13) = safe transport → perturbation-tolerant
  - DIVERGENT (L14-L21) = amplifier → moderate sensitivity
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "phase_a_results"
RESULTS_DIR.mkdir(exist_ok=True)

TINYLLAMA_N_LAYERS = 22
TINYLLAMA_HIDDEN = 2048

# ── Quantization with Stochastic Rounding ────────────────────────────────────

def quantize_tensor_stochastic(w: torch.Tensor, bits: int, seed: int) -> torch.Tensor:
    """Quantize with stochastic rounding (dithering).

    Stochastic rounding: w_q = floor(w/Δ + u) * Δ  where u ~ U(0,1)
    Different seeds → different quantization realizations → measures basin stability.
    """
    if bits >= 16:
        return w.float()

    rng = torch.Generator(device='cpu').manual_seed(seed)
    w_f = w.float()
    n_levels = 2 ** bits
    w_q = torch.zeros_like(w_f)

    for i in range(w_f.shape[0]):
        row = w_f[i]
        rmin, rmax = row.min(), row.max()
        span = rmax - rmin
        if span < 1e-10:
            w_q[i] = row
            continue
        scale = span / (n_levels - 1)
        # Stochastic rounding
        noise = torch.rand(row.shape, generator=rng)
        q_vals = ((row - rmin) / scale + noise).floor().clamp(0, n_levels - 1)
        w_q[i] = q_vals * scale + rmin

    return w_q


def quantize_layer_weights(model, layer_idx: int, bits: int, seed: int = 42):
    """Quantize FFN + attention weights for a single layer."""
    layer = model.model.layers[layer_idx]
    try:
        for name in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                      "self_attn.q_proj", "self_attn.k_proj",
                      "self_attn.v_proj", "self_attn.o_proj"]:
            w = getattr(getattr(layer, name), "weight")
            w_q = quantize_tensor_stochastic(w.data, bits, seed)
            getattr(layer, name).weight = torch.nn.Parameter(w_q)
    except AttributeError:
        # TinyLlama uses different naming
        for name in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                      "self_attn.q_proj", "self_attn.k_proj",
                      "self_attn.v_proj", "self_attn.o_proj"]:
            try:
                w = getattr(layer, name).weight.data
                getattr(layer, name).weight = torch.nn.Parameter(
                    quantize_tensor_stochastic(w, bits, seed))
            except AttributeError:
                pass


def quantize_all_layers(model, bits_per_layer: dict, seed: int = 42):
    """Quantize all layers to specified bit widths."""
    for l in range(TINYLLAMA_N_LAYERS):
        b = bits_per_layer.get(l, 4)
        quantize_layer_weights(model, l, b, seed + l * 1000)


# ── Model Loading ────────────────────────────────────────────────────────────

def load_tinyllama():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    model.eval()
    return model, tokenizer


# ── Metric 1: Branch Survival Length ─────────────────────────────────────────

def measure_bsl(model_base, model_quant, tokenizer, prompt: str,
                max_new: int = 80, temperature: float = 0.0) -> dict:
    """Branch Survival Length: tokens until quantized trajectory diverges from baseline.

    With temperature=0 (greedy), divergence is deterministic given weights.
    Returns: {bsl, baseline_tokens, quant_tokens, diverged_at, final_match}
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        baseline_ids = model_base.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        quant_ids = model_quant.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)

    baseline_tokens = baseline_ids[0, prompt_len:].tolist()
    quant_tokens = quant_ids[0, prompt_len:].tolist()

    # Find first divergence
    bsl = 0
    for i, (b, q) in enumerate(zip(baseline_tokens, quant_tokens)):
        if b != q:
            bsl = i
            break
    else:
        bsl = len(baseline_tokens)  # never diverged

    return {
        "bsl": bsl,
        "total_tokens": len(baseline_tokens),
        "baseline_tokens": baseline_tokens[:20],
        "quant_tokens": quant_tokens[:20],
        "diverged_at": bsl if bsl < len(baseline_tokens) else None,
        "final_match": bsl == len(baseline_tokens),
    }


# ── Metric 2: Basin Flip Probability ─────────────────────────────────────────

def measure_basin_flip_probability(model_ref, tokenizer, prompts: list[str],
                                    bits_per_layer: dict, n_seeds: int = 10,
                                    max_new: int = 50) -> dict:
    """Probability that stochastic quantization flips the trajectory basin.

    For each seed, quantize the model and measure if BSL < max_new (divergence).
    Basin flip = BSL < max_new (trajectory leaves baseline attractor).
    """
    flip_counts = defaultdict(int)
    seed_bsls = defaultdict(list)

    for seed in range(n_seeds):
        m, _ = load_tinyllama()
        quantize_all_layers(m, bits_per_layer, seed)

        for prompt in prompts:
            result = measure_bsl(model_ref, m, tokenizer, prompt, max_new)
            seed_bsls[seed].append(result["bsl"])
            if result["bsl"] < max_new:
                flip_counts[seed] += 1

        del m

    total_prompts = len(prompts)
    flip_probs = [flip_counts[s] / total_prompts for s in range(n_seeds)]

    return {
        "n_seeds": n_seeds,
        "n_prompts": total_prompts,
        "mean_flip_prob": float(np.mean(flip_probs)),
        "std_flip_prob": float(np.std(flip_probs)),
        "mean_bsl": float(np.mean([np.mean(v) for v in seed_bsls.values()])),
        "min_bsl": float(np.min([np.min(v) for v in seed_bsls.values()])),
        "seed_bsls": {str(k): v for k, v in seed_bsls.items()},
    }


# ── Metric 3: Per-Layer Perturbation Sensitivity ─────────────────────────────

def measure_per_layer_sensitivity(model_ref, tokenizer, prompts: list[str],
                                   perturbation_bits: int = 4,
                                   max_new: int = 50) -> dict:
    """Quantize ONE layer at a time, measure BSL impact.

    Hypothesis: L2 (UNFOLD) will have dramatically lower BSL than any other layer.
    """
    results = {}

    # Baseline: all fp16 → BSL should be max_new
    print("  Baseline (all fp16)...")
    baseline_bsls = []
    for prompt in prompts:
        r = measure_bsl(model_ref, model_ref, tokenizer, prompt, max_new)
        baseline_bsls.append(r["bsl"])
    baseline_mean = np.mean(baseline_bsls)

    # Test each early layer individually
    for test_layer in range(TINYLLAMA_N_LAYERS):
        print(f"  Layer {test_layer}...", end=" ", flush=True)

        m, _ = load_tinyllama()
        # All layers fp16 except test_layer → perturbation_bits
        bits = {l: 16 for l in range(TINYLLAMA_N_LAYERS)}
        bits[test_layer] = perturbation_bits
        quantize_all_layers(m, bits, seed=42)

        layer_bsls = []
        for prompt in prompts:
            r = measure_bsl(model_ref, m, tokenizer, prompt, max_new)
            layer_bsls.append(r["bsl"])

        results[str(test_layer)] = {
            "mean_bsl": float(np.mean(layer_bsls)),
            "min_bsl": float(np.min(layer_bsls)),
            "bsl_values": layer_bsls,
            "bsl_ratio": float(np.mean(layer_bsls) / baseline_mean if baseline_mean > 0 else 0),
        }

        print(f"BSL={np.mean(layer_bsls):.1f}")
        del m

    return {
        "baseline_mean_bsl": float(baseline_mean),
        "perturbation_bits": perturbation_bits,
        "per_layer": results,
    }


# ── Metric 4: Attention-Inclusive Jacobian ───────────────────────────────────

def estimate_full_jacobian_spectrum(model, tokenizer, prompts: list[str]) -> dict:
    """Estimate ||J_l|| for each layer using finite differences on full forward pass.

    J_l(h) = ∂(h + Attn_l(h) + FFN_l(h)) / ∂h

    Using: ||J_l|| ≈ ||F(h + εv) - F(h)|| / ε  for random unit vector v.
    Run multiple random directions, take max.
    """
    n_directions = 10
    epsilon = 0.01

    all_results = defaultdict(list)

    for prompt in prompts[:3]:  # limit for speed
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)

        # We need intermediate hidden states
        with torch.no_grad():
            # Use output_hidden_states to get per-layer h
            out = model(**inputs, output_hidden_states=True)
            hidden_states = out.hidden_states  # tuple of (n_layers+1) tensors

        for l in range(TINYLLAMA_N_LAYERS):
            h_in = hidden_states[l][:, -1, :].clone().detach()  # last token
            h_out = hidden_states[l + 1][:, -1, :].clone().detach()

            # Run forward for this single layer with perturbations
            layer = model.model.layers[l]
            layer_jnorms = []

            for _ in range(n_directions):
                v = torch.randn_like(h_in)
                v = v / (v.norm() + 1e-12)

                # Compute h_out for perturbed input
                h_perturbed = h_in + epsilon * v

                # Forward through this single layer
                residual = h_perturbed
                # RMSNorm
                rms = layer.input_layernorm(h_perturbed)
                # Attention
                attn_out = layer.self_attn(rms.unsqueeze(0), output_attentions=False)[0]
                h_mid = residual + attn_out.squeeze(0)
                # FFN
                post_rms = layer.post_attention_layernorm(h_mid)
                ffn_out = layer.mlp(post_rms)
                h_perturbed_out = h_mid + ffn_out

                # J applied to v: (F(h+εv) - F(h)) / ε
                delta = (h_perturbed_out - h_out.squeeze(0)) / epsilon
                jnorm = delta.norm().item() / v.norm().item()
                layer_jnorms.append(jnorm)

            all_results[l].append(max(layer_jnorms))  # max over directions

    # Aggregate across prompts
    return {
        str(l): {
            "mean_spectral_norm": float(np.mean(vals)),
            "max_spectral_norm": float(np.max(vals)),
            "std_spectral_norm": float(np.std(vals)),
        }
        for l, vals in all_results.items()
    }


# ── Quantization-Attractor Map ───────────────────────────────────────────────

def measure_attractor_map(model_ref, tokenizer, prompts: list[str],
                           bit_configs: list[dict]) -> dict:
    """For each bit configuration, measure entropy, repetition, diversity.

    Maps quantization configurations to behavioral phase space.
    """
    results = []

    for config in bit_configs:
        name = config.get("name", "unknown")
        bits = config.get("bits", {})
        print(f"  Config: {name}...", end=" ", flush=True)

        m, _ = load_tinyllama()
        quantize_all_layers(m, bits, seed=42)

        entropies = []
        rep_rates = []
        diversities = []

        for prompt in prompts[:5]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)

            with torch.no_grad():
                # Entropy at last prompt token
                out = m(**inputs)
                probs = F.softmax(out.logits[:, -1, :], dim=-1)
                valid = probs[0, :32000]
                valid = valid / valid.sum()
                ent = -(valid * torch.log(valid + 1e-12)).sum().item()
                entropies.append(ent)

                # Generate for repetition/diversity
                gen = m.generate(
                    **inputs, max_new_tokens=40, do_sample=True,
                    temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id)

            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = gen[0, prompt_len:].tolist()

            if len(new_tokens) > 1:
                # Repetition: consecutive duplicates
                dups = sum(1 for i in range(1, len(new_tokens))
                           if new_tokens[i] == new_tokens[i-1])
                rep_rates.append(dups / len(new_tokens))
            else:
                rep_rates.append(0.0)

            # Token diversity: unique / total
            if len(new_tokens) > 0:
                diversities.append(len(set(new_tokens)) / len(new_tokens))
            else:
                diversities.append(0.0)

        results.append({
            "name": name,
            "avg_bits": float(np.mean(list(bits.values()))),
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "mean_repetition": float(np.mean(rep_rates)),
            "mean_diversity": float(np.mean(diversities)),
            "bits": bits,
        })

        print(f"ent={np.mean(entropies):.2f} rep={np.mean(rep_rates):.2f}")
        del m

    return results


# ── Test Prompts ─────────────────────────────────────────────────────────────

def get_test_prompts() -> list[str]:
    return [
        "The capital of France is Paris, a city known for",
        "Machine learning is a subset of artificial intelligence that",
        "The quick brown fox jumps over the lazy dog and then",
        "In the beginning, God created the heavens and the",
        "The history of science shows that major breakthroughs often",
        "Climate change is one of the most pressing challenges",
        "Shakespeare's plays continue to be performed around the",
        "The human brain contains approximately 86 billion neurons",
        "Quantum mechanics describes the behavior of matter at",
        "The Industrial Revolution began in Britain in the late",
    ]


# ── Main Experiment ──────────────────────────────────────────────────────────

def run_phase_a():
    print("=" * 70)
    print("  Phase A: Trajectory-Centric Perturbation Analysis")
    print("=" * 70)
    print()

    print("Loading TinyLlama-1.1B-Chat-v1.0...")
    model, tokenizer = load_tinyllama()
    print(f"Loaded: {TINYLLAMA_N_LAYERS}L, {TINYLLAMA_HIDDEN}D")
    print()

    prompts = get_test_prompts()

    # ═══════════════════════════════════════════════════════════════════════
    # Experiment 2: UNFOLD Sensitivity (highest priority — quickest signal)
    # ═══════════════════════════════════════════════════════════════════════

    print("=" * 70)
    print("  EXP 2: UNFOLD Sensitivity — Per-Layer Perturbation BSL")
    print("=" * 70)
    print()
    print("  Hypothesis: L2 perturbation → drastically lower BSL than any other layer")
    print("  Method: Quantize ONE layer at a time to q4, all others fp16")
    print("  Metric: Branch Survival Length (tokens before trajectory diverges)")
    print()

    t0 = time.time()
    sensitivity_results = measure_per_layer_sensitivity(
        model, tokenizer, prompts[:6], perturbation_bits=4, max_new=50)
    print(f"\n  Completed in {time.time()-t0:.0f}s")

    # Find the most sensitive layer
    per_layer = sensitivity_results["per_layer"]
    sorted_layers = sorted(per_layer.items(),
                           key=lambda x: x[1]["mean_bsl"])
    most_sensitive = sorted_layers[0]
    least_sensitive = sorted_layers[-1]

    print(f"\n  Most sensitive layer:  L{most_sensitive[0]} (BSL={most_sensitive[1]['mean_bsl']:.1f})")
    print(f"  Least sensitive layer: L{least_sensitive[0]} (BSL={least_sensitive[1]['mean_bsl']:.1f})")
    print(f"  Sensitivity ratio:     {least_sensitive[1]['mean_bsl'] / max(most_sensitive[1]['mean_bsl'], 1):.1f}x")
    print(f"  Baseline BSL:          {sensitivity_results['baseline_mean_bsl']:.1f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Experiment 1: Jacobian-Aware Validation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  EXP 1: Jacobian-Aware Quantization — Basin Flip & BSL")
    print("=" * 70)
    print()

    # Define bit allocations (same total budget ≈ 4 bits avg)
    uniform_q4 = {l: 4 for l in range(TINYLLAMA_N_LAYERS)}
    lko_aware = {}
    for l in range(TINYLLAMA_N_LAYERS):
        if l <= 1:      lko_aware[l] = 4   # SYNC
        elif l == 2:    lko_aware[l] = 16  # UNFOLD — fp16 mandatory
        elif l <= 13:   lko_aware[l] = 3   # ISOMETRIC — aggressive
        elif l <= 20:   lko_aware[l] = 5   # DIVERGENT — conservative
        else:           lko_aware[l] = 4

    # Hessian-aware: protect top-3 Hessian layers (estimated via gradient norm)
    print("  Estimating Hessian trace for Hessian-aware config...")
    hessian = estimate_hessian_trace_simple(model, tokenizer)
    sorted_hess = sorted(hessian.items(), key=lambda x: x[1], reverse=True)
    hessian_aware = {l: 4 for l in range(TINYLLAMA_N_LAYERS)}
    for l, _ in sorted_hess[:3]:
        hessian_aware[l] = 8

    # Random: protect 3 random layers
    rng = np.random.RandomState(123)
    random_layers = set(rng.choice(TINYLLAMA_N_LAYERS, 3, replace=False))
    random_aware = {l: 8 if l in random_layers else 4 for l in range(TINYLLAMA_N_LAYERS)}

    configs = [
        {"name": "uniform_q4", "bits": uniform_q4},
        {"name": "random_q8", "bits": random_aware},
        {"name": "hessian_aware", "bits": hessian_aware},
        {"name": "lko_aware", "bits": lko_aware},
    ]

    # For each config: BSL with deterministic quantization + basin flip prob
    exp1_results = []
    for cfg in configs:
        name = cfg["name"]
        bits = cfg["bits"]
        avg_b = np.mean(list(bits.values()))
        print(f"\n  ── {name} (avg={avg_b:.1f}bit) ──")

        # BSL with deterministic quantization
        m, _ = load_tinyllama()
        quantize_all_layers(m, bits, seed=42)
        bsls = []
        for prompt in prompts[:6]:
            r = measure_bsl(model, m, tokenizer, prompt, max_new=50)
            bsls.append(r["bsl"])
        del m

        # Basin flip probability (stochastic seeds)
        print("    Measuring basin flip probability...")
        flip_result = measure_basin_flip_probability(
            model, tokenizer, prompts[:4], bits, n_seeds=8, max_new=50)

        entry = {
            "name": name,
            "avg_bits": float(avg_b),
            "mean_bsl": float(np.mean(bsls)),
            "min_bsl": float(np.min(bsls)),
            "bsl_values": bsls,
            "basin_flip_prob": flip_result["mean_flip_prob"],
            "basin_flip_std": flip_result["std_flip_prob"],
            "bits_protected": {str(l): b for l, b in bits.items() if b >= 8},
        }
        exp1_results.append(entry)

        print(f"    BSL={entry['mean_bsl']:.1f}  flip_prob={entry['basin_flip_prob']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Metric 4: Attention-Inclusive Jacobian
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  EXP 4: Attention-Inclusive Jacobian Spectrum")
    print("=" * 70)
    print("  Estimating ||J_l|| via finite differences on full forward pass...")
    print()

    jacobian = estimate_full_jacobian_spectrum(model, tokenizer, prompts[:3])

    # ═══════════════════════════════════════════════════════════════════════
    # Attractor Map
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  Attractor Map: Phase-Space Behavior per Configuration")
    print("=" * 70)
    print()

    attractor_configs = configs + [
        {"name": "all_q2", "bits": {l: 2 for l in range(TINYLLAMA_N_LAYERS)}},
        {"name": "all_q3", "bits": {l: 3 for l in range(TINYLLAMA_N_LAYERS)}},
        {"name": "all_q8", "bits": {l: 8 for l in range(TINYLLAMA_N_LAYERS)}},
        {"name": "all_fp16", "bits": {l: 16 for l in range(TINYLLAMA_N_LAYERS)}},
    ]
    attractor_results = measure_attractor_map(model, tokenizer, prompts, attractor_configs)

    # ═══════════════════════════════════════════════════════════════════════
    # Save & Print Results
    # ═══════════════════════════════════════════════════════════════════════

    all_results = {
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "n_layers": TINYLLAMA_N_LAYERS,
        "hidden_dim": TINYLLAMA_HIDDEN,
        "exp2_per_layer_sensitivity": sensitivity_results,
        "exp1_jacobian_aware": exp1_results,
        "jacobian_spectrum": jacobian,
        "attractor_map": attractor_results,
    }

    path = RESULTS_DIR / "phase_a_results.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved: {path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  Phase A Results Summary")
    print("=" * 70)

    print("\n  ── EXP 2: UNFOLD Sensitivity (BSL under per-layer perturbation) ──")
    print(f"  {'Layer':<8} {'BSL':>8} {'Ratio':>8}")
    for l_str, data in sorted(per_layer.items(), key=lambda x: x[1]["mean_bsl"]):
        marker = " ★" if float(l_str) == 2 else ""
        print(f"  L{l_str:<7} {data['mean_bsl']:>8.1f} {data['bsl_ratio']:>8.2f}{marker}")

    print("\n  ── EXP 1: Jacobian-Aware Quantization ──")
    print(f"  {'Config':<20} {'AvgBit':>6} {'BSL':>8} {'FlipProb':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*10}")
    best_bsl = max(exp1_results, key=lambda x: x["mean_bsl"])
    best_flip = min(exp1_results, key=lambda x: x["basin_flip_prob"])
    for e in exp1_results:
        bsl_mark = " ← BEST" if e["name"] == best_bsl["name"] else ""
        flip_mark = " ← LOWEST FLIP" if e["name"] == best_flip["name"] else ""
        print(f"  {e['name']:<20} {e['avg_bits']:>5.1f}  {e['mean_bsl']:>7.1f}  {e['basin_flip_prob']:>9.2f}{bsl_mark}{flip_mark}")

    print("\n  ── Attractor Map (entropy / repetition / diversity) ──")
    print(f"  {'Config':<20} {'Bits':>5} {'Entropy':>9} {'RepRate':>9} {'Diversity':>10}")
    for a in attractor_results:
        print(f"  {a['name']:<20} {a['avg_bits']:>4.1f}  {a['mean_entropy']:>8.3f}  "
              f"{a['mean_repetition']:>8.3f}  {a['mean_diversity']:>9.3f}")

    print(f"\n  Key Finding: L2 perturbation sensitivity = {most_sensitive[1]['mean_bsl']:.1f} BSL")
    if most_sensitive[0] == "2":
        print("  ✓ L2 is the MOST sensitive layer — confirms LKO UNFOLD hypothesis")
    else:
        print(f"  ⚠ L{most_sensitive[0]} is most sensitive, not L2")

    return all_results


def estimate_hessian_trace_simple(model, tokenizer) -> dict:
    """Estimate Hessian trace via gradient norms on sample forward passes.

    Simple proxy: ||grad||^2 for each layer's parameters.
    """
    hessian = defaultdict(float)
    texts = ["The capital of France is", "Machine learning is a",
             "The quick brown fox"]

    with torch.enable_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            # Need gradients
            model_params = list(model.parameters())
            out = model(**inputs)
            log_probs = F.log_softmax(out.logits[:, -1, :], dim=-1)
            loss = -log_probs[:, log_probs.argmax(dim=-1)].mean()
            loss.backward()

            for name, p in model.named_parameters():
                if p.grad is not None and "layers." in name:
                    try:
                        layer_str = name.split("layers.")[1].split(".")[0]
                        l = int(layer_str)
                        hessian[l] += p.grad.norm().item() ** 2
                    except ValueError:
                        pass

            model.zero_grad()

    # Normalize
    if hessian:
        max_v = max(hessian.values())
        for l in hessian:
            hessian[l] /= max_v
    for l in range(TINYLLAMA_N_LAYERS):
        if l not in hessian:
            hessian[l] = 0.5

    return dict(hessian)


if __name__ == "__main__":
    run_phase_a()
