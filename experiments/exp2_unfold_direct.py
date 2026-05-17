#!/usr/bin/env python3
"""
Experiment 2: UNFOLD Sensitivity Mapping — Direct Quantized Model Comparison

No simulation. No Lyapunov estimation. Real quantized TinyLlama generation.

Configs:
  A: all q4 (baseline)
  B: L2 only fp16, rest q4  ← key test
  C: L0 only fp16, rest q4
  D: L1 only fp16, rest q4
  E: L3 only fp16, rest q4
  F: L0-L2 fp16, rest q4
  G: all fp16 (oracle)

Hypothesis: B ≈ F ≈ G. L2 alone determines trajectory basin stability.

Metrics: perplexity (WikiText-2), repetition rate, entropy, BSL vs all_fp16
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "phase_a_results"
RESULTS_DIR.mkdir(exist_ok=True)

TINYLLAMA_N_LAYERS = 22


def quantize_tensor(w: torch.Tensor, bits: int, orig_dtype=None) -> torch.Tensor:
    """Per-row uniform quantization. Deterministic rounding. Preserves original dtype."""
    if bits >= 16:
        return w.data.clone()
    orig_dtype = orig_dtype or w.dtype
    orig_device = w.device
    w_f = w.float().cpu()
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
        q_vals = ((row - rmin) / scale).round().clamp(0, n_levels - 1)
        w_q[i] = q_vals * scale + rmin
    return w_q.to(orig_dtype).to(orig_device)


def quantize_layer(model, layer_idx: int, bits: int):
    """Quantize all weight matrices in a single layer."""
    layer = model.model.layers[layer_idx]
    weight_names = [
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]
    for name in weight_names:
        parts = name.split(".")
        obj = layer
        for p in parts[:-1]:
            obj = getattr(obj, p)
        w = obj.weight.data
        obj.weight = torch.nn.Parameter(quantize_tensor(w, bits, orig_dtype=w.dtype))


def quantize_model(model, bits_per_layer: dict):
    """Apply per-layer quantization to entire model."""
    for l in range(TINYLLAMA_N_LAYERS):
        bits = bits_per_layer.get(l, 4)
        quantize_layer(model, l, bits)


def load_model(device="mps"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load to CPU first, then move to MPS (device_map="mps" has allocation bugs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    """Perplexity on given texts."""
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if inputs["input_ids"].shape[1] < 2:
                continue
            out = model(**inputs, labels=inputs["input_ids"])
            if out.loss is not None:
                total_loss += out.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
    if total_tokens == 0:
        return float("inf")
    return np.exp(total_loss / total_tokens)


def measure_repetition(model, tokenizer, prompts: list[str],
                       max_new: int = 50, temperature: float = 0.7) -> dict:
    """Measure repetition rate and token diversity."""
    device = next(model.parameters()).device
    rep_rates = []
    diversities = []
    generated_texts = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gen = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = gen[0, prompt_len:].tolist()

            if len(new_tokens) > 1:
                dups = sum(1 for i in range(1, len(new_tokens))
                           if new_tokens[i] == new_tokens[i - 1])
                rep_rates.append(dups / len(new_tokens))
                diversities.append(len(set(new_tokens)) / len(new_tokens))
            else:
                rep_rates.append(0.0)
                diversities.append(0.0)

            generated_texts.append(tokenizer.decode(gen[0], skip_special_tokens=True))

    return {
        "mean_repetition": float(np.mean(rep_rates)),
        "std_repetition": float(np.std(rep_rates)),
        "mean_diversity": float(np.mean(diversities)),
        "samples": generated_texts[:3],
    }


def measure_bsl_vs_oracle(model_test, model_oracle, tokenizer,
                           prompts: list[str], max_new: int = 50) -> dict:
    """Branch Survival Length: tokens until test model diverges from oracle."""
    device = next(model_test.parameters()).device
    bsls = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            oracle_ids = model_oracle.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
            test_ids = model_test.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)

            oracle_tokens = oracle_ids[0, prompt_len:].tolist()
            test_tokens = test_ids[0, prompt_len:].tolist()

            for i, (o, t) in enumerate(zip(oracle_tokens, test_tokens)):
                if o != t:
                    bsls.append(i)
                    break
            else:
                bsls.append(len(oracle_tokens))

    return {
        "mean_bsl": float(np.mean(bsls)),
        "min_bsl": float(np.min(bsls)),
        "max_bsl": float(np.max(bsls)),
        "bsl_values": bsls,
    }


def run_exp2():
    print("=" * 70)
    print("  Exp 2: UNFOLD Sensitivity — Direct Quantized Model Comparison")
    print("=" * 70)
    print()
    print("  Hypothesis: L2 (UNFOLD) alone determines trajectory basin stability.")
    print("  Prediction: B (L2 fp16) ≈ F (L0-L2 fp16) ≈ G (all fp16)")
    print("  All others much worse, especially D (L1 fp16)")
    print()

    # Eval data — minimal for speed (CPU inference)
    eval_texts = [
        "The history of artificial intelligence dates back to the 1950s when researchers first began exploring the possibility of creating machines that could think and learn.",
        "Climate change is one of the most pressing challenges facing humanity today, with rising temperatures and extreme weather events.",
        "The Renaissance was a period of European history marking the transition from the Middle Ages to modernity.",
    ]

    prompts = [
        "The capital of France is Paris, a city known for",
        "Machine learning is a subset of artificial intelligence that",
        "In the beginning, God created the heavens and the",
    ]

    # ── Load oracle (all fp16) ──
    print("\nLoading oracle model (all fp16)...")
    model_oracle, tokenizer = load_model()

    # ── Configs ──
    n = TINYLLAMA_N_LAYERS
    configs = [
        ("A_all_q4",     {l: 4 for l in range(n)}),
        ("B_L2_fp16",    {2: 16, **{l: 4 for l in range(n) if l != 2}}),
        ("C_L0_fp16",    {0: 16, **{l: 4 for l in range(n) if l != 0}}),
        ("D_L1_fp16",    {1: 16, **{l: 4 for l in range(n) if l != 1}}),
        ("E_L3_fp16",    {3: 16, **{l: 4 for l in range(n) if l != 3}}),
        ("F_L0L2_fp16",  {0: 16, 1: 16, 2: 16, **{l: 4 for l in range(3, n)}}),
        ("G_all_fp16",   {l: 16 for l in range(n)}),
    ]

    results = {}
    for name, bits in configs:
        avg_b = np.mean(list(bits.values()))
        fp16_layers = [l for l, b in bits.items() if b >= 16]
        print(f"\n{'='*60}")
        print(f"  {name}  (fp16 layers: {fp16_layers}, avg={avg_b:.1f}bit)")
        print(f"{'='*60}")

        t0 = time.time()

        # Load & quantize
        print("  Loading & quantizing...")
        m, _ = load_model()
        quantize_model(m, bits)

        # Perplexity (3 short texts)
        print("  PPL...", end=" ", flush=True)
        ppl = compute_perplexity(m, tokenizer, eval_texts)
        print(f"{ppl:.2f}", flush=True)

        # Repetition (3 prompts, 30 tokens)
        print("  Repetition...", end=" ", flush=True)
        rep = measure_repetition(m, tokenizer, prompts, max_new=30)
        print(f"rep={rep['mean_repetition']:.3f}", flush=True)

        # BSL vs oracle (3 prompts, 30 tokens, greedy)
        print("  BSL...", end=" ", flush=True)
        bsl = measure_bsl_vs_oracle(m, model_oracle, tokenizer, prompts, max_new=30)
        print(f"BSL={bsl['mean_bsl']:.1f}", flush=True)

        results[name] = {
            "avg_bits": float(avg_b),
            "fp16_layers": fp16_layers,
            "ppl": float(ppl),
            "repetition": rep["mean_repetition"],
            "repetition_std": rep["std_repetition"],
            "diversity": rep["mean_diversity"],
            "bsl_mean": bsl["mean_bsl"],
            "bsl_min": bsl["min_bsl"],
            "bsl_values": bsl["bsl_values"],
            "generated_sample": rep["samples"][0] if rep["samples"] else "",
        }

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s → ppl={ppl:.2f} rep={rep['mean_repetition']:.3f} "
              f"div={rep['mean_diversity']:.3f} BSL={bsl['mean_bsl']:.1f}")

        del m

    # ── Save ──
    out = {
        "experiment": "exp2_unfold_sensitivity_direct",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "n_layers": TINYLLAMA_N_LAYERS,
        "results": results,
    }
    path = RESULTS_DIR / "exp2_direct.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {path}")

    # ── Summary Table ──
    print("\n" + "=" * 70)
    print("  EXP 2: Results")
    print("=" * 70)
    print()
    print(f"  {'Config':<20} {'Bit':>5} {'PPL':>8} {'ΔPPL':>8} {'RepRate':>9} {'BSL':>7} {'Diver':>7}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*7} {'-'*7}")

    all_fp16_ppl = results["G_all_fp16"]["ppl"]
    all_q4_ppl = results["A_all_q4"]["ppl"]
    all_q4_bsl = results["A_all_q4"]["bsl_mean"]
    all_fp16_bsl = results["G_all_fp16"]["bsl_mean"]

    for name in ["A_all_q4", "B_L2_fp16", "C_L0_fp16", "D_L1_fp16",
                  "E_L3_fp16", "F_L0L2_fp16", "G_all_fp16"]:
        r = results[name]
        dppl = r["ppl"] - all_fp16_ppl
        bsl_mark = ""
        if name == "B_L2_fp16":
            bsl_mark = f"  ← {'GOOD' if r['bsl_mean'] > all_q4_bsl * 1.5 else 'CHECK'}"
        print(f"  {name:<20} {r['avg_bits']:>4.1f}  {r['ppl']:>7.2f}  "
              f"{dppl:>+7.2f}  {r['repetition']:>8.3f}  {r['bsl_mean']:>6.1f}  "
              f"{r['diversity']:>6.3f}{bsl_mark}")

    # ── Key Finding ──
    print()
    print("  Key Comparisons:")
    print(f"    all_fp16 → all_q4 PPL drop:    {all_q4_ppl - all_fp16_ppl:+.2f}")
    print(f"    all_fp16 → L2_fp16 PPL drop:   {results['B_L2_fp16']['ppl'] - all_fp16_ppl:+.2f}")
    print(f"    all_fp16 → L1_fp16 PPL drop:   {results['D_L1_fp16']['ppl'] - all_fp16_ppl:+.2f}")

    l2_recovery = (1 - (results['B_L2_fp16']['ppl'] - all_fp16_ppl) /
                   max(all_q4_ppl - all_fp16_ppl, 0.01)) * 100
    print(f"\n    L2 protection recovers {l2_recovery:.0f}% of the all_q4→all_fp16 gap")
    print(f"    L2 BSL = {results['B_L2_fp16']['bsl_mean']:.1f} vs all_q4 BSL = {all_q4_bsl:.1f}")

    if l2_recovery > 50:
        print(f"\n    ✓ L2 is indeed the dominant sensitivity layer ({l2_recovery:.0f}% recovery)")
    else:
        print(f"\n    ⚠ L2 alone is not sufficient ({l2_recovery:.0f}% recovery)")

    return results


if __name__ == "__main__":
    run_exp2()
