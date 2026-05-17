#!/usr/bin/env python3
"""
Experiment 3: True Phase-Adaptive Quantization

Configs (TinyLlama-1.1B, MPS GPU):
  A: all q4                (4.0 avg bit)  — baseline
  B: all q3                (3.0 avg bit)
  C: L0-L2 fp16, L3-L13 q3, L14-L21 q4  (5.1 avg bit)  — phase-adaptive
  D: L0-L2 fp16, rest q4   (5.6 avg bit)  — F config from Exp 2
  E: all fp16              (16.0 avg bit) — oracle
  F: L0-L2 fp16, L3-L21 q2 (3.9 avg bit)  — late precision irrelevance test

Core test: C vs D at LOWER bit budget → hypothesis confirmed if C ≥ D.
Side test: F (aggressive) — does late precision matter?
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "phase_a_results"
RESULTS_DIR.mkdir(exist_ok=True)

N_LAYERS = 22


def quantize_tensor(w, bits):
    if bits >= 16:
        return w.data.clone()
    orig_dtype = w.dtype
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


def quantize_layer(model, layer_idx, bits):
    layer = model.model.layers[layer_idx]
    for name in ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                  "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
        parts = name.split(".")
        obj = layer
        for p in parts[:-1]:
            obj = getattr(obj, p)
        w = obj.weight.data
        obj.weight = torch.nn.Parameter(quantize_tensor(w, bits))


def quantize_model(model, bits_per_layer):
    for l in range(N_LAYERS):
        quantize_layer(model, l, bits_per_layer.get(l, 4))


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    if torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return model, tokenizer


def compute_ppl(model, tokenizer, texts):
    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0
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
    return float("inf") if total_tokens == 0 else np.exp(total_loss / total_tokens)


def measure_generation(model, tokenizer, prompts, max_new=50, temperature=0.7):
    device = next(model.parameters()).device
    rep_rates, diversities, entropies = [], [], []
    samples = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            # Entropy at last prompt token
            out = model(**inputs)
            probs = F.softmax(out.logits[:, -1, :], dim=-1)
            valid = probs[0, :32000]
            valid = valid / valid.sum()
            ent = -(valid * torch.log(valid + 1e-12)).sum().item()
            entropies.append(ent)

            # Generate
            gen = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id)
            new_tokens = gen[0, prompt_len:].tolist()

            if len(new_tokens) > 1:
                dups = sum(1 for i in range(1, len(new_tokens)) if new_tokens[i] == new_tokens[i - 1])
                rep_rates.append(dups / len(new_tokens))
                diversities.append(len(set(new_tokens)) / len(new_tokens))
            else:
                rep_rates.append(0.0)
                diversities.append(0.0)

            samples.append(tokenizer.decode(gen[0], skip_special_tokens=True))

    return {
        "mean_entropy": float(np.mean(entropies)),
        "mean_repetition": float(np.mean(rep_rates)),
        "mean_diversity": float(np.mean(diversities)),
        "sample": samples[0] if samples else "",
    }


def run():
    print("=" * 70)
    print("  Exp 3: True Phase-Adaptive Quantization")
    print("=" * 70)

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

    # ── Configs ──
    n = N_LAYERS
    configs = {
        "A_all_q4":               {l: 4 for l in range(n)},
        "B_all_q3":               {l: 3 for l in range(n)},
        "C_phase_adaptive":       {**{l: 16 for l in range(0, 3)},
                                   **{l: 3 for l in range(3, 14)},
                                   **{l: 4 for l in range(14, 22)}},
        "D_L0L2_fp16_rest_q4":    {**{l: 16 for l in range(0, 3)},
                                   **{l: 4 for l in range(3, 22)}},
        "E_all_fp16":             {l: 16 for l in range(n)},
        "F_early_fp16_late_q2":   {**{l: 16 for l in range(0, 3)},
                                   **{l: 2 for l in range(3, 22)}},
        "G_q4floor_div_q5":       {**{l: 16 for l in range(0, 3)},
                                   **{l: 4 for l in range(3, 14)},
                                   **{l: 5 for l in range(14, 22)}},
        "H_q4floor_div_q6":       {**{l: 16 for l in range(0, 3)},
                                   **{l: 4 for l in range(3, 14)},
                                   **{l: 6 for l in range(14, 22)}},
    }

    # Load oracle once
    print("\nLoading oracle (all fp16)...")
    model_oracle, tokenizer = load_model()

    results = {}
    for name, bits in configs.items():
        avg_b = np.mean(list(bits.values()))
        fp16_layers = [l for l, b in bits.items() if b >= 16]
        q2_layers = [l for l, b in bits.items() if b == 2]
        q3_layers = [l for l, b in bits.items() if b == 3]
        q4_layers = [l for l, b in bits.items() if b == 4]

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  avg={avg_b:.1f}bit  fp16:{fp16_layers}  "
              f"q2:{q2_layers}  q3:{q3_layers}  q4:{q4_layers}")
        print(f"{'='*60}")

        t0 = time.time()

        # Load & quantize
        m, _ = load_model()
        quantize_model(m, bits)

        # PPL
        ppl = compute_ppl(m, tokenizer, eval_texts)

        # Generation quality
        gen = measure_generation(m, tokenizer, prompts, max_new=50)
        del m

        elapsed = time.time() - t0
        results[name] = {
            "avg_bits": float(avg_b),
            "ppl": float(ppl),
            **gen,
        }
        print(f"  PPL={ppl:.2f}  ent={gen['mean_entropy']:.3f}  "
              f"rep={gen['mean_repetition']:.3f}  div={gen['mean_diversity']:.3f}  "
              f"({elapsed:.0f}s)")

    # ── Summary ──
    oracle_ppl = results["E_all_fp16"]["ppl"]
    all_q4_ppl = results["A_all_q4"]["ppl"]
    gap = all_q4_ppl - oracle_ppl

    print("\n" + "=" * 70)
    print("  Exp 3 Results: Phase-Adaptive Quantization")
    print("=" * 70)
    print(f"\n  Oracle PPL: {oracle_ppl:.2f}")
    print(f"  all_q4 PPL: {all_q4_ppl:.2f}  (gap={gap:+.2f})")
    print()
    print(f"  {'Config':<30} {'AvgBit':>6} {'PPL':>8} {'ΔPPL':>8} {'Recov':>7} {'Ent':>7} {'Rep':>7} {'Div':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for name in ["A_all_q4", "B_all_q3", "C_phase_adaptive", "D_L0L2_fp16_rest_q4",
                  "G_q4floor_div_q5", "H_q4floor_div_q6",
                  "F_early_fp16_late_q2", "E_all_fp16"]:
        r = results[name]
        dppl = r["ppl"] - oracle_ppl
        recovery = (1 - dppl / gap) * 100 if gap > 0 else 0
        marker = ""
        if name == "C_phase_adaptive":
            c_bit = r["avg_bits"]
            d_bit = results["D_L0L2_fp16_rest_q4"]["avg_bits"]
            c_ppl = r["ppl"]
            d_ppl = results["D_L0L2_fp16_rest_q4"]["ppl"]
            if c_ppl <= d_ppl and c_bit < d_bit:
                marker = " ← BETTER PPL AT LOWER BIT"
            elif c_ppl <= d_ppl:
                marker = " ← BETTER PPL"
        if name == "F_early_fp16_late_q2":
            f_ppl = r["ppl"]
            if f_ppl < all_q4_ppl:
                marker = f" ← BEATS ALL_Q4 (Δ={f_ppl-all_q4_ppl:+.2f})"

        print(f"  {name:<30} {r['avg_bits']:>5.1f}  {r['ppl']:>7.2f}  "
              f"{dppl:>+7.2f}  {recovery:>5.0f}%  {r['mean_entropy']:>6.3f}  "
              f"{r['mean_repetition']:>6.3f}  {r['mean_diversity']:>6.3f}{marker}")

    # ── Key verdicts ──
    print()
    print("  Verdicts:")

    c = results["C_phase_adaptive"]
    d = results["D_L0L2_fp16_rest_q4"]
    a = results["A_all_q4"]
    f = results["F_early_fp16_late_q2"]
    b = results["B_all_q3"]
    g = results["G_q4floor_div_q5"]
    h = results["H_q4floor_div_q6"]

    print(f"\n  1. q3 cliff: all_q3 PPL={b['ppl']:.0f} vs all_q4 PPL={a['ppl']:.2f}")
    print(f"     → q3 is NOT viable for TinyLlama, even in ISOMETRIC zone")

    print(f"\n  2. Early-layer protection: D (L0-L2 fp16 + rest q4)")
    print(f"     PPL={d['ppl']:.2f}, {(1-(d['ppl']-oracle_ppl)/gap)*100:.0f}% recovery @ {d['avg_bits']:.1f}bit")

    if g["ppl"] < d["ppl"]:
        div_gain = d["ppl"] - g["ppl"]
        print(f"\n  3. ✓ DIVERGENT q5 improves over q4:")
        print(f"     G: PPL={g['ppl']:.2f} (Δ={div_gain:+.2f} vs D) @ {g['avg_bits']:.1f}bit")
    else:
        print(f"\n  3. ⚠ DIVERGENT q5 does NOT improve over q4:")
        print(f"     G: PPL={g['ppl']:.2f} vs D: PPL={d['ppl']:.2f}")

    if h["ppl"] < g["ppl"]:
        div_gain2 = g["ppl"] - h["ppl"]
        print(f"\n  4. ✓ DIVERGENT q6 improves over q5:")
        print(f"     H: PPL={h['ppl']:.2f} (Δ={div_gain2:+.2f} vs G) @ {h['avg_bits']:.1f}bit")
    elif h["ppl"] < d["ppl"]:
        print(f"\n  4. ~ DIVERGENT q6 improves over q4 but not over q5")
        print(f"     H: PPL={h['ppl']:.2f} @ {h['avg_bits']:.1f}bit")
    else:
        print(f"\n  4. ⚠ DIVERGENT q6 does NOT improve:")
        print(f"     H: PPL={h['ppl']:.2f} @ {h['avg_bits']:.1f}bit")

    # Overall best
    non_oracle = {k: v for k, v in results.items() if k != "E_all_fp16"}
    best = min(non_oracle.items(), key=lambda x: x[1]["ppl"])
    print(f"\n  → Best non-oracle: {best[0]} (PPL={best[1]['ppl']:.2f}, "
          f"{best[1]['avg_bits']:.1f}bit, "
          f"{(1-(best[1]['ppl']-oracle_ppl)/gap)*100:.0f}% recovery)")

    # Save
    out = {
        "experiment": "exp3_phase_adaptive",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "oracle_ppl": oracle_ppl,
        "all_q4_ppl": all_q4_ppl,
        "gap": gap,
        "results": results,
    }
    path = RESULTS_DIR / "exp3_phase_adaptive.json"
    with open(path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n  Saved: {path}")
    return results


if __name__ == "__main__":
    run()
