#!/usr/bin/env python3
"""
Exp 1: Pareto-Optimal Phase-Adaptive Quantization — Head-to-Head

Compares allocation strategies at controlled bit budgets.

Budget tier 1 (~5.0 bit):
  A: Uniform q5                         5.00 bit
  D: LKO-bitmatched (L0-L2 q8, iso q4, div q5)  4.91 bit  ← lower bit!

Budget tier 2 (~5.4 bit):
  B: Hessian-aware (top-3 Hessian→q8, rest q5)   5.41 bit
  C: Random (3 random→q8, rest q5)               5.41 bit  — placebo
  D2: LKO-bitmatched (L0-L2 q8, iso q4, div q6)  5.27 bit  ← lower bit!

Budget tier 3 (6.0 bit):
  F: Uniform q6                         6.00 bit
  G: LKO-aware (fp16 early, q4 iso, q5 div)      6.00 bit  ← identical bit!

Reference:
  all_q4                                4.00 bit  (baseline)
  all_fp16                              16.0 bit  (oracle)

Core tests:
  D vs A: LKO allocation beats uniform at LOWER bit budget
  D2 vs B/C: LKO allocation beats Hessian/random at LOWER bit budget
  G vs F: LKO allocation beats uniform at IDENTICAL bit budget
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
    if bits >= 16: return w.data.clone()
    orig_dtype, orig_device = w.dtype, w.device
    w_f = w.float().cpu()
    n_levels = 2 ** bits
    w_q = torch.zeros_like(w_f)
    for i in range(w_f.shape[0]):
        row = w_f[i]
        rmin, rmax = row.min(), row.max()
        span = rmax - rmin
        if span < 1e-10:
            w_q[i] = row; continue
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
        for p in parts[:-1]: obj = getattr(obj, p)
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
            if inputs["input_ids"].shape[1] < 2: continue
            out = model(**inputs, labels=inputs["input_ids"])
            if out.loss is not None:
                total_loss += out.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
    return float("inf") if total_tokens == 0 else np.exp(total_loss / total_tokens)


def measure_generation(model, tokenizer, prompts, max_new=50, temperature=0.7):
    device = next(model.parameters()).device
    rep_rates, diversities, entropies, samples = [], [], [], []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]
            out = model(**inputs)
            probs = F.softmax(out.logits[:, -1, :], dim=-1)
            valid = probs[0, :32000]; valid = valid / valid.sum()
            ent = -(valid * torch.log(valid + 1e-12)).sum().item()
            entropies.append(ent)
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                                  temperature=temperature, top_p=0.9,
                                  pad_token_id=tokenizer.pad_token_id)
            new_tokens = gen[0, prompt_len:].tolist()
            if len(new_tokens) > 1:
                dups = sum(1 for i in range(1, len(new_tokens)) if new_tokens[i] == new_tokens[i-1])
                rep_rates.append(dups / len(new_tokens))
                diversities.append(len(set(new_tokens)) / len(new_tokens))
            else:
                rep_rates.append(0.0); diversities.append(0.0)
            samples.append(tokenizer.decode(gen[0], skip_special_tokens=True))
    return {"mean_entropy": float(np.mean(entropies)),
            "mean_repetition": float(np.mean(rep_rates)),
            "mean_diversity": float(np.mean(diversities)),
            "sample": samples[0] if samples else ""}


def estimate_hessian_trace(model, tokenizer):
    """Hessian trace proxy: gradient norms on sample forward passes."""
    hessian = {l: 0.0 for l in range(N_LAYERS)}
    texts = ["The capital of France is", "Machine learning is a", "The quick brown fox"]
    with torch.enable_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Need gradients on model
            for p in model.parameters(): p.requires_grad = True
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
                    except ValueError: pass
            model.zero_grad()
            for p in model.parameters(): p.requires_grad = False
    # Normalize
    max_v = max(hessian.values()) if max(hessian.values()) > 0 else 1.0
    return {l: v/max_v for l, v in hessian.items()}


def run():
    print("=" * 70)
    print("  Exp 1: Head-to-Head Phase-Adaptive Quantization")
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

    # ── Get Hessian trace for Hessian-aware allocation ──
    print("\nComputing Hessian trace for Hessian-aware config...")
    m_tmp, tokenizer = load_model()
    hessian = estimate_hessian_trace(m_tmp, tokenizer)
    top3_hessian = sorted(hessian.items(), key=lambda x: x[1], reverse=True)[:3]
    hessian_protected = [l for l, _ in top3_hessian]
    print(f"  Top-3 Hessian layers: {hessian_protected}")
    del m_tmp

    # Random protected layers (fixed seed for reproducibility)
    rng = np.random.RandomState(123)
    random_protected = sorted(rng.choice(N_LAYERS, 3, replace=False))
    print(f"  Random protected layers: {random_protected}")

    # ── Configs ──
    n = N_LAYERS

    def make_q8_protect(protected_layers, base=5):
        bits = {l: base for l in range(n)}
        for l in protected_layers: bits[l] = 8
        return bits

    configs = {
        # Tier 1: ~5.0 bit
        "A_uniform_q5": {
            "bits": {l: 5 for l in range(n)},
            "tier": 1, "desc": "Uniform q5 (5.00 bit)",
        },
        "D_lko_5bit": {
            "bits": {**{l: 8 for l in range(0, 3)}, **{l: 4 for l in range(3, 14)}, **{l: 5 for l in range(14, 22)}},
            "tier": 1, "desc": "LKO: L0-L2 q8, iso q4, div q5 (4.91 bit)",
        },
        # Tier 2: ~5.4 bit
        "B_hessian_aware": {
            "bits": make_q8_protect(hessian_protected, base=5),
            "tier": 2, "desc": f"Hessian-aware: protect {hessian_protected} → q8 (5.41 bit)",
        },
        "C_random": {
            "bits": make_q8_protect(random_protected, base=5),
            "tier": 2, "desc": f"Random: protect {random_protected} → q8 (5.41 bit)",
        },
        "D2_lko_5_4bit": {
            "bits": {**{l: 8 for l in range(0, 3)}, **{l: 4 for l in range(3, 14)}, **{l: 6 for l in range(14, 22)}},
            "tier": 2, "desc": "LKO: L0-L2 q8, iso q4, div q6 (5.27 bit)",
        },
        # Tier 3: 6.0 bit
        "F_uniform_q6": {
            "bits": {l: 6 for l in range(n)},
            "tier": 3, "desc": "Uniform q6 (6.00 bit)",
        },
        # References
        "G_lko_6bit": {
            "bits": {**{l: 16 for l in range(0, 3)}, **{l: 4 for l in range(3, 14)}, **{l: 5 for l in range(14, 22)}},
            "tier": 3, "desc": "LKO: L0-L2 fp16, iso q4, div q5 (6.00 bit) [from Exp3]",
        },
        "ref_all_q4": {
            "bits": {l: 4 for l in range(n)},
            "tier": 0, "desc": "Uniform q4 (4.00 bit) [baseline]",
        },
    }

    # Load oracle
    print("\nLoading oracle (all fp16)...")
    model_oracle, tokenizer = load_model()
    oracle_ppl = compute_ppl(model_oracle, tokenizer, eval_texts)
    print(f"  Oracle PPL: {oracle_ppl:.2f}")

    results = {}
    for name, cfg in configs.items():
        bits = cfg["bits"]
        avg_b = np.mean(list(bits.values()))
        fp16_n = sum(1 for b in bits.values() if b >= 16)
        q8_n = sum(1 for b in bits.values() if b == 8)
        q4_n = sum(1 for b in bits.values() if b == 4)

        print(f"\n{'='*60}")
        print(f"  [{cfg['tier']}] {cfg['desc']}")
        print(f"  avg={avg_b:.2f}bit  fp16={fp16_n}  q8={q8_n}  q4={q4_n}")
        print(f"{'='*60}")

        t0 = time.time()
        m, _ = load_model()
        quantize_model(m, bits)

        ppl = compute_ppl(m, tokenizer, eval_texts)
        gen = measure_generation(m, tokenizer, prompts)
        del m

        elapsed = time.time() - t0
        results[name] = {"avg_bits": float(avg_b), "ppl": float(ppl), **cfg, **gen}
        print(f"  PPL={ppl:.2f}  Δ={ppl-oracle_ppl:+.2f}  "
              f"ent={gen['mean_entropy']:.3f}  div={gen['mean_diversity']:.3f}  "
              f"({elapsed:.0f}s)")

    # ── Summary ──
    all_q4_ppl = results["ref_all_q4"]["ppl"]
    gap = all_q4_ppl - oracle_ppl

    print("\n" + "=" * 70)
    print("  Exp 1 Results: Head-to-Head Phase-Adaptive Quantization")
    print("=" * 70)
    print(f"\n  Oracle PPL: {oracle_ppl:.2f}")
    print(f"  all_q4 PPL: {all_q4_ppl:.2f}  (gap={gap:+.2f})")

    for tier_name, tier_configs in [
        ("Tier 1: ~5.0 bit", ["A_uniform_q5", "D_lko_5bit"]),
        ("Tier 2: ~5.4 bit", ["B_hessian_aware", "C_random", "D2_lko_5_4bit"]),
        ("Tier 3: 6.0 bit", ["F_uniform_q6", "G_lko_6bit"]),
    ]:
        print(f"\n  ── {tier_name} ──")
        print(f"  {'Config':<30} {'Bit':>6} {'PPL':>8} {'ΔPPL':>8} {'Recov':>7}")
        print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*7}")
        for name in tier_configs:
            if name not in results: continue
            r = results[name]
            dppl = r["ppl"] - oracle_ppl
            rec = (1 - dppl/gap)*100 if gap > 0 else 0
            best_in_tier = min((results[n]["ppl"] for n in tier_configs if n in results), default=0)
            marker = " ← BEST" if r["ppl"] == best_in_tier else ""
            print(f"  {name:<30} {r['avg_bits']:>5.2f}  {r['ppl']:>7.2f}  "
                  f"{dppl:>+7.2f}  {rec:>5.0f}%{marker}")

    # Verdicts
    print("\n  ── Verdicts ──")
    d = results["D_lko_5bit"]
    a = results["A_uniform_q5"]
    if d["ppl"] <= a["ppl"] and d["avg_bits"] < a["avg_bits"]:
        print(f"  ✓ Tier 1: LKO beats uniform at LOWER bit budget")
        print(f"    D: PPL={d['ppl']:.2f} @ {d['avg_bits']:.2f}bit vs A: PPL={a['ppl']:.2f} @ {a['avg_bits']:.2f}bit")
    else:
        print(f"  ⚠ Tier 1: D PPL={d['ppl']:.2f} vs A PPL={a['ppl']:.2f}")

    d2 = results["D2_lko_5_4bit"]
    b = results["B_hessian_aware"]
    c = results["C_random"]
    best_other = min(b["ppl"], c["ppl"])
    if d2["ppl"] <= best_other and d2["avg_bits"] <= min(b["avg_bits"], c["avg_bits"]):
        print(f"  ✓ Tier 2: LKO beats Hessian/Random at lower bit budget")
    elif d2["ppl"] <= best_other:
        print(f"  ~ Tier 2: LKO matches best competitor")
    else:
        print(f"  ⚠ Tier 2: LKO PPL={d2['ppl']:.2f} vs Hessian={b['ppl']:.2f} / Random={c['ppl']:.2f}")

    g = results["G_lko_6bit"]
    f = results["F_uniform_q6"]
    if g["ppl"] < f["ppl"]:
        print(f"  ✓ Tier 3: LKO beats uniform q6 at IDENTICAL 6.0 bit budget")
        print(f"    G: PPL={g['ppl']:.2f} vs F: PPL={f['ppl']:.2f}")
    else:
        print(f"  ⚠ Tier 3: G PPL={g['ppl']:.2f} vs F PPL={f['ppl']:.2f}")

    # Save
    out = {"experiment": "exp1_head_to_head", "model": "TinyLlama-1.1B-Chat-v1.0",
           "oracle_ppl": oracle_ppl, "all_q4_ppl": all_q4_ppl, "gap": gap,
           "hessian_protected": hessian_protected, "random_protected": random_protected,
           "results": {k: {kk: vv for kk, vv in v.items() if kk != "bits"}
                       for k, v in results.items()}}
    path = RESULTS_DIR / "exp1_head_to_head.json"
    with open(path, "w") as fp: json.dump(out, fp, indent=2)
    print(f"\n  Saved: {path}")
    return results


if __name__ == "__main__":
    run()
