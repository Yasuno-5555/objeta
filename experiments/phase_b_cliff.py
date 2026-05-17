#!/usr/bin/env python3
"""
Phase B: Precision Cliff Mechanism

Exp 7: Component ablation — which component triggers q3→q4 collapse?
  - Isolate FFN vs Attention quantization at the cliff edge
  - Hypothesis: residual stream channel capacity is the bottleneck

Exp 8: Continuous precision sweep q3.0→q5.0
  - Find exact cliff position
  - Measure transition sharpness

TinyLlama-1.1B, MPS GPU.
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
    n_levels = max(2, int(round(2 ** bits)))
    w_q = torch.zeros_like(w_f)
    for i in range(w_f.shape[0]):
        row = w_f[i]
        rmin, rmax = row.min(), row.max()
        span = rmax - rmin
        if span < 1e-10: w_q[i] = row; continue
        scale = span / (n_levels - 1)
        q_vals = ((row - rmin) / scale).round().clamp(0, n_levels - 1)
        w_q[i] = q_vals * scale + rmin
    return w_q.to(orig_dtype).to(orig_device)


def quantize_component(layer, component, bits):
    """Quantize FFN only, Attention only, or both."""
    if component == "ffn":
        names = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    elif component == "attn":
        names = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
    else:  # both
        names = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                  "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight"]

    for name in names:
        parts = name.split(".")
        obj = layer
        for p in parts[:-1]: obj = getattr(obj, p)
        w = obj.weight.data
        obj.weight = torch.nn.Parameter(quantize_tensor(w, bits))


def quantize_model_ablated(model, ffn_bits, attn_bits):
    """Quantize FFN and Attention to different bit widths."""
    for l in range(N_LAYERS):
        layer = model.model.layers[l]
        quantize_component(layer, "ffn", ffn_bits)
        quantize_component(layer, "attn", attn_bits)


def quantize_model_uniform(model, bits):
    quantize_model_ablated(model, bits, bits)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    if torch.backends.mps.is_available(): model = model.to("mps")
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


def compute_per_layer_cos(model_q, model_ref, tokenizer, text):
    """Per-layer hidden state cosine between quantized and reference."""
    device = next(model_ref.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ref = model_ref(**inputs, output_hidden_states=True)
        out_q = model_q(**inputs, output_hidden_states=True)

    cos_vals = {}
    for l in range(len(out_ref.hidden_states)):
        h_ref = out_ref.hidden_states[l][:, -1, :].cpu().float().numpy().flatten()
        h_q = out_q.hidden_states[l][:, -1, :].cpu().float().numpy().flatten()
        cos = np.dot(h_ref, h_q) / (np.linalg.norm(h_ref) * np.linalg.norm(h_q) + 1e-12)
        cos_vals[l] = float(cos)
    return cos_vals


def run_exp7_cliff_ablation():
    """Component ablation: which component triggers q3→q4 collapse?"""
    print("=" * 70)
    print("  Exp 7: Precision Cliff — Component Ablation")
    print("=" * 70)
    print()
    print("  Question: Is FFN or Attention responsible for q3→q4 collapse?")
    print()

    eval_texts = [
        "The history of artificial intelligence dates back to the 1950s when researchers first began exploring the possibility of creating machines that could think and learn.",
        "Climate change is one of the most pressing challenges facing humanity today, with rising temperatures and extreme weather events.",
        "The Renaissance was a period of European history marking the transition from the Middle Ages to modernity.",
    ]
    probe_text = "The capital of France is Paris, a city known for"

    # Reference
    print("Loading reference (all fp16)...")
    model_ref, tokenizer = load_model()
    ref_ppl = compute_ppl(model_ref, tokenizer, eval_texts)
    print(f"  Reference PPL: {ref_ppl:.2f}")

    # Conditions
    conditions = [
        ("A_all_q4",           4, 4,   "Baseline — survives"),
        ("B_all_q3",           3, 3,   "Full collapse — reference"),
        ("C_FFNq3_AttnFP16",   3, 16,  "FFN at cliff, Attention safe"),
        ("D_FFNFP16_Attnq3",   16, 3,  "FFN safe, Attention at cliff"),
        ("E_FFNq3_Attnq4",     3, 4,   "FFN at cliff, Attention marginal"),
        ("F_FFNq4_Attnq3",     4, 3,   "FFN marginal, Attention at cliff"),
        ("G_all_q5",           5, 5,   "Safe regime reference"),
    ]

    results = {}
    for name, ffn_b, attn_b, desc in conditions:
        print(f"\n  {name}: FFN={ffn_b}bit, Attn={attn_b}bit — {desc}")
        t0 = time.time()

        m, _ = load_model()
        quantize_model_ablated(m, ffn_b, attn_b)

        ppl = compute_ppl(m, tokenizer, eval_texts)
        layer_cos = compute_per_layer_cos(m, model_ref, tokenizer, probe_text)
        mean_cos = float(np.mean(list(layer_cos.values())))

        results[name] = {
            "ffn_bits": ffn_b, "attn_bits": attn_b,
            "ppl": float(ppl), "mean_layer_cos": mean_cos,
            "layer_cos": layer_cos, "desc": desc,
        }
        del m

        status = "SURVIVES" if ppl < 20 else "COLLAPSED" if ppl > 100 else "DEGRADED"
        print(f"    PPL={ppl:.2f}  mean_cos={mean_cos:.4f}  [{status}]  ({time.time()-t0:.0f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("  Exp 7 Results: Component Ablation")
    print("=" * 70)
    print(f"\n  Reference PPL: {ref_ppl:.2f}")
    print(f"\n  {'Condition':<25} {'FFN':>5} {'Attn':>6} {'PPL':>10} {'MeanCos':>9}  Status")
    print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*10} {'-'*9}  ------")

    for name in ["A_all_q4", "B_all_q3", "C_FFNq3_AttnFP16", "D_FFNFP16_Attnq3",
                  "E_FFNq3_Attnq4", "F_FFNq4_Attnq3", "G_all_q5"]:
        r = results[name]
        s = "SURVIVES" if r["ppl"] < 20 else "COLLAPSED" if r["ppl"] > 100 else "DEGRADED"
        print(f"  {name:<25} {r['ffn_bits']:>4}bit {r['attn_bits']:>4}bit  "
              f"{r['ppl']:>9.2f}  {r['mean_layer_cos']:>8.4f}  {s}")

    # Verdict
    c = results["C_FFNq3_AttnFP16"]
    d = results["D_FFNFP16_Attnq3"]
    b = results["B_all_q3"]
    a = results["A_all_q4"]

    print("\n  Verdict:")
    if c["ppl"] < 20 and d["ppl"] > 100:
        print("  → Attention q3 triggers collapse. FFN alone at q3 survives.")
        print("    Attention precision is the bottleneck for TinyLlama.")
    elif d["ppl"] < 20 and c["ppl"] > 100:
        print("  → FFN q3 triggers collapse. Attention alone at q3 survives.")
        print("    FFN precision is the bottleneck for TinyLlama.")
    elif c["ppl"] > 100 and d["ppl"] > 100:
        print("  → BOTH components at q3 cause collapse individually.")
        print("    The precision cliff is not component-specific — it's SYSTEMIC.")
        print("    This supports the residual bandwidth hypothesis.")
    else:
        print(f"  → Partial degradation. C={c['ppl']:.1f}, D={d['ppl']:.1f}")

    return results


def run_exp8_continuous_sweep():
    """Continuous precision sweep to find exact cliff position."""
    print("\n" + "=" * 70)
    print("  Exp 8: Continuous Precision Sweep q3.0→q5.0")
    print("=" * 70)
    print()

    eval_texts = [
        "The history of artificial intelligence dates back to the 1950s when researchers first began exploring.",
        "Climate change is one of the most pressing challenges facing humanity today.",
        "The Renaissance was a period of European history marking the transition from the Middle Ages.",
    ]

    # Bit widths to test: continuous from q3.0 to q5.0
    bit_levels = [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

    print(f"  Testing {len(bit_levels)} precision levels: {bit_levels}")
    print()

    results = {}
    for bits in bit_levels:
        levels = max(2, int(round(2 ** bits)))
        print(f"  q{bits:.2f} ({levels} levels)...", end=" ", flush=True)
        t0 = time.time()

        m, tokenizer = load_model()
        quantize_model_uniform(m, bits)

        ppl = compute_ppl(m, tokenizer, eval_texts)
        del m

        results[f"q{bits:.2f}"] = {"bits": bits, "levels": levels, "ppl": float(ppl)}
        status = "✓" if ppl < 20 else "☠" if ppl > 100 else "~"
        print(f"PPL={ppl:.2f} {status}  ({time.time()-t0:.0f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("  Exp 8 Results: Continuous Precision Sweep")
    print("=" * 70)
    print(f"\n  {'Bits':<8} {'Levels':>7} {'PPL':>10}  Status")
    print(f"  {'-'*8} {'-'*7} {'-'*10}  ------")

    prev_ppl = None
    cliff_point = None
    max_jump = 0

    for bits in bit_levels:
        key = f"q{bits:.2f}"
        r = results[key]
        s = "✓" if r["ppl"] < 20 else "☠" if r["ppl"] > 100 else "~"

        jump = ""
        if prev_ppl is not None and prev_ppl > 0:
            ratio = r["ppl"] / prev_ppl
            if ratio > 3:
                jump = f"  ← {ratio:.1f}x JUMP"
                if ratio > max_jump:
                    max_jump = ratio
                    cliff_point = bits

        print(f"  {bits:<8.2f} {r['levels']:>6}  {r['ppl']:>9.2f}  {s}{jump}")
        prev_ppl = r["ppl"]

    if cliff_point:
        print(f"\n  → Precision cliff detected at ~q{cliff_point:.2f}")
        print(f"    Max PPL jump: {max_jump:.1f}x")

    # Identify the sharpest transition
    ppls = [results[f"q{b:.2f}"]["ppl"] for b in bit_levels]
    log_ppls = np.log([max(p, 0.01) for p in ppls])
    slopes = np.diff(log_ppls) / np.diff(bit_levels)
    max_slope_idx = np.argmax(np.abs(slopes))
    print(f"    Sharpest transition: q{bit_levels[max_slope_idx]:.2f}→q{bit_levels[max_slope_idx+1]:.2f}")
    print(f"    Slope: {slopes[max_slope_idx]:.1f} (log PPL / bit)")

    return results


def run():
    results_7 = run_exp7_cliff_ablation()
    results_8 = run_exp8_continuous_sweep()

    # Save
    out = {
        "experiment": "phase_b_cliff_mechanism",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "exp7_component_ablation": results_7,
        "exp8_continuous_sweep": results_8,
    }
    path = RESULTS_DIR / "phase_b_cliff.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    run()
