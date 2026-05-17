#!/usr/bin/env python3
"""
Cross-Family Validation: TinyLlama (Family A) vs Qwen2.5-0.5B (Family B)

Key LKO findings about Family B:
  - Spherical Steering: h_{l+1} ⟂ h_l (cos ≈ 0)
  - Aligned field: intra cos(attn, ffn) ≈ 0.999
  - Phase 1: effective rank = 1/48 (collapse-prone)
  - Stronger layer anisotropy
  - ||J|| ≈ 0.05 (contractive, not isometric)

Hypothesis:
  Family B has DIFFERENT precision sensitivity than Family A.
  Specifically: attention backbone may matter MORE in Family B
  because attention is the only diversity injection mechanism.

Tests:
  1. Precision sweep (q3.0→q5.0) — cliff position
  2. Attention backbone (Attn q5 + FFN sweep) — asymmetry
  3. QO vs KV split — transport routing bottleneck
"""
import torch, torch.nn.functional as F, numpy as np, json, time
from pathlib import Path
from collections import defaultdict
import warnings; warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "phase_a_results"
RESULTS_DIR.mkdir(exist_ok=True)

QWEN_N_LAYERS = 24
QWEN_HIDDEN = 896
QWEN_FFN = 4864  # intermediate_size for Qwen2.5-0.5B
QWEN_VOCAB = 151936

# ═══════════════════════════════════════════════════════════════
# Quantization
# ═══════════════════════════════════════════════════════════════

def quantize_tensor_fast(w, bits):
    if bits >= 16: return w.clone()
    n_levels = max(2, int(round(2 ** bits)))
    w_f = w.float()
    rmin = w_f.min(dim=1, keepdim=True).values
    rmax = w_f.max(dim=1, keepdim=True).values
    span = (rmax - rmin).clamp(min=1e-10)
    scale = span / (n_levels - 1)
    q = ((w_f - rmin) / scale).round().clamp(0, n_levels - 1)
    return (q * scale + rmin).to(w.dtype)

def quantize_weights(model, ffn_bits, attn_bits, n_layers):
    for l in range(n_layers):
        layer = model.model.layers[l]
        for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            try:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, ffn_bits))
            except AttributeError: pass
        for key in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]:
            try:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, attn_bits))
            except AttributeError: pass

def quantize_qo_kv_split(model, ffn_bits, qo_bits, kv_bits, n_layers):
    for l in range(n_layers):
        layer = model.model.layers[l]
        for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            try:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, ffn_bits))
            except AttributeError: pass
        for key in ["self_attn.q_proj","self_attn.o_proj"]:
            try:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, qo_bits))
            except AttributeError: pass
        for key in ["self_attn.k_proj","self_attn.v_proj"]:
            try:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, kv_bits))
            except AttributeError: pass

def save_weights(model, n_layers):
    saved = {}
    for l in range(n_layers):
        layer = model.model.layers[l]
        for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj",
                     "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]:
            try:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                saved[f"L{l}_{key}"] = obj.weight.data.clone().cpu()
            except AttributeError: pass
    return saved

def restore_weights(model, saved, n_layers):
    for l in range(n_layers):
        layer = model.model.layers[l]
        for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj",
                     "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]:
            k = f"L{l}_{key}"
            if k not in saved: continue
            try:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(saved[k].to(obj.weight.device))
            except AttributeError: pass

def load_qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    if torch.backends.mps.is_available(): model = model.to("mps")
    model.eval()
    return model, tokenizer, QWEN_N_LAYERS

def fast_ppl(model, tokenizer, texts):
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

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * 66)
    print("  Cross-Family Validation: Qwen2.5-0.5B (Family B)")
    print("=" * 66)
    print()
    print("  LKO classification: Spherical Steering, Phase 1")
    print("  intra_cos ≈ 0.999, effective_rank ≈ 1/48")
    print("  Hyp: Family B has different precision sensitivity")
    print()

    texts = [
        "The history of artificial intelligence dates back to the 1950s when researchers first began exploring.",
        "Climate change is one of the most pressing challenges facing humanity today.",
        "The Renaissance was a period of European history marking the transition from the Middle Ages.",
    ]

    print("Loading Qwen2.5-0.5B...")
    model, tokenizer, n_layers = load_qwen()
    print(f"  Loaded: {n_layers} layers, {QWEN_HIDDEN}D hidden, {QWEN_FFN}D FFN")
    pristine = save_weights(model, n_layers)
    ref_ppl = fast_ppl(model, tokenizer, texts)
    print(f"  Reference PPL (bf16): {ref_ppl:.2f}")

    # ═══════════════════════════════════════════════════════════
    # Test 1: Precision sweep
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  Test 1: Precision Sweep (Qwen2.5)")
    print("=" * 66)
    print(f"  {'Bits':<8} {'Levels':>7} {'PPL':>10}  vs fp16")
    print(f"  {'-'*8} {'-'*7} {'-'*10}  ------")

    sweep_results = {}
    for bits in [3.0, 3.25, 3.5, 4.0, 5.0]:
        restore_weights(model, pristine, n_layers)
        quantize_weights(model, bits, bits, n_layers)
        ppl = fast_ppl(model, tokenizer, texts)
        s = "✓" if ppl < 30 else "☠" if ppl > 200 else "~"
        sweep_results[f"q{bits:.2f}"] = float(ppl)
        d = ppl - ref_ppl
        print(f"  {bits:.2f}    {int(2**bits):>6}  {ppl:>9.2f}  {d:>+6.1f}  {s}")

    # ═══════════════════════════════════════════════════════════
    # Test 2: Attention backbone
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  Test 2: Attention Backbone (Qwen2.5)")
    print("=" * 66)

    backbone_tests = [
        ("Backbone: Attn5+FFN3", 3.0, 5.0),
        ("Backbone: Attn5+FFN3.5", 3.5, 5.0),
        ("Inverse:  Attn3+FFN5", 5.0, 3.0),
        ("Inverse:  Attn3.5+FFN5", 5.0, 3.5),
        ("Uniform q4", 4.0, 4.0),
        ("Uniform q5", 5.0, 5.0),
    ]

    backbone_results = {}
    for name, ffn_b, attn_b in backbone_tests:
        restore_weights(model, pristine, n_layers)
        quantize_weights(model, ffn_b, attn_b, n_layers)
        ppl = fast_ppl(model, tokenizer, texts)
        s = "✓" if ppl < 30 else "☠" if ppl > 200 else "~"
        backbone_results[name] = float(ppl)
        print(f"  {name:<28} FFN={ffn_b} Attn={attn_b} → PPL={ppl:.2f} {s}")

    # Asymmetry
    if "Backbone: Attn5+FFN3" in backbone_results and "Inverse:  Attn3+FFN5" in backbone_results:
        a = backbone_results["Backbone: Attn5+FFN3"]
        b = backbone_results["Inverse:  Attn3+FFN5"]
        ratio = b / a if a > 0 else float("inf")
        winner = "ATTN PRIORITY" if a < b else "FFN PRIORITY" if b < a else "EQUAL"
        print(f"\n  Asymmetry: Attn5+FFN3={a:.1f} vs Attn3+FFN5={b:.1f} ({ratio:.1f}x) → {winner}")

    # ═══════════════════════════════════════════════════════════
    # Test 3: QO vs KV split
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  Test 3: QO vs KV Split (Qwen2.5)")
    print("=" * 66)

    split_tests = [
        ("KVq3_QOq5_FFNq3", 3.0, 5, 3),
        ("KVq5_QOq3_FFNq3", 3.0, 3, 5),
    ]

    for name, ffn_b, qo_b, kv_b in split_tests:
        restore_weights(model, pristine, n_layers)
        quantize_qo_kv_split(model, ffn_b, qo_b, kv_b, n_layers)
        ppl = fast_ppl(model, tokenizer, texts)
        s = "✓" if ppl < 30 else "☠"
        print(f"  {name}: FFN={ffn_b} QO={qo_b} KV={kv_b} → PPL={ppl:.2f} {s}")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  Cross-Family Comparison")
    print("=" * 66)

    # TinyLlama data (from Phase C experiments)
    tinyllama = {
        "q3.00": 949.68, "q3.25": 41.99, "q3.50": 7.13, "q4.00": 5.83, "q5.00": 4.34,
        "FFN3_Attn5": 14.4, "FFN5_Attn3": 127.6,
        "family": "Family A (Residual Transport)",
        "ref_ppl": 4.09,
    }

    print(f"\n  {'Metric':<25} {'TinyLlama (A)':>16} {'Qwen2.5 (B)':>16}")
    print(f"  {'-'*25} {'-'*16} {'-'*16}")
    print(f"  {'Family':<25} {tinyllama['family']:>16} {'Family B (Spherical)':>16}")
    print(f"  {'Reference PPL':<25} {tinyllama['ref_ppl']:>15.2f}  {ref_ppl:>15.2f}")

    for bits in [3.0, 3.25, 3.5, 4.0, 5.0]:
        key = f"q{bits:.2f}"
        tv = tinyllama.get(key, float("nan"))
        qv = sweep_results.get(key, float("nan"))
        print(f"  {f'q{bits:.2f} PPL':<25} {tv:>15.2f}  {qv:>15.2f}")

    # Cliff comparison
    t_cliff = tinyllama["q3.25"] / tinyllama["q3.00"] if tinyllama["q3.00"] > 0 else 0
    q_cliff_3 = sweep_results.get("q3.25", 1) / max(sweep_results.get("q3.00", 1), 1)
    print(f"\n  Cliff ratio (q3.25/q3.00): TinyLlama={t_cliff:.1f}x  Qwen2.5={q_cliff_3:.1f}x")

    # Attention backbone asymmetry
    t_ratio = tinyllama["FFN5_Attn3"] / tinyllama["FFN3_Attn5"] if tinyllama["FFN3_Attn5"] > 0 else 0
    q_a = backbone_results.get("Backbone: Attn5+FFN3", 1)
    q_b = backbone_results.get("Inverse:  Attn3+FFN5", 1)
    q_ratio = q_b / q_a if q_a > 0 else 0
    print(f"  Attn asymmetry (inv/backbone): TinyLlama={t_ratio:.1f}x  Qwen2.5={q_ratio:.1f}x")

    # Save
    out = {"experiment": "cross_family_qwen25",
           "model": "Qwen2.5-0.5B", "family": "Family B (Spherical Steering)",
           "n_layers": n_layers, "hidden_dim": QWEN_HIDDEN,
           "ref_ppl": ref_ppl,
           "precision_sweep": sweep_results,
           "attention_backbone": backbone_results,
           "tinyllama_reference": tinyllama}
    path = RESULTS_DIR / "cross_family_qwen25.json"
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    run()
