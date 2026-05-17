#!/usr/bin/env python3
"""
Exp 13: Ultra-Low FFN Survival Frontier

Core hypothesis (post Phase C):
  Attention = transport routing (determines trajectory continuity)
  FFN = local field modulation (can degrade without breaking transport)

Test:
  1. Fix Attn at q5 (safe transport), sweep FFN down to find survival floor
  2. Compare "Attention Backbone" configs vs uniform baselines
  3. If Attn(q5)+FFN(q2.5) ≈ all_q4 PPL at LOWER bit → paradigm shift confirmed

Bonus: Tangent rank tracking during rollout
  - Does Δh effective rank stay low even with ultra-low FFN precision?
"""
import torch, torch.nn.functional as F, numpy as np, json, time
from pathlib import Path
from collections import defaultdict
import warnings; warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "phase_a_results"
RESULTS_DIR.mkdir(exist_ok=True)
N_LAYERS, HIDDEN, VOCAB = 22, 2048, 32000

# ═══════════════════════════════════════════════════════════════════
# Quantization primitives
# ═══════════════════════════════════════════════════════════════════

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

def quantize_weights(model, ffn_bits, attn_bits):
    for l in range(N_LAYERS):
        layer = model.model.layers[l]
        for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            obj = layer; [obj:=getattr(obj,p) for p in key.split(".")[:-1]]
            obj = layer
            for p in key.split("."): obj = getattr(obj, p)
            obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, ffn_bits))
        for key in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]:
            obj = layer
            for p in key.split("."): obj = getattr(obj, p)
            obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, attn_bits))

def quantize_qo_kv_split(model, ffn_bits, qo_bits, kv_bits):
    for l in range(N_LAYERS):
        layer = model.model.layers[l]
        for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            obj = layer
            for p in key.split("."): obj = getattr(obj, p)
            obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, ffn_bits))
        for key in ["self_attn.q_proj","self_attn.o_proj"]:
            obj = layer
            for p in key.split("."): obj = getattr(obj, p)
            obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, qo_bits))
        for key in ["self_attn.k_proj","self_attn.v_proj"]:
            obj = layer
            for p in key.split("."): obj = getattr(obj, p)
            obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, kv_bits))

def save_weights(model):
    saved = {}
    for l in range(N_LAYERS):
        layer = model.model.layers[l]
        for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj",
                     "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]:
            obj = layer
            for p in key.split("."): obj = getattr(obj, p)
            saved[f"L{l}_{key}"] = obj.weight.data.clone().cpu()
    return saved

def restore_weights(model, saved):
    for l in range(N_LAYERS):
        layer = model.model.layers[l]
        for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj",
                     "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]:
            obj = layer
            for p in key.split("."): obj = getattr(obj, p)
            obj.weight = torch.nn.Parameter(saved[f"L{l}_{key}"].to(obj.weight.device))

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

# ═══════════════════════════════════════════════════════════════════
# Tangent Rank Tracking
# ═══════════════════════════════════════════════════════════════════

def measure_tangent_rank(model, tokenizer, text):
    """Track Δh effective rank during autoregressive rollout."""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # Generate and collect hidden states
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=30, do_sample=True, temperature=0.7, top_p=0.9,
            output_hidden_states=True, return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id)

    layer_deltas = defaultdict(list)

    for step, step_out in enumerate(gen_out.hidden_states):
        # step_out is tuple of (n_layers+1) tensors, each [1, 1, hidden]
        for l in range(len(step_out) - 1):
            h_now = step_out[l][0, 0, :].float().cpu().numpy()
            h_next = step_out[l + 1][0, 0, :].float().cpu().numpy()
            delta = h_next - h_now
            layer_deltas[l].append(delta)

    # Compute effective rank of Δ matrix per layer
    results = {}
    for l in sorted(layer_deltas.keys()):
        deltas = np.array(layer_deltas[l])  # [n_steps, hidden]
        if len(deltas) < 4: continue

        # SVD of delta matrix
        U, S, Vt = np.linalg.svd(deltas, full_matrices=False)
        total_var = (S ** 2).sum()
        cumsum = np.cumsum(S ** 2) / total_var

        # Effective rank: dimensions for 50%, 90%, 95% variance
        eff_50 = int(np.searchsorted(cumsum, 0.50) + 1) if cumsum[-1] > 0 else 0
        eff_90 = int(np.searchsorted(cumsum, 0.90) + 1) if cumsum[-1] > 0 else 0
        eff_95 = int(np.searchsorted(cumsum, 0.95) + 1) if cumsum[-1] > 0 else 0

        results[str(l)] = {
            "eff_rank_50": eff_50, "eff_rank_90": eff_90, "eff_rank_95": eff_95,
            "sv_ratio_s1_s2": float(S[0] / (S[1] + 1e-12)),
            "n_steps": len(deltas),
        }

    return results

# ═══════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════

def run():
    print("=" * 66)
    print("  Exp 13: Ultra-Low FFN Survival Frontier")
    print("=" * 66)
    print()
    print("  Hypothesis: Attention = transport backbone.")
    print("  FFN precision can go very low if Attn precision is maintained.")
    print()

    texts = [
        "The history of artificial intelligence dates back to the 1950s when researchers first began exploring the possibility of creating machines that could think and learn.",
        "Climate change is one of the most pressing challenges facing humanity today, with rising temperatures and extreme weather events.",
        "The Renaissance was a period of European history marking the transition from the Middle Ages to modernity.",
    ]

    print("Loading model ONCE...")
    model, tokenizer = load_model()
    pristine = save_weights(model)
    ref_ppl = fast_ppl(model, tokenizer, texts)
    print(f"  Ref PPL (fp16): {ref_ppl:.2f}")
    print(f"  all_q4 PPL: 7.54 (from Exp 3)")

    # ═══════════════════════════════════════════════════════════════
    # Test 1: FFN Sweep with Attn fixed at q5
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  Test 1: FFN Sweep (Attn q5 fixed)")
    print("=" * 66)
    print(f"\n  {'FFN bits':<10} {'Levels':>7} {'Avg bit':>8} {'PPL':>9}  Status")
    print(f"  {'-'*10} {'-'*7} {'-'*8} {'-'*9}  ------")

    ffn_results = {}
    ffn_bits_list = [4.0, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25, 2.0]

    for ffn_b in ffn_bits_list:
        restore_weights(model, pristine)
        quantize_weights(model, ffn_b, 5.0)  # FFN=f, Attn=q5
        ppl = fast_ppl(model, tokenizer, texts)
        n_lev = int(round(2**ffn_b))
        avg_bit = (ffn_b * 5632 + 5.0 * 2048) / (5632 + 2048)  # weighted by param count approx
        # Actually simpler: avg = (FFN_bits * FFN_params + Attn_bits * Attn_params) / total
        # For TinyLlama: FFN = gate(5632*2048) + up(5632*2048) + down(2048*5632) ≈ 3*11.5M
        # Attn = Q(2048*256*32) + K(2048*64*32) + V(2048*64*32) + O(2048*256*32)
        # Not exact but roughly: FFN is ~60% of params, Attn is ~40%
        # But we're quantizing all weights so avg = (ffn_b * weight_ffn + attn_b * weight_attn) / total
        # Simplified: avg_bit estimate
        w_ffn = 3 * 5632 * 2048  # gate + up + down
        w_attn = 4 * 2048 * 2048  # Q + K + V + O (simplified, head structure ignored)
        total_w = w_ffn + w_attn
        avg_bit = (ffn_b * w_ffn + 5.0 * w_attn) / total_w

        s = "✓" if ppl < 20 else "☠" if ppl > 100 else "~"
        ffn_results[f"FFN{ffn_b}_Attn5"] = {"ffn": ffn_b, "attn": 5.0,
                                              "ppl": float(ppl), "avg_bit": avg_bit}
        print(f"  {ffn_b:<10.2f} {n_lev:>6}  {avg_bit:>7.2f}  {ppl:>8.2f}  {s}")

    # ═══════════════════════════════════════════════════════════════
    # Test 2: Attention Backbone configs vs uniform baselines
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  Test 2: Attention Backbone vs Uniform Baselines")
    print("=" * 66)

    backbone_configs = [
        # (name, ffn_b, attn_b)
        ("Backbone_A: Attn5+FFN3.5", 3.5, 5.0),
        ("Backbone_B: Attn5+FFN3.0", 3.0, 5.0),
        ("Backbone_C: Attn6+FFN2.75", 2.75, 6.0),
        ("Backbone_D: AttnQO5+KVq4+FFN3", 3.0, 5.0),  # with KV split
        ("Uniform_q4 (ref)", 4.0, 4.0),
        ("Uniform_q5 (ref)", 5.0, 5.0),
    ]

    print(f"\n  {'Config':<30} {'FFN':>5} {'Attn':>6} {'AvgBit':>7} {'PPL':>9}  vs q4")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*7} {'-'*9}  ------")

    backbone_results = {}
    all_q4_ppl = 7.54  # known from Exp 3

    for name, ffn_b, attn_b in backbone_configs:
        if name.startswith("Backbone_D"):
            restore_weights(model, pristine)
            quantize_qo_kv_split(model, ffn_b, attn_b, 4.0)  # QO=attn_b, KV=q4
        else:
            restore_weights(model, pristine)
            quantize_weights(model, ffn_b, attn_b)

        ppl = fast_ppl(model, tokenizer, texts)
        w_ffn = 3 * 5632 * 2048
        w_attn = 4 * 2048 * 2048
        avg_bit = (ffn_b * w_ffn + attn_b * w_attn) / (w_ffn + w_attn)

        s = "✓" if ppl < 20 else "☠"
        delta_q4 = ppl - all_q4_ppl
        backbone_results[name] = {"ffn": ffn_b, "attn": attn_b,
                                   "ppl": float(ppl), "avg_bit": avg_bit}

        better = "← BEATS q4" if ppl < all_q4_ppl else ""
        print(f"  {name:<30} {ffn_b:>4.1f}  {attn_b:>4.1f}  {avg_bit:>6.2f}  {ppl:>8.2f}  {delta_q4:>+6.2f}  {better}")

    # ═══════════════════════════════════════════════════════════════
    # Test 3: Tangent Rank (on best config)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  Test 3: Tangent Rank Stability (Δh effective rank)")
    print("=" * 66)

    tangent_text = ("The history of artificial intelligence dates back to the 1950s "
                    "when researchers first began exploring the possibility of creating "
                    "machines that could think and learn like humans.")

    # Reference: all fp16
    restore_weights(model, pristine)
    print("\n  Reference (fp16)...")
    tan_ref = measure_tangent_rank(model, tokenizer, tangent_text)

    # Best backbone config
    best_name = min(backbone_results.items(),
                    key=lambda x: x[1]["ppl"] if x[1]["ppl"] < 100 else float("inf"))[0]
    best_cfg = backbone_results[best_name]
    print(f"\n  Best backbone ({best_name}): FFN={best_cfg['ffn']}, Attn={best_cfg['attn']}")

    restore_weights(model, pristine)
    if "KV" in best_name:
        quantize_qo_kv_split(model, best_cfg["ffn"], best_cfg["attn"], 4.0)
    else:
        quantize_weights(model, best_cfg["ffn"], best_cfg["attn"])
    tan_backbone = measure_tangent_rank(model, tokenizer, tangent_text)

    # Uniform q4 for comparison
    restore_weights(model, pristine)
    quantize_weights(model, 4.0, 4.0)
    tan_q4 = measure_tangent_rank(model, tokenizer, tangent_text)

    # Print tangent rank summary
    key_layers = [0, 2, 8, 13, 18, 21]
    print(f"\n  {'L':<4}  {'fp16_90':>8}  {'fp16_95':>8}  {'best_90':>8}  {'q4_90':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for l in key_layers:
        ls = str(l)
        r90 = tan_ref.get(ls, {}).get("eff_rank_90", 0)
        r95 = tan_ref.get(ls, {}).get("eff_rank_95", 0)
        b90 = tan_backbone.get(ls, {}).get("eff_rank_90", 0)
        q90 = tan_q4.get(ls, {}).get("eff_rank_90", 0)
        print(f"  L{l:<3}  {r90:>8}  {r95:>8}  {b90:>8}  {q90:>8}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  Exp 13 Verdict")
    print("=" * 66)

    # Find FFN survival floor: lowest FFN bits where PPL < 20
    surviving = [(b, r["ppl"]) for b, r in ffn_results.items()
                 if r["ppl"] < 20]
    if surviving:
        floor = min(surviving, key=lambda x: x[1])
        print(f"\n  FFN Survival Floor: {floor[0]} (PPL={floor[1]:.2f})")
        print(f"  → FFN can go down to {floor[0].split('_')[0].replace('FFN','')} "
              f"with Attn q5")

    # Best backbone
    best_bb = min(backbone_results.items(),
                  key=lambda x: x[1]["ppl"] if x[1]["ppl"] < 100 else float("inf"))
    print(f"\n  Best Attention Backbone: {best_bb[0]}")
    print(f"    PPL={best_bb[1]['ppl']:.2f} @ {best_bb[1]['avg_bit']:.2f} avg bit")
    print(f"    vs all_q4: PPL=7.54 @ 4.0 bit")

    if best_bb[1]["ppl"] < all_q4_ppl and best_bb[1]["avg_bit"] < 4.0:
        print(f"\n  ✓ PARADIGM SHIFT: Attention Backbone beats uniform q4")
        print(f"    at LOWER bit budget with BETTER quality")
        print(f"    Transport continuity preservation > weight approximation")
    elif best_bb[1]["ppl"] < all_q4_ppl:
        print(f"\n  ~ Attention Backbone beats all_q4 PPL")
        print(f"    but at higher bit budget ({best_bb[1]['avg_bit']:.1f} vs 4.0)")
    else:
        print(f"\n  ⚠ Attention Backbone does NOT beat all_q4")
        print(f"    PPL={best_bb[1]['ppl']:.2f} vs 7.54")

    # Save
    out = {"experiment": "exp13_ffn_survival",
           "ffn_sweep": ffn_results,
           "backbone_configs": backbone_results,
           "tangent_rank_ref": tan_ref,
           "tangent_rank_backbone": tan_backbone,
           "tangent_rank_q4": tan_q4}
    path = RESULTS_DIR / "exp13_ffn_survival.json"
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    run()
