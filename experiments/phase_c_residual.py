#!/usr/bin/env python3
"""
Phase C: Residual Continuity Preservation

Exp 9: Residual Mutual Information Sweep
  - Per-layer: token prediction agreement (logit lens @ each layer)
  - Neighborhood topology preservation
  - NOT cosine — we already proved cos≈1 doesn't guarantee rollout stability

Exp 10: Attention Bandwidth Hypothesis
  - Sweep FFN vs Attn precision independently
  - Test: can high Attn precision compensate for low FFN precision?
  - "Attention determines residual stream transport capacity"

TinyLlama-1.1B, MPS GPU.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "phase_a_results"
RESULTS_DIR.mkdir(exist_ok=True)

N_LAYERS = 22
HIDDEN_DIM = 2048
VOCAB_SIZE = 32000


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
    if component == "ffn":
        names = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    else:
        names = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
    for name in names:
        parts = name.split(".")
        obj = layer
        for p in parts[:-1]: obj = getattr(obj, p)
        w = obj.weight.data
        obj.weight = torch.nn.Parameter(quantize_tensor(w, bits))


def quantize_model_ablated(model, ffn_bits, attn_bits):
    for l in range(N_LAYERS):
        quantize_component(model.model.layers[l], "ffn", ffn_bits)
        quantize_component(model.model.layers[l], "attn", attn_bits)


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


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 9: Residual Mutual Information Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def compute_logit_lens_agreement(model_q, model_ref, tokenizer, texts, bits_name):
    """Per-layer token prediction agreement via logit lens.

    At each layer, apply lm_head to hidden state → compare top-k with reference.
    This measures whether the TRAJECTORY (not just vector cos) is preserved.
    """
    device = next(model_ref.parameters()).device
    lm_head = model_ref.lm_head.weight.data  # [vocab, hidden]

    agreements = defaultdict(list)

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out_ref = model_ref(**inputs, output_hidden_states=True)
            out_q = model_q(**inputs, output_hidden_states=True)

            for l in range(len(out_ref.hidden_states)):
                h_ref = out_ref.hidden_states[l][:, -1, :]  # last token
                h_q = out_q.hidden_states[l][:, -1, :]

                # Logit lens: hidden → lm_head → logits
                logits_ref = F.linear(h_ref.float(), lm_head.float())
                logits_q = F.linear(h_q.float(), lm_head.float())

                # Top-10 agreement
                top10_ref = set(logits_ref[0].topk(10).indices.tolist())
                top10_q = set(logits_q[0].topk(10).indices.tolist())
                top10_overlap = len(top10_ref & top10_q) / 10

                # Top-1 agreement (hardest test)
                top1_ref = logits_ref[0].argmax().item()
                top1_q = logits_q[0].argmax().item()
                top1_match = 1.0 if top1_ref == top1_q else 0.0

                agreements[l].append({
                    "top10_overlap": float(top10_overlap),
                    "top1_match": float(top1_match),
                })

    # Aggregate
    result = {}
    for l in sorted(agreements.keys()):
        vals = agreements[l]
        result[str(l)] = {
            "top10_overlap_mean": float(np.mean([v["top10_overlap"] for v in vals])),
            "top1_accuracy": float(np.mean([v["top1_match"] for v in vals])),
            "n_samples": len(vals),
        }

    return result


def compute_neighborhood_preservation(model_q, model_ref, tokenizer, texts):
    """Neighborhood topology: for N tokens, does relative distance structure survive?

    For a set of token hidden states, compute pairwise distance matrix.
    Measure correlation between reference and quantized distance matrices.
    """
    device = next(model_ref.parameters()).device
    layer_corrs = defaultdict(list)

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out_ref = model_ref(**inputs, output_hidden_states=True)
            out_q = model_q(**inputs, output_hidden_states=True)

            for l in range(len(out_ref.hidden_states)):
                # All token hidden states: [seq_len, hidden]
                h_ref = out_ref.hidden_states[l][0].float()
                h_q = out_q.hidden_states[l][0].float()
                seq_len = h_ref.shape[0]

                if seq_len < 5: continue

                # Pairwise cosine distance matrices
                h_ref_n = h_ref / (h_ref.norm(dim=1, keepdim=True) + 1e-12)
                h_q_n = h_q / (h_q.norm(dim=1, keepdim=True) + 1e-12)

                D_ref = h_ref_n @ h_ref_n.T  # [seq, seq]
                D_q = h_q_n @ h_q_n.T

                # Correlation between upper triangles
                triu_idx = torch.triu_indices(seq_len, seq_len, offset=1)
                d_ref_flat = D_ref[triu_idx[0], triu_idx[1]].cpu().numpy()
                d_q_flat = D_q[triu_idx[0], triu_idx[1]].cpu().numpy()

                corr = np.corrcoef(d_ref_flat, d_q_flat)[0, 1]
                if not np.isnan(corr):
                    layer_corrs[l].append(float(corr))

    result = {}
    for l in sorted(layer_corrs.keys()):
        vals = layer_corrs[l]
        result[str(l)] = {
            "neighborhood_corr_mean": float(np.mean(vals)),
            "neighborhood_corr_std": float(np.std(vals)),
        }

    return result


def run_exp9_residual_mi():
    """Residual mutual information sweep across precision levels."""
    print("=" * 70)
    print("  Exp 9: Residual Continuity — Logit Lens + Neighborhood Topology")
    print("=" * 70)
    print()
    print("  Measuring what cosine misses:")
    print("    1. Logit-lens token prediction agreement (per layer)")
    print("    2. Neighborhood topology preservation (pairwise distance corr)")
    print()

    eval_texts = [
        "The history of artificial intelligence dates back to the 1950s when researchers first began exploring the possibility of creating machines that could think and learn.",
        "Climate change is one of the most pressing challenges facing humanity today, with rising temperatures and extreme weather events.",
        "The Renaissance was a period of European history marking the transition from the Middle Ages to modernity.",
    ]

    # Reference
    print("Loading reference (all fp16)...")
    model_ref, tokenizer = load_model()

    # Precision levels
    bit_levels = [3.0, 3.25, 3.5, 4.0, 5.0, 16.0]

    all_logit_lens = {}
    all_neighborhood = {}

    for bits in bit_levels:
        name = f"q{bits:.2f}" if bits < 16 else "fp16"
        print(f"\n  {name} ({int(2**bits) if bits < 16 else 'full'} levels)...", end=" ", flush=True)
        t0 = time.time()

        m, _ = load_model()
        for l in range(N_LAYERS):
            quantize_component(m.model.layers[l], "ffn", bits)
            quantize_component(m.model.layers[l], "attn", bits)

        logit_lens = compute_logit_lens_agreement(m, model_ref, tokenizer, eval_texts, name)
        neighborhood = compute_neighborhood_preservation(m, model_ref, tokenizer, eval_texts)

        all_logit_lens[name] = logit_lens
        all_neighborhood[name] = neighborhood
        del m

        # Quick summary
        mean_top10 = np.mean([v["top10_overlap_mean"] for v in logit_lens.values()])
        mean_topo = np.mean([v["neighborhood_corr_mean"] for v in neighborhood.values()])
        print(f"logit_top10={mean_top10:.3f}  topo_corr={mean_topo:.3f}  ({time.time()-t0:.0f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("  Exp 9 Results: Residual Continuity")
    print("=" * 70)

    # Find key layers: UNFOLD (L2), mid-ISOMETRIC (L8), DIVERGENT (L18), output (L21)
    key_layers = [0, 2, 8, 13, 18, 21]

    for metric_name, data, is_pct in [
        ("Logit-Lens Top-1 Accuracy", all_logit_lens, True),
        ("Neighborhood Topology Correlation", all_neighborhood, False),
    ]:
        print(f"\n  ── {metric_name} ──")
        header = f"  {'Layer':<6}"
        for bits in bit_levels:
            name = f"q{bits:.2f}" if bits < 16 else "fp16"
            header += f"  {name:>8}"
        print(header)
        print(f"  {'-'*6}" + f"  {'-'*8}" * len(bit_levels))

        for l in range(N_LAYERS):
            ls = str(l)
            line = f"  L{l:<5}"
            for bits in bit_levels:
                name = f"q{bits:.2f}" if bits < 16 else "fp16"
                if ls in data[name]:
                    if "top1_accuracy" in data[name][ls]:
                        v = data[name][ls]["top1_accuracy"]
                    else:
                        v = data[name][ls]["neighborhood_corr_mean"]
                    if is_pct: line += f"  {v:>7.1%}"
                    else:      line += f"  {v:>8.3f}"
                else:
                    line += f"  {'—':>8}"
            if l in key_layers:
                print(line)
            elif l <= 3 or l >= 20:
                print(line)

    return all_logit_lens, all_neighborhood


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 10: Attention Bandwidth Hypothesis
# ═══════════════════════════════════════════════════════════════════════════════

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


def run_exp10_attention_bandwidth():
    """Sweep FFN vs Attn precision independently.

    Hypothesis: Attention precision determines residual transport capacity.
    Prediction: (FFN=q3, Attn=q5) >> (FFN=q5, Attn=q3)
    """
    print("\n" + "=" * 70)
    print("  Exp 10: Attention Bandwidth Hypothesis")
    print("=" * 70)
    print()
    print("  Hypothesis: Attention determines residual stream transport capacity.")
    print("  Prediction: Attn precision >> FFN precision for quality preservation.")
    print()

    eval_texts = [
        "The history of artificial intelligence dates back to the 1950s when researchers first began exploring.",
        "Climate change is one of the most pressing challenges facing humanity today.",
        "The Renaissance was a period of European history marking the transition from the Middle Ages.",
    ]

    # Reference
    print("Loading reference...")
    model_ref, tokenizer = load_model()
    ref_ppl = compute_ppl(model_ref, tokenizer, eval_texts)
    print(f"  Reference PPL: {ref_ppl:.2f}")

    # Sweep grid
    ffn_levels = [3, 4, 5, 16]
    attn_levels = [3, 4, 5, 16]

    results = {}
    total = len(ffn_levels) * len(attn_levels)
    n = 0

    for ffn_b in ffn_levels:
        for attn_b in attn_levels:
            n += 1
            name = f"FFN{ffn_b}_Attn{attn_b}"
            avg_bits = (ffn_b + attn_b) / 2 if ffn_b < 16 and attn_b < 16 else "mixed"
            print(f"\n  [{n}/{total}] {name} (FFN={ffn_b}bit, Attn={attn_b}bit)...",
                  end=" ", flush=True)
            t0 = time.time()

            m, _ = load_model()
            quantize_model_ablated(m, ffn_b, attn_b)

            ppl = compute_ppl(m, tokenizer, eval_texts)
            del m

            results[name] = {"ffn_bits": ffn_b, "attn_bits": attn_b,
                             "ppl": float(ppl)}
            s = "✓" if ppl < 20 else "☠" if ppl > 100 else "~"
            print(f"PPL={ppl:.2f} {s}  ({time.time()-t0:.0f}s)")

    # Summary matrix
    print("\n" + "=" * 70)
    print("  Exp 10 Results: Attention Bandwidth Matrix")
    print("=" * 70)
    print(f"\n  Reference PPL: {ref_ppl:.2f}")
    print(f"\n  {'FFN ↓ / Attn →':<16}", end="")
    for ab in attn_levels:
        print(f"  {'q'+str(ab) if ab<16 else 'fp16':>10}", end="")
    print(f"\n  {'-'*16}", end="")
    for _ in attn_levels: print(f"  {'-'*10}", end="")
    print()

    for ffn_b in ffn_levels:
        label = f"  {'q'+str(ffn_b) if ffn_b<16 else 'fp16':<16}"
        print(label, end="")
        for attn_b in attn_levels:
            name = f"FFN{ffn_b}_Attn{attn_b}"
            ppl = results[name]["ppl"]
            s = "✓" if ppl < 20 else "☠" if ppl > 100 else "~"
            print(f"  {ppl:>8.2f} {s}", end="")
        print()

    # Asymmetry test
    print("\n  Asymmetry Test (FFN vs Attn importance):")
    for ffn_b, attn_b in [(3, 5), (5, 3), (3, 4), (4, 3)]:
        name_a = f"FFN{ffn_b}_Attn{attn_b}"
        name_b = f"FFN{attn_b}_Attn{ffn_b}"
        if name_a in results and name_b in results:
            ppl_a = results[name_a]["ppl"]
            ppl_b = results[name_b]["ppl"]
            better = name_a if ppl_a < ppl_b else name_b
            print(f"    {name_a}={ppl_a:.1f} vs {name_b}={ppl_b:.1f} → {better} wins")

    # Key verdict
    f3a5 = results.get("FFN3_Attn5", {}).get("ppl", float("inf"))
    f5a3 = results.get("FFN5_Attn3", {}).get("ppl", float("inf"))
    print(f"\n  Verdict:")
    if f3a5 < f5a3:
        print(f"  ✓ Attention bandwidth hypothesis SUPPORTED")
        print(f"    FFN3+Attn5 (PPL={f3a5:.1f}) > FFN5+Attn3 (PPL={f5a3:.1f})")
        print(f"    Attn precision dominates FFN precision.")
    else:
        print(f"  ⚠ Attention bandwidth hypothesis NOT supported")
        print(f"    FFN3+Attn5 (PPL={f3a5:.1f}) vs FFN5+Attn3 (PPL={f5a3:.1f})")

    return results


def run():
    logit_lens, neighborhood = run_exp9_residual_mi()
    attn_bandwidth = run_exp10_attention_bandwidth()

    # Save
    out = {
        "experiment": "phase_c_residual_continuity",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "exp9_logit_lens": logit_lens,
        "exp9_neighborhood_topology": neighborhood,
        "exp10_attention_bandwidth": attn_bandwidth,
    }
    path = RESULTS_DIR / "phase_c_residual.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    run()
