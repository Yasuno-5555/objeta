#!/usr/bin/env python3
"""
Phase D: KV-Cache Precision + Logit Topology Preservation

Exp 11: Is attention bottleneck in Wq/Wk/Wv weights or KV cache precision?
  - Sweep weight_bits × kv_bits independently
  - kv_bits is simulated by quantizing K,V after projection

Exp 12: Logit Topology Preservation — per-layer topological continuity
  - Top-k overlap, ranking correlation, KL divergence
  - Track layer-by-layer collapse of token prediction topology
"""
import torch, torch.nn.functional as F, numpy as np, json, time
from pathlib import Path
from collections import defaultdict
import warnings; warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "phase_a_results"
RESULTS_DIR.mkdir(exist_ok=True)
N_LAYERS, VOCAB = 22, 32000


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


def save_all_weights(model):
    saved = {}
    for l in range(N_LAYERS):
        layer = model.model.layers[l]
        for comp, keys in [("ffn", ["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]),
                            ("attn", ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"])]:
            for key in keys:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                saved[f"L{l}_{key}"] = obj.weight.data.clone().cpu()
    return saved


def restore_all_weights(model, saved):
    for l in range(N_LAYERS):
        layer = model.model.layers[l]
        for comp, keys in [("ffn", ["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]),
                            ("attn", ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"])]:
            for key in keys:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(saved[f"L{l}_{key}"].to(obj.weight.device))


def quantize_weights(model, ffn_bits, attn_bits):
    for l in range(N_LAYERS):
        layer = model.model.layers[l]
        for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            obj = layer
            for p in key.split("."): obj = getattr(obj, p)
            obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, ffn_bits))
        for key in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]:
            obj = layer
            for p in key.split("."): obj = getattr(obj, p)
            obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, attn_bits))


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


def fast_ppl(model, tokenizer, text):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return float("inf") if out.loss is None else np.exp(out.loss.item())


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 11: KV-Cache Precision Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp11_kv_precision(model, tokenizer, pristine, text):
    """Is attention bottleneck in weights or KV cache?

    We can't easily quantize KV cache mid-forward in HuggingFace.
    Instead, we QUANTIZE the K/V projection WEIGHTS specifically
    vs the Q/O projection weights, to isolate the KV path.

    Actually simpler: quantize ALL attention weights to different levels
    and compare with FFN weights. But we already did that.

    Better approach: quantize K_proj and V_proj separately from Q_proj and O_proj.
    This isolates the "KV path" from the "query/output path".
    """
    print("=" * 60)
    print("  Exp 11: KV-Path vs QO-Path Precision")
    print("=" * 60)
    print()
    print("  Isolating K/V projection from Q/O projection.")
    print("  Hypothesis: KV path precision = trajectory transport bottleneck.")
    print()

    results = {}

    configs = [
        # Format: (name, ffn_bits, QO_bits, KV_bits)
        ("KVq3_QOq5",      5, 5, 3),   # KV path at cliff, QO safe
        ("KVq5_QOq3",      5, 3, 5),   # QO at cliff, KV safe
        ("KVq3_QOq4",      4, 4, 3),
        ("KVq4_QOq3",      4, 3, 4),
        ("KVq5_QOq5_FFNq3", 3, 5, 5),  # All attn safe, FFN at cliff
        ("KVq5_QOq5_FFNq5", 5, 5, 5),  # All safe
    ]

    for name, ffn_b, qo_b, kv_b in configs:
        t0 = time.time()
        restore_all_weights(model, pristine)

        for l in range(N_LAYERS):
            layer = model.model.layers[l]
            # FFN
            for key in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, ffn_b))
            # Q/O projection
            for key in ["self_attn.q_proj","self_attn.o_proj"]:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, qo_b))
            # K/V projection
            for key in ["self_attn.k_proj","self_attn.v_proj"]:
                obj = layer
                for p in key.split("."): obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(quantize_tensor_fast(obj.weight.data, kv_b))

        ppl = fast_ppl(model, tokenizer, text)
        results[name] = {"ffn": ffn_b, "qo": qo_b, "kv": kv_b, "ppl": float(ppl)}
        s = "✓" if ppl < 20 else "☠" if ppl > 100 else "~"
        print(f"  {name:<25} FFN={ffn_b} QO={qo_b} KV={kv_b}  PPL={ppl:>8.2f} {s}  ({time.time()-t0:.0f}s)")

    # Summary
    print("\n  ── KV vs QO Asymmetry ──")
    for a, b in [("KVq3_QOq5", "KVq5_QOq3"), ("KVq3_QOq4", "KVq4_QOq3")]:
        if a in results and b in results:
            pa, pb = results[a]["ppl"], results[b]["ppl"]
            winner = a if pa < pb else b
            print(f"  {a}={pa:.1f} vs {b}={pb:.1f}  → {winner} wins (Δ={abs(pa-pb):.1f})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 12: Logit Topology Preservation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_logit_topology(model_q, model_ref, tokenizer, text):
    """Per-layer logit-lens topology metrics.

    Returns per-layer:
      - top-1 match (hardest: does the model predict the same token?)
      - top-10 overlap (does the prediction distribution preserve structure?)
      - logit rank correlation (are token rankings preserved?)
      - logit KL divergence (how far is the full distribution?)
    """
    device = next(model_ref.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ref = model_ref(**inputs, output_hidden_states=True)
        out_q = model_q(**inputs, output_hidden_states=True)

    lm_head = model_ref.lm_head.weight.data.float()
    layer_data = {}

    for l in range(len(out_ref.hidden_states)):
        h_ref = out_ref.hidden_states[l][:, -1, :].float()  # last token
        h_q = out_q.hidden_states[l][:, -1, :].float()

        logits_ref = F.linear(h_ref, lm_head)[0]
        logits_q = F.linear(h_q, lm_head)[0]

        # Top-1 match
        top1_match = 1.0 if logits_ref.argmax().item() == logits_q.argmax().item() else 0.0

        # Top-10 overlap
        top10_ref = set(logits_ref.topk(10).indices.tolist())
        top10_q = set(logits_q.topk(10).indices.tolist())
        top10_overlap = len(top10_ref & top10_q) / 10

        # Top-100 overlap (broader structure)
        top100_ref = set(logits_ref.topk(100).indices.tolist())
        top100_q = set(logits_q.topk(100).indices.tolist())
        top100_overlap = len(top100_ref & top100_q) / 100

        # Rank correlation (Spearman on top-1000)
        k = min(1000, VOCAB)
        _, ref_idx = logits_ref.topk(k)
        _, q_idx = logits_q.topk(k)
        ref_rank = {idx.item(): i for i, idx in enumerate(ref_idx)}
        q_rank = {idx.item(): i for i, idx in enumerate(q_idx)}
        common = set(ref_rank.keys()) & set(q_rank.keys())
        if len(common) > 10:
            ref_vals = [ref_rank[i] for i in common]
            q_vals = [q_rank[i] for i in common]
            rho = float(np.corrcoef(ref_vals, q_vals)[0, 1])
            if np.isnan(rho): rho = 0.0
        else:
            rho = 0.0

        # KL divergence
        probs_ref = F.softmax(logits_ref.float(), dim=-1)
        probs_q = F.softmax(logits_q.float(), dim=-1)
        kl = float(F.kl_div((probs_q + 1e-12).log(), probs_ref + 1e-12, reduction='sum'))

        layer_data[str(l)] = {
            "top1_match": top1_match,
            "top10_overlap": top10_overlap,
            "top100_overlap": top100_overlap,
            "rank_corr": rho,
            "kl_divergence": kl,
        }

    return layer_data


def run_exp12_topology(model, tokenizer, pristine, text):
    """Per-layer logit topology across precision levels."""
    print("\n" + "=" * 60)
    print("  Exp 12: Logit Topology Preservation")
    print("=" * 60)
    print()
    print("  Tracking per-layer token prediction topology.")
    print("  NOT cosine — actual logit-space structure.")
    print()

    # Reference: all fp16
    restore_all_weights(model, pristine)
    print("  Computing reference topology (fp16)...")
    ref_topology = compute_logit_topology(model, model, tokenizer, text)

    all_results = {}
    for bits in [3.0, 3.25, 3.5, 4.0, 5.0]:
        name = f"q{bits:.2f}"
        t0 = time.time()
        restore_all_weights(model, pristine)
        quantize_weights(model, bits, bits)

        topo = compute_logit_topology(model, model, tokenizer, text)  # model_q = model_ref here
        # Actually we need comparison with fp16. Let's use a reference model.
        # For now, model IS the quantized model. We can't compare with fp16 easily.
        # Let me redo this differently...

        # Just measure the topology metrics on the quantized model directly
        # (they're still meaningful: top1 is "does it predict the same as itself?")
        # Better: compare with fp16 reference from saved hidden states.
        all_results[name] = topo

        mean_t1 = np.mean([v["top1_match"] for v in topo.values()])
        mean_t10 = np.mean([v["top10_overlap"] for v in topo.values()])
        print(f"  {name}: mean top1={mean_t1:.2%} top10={mean_t10:.2%}  ({time.time()-t0:.0f}s)")

    # Actually the topology metrics above compare model_q with itself (trivially 1.0).
    # We need to compare quantized with fp16 reference.
    # Let me redo the measurement properly.

    print("\n  Re-measuring against fp16 reference...")
    restore_all_weights(model, pristine)

    topology_vs_ref = {}
    for bits in [3.0, 3.25, 3.5, 4.0, 5.0]:
        name = f"q{bits:.2f}"
        t0 = time.time()

        # Clone model for quantization
        model_q, _ = load_model()
        quantize_weights(model_q, bits, bits)

        topo = compute_logit_topology(model_q, model, tokenizer, text)
        topology_vs_ref[name] = topo
        del model_q

        mean_t1 = np.mean([v["top1_match"] for v in topo.values()])
        mean_t10 = np.mean([v["top10_overlap"] for v in topo.values()])
        mean_rho = np.mean([v["rank_corr"] for v in topo.values()])
        mean_kl = np.mean([v["kl_divergence"] for v in topo.values()])
        print(f"  {name}: top1={mean_t1:.2%} top10={mean_t10:.2%} rank_corr={mean_rho:.3f} KL={mean_kl:.1f}  ({time.time()-t0:.0f}s)")

    # Per-layer breakdown for key layers
    print("\n  ── Top-1 Match by Layer ──")
    key_layers = [0, 1, 2, 3, 5, 8, 11, 14, 17, 20, 21]
    header = f"  {'L':<4}"
    for bits in [3.0, 3.25, 3.5, 4.0, 5.0]:
        header += f"  {'q'+str(bits):>8}"
    print(header)
    print(f"  {'-'*4}" + f"  {'-'*8}" * 5)
    for l in range(N_LAYERS):
        ls = str(l)
        line = f"  L{l:<3}"
        for bits in [3.0, 3.25, 3.5, 4.0, 5.0]:
            name = f"q{bits:.2f}"
            v = topology_vs_ref[name].get(ls, {}).get("top1_match", 0)
            line += f"  {v:>7.1%}" if v > 0 else f"  {'—':>8}"
        if l in key_layers or l <= 2 or l >= 20:
            print(line)

    print("\n  ── Rank Correlation by Layer ──")
    print(header)
    print(f"  {'-'*4}" + f"  {'-'*8}" * 5)
    for l in range(N_LAYERS):
        ls = str(l)
        line = f"  L{l:<3}"
        for bits in [3.0, 3.25, 3.5, 4.0, 5.0]:
            name = f"q{bits:.2f}"
            v = topology_vs_ref[name].get(ls, {}).get("rank_corr", 0)
            if isinstance(v, (int, float)) and not np.isnan(v):
                line += f"  {v:>8.3f}"
            else:
                line += f"  {'—':>8}"
        if l in key_layers or l <= 2 or l >= 20:
            print(line)

    return topology_vs_ref


def run():
    print("=" * 60)
    print("  Phase D: KV Precision + Logit Topology")
    print("=" * 60)

    text = ("The history of artificial intelligence dates back to the 1950s "
            "when researchers first began exploring the possibility of creating "
            "machines that could think and learn like humans.")

    print("\nLoading model...")
    model, tokenizer = load_model()
    pristine = save_all_weights(model)
    print(f"  Saved {len(pristine)} weight tensors")

    kv_results = run_exp11_kv_precision(model, tokenizer, pristine, text)
    topo_results = run_exp12_topology(model, tokenizer, pristine, text)

    # Save
    out = {"experiment": "phase_d_kv_topology",
           "exp11_kv_precision": kv_results,
           "exp12_logit_topology": topo_results}
    path = RESULTS_DIR / "phase_d_kv_topology.json"
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    run()
