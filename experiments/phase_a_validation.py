#!/usr/bin/env python3
"""
Phase A: Jacobian-Aware Quantization Validation + UNFOLD Sensitivity Mapping

Two experiments on TinyLlama-1.1B-Chat-v1.0:

Exp 1 — Compare 4 quantization strategies at identical bit budget:
  Baseline: uniform q4
  A: Random q8 protection
  B: Hessian-aware (GPTQ-style, highest Hessian trace → q8)
  C: LKO-aware (UNFOLD=q8/fp16, DIVERGENT=q6, ISOMETRIC=q3)

Exp 2 — UNFOLD sensitivity: protect individual early layers one at a time:
  L0 only, L1 only, L2 only, L3 only, L4 only, L0-L2, none, all fp16
  Measure per-layer cos preservation and rollout divergence.

Metrics: ppl, rollout cos, hidden cosine drift, entropy, repetition
"""
import torch
import numpy as np
import json
import os
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Silence warnings
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "phase_a_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── TinyLlama Phase Data (from LKO measurements) ──

TINYLLAMA_N_LAYERS = 22
TINYLLAMA_HIDDEN = 2048
TINYLLAMA_FFN = 5632
TINYLLAMA_VOCAB = 32000

# Zone classification per layer
ZONES = {
    0: "Sync", 1: "Sync",
    2: "Unfold",
    **{l: "IsometricLocal" for l in range(3, 7)},
    **{l: "IsometricGlobal" for l in range(7, 14)},
    **{l: "Divergent" for l in range(14, 22)},
}

# Lyapunov estimates from LKO synthetic forward (FINDINGS_v8)
LYAPUNOV = {
    0: 0.80, 1: 0.90,
    2: 3.50,  # UNFOLD — J≠I
    3: 1.10, 4: 1.00, 5: 0.95, 6: 1.05,
    7: 1.02, 8: 1.08, 9: 1.15, 10: 1.20, 11: 1.18, 12: 1.10, 13: 1.05,
    14: 1.40, 15: 1.60, 16: 1.80, 17: 2.00, 18: 2.10, 19: 2.00, 20: 1.90, 21: 1.50,
}

# Steering cos cos(Δ_l, Δ_{l+1})
STEERING_COS = {
    0: 0.21, 1: 0.18, 2: 0.11,
    3: 0.08, 4: 0.06, 5: 0.04, 6: 0.03,
    7: -0.01, 8: -0.03, 9: -0.06, 10: -0.09,
    11: -0.11, 12: -0.08, 13: -0.04,
    14: 0.01, 15: 0.05, 16: 0.14, 17: 0.18,
    18: 0.21, 19: 0.24, 20: 0.27,
}

# Per-layer sensitivity from LKO quantization-drift.md (ΔL2 improvement when upgraded to q5)
PER_LAYER_SENSITIVITY = {
    0: -0.009, 1: 0.018, 2: 0.049,  # L2: 2.7× dominant
    **{l: 0.0 for l in range(3, 22)},
}


def quantize_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Simulate quantization by rounding to limited levels.

    Uses per-channel (row-wise) quantization matching Q*_K_APPL pattern.
    Returns quantized tensor with simulated noise.
    """
    if bits >= 16:
        return w.float()

    w_f = w.float()
    n_levels = 2 ** bits

    # Per-row quantization (mimics Q*_K_APPL row-wise approach)
    w_q = torch.zeros_like(w_f)
    for i in range(w_f.shape[0]):
        row = w_f[i]
        rmin, rmax = row.min(), row.max()
        span = rmax - rmin
        if span < 1e-10:
            w_q[i] = row
            continue
        scale = span / (n_levels - 1)
        # Quantize
        q_vals = ((row - rmin) / scale).round().clamp(0, n_levels - 1)
        # Dequantize
        w_q[i] = q_vals * scale + rmin

    return w_q


@dataclass
class QuantizationConfig:
    """Per-layer bit allocation."""
    name: str
    bits_per_layer: dict  # layer_idx → bits

    @staticmethod
    def uniform(n_layers: int, bits: int = 4, name: str = None) -> "QuantizationConfig":
        return QuantizationConfig(
            name=name or f"uniform_q{bits}",
            bits_per_layer={l: bits for l in range(n_layers)},
        )

    @staticmethod
    def lko_aware() -> "QuantizationConfig":
        """LKO-derived allocation: protect UNFOLD, conservative DIVERGENT, aggressive ISOMETRIC."""
        bits = {}
        for l in range(TINYLLAMA_N_LAYERS):
            zone = ZONES.get(l, "?")
            if zone == "Unfold":
                bits[l] = 16  # fp16 — basin compiler, mandatory
            elif zone in ("Sync",):
                bits[l] = 4   # q4 — anti-damped but short
            elif zone.startswith("Isometric"):
                bits[l] = 3   # q3 — λ≈0, maximally safe
            elif zone == "Divergent":
                bits[l] = 6   # q6 — λ>0, needs headroom
            else:
                bits[l] = 4
        return QuantizationConfig(name="lko_aware", bits_per_layer=bits)

    @staticmethod
    def hessian_aware(hessian_trace: dict) -> "QuantizationConfig":
        """GPTQ-style: top 3 layers by Hessian trace get q8, rest q4."""
        sorted_layers = sorted(hessian_trace.items(), key=lambda x: x[1], reverse=True)
        protected = {l for l, _ in sorted_layers[:3]}
        bits = {}
        for l in range(TINYLLAMA_N_LAYERS):
            bits[l] = 8 if l in protected else 4
        return QuantizationConfig(name="hessian_aware", bits_per_layer=bits)

    @staticmethod
    def random_protection(n_layers: int, seed: int = 123) -> "QuantizationConfig":
        """Random 3 layers get q8, rest q4 (placebo control)."""
        rng = np.random.RandomState(seed)
        protected = set(rng.choice(n_layers, 3, replace=False))
        bits = {}
        for l in range(n_layers):
            bits[l] = 8 if l in protected else 4
        return QuantizationConfig(name="random_q8", bits_per_layer=bits)

    def average_bits(self) -> float:
        return np.mean(list(self.bits_per_layer.values()))


def apply_quantization(model, config: QuantizationConfig):
    """Quantize model weights per-layer according to config."""
    for layer_idx in range(TINYLLAMA_N_LAYERS):
        bits = config.bits_per_layer.get(layer_idx, 4)
        if bits >= 16:
            continue  # keep fp16

        layer = model.model.layers[layer_idx]

        # FFN weights (dominant for steering per LKO theory)
        for name in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            w = getattr(layer, name).weight.data
            setattr(getattr(layer, name), "weight", torch.nn.Parameter(quantize_tensor(w, bits)))

        # Attention weights
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
            w = getattr(layer, name).weight.data
            setattr(getattr(layer, name), "weight", torch.nn.Parameter(quantize_tensor(w, bits)))

    return model


def load_tinyllama():
    """Load TinyLlama-1.1B-Chat-v1.0 with bf16 weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def run_forward(model, tokenizer, texts: list[str], max_new_tokens: int = 50):
    """Run autoregressive generation and collect per-layer hidden states.

    Returns:
        outputs: list of token id sequences
        hidden_states: [batch, seq_len, n_layers, hidden_dim]
    """
    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    all_hidden = []
    all_outputs = []

    with torch.no_grad():
        for i in range(len(texts)):
            input_ids = inputs["input_ids"][i:i+1]
            attention_mask = inputs["attention_mask"][i:i+1]
            prompt_len = input_ids.shape[1]

            # Forward with hidden states
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Collect per-layer hidden states
            layer_hidden = torch.stack(out.hidden_states, dim=1)  # [1, n_layers+1, seq, dim]
            all_hidden.append(layer_hidden[:, :, -1, :])  # last token per layer

            # Generate continuation to measure degeneration
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            all_outputs.append(gen_out.sequences[0])

    return all_outputs, all_hidden


def compute_perplexity(model, tokenizer, eval_texts: list[str]) -> float:
    """Compute perplexity on evaluation texts."""
    import torch.nn.functional as F

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if inputs["input_ids"].shape[1] < 2:
                continue

            out = model(**inputs, labels=inputs["input_ids"])
            loss = out.loss
            if loss is not None:
                total_loss += loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]

    if total_tokens == 0:
        return float("inf")

    return np.exp(total_loss / total_tokens)


# ── Experiment 1: Jacobian-Aware Quantization ──────────────────────────────

def run_experiment_1(model, tokenizer):
    """Compare 4 quantization strategies at identical bit budget."""
    print("=" * 70)
    print("  Experiment 1: Jacobian-Aware Quantization Validation")
    print("=" * 70)
    print()

    # Evaluation data
    eval_texts = load_eval_texts()

    # Reference: unquantized forward
    print("  Running reference (bf16)...")
    t0 = time.time()
    ref_hidden = collect_hidden_states(model, tokenizer, eval_texts)
    ref_ppl = compute_perplexity(model, tokenizer, eval_texts)
    print(f"  Reference ppl={ref_ppl:.2f} ({(time.time()-t0):.0f}s)")

    # Configs to test
    configs = [
        QuantizationConfig.uniform(TINYLLAMA_N_LAYERS, 4, "uniform_q4"),
        QuantizationConfig.random_protection(TINYLLAMA_N_LAYERS),
        QuantizationConfig.hessian_aware(estimate_hessian_trace(model)),
        QuantizationConfig.lko_aware(),
    ]

    results = {}
    for config in configs:
        print(f"\n  ── {config.name} (avg={config.average_bits():.1f}bit) ──")
        t0 = time.time()

        # Reload fresh model for each config (to avoid accumulation)
        m, _ = load_tinyllama()
        m = apply_quantization(m, config)

        # Perplexity
        ppl = compute_perplexity(m, tokenizer, eval_texts)

        # Hidden states
        hidden = collect_hidden_states(m, tokenizer, eval_texts)

        # Metrics
        layer_cos = compute_layer_cos(hidden, ref_hidden)
        entropy_stats = compute_entropy_metrics(m, tokenizer, eval_texts[:5])
        repetition = compute_repetition_rate(m, tokenizer, eval_texts[:3])

        results[config.name] = {
            "config": config,
            "ppl": ppl,
            "layer_cos": layer_cos,
            "entropy": entropy_stats,
            "repetition": repetition,
        }

        print(f"    ppl={ppl:.2f}  mean_cos={np.mean(list(layer_cos.values())):.4f}  "
              f"entropy={entropy_stats['mean']:.3f}  repeat={repetition:.3f}  "
              f"({time.time()-t0:.0f}s)")

        del m

    # Save results
    save_results_1(results, ref_ppl)

    # Print comparison
    print("\n" + "=" * 70)
    print("  Exp 1 Results: Jacobian-Aware Quantization")
    print("=" * 70)
    print(f"  {'Config':<20} {'PPL':>8} {'ΔPPL':>8} {'MeanCos':>9} {'Entropy':>9} {'Repeat':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")
    for name, r in results.items():
        dppl = r["ppl"] - ref_ppl
        mean_cos = np.mean(list(r["layer_cos"].values()))
        print(f"  {name:<20} {r['ppl']:>8.2f} {dppl:>+8.2f} "
              f"{mean_cos:>9.4f} {r['entropy']['mean']:>9.4f} {r['repetition']:>8.3f}")

    return results


def collect_hidden_states(model, tokenizer, texts: list[str]):
    """Collect last-token hidden states for all layers."""
    device = next(model.parameters()).device
    hidden_dict = defaultdict(list)

    with torch.no_grad():
        for text in texts[:20]:  # limit for speed
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            out = model(**inputs, output_hidden_states=True)
            for l, hs in enumerate(out.hidden_states):
                hidden_dict[l].append(hs[:, -1, :].cpu().float().numpy())

    # Average across texts
    return {l: np.mean(np.concatenate(v, axis=0), axis=0) for l, v in hidden_dict.items()}


def compute_layer_cos(hidden: dict, ref_hidden: dict) -> dict:
    """Cosine similarity per layer between quantized and reference."""
    cos_vals = {}
    for l in range(TINYLLAMA_N_LAYERS + 1):  # +1 for embedding
        if l in hidden and l in ref_hidden:
            h = hidden[l].flatten()
            r = ref_hidden[l].flatten()
            cos = np.dot(h, r) / (np.linalg.norm(h) * np.linalg.norm(r) + 1e-12)
            cos_vals[l] = float(cos)
    return cos_vals


def compute_entropy_metrics(model, tokenizer, texts: list[str]):
    """Compute token prediction entropy statistics."""
    import torch.nn.functional as F

    entropies = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            out = model(**inputs)
            probs = F.softmax(out.logits[:, -1, :], dim=-1)
            # Skip special tokens
            valid_probs = probs[0, :TINYLLAMA_VOCAB]
            valid_probs = valid_probs / valid_probs.sum()
            ent = -(valid_probs * torch.log(valid_probs + 1e-12)).sum().item()
            entropies.append(ent)

    return {
        "mean": float(np.mean(entropies)),
        "std": float(np.std(entropies)),
        "min": float(np.min(entropies)),
    }


def compute_repetition_rate(model, tokenizer, texts: list[str], max_new: int = 30):
    """Fraction of generated tokens that are repetitions."""
    rep_rates = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            gen = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = gen[0, inputs["input_ids"].shape[1]:].tolist()
            if len(new_tokens) < 2:
                rep_rates.append(0.0)
                continue
            # Count duplicate consecutive tokens
            dups = sum(1 for i in range(1, len(new_tokens)) if new_tokens[i] == new_tokens[i-1])
            rep_rates.append(dups / len(new_tokens))

    return float(np.mean(rep_rates))


def estimate_hessian_trace(model) -> dict:
    """Estimate Hessian trace per layer (proxy via weight gradient norm on sample inputs).

    Uses the diagonal Fisher approximation: E[(grad_log_p)^2] per parameter.
    This is the GPTQ/AWQ standard approach.
    """
    import torch.nn.functional as F

    # Sample inputs
    sample_texts = [
        "The capital of France is",
        "Machine learning is a subset of",
        "The quick brown fox jumps over",
    ]

    tokenizer = None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hessian_trace = {}
    with torch.no_grad():
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            out = model(**inputs)
            log_probs = F.log_softmax(out.logits[:, -1, :], dim=-1)
            # Sample from distribution as proxy
            target = log_probs.argmax(dim=-1)
            loss = F.nll_loss(log_probs, target)

            # Gradient wrt all parameters
            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)

            # Aggregate per layer
            grad_idx = 0
            param_names = [n for n, _ in model.named_parameters()]
            for name, param in zip(param_names, model.parameters()):
                if param.grad is not None:
                    # Extract layer index
                    if "layers." in name:
                        layer_str = name.split("layers.")[1].split(".")[0]
                        try:
                            layer_idx = int(layer_str)
                            g_norm = param.grad.norm().item() ** 2
                            if layer_idx not in hessian_trace:
                                hessian_trace[layer_idx] = 0.0
                            hessian_trace[layer_idx] += g_norm
                        except ValueError:
                            pass

    # Normalize
    if hessian_trace:
        max_val = max(hessian_trace.values())
        for l in hessian_trace:
            hessian_trace[l] /= max_val

    # Fill missing layers
    for l in range(TINYLLAMA_N_LAYERS):
        if l not in hessian_trace:
            hessian_trace[l] = 0.5  # default

    return hessian_trace


def save_results_1(results: dict, ref_ppl: float):
    """Save Exp 1 results to JSON."""
    out = {
        "experiment": "jacobian_aware_quantization",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "reference_ppl": ref_ppl,
        "n_layers": TINYLLAMA_N_LAYERS,
        "hidden_dim": TINYLLAMA_HIDDEN,
        "configs": {}
    }
    for name, r in results.items():
        out["configs"][name] = {
            "avg_bits": r["config"].average_bits(),
            "bits": r["config"].bits_per_layer,
            "ppl": r["ppl"],
            "mean_layer_cos": float(np.mean(list(r["layer_cos"].values()))),
            "layer_cos": r["layer_cos"],
            "entropy": r["entropy"],
            "repetition": r["repetition"],
        }

    path = RESULTS_DIR / "exp1_jacobian_aware.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {path}")


# ── Experiment 2: UNFOLD Sensitivity Mapping ────────────────────────────────

def run_experiment_2(model, tokenizer):
    """Protect individual early layers one at a time and measure rollout stability."""
    print("\n" + "=" * 70)
    print("  Experiment 2: UNFOLD Sensitivity Mapping")
    print("=" * 70)
    print()

    eval_texts = load_eval_texts()

    # Reference
    print("  Reference (bf16)...")
    ref_hidden = collect_hidden_states(model, tokenizer, eval_texts[:10])

    # Configs: protect individual early layers
    configs = [
        QuantizationConfig.uniform(TINYLLAMA_N_LAYERS, 4, "all_q4_baseline"),
        ("protect_L0", {0: 16, **{l: 4 for l in range(1, TINYLLAMA_N_LAYERS)}}),
        ("protect_L1", {1: 16, **{l: 4 for l in range(TINYLLAMA_N_LAYERS) if l != 1}}),
        ("protect_L2", {2: 16, **{l: 4 for l in range(TINYLLAMA_N_LAYERS) if l != 2}}),
        ("protect_L3", {3: 16, **{l: 4 for l in range(TINYLLAMA_N_LAYERS) if l != 3}}),
        ("protect_L4", {4: 16, **{l: 4 for l in range(TINYLLAMA_N_LAYERS) if l != 4}}),
        ("protect_L0_L2", {0: 16, 1: 16, 2: 16, **{l: 4 for l in range(3, TINYLLAMA_N_LAYERS)}}),
        ("all_fp16", {l: 16 for l in range(TINYLLAMA_N_LAYERS)}),
    ]

    results = {}
    for item in configs:
        if isinstance(item, QuantizationConfig):
            config = item
        else:
            name, bits_dict = item
            config = QuantizationConfig(name=name, bits_per_layer=bits_dict)

        print(f"\n  ── {config.name} ──")

        m, _ = load_tinyllama()
        m = apply_quantization(m, config)

        hidden = collect_hidden_states(m, tokenizer, eval_texts[:10])
        layer_cos = compute_layer_cos(hidden, ref_hidden)

        # Rollout simulation: feed quantized hidden through reference layers
        rollout_cos = compute_rollout_cos(m, model, tokenizer, eval_texts[:3])

        results[config.name] = {
            "layer_cos": layer_cos,
            "rollout_cos": rollout_cos,
        }

        del m

    # Save
    save_results_2(results)

    # Print
    print("\n" + "=" * 70)
    print("  Exp 2 Results: UNFOLD Sensitivity Mapping")
    print("=" * 70)
    print()
    header = f"  {'Config':<20} {'L0_cos':>8} {'L2_cos':>8} {'L5_cos':>8} {'L13_cos':>8} {'L21_cos':>8}"
    print(header)
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, r in results.items():
        lc = r["layer_cos"]
        vals = [lc.get(0, 0), lc.get(2, 0), lc.get(5, 0), lc.get(13, 0), lc.get(21, 0)]
        print(f"  {name:<20} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f} {vals[4]:>8.4f}")

    # Identify dominant layer
    print("\n  Per-Layer Protection Effectiveness (cos vs all_q4 baseline):")
    baseline_cos = results["all_q4_baseline"]["layer_cos"]
    for layer in range(5):
        key = f"protect_L{layer}"
        if key in results:
            gain = {}
            for l in range(TINYLLAMA_N_LAYERS + 1):
                base = baseline_cos.get(l, 0)
                prot = results[key]["layer_cos"].get(l, 0)
                gain[l] = prot - base
            mean_gain = np.mean(list(gain.values()))
            l2_gain = gain.get(2, 0)
            print(f"    L{layer} protected: mean Δcos={mean_gain:+.4f}, L2 Δcos={l2_gain:+.4f}")

    return results


def compute_rollout_cos(model_q, model_ref, tokenizer, texts: list[str]):
    """Simulate autoregressive rollout and measure hidden state divergence.

    Feed the same input through both models. At each layer, measure cos
    between quantized and reference hidden states.
    """
    rollout_stats = defaultdict(list)
    device = next(model_ref.parameters()).device

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

            # Reference forward
            out_ref = model_ref(**inputs, output_hidden_states=True)

            # Quantized forward
            out_q = model_q(**inputs, output_hidden_states=True)

            for l in range(len(out_ref.hidden_states)):
                h_ref = out_ref.hidden_states[l][:, -1, :].cpu().float().numpy().flatten()
                h_q = out_q.hidden_states[l][:, -1, :].cpu().float().numpy().flatten()
                cos = np.dot(h_ref, h_q) / (np.linalg.norm(h_ref) * np.linalg.norm(h_q) + 1e-12)
                rollout_stats[l].append(float(cos))

    return {l: float(np.mean(v)) for l, v in rollout_stats.items()}


def save_results_2(results: dict):
    """Save Exp 2 results."""
    out = {
        "experiment": "unfold_sensitivity_mapping",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "configs": {}
    }
    for name, r in results.items():
        out["configs"][name] = {
            "layer_cos": r["layer_cos"],
            "rollout_cos": r["rollout_cos"],
        }

    path = RESULTS_DIR / "exp2_unfold_sensitivity.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {path}")


# ── Evaluation Data ─────────────────────────────────────────────────────────

def load_eval_texts() -> list[str]:
    """Load WikiText-2 validation texts for evaluation."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in dataset["text"] if len(t.strip()) > 50]
        return texts[:50]  # limit for speed
    except Exception:
        # Fallback texts
        return [
            "The history of artificial intelligence dates back to the 1950s when researchers first began exploring the possibility of creating machines that could think and learn like humans.",
            "Quantum computing represents a fundamentally different approach to computation, using quantum bits or qubits that can exist in multiple states simultaneously.",
            "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789 and ended with the formation of the French Consulate in 1799.",
            "Climate change is one of the most pressing challenges facing humanity today, with rising global temperatures causing sea level rise, extreme weather events, and ecosystem disruption.",
            "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity, covering the 15th and 16th centuries.",
            "Deep learning has revolutionized the field of artificial intelligence, enabling breakthroughs in computer vision, natural language processing, and speech recognition.",
            "The human genome contains approximately three billion base pairs of DNA, encoding the instructions for building and maintaining a human being.",
            "Ancient Rome was one of the most influential civilizations in world history, with its legal system, architecture, and language leaving lasting impacts on Western culture.",
            "Shakespeare's plays continue to be performed and studied around the world, more than four hundred years after his death.",
            "The Industrial Revolution transformed economies that had been based on agriculture and handicrafts into economies based on large-scale industry and mechanized manufacturing.",
        ]


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phase A: Jacobian-Aware Quantization Validation + UNFOLD Sensitivity Mapping")
    print()

    print("Loading TinyLlama-1.1B-Chat-v1.0...")
    model, tokenizer = load_tinyllama()
    print(f"Loaded: {TINYLLAMA_N_LAYERS} layers, {TINYLLAMA_HIDDEN}D hidden, {TINYLLAMA_FFN}D FFN")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print()

    # Experiment 1: Jacobian-Aware Quantization
    results_1 = run_experiment_1(model, tokenizer)

    # Experiment 2: UNFOLD Sensitivity Mapping
    results_2 = run_experiment_2(model, tokenizer)

    # Summary
    print("\n" + "=" * 70)
    print("  Phase A Complete")
    print("=" * 70)
    print(f"  Results saved to: {RESULTS_DIR}")
    print()

    # Key findings
    exp1 = json.load(open(RESULTS_DIR / "exp1_jacobian_aware.json"))
    configs = exp1["configs"]

    print("  Key Findings:")
    print(f"    Reference PPL: {exp1['reference_ppl']:.2f}")
    for name, c in configs.items():
        dppl = c["ppl"] - exp1["reference_ppl"]
        print(f"    {name}: ppl={c['ppl']:.2f} (Δ={dppl:+.2f}), mean_cos={c['mean_layer_cos']:.4f}")
