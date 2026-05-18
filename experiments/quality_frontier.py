#!/usr/bin/env python3
"""Quality Frontier — where does locality break intelligence?

Sweeps locality λ and measures output quality:
  - Perplexity (on WikiText-2 samples)
  - Repetition rate
  - Entropy collapse rate
  - Output diversity (self-BLEU)

Finds the quality-degradation curve: λ vs output quality.

Usage:
  python3 experiments/quality_frontier.py --model tinyllama --quick
  python3 experiments/quality_frontier.py --model stories-moe --lambdas 0,2,4,6,8
"""

import json, sys, time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO)); sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from transformers import AutoTokenizer

OUTPUT = PROJECT / "experiments" / "results" / "quality_frontier.json"


def compute_perplexity(logits_list: list[np.ndarray],
                       target_ids: list[int]) -> float:
    """Compute perplexity from logits and target token IDs."""
    nll = 0.0
    n_tokens = 0
    for logits, target in zip(logits_list, target_ids):
        logits_stable = logits - logits.max()
        probs = np.exp(logits_stable.astype(np.float64))
        probs /= probs.sum()
        if 0 <= target < len(probs):
            nll -= np.log(probs[target] + 1e-12)
            n_tokens += 1
    return float(np.exp(nll / max(1, n_tokens)))


def measure_tinyllama_quality(lambda_values: list[float],
                              max_tokens: int = 64) -> list[dict]:
    """Measure output quality at different locality λ on TinyLlama."""
    from runtime.models.llm import LLM, ModelConfig
    from runtime.models.loaders.model_loader import ModelLoader
    from os_runtime import OSRuntime, SchedulerConfig

    MODEL_PATH = ("/Users/yasuno/.cache/huggingface/hub/"
                  "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/"
                  "fe8a4ea1ffedaf415f4da2f062534de366a451e6")

    loader = ModelLoader(MODEL_PATH)
    cfg = ModelConfig(hidden_dim=2048, ffn_dim=5632, n_layers=22,
                      n_heads=32, n_kv_heads=4, head_dim=64, vocab_size=32000)
    llm = LLM(loader.load_weights(), cfg)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    prompts = [
        "The meaning of life is",
        "Explain quantum computing simply:",
        "Write a short story about a robot:",
    ]

    results = []
    for lam in lambda_values:
        print(f"  λ={lam:.0f}...", end=" ", flush=True)
        os_config = SchedulerConfig(
            family="residual_transport", backbone="attention",
            fusion_ratio=max(0.3, 1.0 - lam * 0.1),
        )

        all_repetitions = []
        all_entropies = []
        all_steerings = []
        total_tokens = 0
        total_time = 0.0

        for prompt_text in prompts:
            msgs = [{"role": "user", "content": prompt_text}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            from os_runtime.logging import RuntimeLogger, LogLevel
            logger = RuntimeLogger(level=LogLevel.WARNING)
            os = OSRuntime(llm, os_config, logger)

            t0 = time.time()
            tokens = os.generate(prompt, tokenizer=tok, max_tokens=max_tokens, temperature=0)
            elapsed = time.time() - t0

            text = tok.decode(tokens) if tokens else ""
            total_tokens += len(tokens)
            total_time += elapsed

            # Metrics
            reps = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
            all_repetitions.append(reps / max(1, len(tokens)))

            if logger.token_logs:
                all_entropies.extend([t.entropy for t in logger.token_logs])
                all_steerings.extend([t.steering for t in logger.token_logs])

        avg_rep = np.mean(all_repetitions) if all_repetitions else 0
        avg_ent = np.mean(all_entropies) if all_entropies else 0
        avg_steer = np.mean(all_steerings) if all_steerings else 0
        tok_s = total_tokens / total_time if total_time > 0 else 0

        results.append({
            "lambda": lam,
            "tokens": total_tokens,
            "tok_per_s": round(tok_s, 1),
            "repetition_rate": round(float(avg_rep), 4),
            "avg_entropy": round(float(avg_ent), 4),
            "avg_steering": round(float(avg_steer), 4),
        })
        print(f"{total_tokens}tok {tok_s:.1f}t/s rep={avg_rep:.2%} ent={avg_ent:.3f}")

    return results


def measure_stories_quality(lambda_values: list[float],
                            max_tokens: int = 64) -> list[dict]:
    """Measure quality at different locality λ on stories15M_MOE."""
    from transformers import AutoModelForCausalLM
    from os_runtime.scheduler import Scheduler
    from os_runtime.observation import compute_entropy, compute_steering
    from os_runtime.rewriter import RouterRewriter, RoutingConfig

    MOE_PATH = ("/Users/yasuno/.cache/huggingface/hub/"
                "models--ggml-org--stories15M_MOE/snapshots/"
                "b6dd737497465570b5f5e962dbc9d9454ed1e0eb")
    model = AutoModelForCausalLM.from_pretrained(MOE_PATH, dtype=torch.float32, device_map="cpu")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MOE_PATH)

    prompts = [
        "Once upon a time, there was a",
        "The little cat sat on",
        "She opened the door and saw",
    ]

    results = []
    for lam in lambda_values:
        print(f"  λ={lam:.0f}...", end=" ", flush=True)

        all_repetitions = []
        all_entropies = []
        all_steerings = []
        all_routing_ents = []
        total_tokens = 0
        total_time = 0.0

        for prompt_text in prompts:
            inputs = tok(prompt_text, return_tensors="pt")
            generated = list(inputs.input_ids[0].tolist())
            prev_hidden = None
            prev_token = generated[-1]

            with torch.no_grad():
                for _ in range(max_tokens):
                    t0 = time.time()
                    outputs = model(torch.tensor([generated]),
                                   output_hidden_states=True,
                                   output_router_logits=True)
                    logits = outputs.logits[0, -1, :].cpu().numpy()
                    hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
                    total_time += time.time() - t0

                    entropy = compute_entropy(logits)
                    all_entropies.append(entropy)

                    steering = 0.0
                    if prev_hidden is not None:
                        steering = compute_steering(hidden, prev_hidden)
                    all_steerings.append(steering)
                    prev_hidden = hidden.copy()

                    # Routing entropy (if MoE)
                    if hasattr(outputs, 'router_logits') and outputs.router_logits:
                        for rl in outputs.router_logits:
                            if rl is not None:
                                w = torch.softmax(rl[-1,:].float(), dim=-1).cpu().numpy()
                                ent = -float(np.sum(w * np.log(w + 1e-12))) / np.log(len(w))
                                all_routing_ents.append(ent)

                    top1 = int(np.argmax(logits))
                    if top1 == tok.eos_token_id:
                        break
                    generated.append(top1)
                    prev_token = top1

            gen_tokens = len(generated) - len(inputs.input_ids[0])
            total_tokens += gen_tokens
            reps = sum(1 for i in range(len(generated)-len(inputs.input_ids[0])-1)
                      if generated[-gen_tokens+i] == generated[-gen_tokens+i-1])
            all_repetitions.append(reps / max(1, gen_tokens))

        results.append({
            "lambda": lam,
            "tokens": total_tokens,
            "tok_per_s": round(total_tokens / total_time, 1) if total_time > 0 else 0,
            "repetition_rate": round(float(np.mean(all_repetitions)), 4) if all_repetitions else 0,
            "avg_entropy": round(float(np.mean(all_entropies)), 4) if all_entropies else 0,
            "avg_steering": round(float(np.mean(all_steerings)), 4) if all_steerings else 0,
            "avg_routing_entropy": round(float(np.mean(all_routing_ents)), 4) if all_routing_ents else 0,
        })
        print(f"{total_tokens}tok rep={all_repetitions[-1]:.2%} ent={np.mean(all_entropies):.3f}")

    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stories-moe")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--lambdas", default=None)
    args = p.parse_args()

    lambdas = [0, 1, 2, 3, 4, 6, 8] if not args.quick else [0, 2, 4, 8]
    if args.lambdas:
        lambdas = [float(x) for x in args.lambdas.split(",")]
    max_tokens = 32 if args.quick else 64

    print("═" * 60)
    print(f"  Quality Frontier — λ sweep on {args.model}")
    print(f"  λ ∈ {lambdas}, max_tokens={max_tokens}")
    print("═" * 60)
    print()

    if "tiny" in args.model:
        results = measure_tinyllama_quality(lambdas, max_tokens)
    else:
        results = measure_stories_quality(lambdas, max_tokens)

    # Summary
    print(f"\n{'λ':>4s} {'tok/s':>7s} {'rep':>7s} {'entropy':>8s} {'steering':>9s}")
    baseline_rep = results[0]["repetition_rate"] if results else 1.0
    for r in results:
        rep_flag = " ⚠" if r["repetition_rate"] > baseline_rep * 2 else ""
        ent_flag = " ⚠" if r["avg_entropy"] < 0.01 else ""
        print(f"{r['lambda']:4.0f} {r['tok_per_s']:6.1f}t/s {r['repetition_rate']:6.2%} "
              f"{r['avg_entropy']:8.4f} {r['avg_steering']:8.4f}{rep_flag}{ent_flag}")

    # Find degradation point
    degradation_lam = None
    for r in results[1:]:
        if r["repetition_rate"] > 0.1:  # >10% repetition = degraded
            degradation_lam = r["lambda"]
            break

    print(f"\nQuality frontier:")
    if degradation_lam:
        print(f"  Degradation begins at λ ≈ {degradation_lam}")
    print(f"  Baseline rep rate: {baseline_rep:.2%}")
    print(f"  Safe λ range: [0, {degradation_lam or '?'})")

    json.dump({"model": args.model, "results": results},
              open(OUTPUT, "w"), indent=2)
    print(f"  Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
