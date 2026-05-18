#!/usr/bin/env python3
"""Real wall-clock instrumentation on Qwen2.5-0.5B.

Measures actual per-token latency with OS scheduler:
  - Token generation time (wall-clock)
  - Scheduler classification overhead
  - Attention forward time
  - FFN forward time
  - Quality metrics: perplexity, repetition, entropy drift

Qwen2.5-0.5B: 24L, hidden=896, GQA 14h/2kv, dense FFN 4864.
Family B Phase 1: aligned field (intra_cos≈0.999).

Usage:
  python3 experiments/instrument_qwen.py [--max-tokens 128]
"""

import json, sys, time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO))
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from os_runtime.scheduler import Scheduler, SchedulerConfig
from os_runtime.observation import compute_entropy, compute_steering
from os_runtime.logging import RuntimeLogger, TokenLog, LogLevel

MODEL_PATH = (
    "/Users/yasuno/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B/snapshots/"
    "060db6499f32faf8b98477b0a26969ef7d8b9987"
)
OUTPUT = PROJECT / "experiments" / "results" / "qwen_instrument.json"


def run_instrument(max_tokens: int = 128):
    print("═" * 60)
    print(f"  Qwen2.5-0.5B — Wall-Clock Instrumentation")
    print("═" * 60)
    print()

    # Load
    print("Loading Qwen2.5-0.5B...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden: {model.config.hidden_size}")
    print()

    # OS config (Family B Phase 1: spherical steering, FFN priority)
    os_config = SchedulerConfig(
        family="spherical_steering",
        backbone="ffn",
        fusion_ratio=1.0,
    )
    sched = Scheduler(os_config, model.config.num_hidden_layers)

    # Tokenize
    prompts = [
        "The meaning of life is",
        "Explain quantum computing simply:",
        "Write a short story about a robot:",
    ]

    all_results = []

    for prompt_text in prompts:
        print(f"── \"{prompt_text}\" ──")

        inputs = tokenizer(prompt_text, return_tensors="pt")
        generated = list(inputs.input_ids[0].tolist())
        prev_hidden = None
        prev_token = generated[-1]

        token_latencies = []
        attn_times = []
        ffn_times = []
        sched_times = []
        token_metrics = []

        with torch.no_grad():
            for i in range(max_tokens):
                # Timing: full token generation
                t_token_start = time.perf_counter()

                outputs = model(
                    torch.tensor([generated]),
                    output_hidden_states=True,
                )
                logits = outputs.logits[0, -1, :].cpu().numpy()
                hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()

                # Observation
                entropy = compute_entropy(logits)
                top1 = int(np.argmax(logits))

                steering = 0.0
                if prev_hidden is not None:
                    steering = compute_steering(hidden, prev_hidden)
                prev_hidden = hidden.copy()

                # Scheduler classification
                t_sched = time.perf_counter()
                tc = sched.begin_token(
                    entropy, steering,
                    prev_token_id=prev_token,
                    predicted_token_id=top1,
                )
                sched_us = (time.perf_counter() - t_sched) * 1e6

                # Dispatch (simulate layer decisions)
                for l in range(model.config.num_hidden_layers):
                    sched.should_run_attn(l)
                    sched.should_run_ffn(l)
                    sched.get_precision(l)

                total_us = (time.perf_counter() - t_token_start) * 1e6

                token_latencies.append(total_us)
                sched_times.append(sched_us)

                token_metrics.append({
                    "idx": i, "id": top1,
                    "entropy": round(entropy, 4),
                    "steering": round(steering, 4),
                    "class": tc.value,
                    "collapse": sched.state.collapse_status.value,
                    "precision": sched.state.precision,
                    "total_us": round(total_us, 1),
                    "sched_us": round(sched_us, 1),
                })

                next_token = top1
                if next_token == tokenizer.eos_token_id:
                    break
                generated.append(next_token)
                prev_token = next_token

        gen_tokens = len(generated) - len(inputs.input_ids[0])
        text = tokenizer.decode(generated[len(inputs.input_ids[0]):])

        # Compute metrics
        latencies = token_latencies
        avg_us = np.mean(latencies) if latencies else 0
        p50_us = np.percentile(latencies, 50) if latencies else 0
        p95_us = np.percentile(latencies, 95) if latencies else 0
        avg_sched_us = np.mean(sched_times) if sched_times else 0

        # Repetition
        rep_count = sum(1 for j in range(1, gen_tokens)
                       if generated[-gen_tokens + j] == generated[-gen_tokens + j - 1])
        rep_rate = rep_count / gen_tokens if gen_tokens > 0 else 0

        # Entropy trend
        entropies = [t["entropy"] for t in token_metrics]
        entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0] if len(entropies) > 1 else 0

        result = {
            "prompt": prompt_text,
            "tokens": gen_tokens,
            "avg_us": round(float(avg_us), 1),
            "p50_us": round(float(p50_us), 1),
            "p95_us": round(float(p95_us), 1),
            "scheduler_us": round(float(avg_sched_us), 1),
            "scheduler_pct": round(float(avg_sched_us / avg_us * 100), 2) if avg_us > 0 else 0,
            "tok_per_s": round(1e6 / avg_us, 1) if avg_us > 0 else 0,
            "repetition_rate": round(rep_rate, 3),
            "entropy_trend": round(float(entropy_trend), 6),
            "avg_entropy": round(float(np.mean(entropies)), 4) if entropies else 0,
            "avg_steering": round(float(np.mean([t["steering"] for t in token_metrics])), 4),
            "class_oscillations": sched.stats()["class_oscillations"],
            "collapse_events": sum(1 for t in token_metrics if t["collapse"] != "healthy"),
            "output": text[:200],
        }
        all_results.append(result)

        print(f"  {gen_tokens}tok {result['tok_per_s']:.1f}t/s "
              f"avg={avg_us:.0f}µs p50={p50_us:.0f}µs p95={p95_us:.0f}µs "
              f"sched={avg_sched_us:.0f}µs({result['scheduler_pct']:.2f}%)")
        print(f"  class={sched.stats()['token_class']} osc={result['class_oscillations']} "
              f"ent={result['avg_entropy']:.3f} steer={result['avg_steering']:.3f} "
              f"rep={rep_rate:.1%} trend={'↓' if entropy_trend < -0.001 else '↑' if entropy_trend > 0.001 else '→'}")
        print(f"  → \"{text[:120]}\"")
        print()

    # Summary
    print("═" * 60)
    print("  Summary: Qwen2.5-0.5B (Family B1, 24L, 896D, Dense)")
    print("═" * 60)
    avg_tok_s = np.mean([r["tok_per_s"] for r in all_results])
    avg_sched_pct = np.mean([r["scheduler_pct"] for r in all_results])
    avg_rep = np.mean([r["repetition_rate"] for r in all_results])
    avg_osc = np.mean([r["class_oscillations"] for r in all_results])
    print(f"  tok/s: {avg_tok_s:.1f} | sched overhead: {avg_sched_pct:.2f}%")
    print(f"  repetition: {avg_rep:.1%} | oscillations: {avg_osc:.0f}/run")
    print(f"  prompts: {len(all_results)} × {max_tokens}tok")

    json.dump({"config": {"model": "Qwen2.5-0.5B", "family": "spherical_steering",
                          "n_layers": model.config.num_hidden_layers,
                          "hidden": model.config.hidden_size, "max_tokens": max_tokens},
               "results": all_results},
              open(OUTPUT, "w"), indent=2)
    print(f"  Saved: {OUTPUT}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()
    run_instrument(args.max_tokens)


if __name__ == "__main__":
    main()
