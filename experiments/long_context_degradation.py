#!/usr/bin/env python3
"""Long-Context Degradation Test.

Generates 512+ tokens and tracks:
  - collapse propagation (steering misclassify → precision drop → entropy collapse → repetition lock)
  - CollapseMemory risk score over time
  - conservative mode activation
  - repetition attractor formation

Usage:
  python3 experiments/long_context_degradation.py [--max-tokens 1024]
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

from os_runtime.scheduler import Scheduler, SchedulerConfig, CollapseStatus
from os_runtime.observation import compute_entropy, compute_steering
from os_runtime.logging import RuntimeLogger, TokenLog, LogLevel
from os_runtime.moe import MoeSchedulerExtension

MODEL_PATH = (
    "/Users/yasuno/.cache/huggingface/hub/"
    "models--ggml-org--stories15M_MOE/snapshots/"
    "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
)

OUTPUT_DIR = PROJECT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_long_context(max_tokens: int = 512):
    print("═" * 60)
    print(f"  Long-Context Degradation Test — {max_tokens} tokens")
    print("═" * 60)
    print()

    # Load model
    print("Loading stories15M_MOE...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, device_map="cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"  Loaded ({model.config.num_hidden_layers}L, "
          f"{model.config.num_local_experts} experts)")
    print()

    # OS setup
    os_config = SchedulerConfig(
        family="spherical_steering",
        backbone="steering",
        fusion_ratio=1.0,
    )
    sched = Scheduler(os_config, model.config.num_hidden_layers)
    moe_ext = MoeSchedulerExtension(
        n_layers=model.config.num_hidden_layers,
        n_experts=model.config.num_local_experts,
        default_top_k=model.config.num_experts_per_tok,
    )

    # Tokenize
    prompt = "Once upon a time, in a land far away, there lived a"
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = list(inputs.input_ids[0].tolist())
    prev_hidden = None
    prev_token = -1

    # Tracking
    snapshots = []  # every N tokens, capture full state
    token_logs = []
    conservative_events = []

    print(f"Prompt: \"{prompt}\"")
    print(f"Generating {max_tokens} tokens...")
    t0 = time.time()

    with torch.no_grad():
        for i in range(max_tokens):
            outputs = model(
                torch.tensor([generated]),
                output_router_logits=True,
                output_hidden_states=True,
            )
            logits = outputs.logits[0, -1, :].cpu().numpy()
            hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()

            entropy = compute_entropy(logits)
            top1 = int(np.argmax(logits))

            steering = 0.0
            if prev_hidden is not None:
                steering = compute_steering(hidden, prev_hidden)
            prev_hidden = hidden.copy()

            # Classify through scheduler
            tc = sched.begin_token(
                entropy, steering,
                prev_token_id=prev_token,
                predicted_token_id=top1,
            )

            # Dispatch
            for l in range(model.config.num_hidden_layers):
                sched.should_run_attn(l)
                sched.should_run_ffn(l)
                sched.get_precision(l)

            # MoE routing observation
            if hasattr(outputs, 'router_logits') and outputs.router_logits:
                token_routing_ent = []
                for l, rl in enumerate(outputs.router_logits):
                    if rl is not None:
                        w = torch.softmax(rl[-1, :].float(), dim=-1).cpu().numpy()
                        obs = moe_ext.observe_routing(l, w)
                        token_routing_ent.append(obs.routing_entropy)
                avg_routing_ent = float(np.mean(token_routing_ent)) if token_routing_ent else 0.0
            else:
                avg_routing_ent = 0.0

            # Token log
            token_logs.append({
                "idx": i, "id": top1,
                "entropy": round(entropy, 4),
                "steering": round(steering, 4),
                "class": tc.value,
                "collapse": sched.state.collapse_status.value,
                "precision": sched.state.precision,
                "routing_ent": round(avg_routing_ent, 4),
                "risk_score": round(sched.collapse_memory.risk_score, 4),
                "conservative": sched.collapse_memory.conservative_mode,
            })

            # Track conservative mode transitions
            mem = sched.collapse_memory
            if mem.conservative_mode and (
                not conservative_events or
                conservative_events[-1]["type"] != "enter"
            ):
                conservative_events.append({
                    "type": "enter", "token": i,
                    "risk": mem.risk_score,
                })

            # Snapshot every 64 tokens
            if i % 64 == 0 or i == max_tokens - 1:
                snapshots.append({
                    "token_idx": i,
                    "class": tc.value,
                    "collapse": sched.state.collapse_status.value,
                    "risk_score": round(mem.risk_score, 4),
                    "conservative": mem.conservative_mode,
                    "total_collapse": mem.total_collapse_tokens,
                    "total_warning": mem.total_warning_tokens,
                    "recent_collapse_rate": mem.stats()["recent_collapse_rate"],
                    "text_snippet": tokenizer.decode(
                        generated[-min(50, len(generated)):]
                    )[:120],
                })

            # Sampling
            next_token = top1
            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)
            prev_token = next_token

    elapsed = time.time() - t0
    gen_tokens = len(generated) - len(inputs.input_ids[0])
    text = tokenizer.decode(generated[len(inputs.input_ids[0]):])

    print(f"  Generated {gen_tokens} tokens in {elapsed:.1f}s "
          f"({gen_tokens/elapsed:.1f} tok/s)")
    print()

    # ── Degradation analysis ──
    print("═" * 60)
    print("  Degradation Analysis")
    print("═" * 60)

    # Find degradation phases
    entropy_series = [t["entropy"] for t in token_logs]
    steering_series = [t["steering"] for t in token_logs]
    risk_series = [t["risk_score"] for t in token_logs]
    collapse_series = [1.0 if t["collapse"] != "healthy" else 0.0
                       for t in token_logs]

    # Break into windows
    window = 64
    windows = []
    for w_start in range(0, len(token_logs), window):
        w_end = min(w_start + window, len(token_logs))
        chunk = token_logs[w_start:w_end]
        if not chunk:
            break
        windows.append({
            "start": w_start,
            "end": w_end,
            "avg_entropy": round(np.mean([t["entropy"] for t in chunk]), 4),
            "avg_steering": round(np.mean([t["steering"] for t in chunk]), 4),
            "avg_risk": round(np.mean([t["risk_score"] for t in chunk]), 4),
            "collapse_rate": round(
                sum(1 for t in chunk if t["collapse"] != "healthy") / len(chunk), 3
            ),
            "repeat_rate": round(
                sum(1 for i in range(1, len(chunk))
                    if chunk[i]["id"] == chunk[i-1]["id"]) / max(1, len(chunk)-1), 3
            ),
            "dominant_class": max(set(t["class"] for t in chunk),
                                  key=lambda c: sum(1 for t in chunk if t["class"] == c)),
        })

    print(f"  Window size: {window} tokens")
    print(f"  Total windows: {len(windows)}")
    print()
    print(f"  {'Window':<12s} {'Entropy':>8s} {'Steer':>8s} {'Risk':>8s} "
          f"{'Collapse':>10s} {'Repeat':>8s} {'Dominant':>10s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")

    for w in windows:
        risk_flag = " ⚠" if w["avg_risk"] > 0.3 else ""
        repeat_flag = " 🔴" if w["repeat_rate"] > 0.3 else ""
        print(f"  {w['start']:4d}-{w['end']:<4d}  "
              f"{w['avg_entropy']:8.4f} {w['avg_steering']:8.4f} "
              f"{w['avg_risk']:8.4f} {w['collapse_rate']:10.3f} "
              f"{w['repeat_rate']:8.3f} {w['dominant_class']:>10s}"
              f"{risk_flag}{repeat_flag}")

    print()

    # Conservative mode events
    if conservative_events:
        print(f"  Conservative mode events: {len(conservative_events)}")
        for ev in conservative_events[:10]:
            print(f"    {ev['type']:6s} @ token {ev['token']:4d} "
                  f"(risk={ev['risk']:.3f})")
    else:
        print("  Conservative mode: never triggered")

    print()

    # Final state
    stats = sched.stats()
    mem_stats = sched.collapse_memory.stats()
    print(f"  Final risk score: {mem_stats['risk_score']:.4f}")
    print(f"  Conservative mode: {mem_stats['conservative_mode']}")
    print(f"  Total collapse tokens: {mem_stats['total_collapse_tokens']}")
    print(f"  Total warning tokens: {mem_stats['total_warning_tokens']}")
    print(f"  Class oscillations: {stats['class_oscillations']}")
    print()

    # Output excerpt
    print(f"  Output excerpt: \"{text[:200]}...\"")
    print()

    # Save
    result = {
        "max_tokens": max_tokens,
        "generated": gen_tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_per_s": round(gen_tokens / elapsed, 1),
        "final_risk_score": mem_stats["risk_score"],
        "conservative_triggered": len(conservative_events) > 0,
        "conservative_events": conservative_events,
        "total_collapse_tokens": mem_stats["total_collapse_tokens"],
        "class_oscillations": stats["class_oscillations"],
        "windows": windows,
        "snapshots": snapshots,
        "output_excerpt": text[:500],
    }
    out_path = OUTPUT_DIR / "long_context_degradation.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    run_long_context(args.max_tokens)


if __name__ == "__main__":
    main()
