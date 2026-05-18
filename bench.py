#!/usr/bin/env python3
"""objeta OS — Fixed Benchmark Harness.

"速くなった気がする" is forbidden.

Every run uses:
  same prompt set
  same seed (42)
  same hardware
  same max_tokens
  same temperature

Output: bench_results.json — machine-readable
        bench_report.md  — human-readable

Metrics:
  Performance: tok/s, TTFT, p50/p95 latency, memory peak
  Quality:     repetition rate, entropy collapse count, output length
  OS:          avg steering, precision histogram, plan distribution,
               class_oscillations, collapse recovery time

Usage:
  python bench.py                        # full suite
  python bench.py --quick                # 3 prompts, 16 tokens
  python bench.py --model stories15m-moe # specific model
"""

import json, os, sys, time, tracemalloc
from pathlib import Path
from dataclasses import dataclass, field

PROJECT = Path(__file__).parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO))
sys.path.insert(0, str(PROJECT))

import numpy as np

from os_runtime.scheduler import SchedulerConfig
from os_runtime.logging import RuntimeLogger, LogLevel


# ═══════════════════════════════════════════════════════════
# Fixed prompt set (locked — never change to keep comparability)
# ═══════════════════════════════════════════════════════════

BENCH_PROMPTS = [
    "The capital of France is",
    "Explain quantum computing in simple terms",
    "Write a haiku about cats",
    "In the beginning, God created",
    "The most important thing to remember is",
    "If I could travel anywhere, I would go to",
    "The difference between love and hate is",
    "Once upon a time, there was a",
    "The meaning of life is",
    "She opened the door and saw",
]

SHORT_PROMPTS = BENCH_PROMPTS[:3]

SEED = 42
DEFAULT_MAX_TOKENS = 32
QUICK_MAX_TOKENS = 16


# ═══════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════

@dataclass
class TokenMetrics:
    token_idx: int
    token_id: int
    entropy: float
    steering: float
    precision: int
    token_class: str
    collapse_status: str
    latency_ms: float


@dataclass
class RunResult:
    model: str
    prompt: str
    tokens_generated: int
    elapsed_s: float
    tok_per_s: float
    ttft_ms: float            # time to first token
    p50_latency_ms: float
    p95_latency_ms: float
    memory_peak_mb: float
    repetition_rate: float    # fraction of tokens that repeat previous
    entropy_collapse_count: int
    output_text: str
    # OS metrics
    avg_steering: float
    avg_entropy: float
    avg_precision: float
    skip_rate: float
    class_oscillations: int
    precision_histogram: dict[str, int]
    token_classes: dict[str, int]
    collapse_events: int
    warnings: int
    errors: int
    # Per-token detail
    token_metrics: list[dict] = field(default_factory=list)


def run_single(model_name: str, model_entry: dict,
               prompt_text: str, max_tokens: int,
               tokenizer, os_config: SchedulerConfig) -> RunResult:
    """Run one prompt through the OS runtime. Measure everything."""

    # Build chat prompt
    if "llm" in model_entry:
        msgs = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

        from os_runtime import OSRuntime
        logger = RuntimeLogger(level=LogLevel.WARNING)
        os_runtime = OSRuntime(model_entry["llm"], os_config, logger)

        # Memory tracking
        tracemalloc.start()
        t0 = time.perf_counter()

        tokens = os_runtime.generate(
            prompt, tokenizer=tokenizer,
            max_tokens=max_tokens, temperature=0,
        )
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Extract per-token metrics from logger
        token_metrics = []
        ttft = 0.0
        latencies = []
        for tl in logger.token_logs:
            latencies.append(tl.forward_ms)
            if ttft == 0.0 and tl.elapsed_ms > 0:
                ttft = tl.elapsed_ms
            token_metrics.append({
                "idx": tl.token_idx,
                "id": tl.token_id,
                "entropy": round(tl.entropy, 4),
                "steering": round(tl.steering, 4),
                "precision": tl.precision,
                "class": tl.token_class,
                "collapse": tl.collapse_status,
                "latency_ms": round(tl.forward_ms, 1),
            })

        summary = logger.run_summary()
        stats = os_runtime.scheduler.stats()

        text = tokenizer.decode(tokens) if tokens else ""

    else:
        # PyTorch path (stories15M)
        import torch
        from os_runtime.scheduler import Scheduler
        from os_runtime.observation import compute_entropy, compute_steering
        from os_runtime.logging import TokenLog, LayerAction

        model = model_entry["model"]
        n_layers = model_entry.get("n_layers", 6)

        msgs = [{"role": "user", "content": prompt_text}]
        prompt = " ".join(m["content"] for m in msgs)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        sched = Scheduler(os_config, n_layers)
        logger = RuntimeLogger(level=LogLevel.WARNING)
        logger.start_run()

        generated = list(input_ids[0].tolist())
        prev_hidden = None
        prev_token = generated[-1]
        token_metrics = []
        latencies = []
        ttft = 0.0

        tracemalloc.start()
        t0 = time.perf_counter()

        with torch.no_grad():
            for gen_idx in range(max_tokens):
                t_token = time.perf_counter()

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

                tc = sched.begin_token(
                    entropy, steering,
                    prev_token_id=prev_token,
                    predicted_token_id=top1,
                )

                for l in range(n_layers):
                    sched.should_run_attn(l)
                    sched.should_run_ffn(l)
                    sched.get_precision(l)

                lat_ms = (time.perf_counter() - t_token) * 1000
                latencies.append(lat_ms)
                if ttft == 0.0 and gen_idx == 0:
                    ttft = lat_ms

                token_metrics.append({
                    "idx": gen_idx,
                    "id": top1,
                    "entropy": round(entropy, 4),
                    "steering": round(steering, 4),
                    "precision": sched.state.precision,
                    "class": tc.value,
                    "collapse": sched.state.collapse_status.value,
                    "latency_ms": round(lat_ms, 1),
                })

                next_token = top1
                if next_token == tokenizer.eos_token_id:
                    break
                generated.append(next_token)
                prev_token = next_token

        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.end_run()

        tokens = generated[len(input_ids[0]):]
        text = tokenizer.decode(tokens) if tokens else ""
        summary = logger.run_summary()
        stats = sched.stats()

    # Compute metrics
    n_tok = len(tokens)
    rep_count = sum(
        1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1]
    ) if n_tok > 1 else 0
    repetition_rate = rep_count / n_tok if n_tok > 0 else 0.0

    # Precision histogram
    prec_hist = {}
    for tm in token_metrics:
        p = str(tm["precision"])
        prec_hist[p] = prec_hist.get(p, 0) + 1

    # Token class distribution
    class_dist = {}
    for tm in token_metrics:
        c = tm["class"]
        class_dist[c] = class_dist.get(c, 0) + 1

    return RunResult(
        model=model_name,
        prompt=prompt_text[:80],
        tokens_generated=n_tok,
        elapsed_s=round(elapsed, 2),
        tok_per_s=round(n_tok / elapsed, 1) if elapsed > 0 else 0,
        ttft_ms=round(ttft, 1),
        p50_latency_ms=round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        memory_peak_mb=round(peak / 1024 / 1024, 1),
        repetition_rate=round(repetition_rate, 3),
        entropy_collapse_count=stats.get("collapse_events", 0),
        output_text=text[:200],
        avg_steering=round(summary.get("avg_steering", 0), 4),
        avg_entropy=round(summary.get("avg_entropy", 0), 4),
        avg_precision=round(summary.get("avg_precision", 16), 1),
        skip_rate=round(stats.get("skip_rate", 0), 3),
        class_oscillations=stats.get("class_oscillations", 0),
        precision_histogram=prec_hist,
        token_classes=class_dist,
        collapse_events=summary.get("collapse_events", 0),
        warnings=summary.get("warnings", 0),
        errors=summary.get("errors", 0),
        token_metrics=token_metrics,
    )


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default="bench_results.json")
    args = parser.parse_args()

    np.random.seed(SEED)

    prompts = SHORT_PROMPTS if args.quick else BENCH_PROMPTS
    max_tokens = QUICK_MAX_TOKENS if args.quick else DEFAULT_MAX_TOKENS

    print("═" * 60)
    print("  objeta OS — Fixed Benchmark")
    print(f"  prompts={len(prompts)} max_tokens={max_tokens} seed={SEED}")
    print("═" * 60)
    print()

    # Register models
    models = {}

    # TinyLlama
    try:
        from runtime.models.llm import LLM, ModelConfig
        from runtime.models.loaders.model_loader import ModelLoader
        from transformers import AutoTokenizer

        MODEL_PATH = (
            "/Users/yasuno/.cache/huggingface/hub/"
            "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/"
            "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
        )
        loader = ModelLoader(MODEL_PATH)
        cfg = ModelConfig(
            hidden_dim=2048, ffn_dim=5632, n_layers=22,
            n_heads=32, n_kv_heads=4, head_dim=64, vocab_size=32000,
        )
        weights = loader.load_weights()
        llm = LLM(weights, cfg)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        models["tinyllama-1.1b"] = {
            "llm": llm, "tokenizer": tokenizer,
            "family": "residual_transport", "n_layers": 22,
        }
        print("✓ tinyllama-1.1b")
    except Exception as e:
        print(f"✗ tinyllama: {e}")

    # stories15M_MOE
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        MOE_PATH = (
            "/Users/yasuno/.cache/huggingface/hub/"
            "models--ggml-org--stories15M_MOE/snapshots/"
            "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
        )
        moe_model = AutoModelForCausalLM.from_pretrained(
            MOE_PATH, dtype=torch.float32, device_map="cpu")
        moe_model.eval()
        moe_tok = AutoTokenizer.from_pretrained(MOE_PATH)
        models["stories15m-moe"] = {
            "model": moe_model, "tokenizer": moe_tok,
            "family": "spherical_steering", "n_layers": 6,
            "n_experts": 4, "top_k": 2,
        }
        print("✓ stories15m-moe")
    except Exception as e:
        print(f"✗ stories-moe: {e}")

    if not models:
        print("No models loaded. Exiting.")
        return

    # Filter if --model specified
    if args.model:
        models = {k: v for k, v in models.items() if args.model in k}
        if not models:
            print(f"Model '{args.model}' not found.")
            return

    print()

    # Run all prompts × models
    all_results = []
    for model_name, entry in models.items():
        tokenizer = entry["tokenizer"]
        family = entry.get("family", "residual_transport")
        os_config = SchedulerConfig(
            family=family,
            backbone="attention" if family == "residual_transport" else "steering",
            fusion_ratio=0.5 if family == "residual_transport" else 1.0,
        )

        print(f"── {model_name} ──")
        for prompt_text in prompts:
            t0 = time.time()
            result = run_single(
                model_name, entry, prompt_text, max_tokens,
                tokenizer, os_config,
            )
            elapsed = time.time() - t0
            status = "✓" if result.tokens_generated > 0 else "✗"
            print(f"  {status} \"{prompt_text[:50]}...\" "
                  f"→ {result.tokens_generated}tok {result.tok_per_s:.1f}t/s "
                  f"osc={result.class_oscillations} "
                  f"ent={result.avg_entropy:.3f} steer={result.avg_steering:.3f}")
            all_results.append(result)
        print()

    # Aggregate
    print("═" * 60)
    print("  Aggregate")
    print("═" * 60)

    for model_name in models:
        model_results = [r for r in all_results if r.model == model_name]
        if not model_results:
            continue
        n = len(model_results)
        avg_tok_s = np.mean([r.tok_per_s for r in model_results])
        avg_ttft = np.mean([r.ttft_ms for r in model_results])
        avg_ent = np.mean([r.avg_entropy for r in model_results])
        avg_steer = np.mean([r.avg_steering for r in model_results])
        avg_skip = np.mean([r.skip_rate for r in model_results])
        avg_osc = np.mean([r.class_oscillations for r in model_results])
        avg_collapse = np.mean([r.collapse_events for r in model_results])
        total_rep = np.mean([r.repetition_rate for r in model_results])

        print(f"  {model_name}:")
        print(f"    tok/s={avg_tok_s:.1f}  TTFT={avg_ttft:.0f}ms  "
              f"ent={avg_ent:.3f}  steer={avg_steer:.3f}")
        print(f"    skip={avg_skip*100:.0f}%  osc={avg_osc:.0f}/run  "
              f"collapse={avg_collapse:.0f}/run  repeat={total_rep*100:.1f}%")

    # Save
    output = {
        "meta": {
            "seed": SEED,
            "max_tokens": max_tokens,
            "n_prompts": len(prompts),
            "models": list(models.keys()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": [
            {k: v for k, v in r.__dict__.items() if k != "token_metrics"}
            for r in all_results
        ],
        "token_details": [
            {"model": r.model, "prompt": r.prompt[:50],
             "tokens": r.token_metrics}
            for r in all_results
        ],
    }
    out_path = PROJECT / args.output
    json.dump(output, open(out_path, "w"), indent=2, default=str)
    print(f"\n  Results: {out_path}")


if __name__ == "__main__":
    main()
