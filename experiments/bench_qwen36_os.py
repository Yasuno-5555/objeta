#!/usr/bin/env python3
"""Qwen3.6-35B-A3B — OS Runtime Instrumentation Benchmark.

Measures every token with the full os_runtime pipeline:
  observe → classify → allocate → execute

All heavy computation (forward + lm_head + entropy) in a single Rust call.
Observation (steering) + scheduler classification in Python (<50µs overhead).

Usage:
  python3 experiments/bench_qwen36_os.py [--max-tokens 32] [--prompt "..."]
"""

import ctypes, json, os, sys, time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
LKO = PROJECT.parent / "LKO"
sys.path.insert(0, str(LKO))
sys.path.insert(0, str(PROJECT))

import numpy as np
from experiments.qwen36_executor import get_lib

from os_runtime.scheduler import Scheduler, SchedulerConfig
from os_runtime.observation import compute_steering

OUTPUT = PROJECT / "experiments" / "results" / "qwen36_os_bench.json"
HDIM = 2048


def init_rust_runner(lib, bin_dir: str, fusion_ratio: float = 1.0,
                     moe_on_deltanet: bool = True, max_seq: int = 256):
    """Initialize Rust executor and Metal GPU."""
    metallib = str(PROJECT / "target" / "objeta.metallib")
    lib.lko_metal_init.argtypes = [ctypes.c_char_p]
    lib.lko_metal_init.restype = ctypes.c_int32
    lib.lko_metal_init(metallib.encode())

    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    if not lib.lko_runner_init(bin_dir.encode(), max_seq):
        raise RuntimeError("Runner init failed")

    lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
    lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
    lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
    lib.lko_runner_set_fusion_ratio(fusion_ratio)
    lib.lko_runner_set_moe_on_deltanet(1 if moe_on_deltanet else 0)

    lib.lko_runner_warmup.argtypes = [ctypes.c_int32]
    lib.lko_runner_warmup.restype = ctypes.c_int32
    print("Warming OS page cache...")
    lib.lko_runner_warmup(100)

    # Step C API — single call: forward + RMSNorm + lm_head + top-k
    lib.lko_runner_step.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_step.restype = ctypes.c_int32

    # Step + Entropy C API — single lm_head pass (no double compute)
    lib.lko_runner_step_with_entropy.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,  # entropy_out
    ]
    lib.lko_runner_step_with_entropy.restype = ctypes.c_int32

    return lib


def rust_step_with_entropy(lib, token_id: int, pos: int, seq_len: int,
                           top_k: int = 50):
    """Forward + RMSNorm + lm_head + top-k + entropy — all in Rust.

    Returns (hn, indices, values, entropy).
    Single parallel pass over 248K vocab for lm_head + entropy.
    """
    hn = np.zeros(HDIM, dtype=np.float32)
    indices = np.zeros(top_k, dtype=np.int32)
    values = np.zeros(top_k, dtype=np.float32)
    entropy = ctypes.c_float(0.0)

    k = lib.lko_runner_step_with_entropy(
        token_id, pos, seq_len,
        hn.ctypes.data, top_k,
        indices.ctypes.data, values.ctypes.data,
        ctypes.byref(entropy),
    )
    indices = indices[:k]
    values = values[:k]
    order = np.argsort(values)[::-1]
    return hn, indices[order], values[order], entropy.value


def sample(indices: np.ndarray, values: np.ndarray,
           temperature: float = 0.7, top_k: int = 40):
    """Python-side sampling from top-k logits."""
    k = min(len(indices), top_k)
    idx = indices[:k]
    val = values[:k].astype(np.float64)
    if temperature == 0:
        return int(idx[0])
    val = val / max(temperature, 0.01)
    val -= val.max()
    probs = np.exp(val)
    probs /= probs.sum()
    return int(idx[np.random.choice(len(probs), p=probs)])


def run_benchmark(lib, tokenizer, prompt: str, max_tokens: int = 32,
                  temperature: float = 0.7, top_k: int = 40):
    """Run generation with full OS telemetry on every token."""
    sched_config = SchedulerConfig(
        family="spherical_steering",
        backbone="steering",
        fusion_ratio=1.0,
    )
    sched = Scheduler(sched_config, n_layers=40)

    if hasattr(tokenizer, 'apply_chat_template'):
        msgs = [{"role": "user", "content": prompt}]
        chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(chat)
    else:
        prompt_ids = tokenizer.encode(prompt)

    tokens = list(prompt_ids)
    n_prompt = len(tokens)
    prev_hn = None
    token_metrics = []

    print(f"\n── Prompt ({n_prompt} tokens): \"{prompt}\" ──")

    # Prefill (no entropy needed during prefill)
    t0 = time.perf_counter()
    indices = values = None
    entropy = 0.0
    for i, tid in enumerate(tokens):
        hn, indices, values, entropy = rust_step_with_entropy(
            lib, tid, i, i + 1, max(50, top_k))
    prefill_s = time.perf_counter() - t0
    print(f"  Prefill: {prefill_s:.1f}s")

    # The final prefill step already produced the first next-token logits.
    if temperature == 0:
        next_token = int(indices[0])
    else:
        next_token = sample(indices, values, temperature, top_k)

    generated = []
    t_start = time.perf_counter()

    for step in range(max_tokens):
        pos = n_prompt + step
        generated.append(next_token)

        # ── OS Observation (cheap, Python <50µs) ──
        steering = 0.0
        if prev_hn is not None:
            steering = compute_steering(hn, prev_hn)
        prev_hn = hn.copy()

        is_repeat = (next_token == (tokens[-1] if step == 0
                     else generated[-2] if len(generated) > 1 else -1))

        # ── Scheduler Classification (<10µs) ──
        tc = sched.begin_token(entropy, steering,
                               prev_token_id=(tokens[-1] if step == 0
                                              else generated[-2] if len(generated) > 1 else -1),
                               predicted_token_id=next_token)

        metric = {
            "step": step,
            "token_id": int(generated[-1]),
            "entropy": round(float(entropy), 6),
            "steering": round(float(steering), 6),
            "token_class": tc.value,
            "collapse": sched.state.collapse_status.value,
            "precision": sched.state.precision,
            "is_repeat": is_repeat,
        }
        token_metrics.append(metric)

        if step < 5 or step % 10 == 0 or step == max_tokens - 1:
            e = time.perf_counter() - t_start
            tok_s = (step + 1) / e if e > 0 else 0
            print(f"  [{step + 1}/{max_tokens}] {tok_s:.2f} tok/s "
                  f"ent={entropy:.4f} steer={steering:.4f} "
                  f"class={tc.value} prec={sched.state.precision}b", flush=True)

        # ── Next step (Rust: forward + lm_head + entropy, ~3s) ──
        if next_token == 2:  # eos
            break

        hn, indices, values, entropy = rust_step_with_entropy(
            lib, next_token, pos, pos + 1, max(50, top_k))

        if temperature == 0:
            next_token = int(indices[0])
        else:
            next_token = sample(indices, values, temperature, top_k)

    total_s = time.perf_counter() - t_start
    n_gen = len(generated)

    # ── Summary ──
    text = tokenizer.decode(generated, skip_special_tokens=True) if generated else ""

    entropies = [m["entropy"] for m in token_metrics]
    steerings = [m["steering"] for m in token_metrics]

    classes = {}
    for m in token_metrics:
        c = m["token_class"]
        classes[c] = classes.get(c, 0) + 1

    sched_stats = sched.stats()

    result = {
        "model": "Qwen3.6-35B-A3B",
        "prompt": prompt,
        "n_prompt": n_prompt,
        "n_generated": n_gen,
        "total_s": round(total_s, 2),
        "prefill_s": round(prefill_s, 1),
        "tok_per_s": round(n_gen / total_s, 2) if total_s > 0 else 0,
        "avg_entropy": round(float(np.mean(entropies)), 6) if entropies else 0,
        "avg_steering": round(float(np.mean(steerings)), 6) if steerings else 0,
        "token_classes": classes,
        "class_oscillations": sched_stats.get("class_oscillations", 0),
        "collapse_events": sum(1 for m in token_metrics if m["collapse"] != "healthy"),
        "output": text[:200],
        "token_metrics": token_metrics,
    }

    print(f"\n  {n_gen} tokens in {total_s:.1f}s ({n_gen / total_s:.2f} tok/s)")
    print(f"  Entropy: avg={result['avg_entropy']:.4f}")
    print(f"  Steering: avg={result['avg_steering']:.4f}")
    print(f"  Classes: {classes}")
    print(f"  Oscillations: {result['class_oscillations']}")
    print(f"  Collapse events: {result['collapse_events']}")
    print(f"  Output: \"{text[:150]}\"")

    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--prompt", default="The meaning of life is")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--fusion", type=float, default=1.0)
    p.add_argument("--moe-on-deltanet", type=int, default=1)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    lib = get_lib()
    if lib is None:
        print("Rust library not found. Build: cargo build --release -p objeta-qwen36-executor")
        sys.exit(1)

    BIN_DIR = str(PROJECT / "models" / "qwen36_bin")
    if not Path(BIN_DIR).exists():
        print(f"Model directory not found: {BIN_DIR}")
        sys.exit(1)

    from transformers import AutoTokenizer
    snap = sorted(os.listdir(
        "/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"))[-1]
    tokenizer = AutoTokenizer.from_pretrained(
        f"/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{snap}")
    print(f"Vocab: {tokenizer.vocab_size}")

    print(f"Fusion={args.fusion}, MoE on DeltaNet={'yes' if args.moe_on_deltanet else 'no'}")

    init_rust_runner(lib, BIN_DIR, fusion_ratio=args.fusion,
                     moe_on_deltanet=bool(args.moe_on_deltanet))

    prompts = [args.prompt]
    if args.prompt == "The meaning of life is":
        prompts = [
            "The meaning of life is",
            "Explain quantum computing simply:",
            "Once upon a time in a distant galaxy,",
        ]

    all_results = []
    for prompt in prompts:
        result = run_benchmark(lib, tokenizer, prompt,
                               max_tokens=args.max_tokens,
                               temperature=args.temperature)
        all_results.append(result)

    if len(all_results) > 1:
        print("\n" + "═" * 60)
        print("  Multi-Prompt Summary")
        print("═" * 60)
        avg_tok_s = np.mean([r["tok_per_s"] for r in all_results])
        avg_ent = np.mean([r["avg_entropy"] for r in all_results])
        avg_steer = np.mean([r["avg_steering"] for r in all_results])
        print(f"  tok/s: {avg_tok_s:.2f} | entropy: {avg_ent:.4f} | steering: {avg_steer:.4f}")

    out_path = args.output or OUTPUT
    json.dump({
        "config": {
            "model": "Qwen3.6-35B-A3B",
            "family": "spherical_steering",
            "fusion_ratio": args.fusion,
            "moe_on_deltanet": bool(args.moe_on_deltanet),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "results": all_results,
    }, open(out_path, "w"), indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
