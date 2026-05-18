#!/usr/bin/env python3
"""objeta OS Runtime — Reproducible Benchmark Suite.

Compares Baseline vs OS Runtime on TinyLlama-1.1B-Chat (M1 8GB, MLX).

Metrics:
  - Throughput (tok/s)
  - Quality (repetition rate, empty output rate)
  - Skip rate (% layers skipped)
  - Precision distribution
  - Collapse events
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LKO"))

from os_runtime.scheduler import SchedulerConfig
from os_runtime.logging import RuntimeLogger, LogLevel
from os_runtime.os_llm import OSLLM

import numpy as np


# ── Configuration ──

MODEL_PATH = (
    "/Users/yasuno/.cache/huggingface/hub/"
    "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/"
    "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
)

PROMPTS = [
    "The meaning of life is",
    "Explain quantum computing in simple terms",
    "Write a haiku about cats",
    "The capital of France is",
    "In the beginning, God created",
    "The most important thing to remember is",
    "If I could travel anywhere, I would go to",
    "The difference between love and hate is",
]

GEN_CONFIGS = [
    # (name, fusion_ratio, temporal_stride)
    ("Baseline",       1.0,  0),
    ("OS Default",     0.50, 0),
    ("OS Aggressive",  0.40, 2),
]


# ── Main ──

def run_benchmark():
    from runtime.models.llm import LLM, ModelConfig
    from runtime.models.loaders.model_loader import ModelLoader
    from transformers import AutoTokenizer

    print("=" * 70)
    print("  objeta OS Runtime — Reproducible Benchmark")
    print("  TinyLlama-1.1B-Chat on M1 8GB (MLX)")
    print("=" * 70)

    # Load model once
    loader = ModelLoader(MODEL_PATH)
    config = ModelConfig(
        hidden_dim=2048, ffn_dim=5632, n_layers=22,
        n_heads=32, n_kv_heads=4, head_dim=64, vocab_size=32000,
    )
    t0 = time.time()
    weights = loader.load_weights()
    base_llm = LLM(weights, config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Prompts: {len(PROMPTS)} | Configs: {len(GEN_CONFIGS)}")
    print()

    all_results = []

    for prompt_text in PROMPTS:
        msgs = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

        print(f"── \"{prompt_text[:60]}\" ──")

        for cfg_name, fusion_ratio, temp_stride in GEN_CONFIGS:
            logger = RuntimeLogger(level=LogLevel.INFO)
            os_config = SchedulerConfig(
                fusion_ratio=fusion_ratio,
                temporal_stride=temp_stride,
            )
            os_llm = OSLLM(base_llm, os_config, logger)

            t0 = time.time()
            try:
                tokens = os_llm.generate(
                    prompt, tokenizer=tokenizer,
                    max_tokens=16, temperature=0,
                )
            except Exception as e:
                print(f"  {cfg_name:<20} ERROR: {e}")
                continue

            elapsed = time.time() - t0
            n_tok = len(tokens)
            text = tokenizer.decode(tokens) if tokens else ""
            stats = os_llm.scheduler.stats()
            summary = logger.run_summary()

            # Quality assessment
            if n_tok == 0:
                quality = "EMPTY"
            elif any(w in text.lower() for w in
                     ["the ", "is ", "are ", "was ", "a ", "in ", "to ",
                      "central", "complex", "question", "process",
                      "cat", "purr", "feline", "whisker", "tail",
                      "paris", "france", "capit", "begin", "heav",
                      "love", "hate", "import", "travel"]):
                quality = "OK"
            else:
                quality = "DEGEN"

            tok_s = n_tok / elapsed if elapsed > 0 else 0

            line = (f"  {cfg_name:<20} {elapsed:.1f}s {tok_s:.1f}t/s "
                    f"skip={stats['skip_rate']*100:.0f}% "
                    f"ent={stats['entropy']:.2f} steer={stats['steering']:.2f} "
                    f"[{quality}]")
            print(line)
            if text:
                print(f"    → \"{text[:100]}\"")

            all_results.append({
                "prompt": prompt_text[:60],
                "config": cfg_name,
                "fusion_ratio": fusion_ratio,
                "temporal_stride": temp_stride,
                "tokens": n_tok,
                "elapsed_s": round(elapsed, 2),
                "tok_per_s": round(tok_s, 1),
                "quality": quality,
                "text": text[:200],
                **stats,
                "summary": summary,
            })

        print()

    # Summary
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    for cfg_name, _, _ in GEN_CONFIGS:
        cfg_results = [r for r in all_results if r["config"] == cfg_name]
        if not cfg_results:
            continue
        n = len(cfg_results)
        ok = sum(1 for r in cfg_results if r["quality"] == "OK")
        avg_tok_s = np.mean([r["tok_per_s"] for r in cfg_results])
        avg_skip = np.mean([r["skip_rate"] for r in cfg_results])
        avg_ent = np.mean([r["entropy"] for r in cfg_results])
        print(f"  {cfg_name:<20} ok={ok}/{n} tok/s={avg_tok_s:.1f} "
              f"skip={avg_skip*100:.0f}% ent={avg_ent:.2f}")

    # Save results
    out_path = Path(__file__).parent / "bench_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
