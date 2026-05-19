#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.compare_moe_impl import (
    MODEL_PATH,
    chat_ids,
    init_runner,
    rust_prefill,
    rust_trace_components,
    rust_trace_router,
    rust_selected_expert_q4,
)
from transformers import AutoTokenizer


def run(layer_idx: int, prompt: str, iters: int):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = chat_ids(tok, prompt)

    lib = init_runner()
    rust_prefill(lib, ids)
    comp = rust_trace_components(lib, ids, layer_idx)
    _, expert_ids, routing_weights, _ = rust_trace_router(lib, ids, layer_idx)
    x = comp["h_norm2"].astype(np.float32)
    expert_ids = expert_ids.astype(np.int32)
    routing_weights = routing_weights.astype(np.float32)

    print(f"Layer {layer_idx}, prompt={prompt!r}, iterations={iters}")
    print("N | ms/iter")
    for n in range(len(expert_ids), 0, -1):
        ids_n = expert_ids[:n]
        w_n = routing_weights[:n]
        w_n = w_n / np.sum(w_n)
        t0 = time.perf_counter()
        for _ in range(iters):
            rust_selected_expert_q4(lib, layer_idx, x, ids_n, w_n)
        dt = (time.perf_counter() - t0) * 1000.0 / iters
        print(f"{n} | {dt:.3f}")


if __name__ == "__main__":
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    run(layer, prompt, iters)
