#!/usr/bin/env python3
"""Qwen3.6 generation — Rust executor (all ops in Rust: forward + lm_head + top-k)."""
import ctypes, numpy as np, math, time, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
lib = get_lib()

HDIM = 2048

# Init Metal (for fused GQA, optional)
lib.lko_metal_init.argtypes = [ctypes.c_char_p]
lib.lko_metal_init.restype = ctypes.c_int32
METALLIB = str(Path(__file__).parent.parent / "target" / "objeta.metallib")
lib.lko_metal_init(METALLIB.encode())

# Init Rust runner
lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
lib.lko_runner_init.restype = ctypes.c_int32
BIN_DIR = str(Path(__file__).parent.parent / "models" / "qwen36_bin")
assert lib.lko_runner_init(BIN_DIR.encode(), 256), "Runner init failed"

# Set DeltaNet fusion ratio + MoE skip (SteeringBackbone defaults)
lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32

FUSION_RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.33
MOE_ON_DELTANET = int(sys.argv[2]) if len(sys.argv) > 2 else 0

lib.lko_runner_set_fusion_ratio(FUSION_RATIO)
lib.lko_runner_set_moe_on_deltanet(MOE_ON_DELTANET)
print(f"Strategy: ΔN={FUSION_RATIO:.0%} ({int(30*FUSION_RATIO)}/30 layers), MoE on ΔN={'yes' if MOE_ON_DELTANET else 'no'}")

# Warmup: touch q4 pages to bring them into OS page cache
lib.lko_runner_warmup.argtypes = [ctypes.c_int32]
lib.lko_runner_warmup.restype = ctypes.c_int32
print("Warming OS page cache...")
lib.lko_runner_warmup(100)

# C API: single step = forward + RMSNorm + lm_head + top-k
lib.lko_runner_step.argtypes = [
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # token_id, pos, seq_len
    ctypes.c_void_p,                                   # hn_out
    ctypes.c_int32,                                    # top_k
    ctypes.c_void_p, ctypes.c_void_p,                  # indices, values out
]
lib.lko_runner_step.restype = ctypes.c_int32

def rust_step(token_id, pos, seq_len, top_k=50):
    """One full step: forward 40 layers + RMSNorm + lm_head + top-k. All in Rust."""
    hn = np.zeros(HDIM, dtype=np.float32)
    indices = np.zeros(top_k, dtype=np.int32)
    values = np.zeros(top_k, dtype=np.float32)
    k = lib.lko_runner_step(token_id, pos, seq_len, hn.ctypes.data, top_k, indices.ctypes.data, values.ctypes.data)
    return hn, indices[:k], values[:k]

def sample(indices, values, temperature=0.7, top_k=40):
    """Python-side sampling from top-k logits."""
    k = min(len(indices), top_k)
    idx = indices[:k]
    val = values[:k]
    if temperature == 0:
        return int(idx[0])
    val = val / max(temperature, 0.01)
    val -= np.max(val)
    probs = np.exp(val); probs /= np.sum(probs)
    return int(idx[np.random.choice(len(probs), p=probs)])

def generate(prompt_ids, max_tokens=20, temperature=0.7, top_k=40):
    tokens = list(prompt_ids)
    n_prompt = len(tokens)
    print(f"Prefilling {n_prompt} tokens...")
    t0 = time.perf_counter()

    for i, tid in enumerate(tokens):
        _, _, _ = rust_step(tid, i, i+1, 10)
        if i % 5 == 0 or i == n_prompt-1:
            print(f"  [{i+1}/{n_prompt}] {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"  Prefill done in {time.perf_counter()-t0:.1f}s")

    # First token
    _, indices, values = rust_step(tokens[-1], n_prompt-1, n_prompt, max(50, top_k))
    if temperature == 0:
        next_token = int(indices[0])
    else:
        next_token = sample(indices, values, temperature, top_k)

    generated = []
    t_start = time.perf_counter()
    for step in range(max_tokens):
        generated.append(next_token)
        pos = n_prompt + step
        if next_token == 2: break

        _, indices, values = rust_step(next_token, pos, pos+1, max(50, top_k))
        if temperature == 0:
            next_token = int(indices[0])
        else:
            next_token = sample(indices, values, temperature, top_k)

        if step % 5 == 0 or step == max_tokens-1:
            e = time.perf_counter() - t_start
            print(f"  [{step+1}/{max_tokens}] {e:.0f}s ({ (step+1)/e:.2f} tok/s)" if e > 0 else f"  [{step+1}/{max_tokens}]", flush=True)

    total_s = time.perf_counter() - t_start
    n_gen = len(generated)
    print(f"\n  {n_gen} tokens in {total_s:.1f}s ({n_gen/total_s:.2f} tok/s)")
    return generated

if __name__ == "__main__":
    from transformers import AutoTokenizer
    snap = sorted(os.listdir(
        "/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"))[-1]
    tok = AutoTokenizer.from_pretrained(
        f"/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{snap}")
    print(f"Vocab: {tok.vocab_size}\n")

    for prompt in ["The meaning of life is"]:
        msgs = [{"role": "user", "content": prompt}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(chat)
        print(f"── Prompt: \"{prompt}\" ──")
        gen = generate(ids, max_tokens=15, temperature=0, top_k=0)
        text = tok.decode(gen, skip_special_tokens=True)
        print(f"  Output: {text}")
