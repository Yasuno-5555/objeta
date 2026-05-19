#!/usr/bin/env python3
"""Compare Rust prefill next-token logits against HF for a short prompt."""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.qwen36_executor import get_lib

BIN = PROJECT / "models" / "qwen36_bin"
SNAP_ROOT = Path("/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots")
SNAPSHOT = str(sorted(os.listdir(SNAP_ROOT))[-1])
MODEL_PATH = str(SNAP_ROOT / SNAPSHOT)
HDIM = 2048


def init_runner():
    lib = get_lib()
    assert lib is not None, "Rust library not found"
    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
    lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
    lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
    lib.lko_runner_step.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_step.restype = ctypes.c_int32
    assert lib.lko_runner_init(str(BIN).encode(), 256), "runner init failed"
    lib.lko_runner_set_fusion_ratio(1.0)
    lib.lko_runner_set_moe_on_deltanet(1)
    return lib


def rust_prefill(lib, token_ids, top_k=10):
    hn = np.zeros(HDIM, dtype=np.float32)
    idx = np.zeros(max(top_k, 64), dtype=np.int32)
    val = np.zeros(max(top_k, 64), dtype=np.float32)
    for pos, tid in enumerate(token_ids):
        lib.lko_runner_step(tid, pos, pos + 1, hn.ctypes.data, len(idx), idx.ctypes.data, val.ctypes.data)
    order = np.argsort(val)[::-1]
    return hn.copy(), idx[order][:top_k].copy(), val[order][:top_k].copy()


def hf_prefill(model, token_ids, top_k=10):
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0, -1].float().cpu()
    vals, idx = torch.topk(logits, k=top_k)
    return logits.numpy(), idx.numpy(), vals.numpy()


def decode_tokens(tok, ids):
    return [tok.decode([int(i)]) for i in ids]


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    lib = init_runner()

    msgs = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat)
    print(f"Prompt: {prompt}")
    print(f"Chat length: {len(ids)} tokens")
    print(f"Token ids: {ids}")

    rust_hn, rust_idx, rust_val = rust_prefill(lib, ids, top_k=10)
    hf_logits, hf_idx, hf_val = hf_prefill(model, ids, top_k=10)

    embed = np.memmap(BIN / "embed_tokens.bin", dtype=np.float32, mode="r").reshape(-1, HDIM)
    rust_logits_top = rust_val
    hf_hn = None
    with torch.no_grad():
        hidden = model.model(input_ids=torch.tensor([ids], dtype=torch.long), output_hidden_states=False)[0][0, -1].float()
        norm_w = model.model.norm.weight.float()
        hf_hn = F.rms_norm(hidden, (HDIM,), norm_w, 1e-6).cpu().numpy()

    cos_hn = float(np.dot(rust_hn, hf_hn) / (np.linalg.norm(rust_hn) * np.linalg.norm(hf_hn) + 1e-12))
    print(f"hn cosine: {cos_hn:.6f}")
    print("\nRust top-10:")
    for i, v in zip(rust_idx, rust_logits_top):
        print(f"  {int(i):>6}  {v:>10.4f}  {tok.decode([int(i)])!r}")
    print("\nHF top-10:")
    for i, v in zip(hf_idx, hf_val):
        print(f"  {int(i):>6}  {float(v):>10.4f}  {tok.decode([int(i)])!r}")

    overlap = set(map(int, rust_idx)) & set(map(int, hf_idx))
    print(f"\nTop-10 overlap: {len(overlap)}/10")


if __name__ == "__main__":
    main()
