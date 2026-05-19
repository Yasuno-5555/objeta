#!/usr/bin/env python3
"""Compare routed MoE router parity between HF and Rust for one target layer."""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.qwen36_executor import get_lib

BIN = PROJECT / "models" / "qwen36_bin"
SNAP_ROOT = Path("/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots")
SNAPSHOT = str(sorted(os.listdir(SNAP_ROOT))[-1])
MODEL_PATH = str(SNAP_ROOT / SNAPSHOT)
HDIM = 2048
TOP_K = 8


def init_runner():
    lib = get_lib()
    assert lib is not None, "Rust library not found"
    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
    lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
    lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
    lib.lko_runner_set_moe_enabled.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_enabled.restype = ctypes.c_int32
    lib.lko_runner_step.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_step.restype = ctypes.c_int32
    lib.lko_runner_trace_router.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_trace_router.restype = ctypes.c_int32

    assert lib.lko_runner_init(str(BIN).encode(), 256), "runner init failed"
    lib.lko_runner_set_fusion_ratio(1.0)
    lib.lko_runner_set_moe_on_deltanet(1)
    lib.lko_runner_set_moe_enabled(1)
    return lib


def apply_chat_template(tok, prompt: str):
    msgs = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(chat)


def prefill_rust(lib, ids):
    hn = np.zeros(HDIM, dtype=np.float32)
    idx = np.zeros(64, dtype=np.int32)
    val = np.zeros(64, dtype=np.float32)
    for pos, tid in enumerate(ids[:-1]):
        lib.lko_runner_step(tid, pos, pos + 1, hn.ctypes.data, len(idx), idx.ctypes.data, val.ctypes.data)


def trace_router_rust(lib, ids, target_layer):
    last_pos = len(ids) - 1
    last_tid = ids[last_pos]
    logits = np.zeros(256, dtype=np.float32)
    top_idx = np.zeros(TOP_K, dtype=np.int32)
    top_w = np.zeros(TOP_K, dtype=np.float32)
    entropy = ctypes.c_float(0.0)
    n = lib.lko_runner_trace_router(
        last_tid, last_pos, last_pos + 1, target_layer, TOP_K,
        logits.ctypes.data, top_idx.ctypes.data, top_w.ctypes.data, ctypes.byref(entropy)
    )
    assert n > 0, f"trace_router failed: {n}"
    return logits[:n].copy(), top_idx.copy(), top_w.copy(), float(entropy.value)


def trace_router_hf(model, ids, target_layer):
    layer = model.model.layers[target_layer]
    captured = {}

    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        captured["h_norm2"] = t[0, -1].detach().float().cpu()

    handle = layer.post_attention_layernorm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], dtype=torch.long))
    finally:
        handle.remove()

    assert "h_norm2" in captured, "failed to capture HF h_norm2"
    h_norm2 = captured["h_norm2"]
    router_w = layer.mlp.gate.weight.detach().float().cpu()
    logits = torch.mv(router_w, h_norm2)
    vals, idx = torch.topk(logits, k=TOP_K)
    weights = torch.softmax(vals, dim=0)
    entropy = float(-(weights * torch.log(weights.clamp_min(1e-10))).sum().item())
    return logits.numpy(), idx.numpy(), weights.numpy(), entropy


def compare_one(model, ids, target_layer):
    lib = init_runner()
    prefill_rust(lib, ids)

    rust_logits, rust_idx, rust_w, rust_ent = trace_router_rust(lib, ids, target_layer)
    hf_logits, hf_idx, hf_w, hf_ent = trace_router_hf(model, ids, target_layer)

    cos = float(np.dot(rust_logits, hf_logits) / (np.linalg.norm(rust_logits) * np.linalg.norm(hf_logits) + 1e-12))
    overlap = len(set(map(int, rust_idx)) & set(map(int, hf_idx)))
    rust_map = {int(i): float(w) for i, w in zip(rust_idx, rust_w)}
    hf_map = {int(i): float(w) for i, w in zip(hf_idx, hf_w)}
    shared_ids = sorted(set(rust_map) | set(hf_map))
    max_abs = max(abs(rust_map.get(i, 0.0) - hf_map.get(i, 0.0)) for i in shared_ids)
    ent_diff = abs(rust_ent - hf_ent)

    print(f"\n=== Layer {target_layer} ===")
    print(f"router logits cos: {cos:.6f}")
    print(f"top-k overlap: {overlap}/{TOP_K}")
    print(f"gate weight max_abs: {max_abs:.6e}")
    print(f"entropy diff: {ent_diff:.6e}")
    print(f"rust entropy: {rust_ent:.6f}")
    print(f"hf entropy:   {hf_ent:.6f}")

    print("\nRust top-k:")
    for i, w in zip(rust_idx, rust_w):
        print(f"  eid={int(i):3d} weight={float(w):.6f}")

    print("\nHF top-k:")
    for i, w in zip(hf_idx, hf_w):
        print(f"  eid={int(i):3d} weight={float(w):.6f}")


def main():
    layer_arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    target_layers = [int(x) for x in layer_arg.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = apply_chat_template(tok, prompt)
    print(f"Target layers: {target_layers}")
    print(f"Prompt: {prompt}")
    print(f"Chat length: {len(ids)} tokens")

    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    model.eval()
    for target_layer in target_layers:
        print("\nInitializing Rust runner...")
        compare_one(model, ids, target_layer)


if __name__ == "__main__":
    main()
