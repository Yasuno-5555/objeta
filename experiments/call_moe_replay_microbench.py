#!/usr/bin/env python3
"""Replay call_moe microbench using actual E2E hidden states from prompt prefill."""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.qwen36_executor import get_lib

BIN = PROJECT / "models" / "qwen36_bin"
HDIM = 2048
TOP_K = 8


def init_lib():
    lib = get_lib()
    assert lib is not None, "Rust library not found"
    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
    lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
    lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
    lib.lko_runner_reset_moe_stats.argtypes = []
    lib.lko_runner_reset_moe_stats.restype = ctypes.c_int32
    lib.lko_runner_build_caches.argtypes = [ctypes.c_int32]
    lib.lko_runner_build_caches.restype = ctypes.c_int32
    lib.lko_runner_step.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_step.restype = ctypes.c_int32
    lib.lko_runner_trace_layer_components.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_trace_layer_components.restype = ctypes.c_int32
    lib.lko_runner_selected_expert_q4_path.argtypes = [
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.lko_runner_selected_expert_q4_path.restype = ctypes.c_int32
    lib.lko_runner_get_moe_stats_json.argtypes = []
    lib.lko_runner_get_moe_stats_json.restype = ctypes.c_void_p

    assert lib.lko_runner_init(str(BIN).encode(), 256), "runner init failed"
    return lib


def load_stats(lib):
    ptr = lib.lko_runner_get_moe_stats_json()
    if not ptr:
        return {}
    raw = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
    return json.loads(raw)


def prefill(lib, ids):
    hn = np.zeros(HDIM, dtype=np.float32)
    idx = np.zeros(64, dtype=np.int32)
    val = np.zeros(64, dtype=np.float32)
    for pos, tid in enumerate(ids[:-1]):
        ret = lib.lko_runner_step(
            int(tid), pos, pos + 1, hn.ctypes.data, len(idx), idx.ctypes.data, val.ctypes.data
        )
        assert ret > 0, f"prefill step failed at pos={pos}: {ret}"


def trace_h_and_router(lib, ids, layer):
    last_pos = len(ids) - 1
    last_tid = int(ids[last_pos])
    after_attn = np.zeros(HDIM, dtype=np.float32)
    h_norm2 = np.zeros(HDIM, dtype=np.float32)
    shared = np.zeros(HDIM, dtype=np.float32)
    moe = np.zeros(HDIM, dtype=np.float32)
    after_mlp = np.zeros(HDIM, dtype=np.float32)
    ret = lib.lko_runner_trace_layer_components(
        last_tid, last_pos, last_pos + 1, layer,
        after_attn.ctypes.data, h_norm2.ctypes.data, shared.ctypes.data, moe.ctypes.data, after_mlp.ctypes.data,
    )
    assert ret == HDIM, f"trace_layer_components failed for layer {layer}: {ret}"
    stats = load_stats(lib)
    layer_stats = next((x for x in stats.get("layers", []) if x.get("layer") == layer), {})
    top_idx = np.asarray(layer_stats.get("last_selected_ids", []), dtype=np.int32)
    top_w = np.asarray(layer_stats.get("last_selected_weights", []), dtype=np.float32)
    entropy = 0.0
    if len(top_w):
        clipped = np.clip(top_w, 1e-10, None)
        entropy = float(-(clipped * np.log(clipped)).sum())
    assert len(top_idx) > 0, f"no selected experts captured for layer {layer}"
    return h_norm2, top_idx, top_w, entropy


def bench_selected(lib, layer, h_norm2, expert_ids, expert_weights, use_fused, down_mode_kind, chunk_rows, iters):
    lib.lko_runner_reset_moe_stats()
    lib.lko_runner_build_caches(0)
    out = np.zeros(HDIM, dtype=np.float32)

    for _ in range(3):
        ret = lib.lko_runner_selected_expert_q4_path(
            layer,
            h_norm2.ctypes.data,
            expert_ids.ctypes.data,
            expert_weights.ctypes.data,
            len(expert_ids),
            int(use_fused),
            down_mode_kind,
            chunk_rows,
            out.ctypes.data,
        )
        assert ret == len(expert_ids), f"selected_expert_q4_path warmup failed: {ret}"

    t0 = time.perf_counter()
    for _ in range(iters):
        ret = lib.lko_runner_selected_expert_q4_path(
            layer,
            h_norm2.ctypes.data,
            expert_ids.ctypes.data,
            expert_weights.ctypes.data,
            len(expert_ids),
            int(use_fused),
            down_mode_kind,
            chunk_rows,
            out.ctypes.data,
        )
        assert ret == len(expert_ids), f"selected_expert_q4_path failed: {ret}"
    external_ms = (time.perf_counter() - t0) * 1000.0 / iters

    stats = load_stats(lib)
    layer_stats = next((x for x in stats.get("layers", []) if x.get("layer") == layer), {})
    return external_ms, layer_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--strategy", default="configs/safe_exact.json")
    ap.add_argument("--layers", default="0,7,31")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    strategy = json.loads(Path(args.strategy).read_text())
    fusion_ratio = float(strategy.get("fusion_ratio", 0.80))
    moe_on_deltanet = bool(strategy.get("moe_on_deltanet", True))

    lib = init_lib()
    lib.lko_runner_set_fusion_ratio(fusion_ratio)
    lib.lko_runner_set_moe_on_deltanet(1 if moe_on_deltanet else 0)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
    chat = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok.encode(chat)

    prefill(lib, ids)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    ts = time.strftime("%Y%m%d_%H%M%S")
    replay_dir = PROJECT / "runs" / f"replay_microbench_{ts}"
    replay_dir.mkdir(parents=True, exist_ok=True)

    print("# call_moe replay microbench (actual E2E hidden states)")
    print(f"prompt={args.prompt!r}")
    print(f"layers={layers}, iters={args.iters}, cache=off")
    print()
    print("| layer | selected | entropy_top8 | path | external_ms | routed_exec_wall_ms | call_moe_wall_ms |")
    print("|---|---:|---:|---|---:|---:|---:|")

    artifact = {"prompt": args.prompt, "layers": {}}
    for layer in layers:
        lib.lko_runner_reset_moe_stats()
        h_norm2, top_idx, top_w, entropy = trace_h_and_router(lib, ids, layer)
        artifact["layers"][str(layer)] = {
            "h_norm2": h_norm2.tolist(),
            "expert_ids": top_idx.tolist(),
            "expert_weights": top_w.tolist(),
            "routing_entropy_top8": entropy,
        }
        for label, use_fused, mode_kind, chunk_rows in (
            ("legacy", False, 1, 1),
            ("fused_row_parallel", True, 1, 1),
        ):
            external_ms, ls = bench_selected(
                lib, layer, h_norm2, top_idx.astype(np.int32), top_w.astype(np.float32),
                use_fused, mode_kind, chunk_rows, args.iters,
            )
            print(
                f"| {layer} | {len(top_idx)} | {entropy:.3f} | {label} | "
                f"{external_ms:.3f} | "
                f"{float(ls.get('avg_routed_exec_wall_ms', 0.0)):.3f} | "
                f"{float(ls.get('avg_call_moe_wall_ms', 0.0)):.3f} |"
            )

    with open(replay_dir / "replay_hidden_artifact.json", "w") as f:
        json.dump(artifact, f)
    print()
    print(f"Saved replay artifact to {replay_dir / 'replay_hidden_artifact.json'}")


if __name__ == "__main__":
    main()
