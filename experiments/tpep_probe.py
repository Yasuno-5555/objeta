#!/usr/bin/env python3
"""Lightweight Topology-Preserving Expert Pruning probe.

Runs layer-local pruning experiments against the current prompt state using
existing HF/Rust comparison utilities. This is intended as a fast screening
tool before wiring pruning into the main executor.
"""
import argparse
import sys
from typing import List, Tuple
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.compare_moe_impl import (
    MODEL_PATH,
    TOP_K,
    chat_ids,
    cos,
    extract_selected_expert_weights,
    hf_capture_h_norm2,
    hf_router,
    hf_selected_experts,
    init_runner,
    rust_eval_layer_from_hidden,
    rust_prefill,
    rust_selected_expert_q4,
    rust_trace_components,
)


def build_policy_indices(
    mode: str,
    weights: np.ndarray,
    weighted_vecs: np.ndarray,
    value: float,
) -> Tuple[np.ndarray, str]:
    n = len(weights)
    order_by_weight = np.argsort(weights)[::-1]
    contrib_norms = np.array([float(np.linalg.norm(v)) for v in weighted_vecs], dtype=np.float32)
    total_contrib = float(np.sum(contrib_norms))
    order_by_contrib = np.argsort(contrib_norms)[::-1]

    if mode == "topk":
        k = max(1, min(n, int(value)))
        return order_by_weight[:k], f"topk={k}"

    if mode == "topp":
        target = float(value)
        chosen: List[int] = []
        cum = 0.0
        for idx in order_by_weight:
            chosen.append(int(idx))
            cum += float(weights[idx])
            if cum >= target and len(chosen) >= 1:
                break
        return np.array(chosen, dtype=np.int64), f"topp={target:.2f}"

    if mode == "contrib":
        threshold = float(value)
        chosen = [
            int(idx)
            for idx in order_by_contrib
            if total_contrib <= 1e-12 or (contrib_norms[idx] / total_contrib) >= threshold
        ]
        if not chosen:
            chosen = [int(order_by_contrib[0])]
        return np.array(chosen, dtype=np.int64), f"contrib>={threshold:.3f}"

    raise ValueError(f"unknown mode: {mode}")


def renorm_weights(weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    kept = weights[indices].astype(np.float32).copy()
    s = float(np.sum(kept))
    if s <= 1e-12:
        kept[:] = 1.0 / max(len(kept), 1)
    else:
        kept /= s
    return kept


def run_probe(model, ids, target_layer: int, mode: str, values: List[float], use_q4: bool):
    layer = model.model.layers[target_layer]

    lib = init_runner()
    rust_prefill(lib, ids)
    rust_comp = rust_trace_components(lib, ids, target_layer)

    hf_h_norm2 = hf_capture_h_norm2(model, ids, target_layer)
    _, hf_idx, hf_w, _ = hf_router(layer, hf_h_norm2)
    expert_ids = hf_idx.astype(np.int32)
    routing_weights = hf_w.astype(np.float32)

    gate_w, up_w, down_w = extract_selected_expert_weights(layer, expert_ids)
    _, _, _, fp_expert_hfinput, fp_weighted_hfinput, _ = hf_selected_experts(
        gate_w, up_w, down_w, rust_comp["h_norm2"].astype(np.float32), routing_weights
    )

    q4_expert, q4_weighted, _ = rust_selected_expert_q4(
        lib, target_layer, rust_comp["h_norm2"].astype(np.float32), expert_ids, routing_weights
    )

    full_shared = rust_comp["shared"].astype(np.float32)
    full_routed = rust_comp["moe"].astype(np.float32) if use_q4 else fp_weighted_hfinput.sum(axis=0).astype(np.float32)
    full_final = full_shared + full_routed

    next_fp = None
    next_layer = None
    if target_layer < len(model.model.layers) - 1:
        next_layer = model.model.layers[target_layer + 1]
        lib_next_fp = init_runner()
        rust_prefill(lib_next_fp, ids)
        next_fp = rust_eval_layer_from_hidden(lib_next_fp, target_layer + 1, ids, full_final)
        baseline_router_w = next_layer.mlp.gate.weight.detach().float().cpu().numpy()
        baseline_logits = baseline_router_w @ next_fp["h_norm2"]
        baseline_idx = np.argsort(baseline_logits)[::-1][:TOP_K]
    else:
        baseline_idx = None

    print(f"\n=== TPEP Probe Layer {target_layer} ({'q4' if use_q4 else 'fp'}) ===")
    print("policy | kept | mass | final_moe_cos | next_hidden_cos | next_router_overlap")

    for value in values:
        keep_idx, label = build_policy_indices(
            mode,
            routing_weights,
            q4_weighted if use_q4 else fp_weighted_hfinput,
            value,
        )
        kept_weights = renorm_weights(routing_weights, keep_idx)

        if use_q4:
            kept_routed = np.sum(q4_weighted[keep_idx] * (kept_weights / routing_weights[keep_idx])[:, None], axis=0)
        else:
            kept_routed = np.sum(fp_weighted_hfinput[keep_idx] * (kept_weights / routing_weights[keep_idx])[:, None], axis=0)

        pruned_final = full_shared + kept_routed.astype(np.float32)
        final_moe_cos = cos(full_final, pruned_final)
        mass = float(np.sum(routing_weights[keep_idx]))

        if next_fp is not None:
            lib_next = init_runner()
            rust_prefill(lib_next, ids)
            next_eval = rust_eval_layer_from_hidden(lib_next, target_layer + 1, ids, pruned_final.astype(np.float32))
            next_hidden_cos = cos(next_fp["after_mlp"], next_eval["after_mlp"])
            router_w = next_layer.mlp.gate.weight.detach().float().cpu().numpy()
            logits = router_w @ next_eval["h_norm2"]
            idx = np.argsort(logits)[::-1][:TOP_K]
            overlap = len(set(map(int, baseline_idx)) & set(map(int, idx)))
            next_hidden_str = f"{next_hidden_cos:.6f}"
            overlap_str = f"{overlap}/{TOP_K}"
        else:
            next_hidden_str = "n/a"
            overlap_str = "n/a"

        print(
            f"{label:16} | {len(keep_idx):4d} | {mass:4.2f} | "
            f"{final_moe_cos:.6f} | {next_hidden_str:>15} | {overlap_str}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", nargs="?", default="0,3,7")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--mode", choices=["topk", "topp", "contrib"], default="topk")
    parser.add_argument("--values", default=None, help="Comma-separated policy values")
    parser.add_argument("--q4", action="store_true", help="Use q4 routed expert outputs instead of fp routed outputs")
    args = parser.parse_args()

    if args.values is None:
        if args.mode == "topk":
            values = [8, 6, 4, 2]
        elif args.mode == "topp":
            values = [0.95, 0.90, 0.85, 0.80]
        else:
            values = [0.01, 0.02, 0.05]
    else:
        values = [float(x) for x in args.values.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = chat_ids(tok, args.prompt)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    model.eval()

    for layer in [int(x) for x in args.layers.split(",")]:
        run_probe(model, ids, layer, args.mode, values, args.q4)


if __name__ == "__main__":
    main()
