#!/usr/bin/env python3
"""Step 2A/2B: isolate MoE implementation and q4 drift with optional lightweight modes."""
import argparse
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
MID = 512
TOP_K = 8


def cos(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def norm_ratio(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return na / max(nb, 1e-12)


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
    lib.lko_runner_trace_layer_components.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.lko_runner_trace_layer_components.restype = ctypes.c_int32
    lib.lko_runner_trace_router.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_trace_router.restype = ctypes.c_int32
    lib.lko_moe_dense_selected.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_moe_dense_selected.restype = ctypes.c_int32
    lib.lko_runner_eval_layer_from_hidden.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_eval_layer_from_hidden.restype = ctypes.c_int32
    lib.lko_runner_selected_expert_q4.argtypes = [
        ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lko_runner_selected_expert_q4.restype = ctypes.c_int32

    assert lib.lko_runner_init(str(BIN).encode(), 256), "runner init failed"
    lib.lko_runner_set_fusion_ratio(1.0)
    lib.lko_runner_set_moe_on_deltanet(1)
    lib.lko_runner_set_moe_enabled(1)
    return lib


def chat_ids(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(chat)


def rust_prefill(lib, ids):
    hn = np.zeros(HDIM, dtype=np.float32)
    idx = np.zeros(64, dtype=np.int32)
    val = np.zeros(64, dtype=np.float32)
    for pos, tid in enumerate(ids[:-1]):
        lib.lko_runner_step(tid, pos, pos + 1, hn.ctypes.data, len(idx), idx.ctypes.data, val.ctypes.data)


def rust_trace_components(lib, ids, target_layer):
    last_pos = len(ids) - 1
    last_tid = ids[last_pos]
    after_attn = np.zeros(HDIM, dtype=np.float32)
    h_norm2 = np.zeros(HDIM, dtype=np.float32)
    shared = np.zeros(HDIM, dtype=np.float32)
    moe = np.zeros(HDIM, dtype=np.float32)
    after_mlp = np.zeros(HDIM, dtype=np.float32)
    ret = lib.lko_runner_trace_layer_components(
        last_tid, last_pos, last_pos + 1, target_layer,
        after_attn.ctypes.data, h_norm2.ctypes.data, shared.ctypes.data,
        moe.ctypes.data, after_mlp.ctypes.data
    )
    assert ret == HDIM, f"trace_layer_components failed: {ret}"
    return {
        "after_attn": after_attn,
        "h_norm2": h_norm2,
        "shared": shared,
        "moe": moe,
        "after_mlp": after_mlp,
    }


def rust_trace_router(lib, ids, target_layer):
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
    return logits[:n], top_idx, top_w, float(entropy.value)


def hf_capture_h_norm2(model, ids, target_layer):
    layer = model.model.layers[target_layer]
    captured = {}

    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        captured["h_norm2"] = t[0, -1].detach().float().cpu().numpy()

    handle = layer.post_attention_layernorm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], dtype=torch.long))
    finally:
        handle.remove()
    return captured["h_norm2"]


def hf_router(layer, h_norm2):
    router_w = layer.mlp.gate.weight.detach().float().cpu().numpy()
    logits = router_w @ h_norm2
    idx = np.argsort(logits)[::-1][:TOP_K]
    vals = logits[idx]
    ex = np.exp(vals - np.max(vals))
    w = ex / np.sum(ex)
    entropy = float(-np.sum(w * np.log(np.clip(w, 1e-10, None))))
    return logits, idx.astype(np.int32), w.astype(np.float32), entropy


def shared_manual(layer, x):
    se = layer.mlp.shared_expert
    gate = se.gate_proj.weight.detach().float().cpu().numpy() @ x
    up = se.up_proj.weight.detach().float().cpu().numpy() @ x
    hidden = (gate / (1.0 + np.exp(-gate))) * up
    se_out = se.down_proj.weight.detach().float().cpu().numpy() @ hidden
    se_gate_w = layer.mlp.shared_expert_gate.weight.detach().float().cpu().numpy().reshape(-1)
    gate_scalar = 1.0 / (1.0 + np.exp(-float(np.dot(se_gate_w, x))))
    return {
        "shared_expert": se_out,
        "shared_gate": gate_scalar,
        "shared_contrib": se_out * gate_scalar,
    }


def extract_selected_expert_weights(layer, expert_ids):
    gate_w = np.zeros((len(expert_ids), MID, HDIM), dtype=np.float32)
    up_w = np.zeros((len(expert_ids), MID, HDIM), dtype=np.float32)
    down_w = np.zeros((len(expert_ids), HDIM, MID), dtype=np.float32)
    gate_up_all = layer.mlp.experts.gate_up_proj.detach().float().cpu().numpy()
    down_all = layer.mlp.experts.down_proj.detach().float().cpu().numpy()
    for i, eid in enumerate(expert_ids):
        gate_w[i] = gate_up_all[int(eid), :MID, :]
        up_w[i] = gate_up_all[int(eid), MID:, :]
        down_w[i] = down_all[int(eid)]
    return gate_w, up_w, down_w


def hf_selected_experts(gate_w, up_w, down_w, x, routing_weights):
    n = len(routing_weights)
    gate = np.zeros((n, MID), dtype=np.float32)
    up = np.zeros((n, MID), dtype=np.float32)
    hidden = np.zeros((n, MID), dtype=np.float32)
    expert = np.zeros((n, HDIM), dtype=np.float32)
    weighted = np.zeros((n, HDIM), dtype=np.float32)
    for i in range(n):
        gate[i] = gate_w[i] @ x
        up[i] = up_w[i] @ x
        hidden[i] = (gate[i] / (1.0 + np.exp(-gate[i]))) * up[i]
        expert[i] = down_w[i] @ hidden[i]
        weighted[i] = expert[i] * routing_weights[i]
    routed_sum = weighted.sum(axis=0)
    return gate, up, hidden, expert, weighted, routed_sum


def rust_dense_selected(lib, gate_w, up_w, down_w, x, routing_weights):
    n = len(routing_weights)
    gate = np.zeros((n, MID), dtype=np.float32)
    up = np.zeros((n, MID), dtype=np.float32)
    hidden = np.zeros((n, MID), dtype=np.float32)
    expert = np.zeros((n, HDIM), dtype=np.float32)
    weighted = np.zeros((n, HDIM), dtype=np.float32)
    routed_sum = np.zeros(HDIM, dtype=np.float32)
    ret = lib.lko_moe_dense_selected(
        x.ctypes.data,
        gate_w.ctypes.data, up_w.ctypes.data, down_w.ctypes.data,
        routing_weights.ctypes.data,
        n,
        gate.ctypes.data, up.ctypes.data, hidden.ctypes.data,
        expert.ctypes.data, weighted.ctypes.data, routed_sum.ctypes.data,
    )
    assert ret == n, f"lko_moe_dense_selected failed: {ret}"
    return gate, up, hidden, expert, weighted, routed_sum


def rust_eval_layer_from_hidden(lib, layer_idx, ids, h_in):
    pos = len(ids) - 1
    after_attn = np.zeros(HDIM, dtype=np.float32)
    h_norm2 = np.zeros(HDIM, dtype=np.float32)
    shared = np.zeros(HDIM, dtype=np.float32)
    moe = np.zeros(HDIM, dtype=np.float32)
    after_mlp = np.zeros(HDIM, dtype=np.float32)
    ret = lib.lko_runner_eval_layer_from_hidden(
        layer_idx, pos, pos + 1,
        h_in.ctypes.data,
        after_attn.ctypes.data, h_norm2.ctypes.data, shared.ctypes.data, moe.ctypes.data, after_mlp.ctypes.data
    )
    assert ret == HDIM, f"lko_runner_eval_layer_from_hidden failed: {ret}"
    return {
        "after_attn": after_attn,
        "h_norm2": h_norm2,
        "shared": shared,
        "moe": moe,
        "after_mlp": after_mlp,
    }


def rust_selected_expert_q4(lib, layer_idx, x, expert_ids, routing_weights):
    n = len(expert_ids)
    expert = np.zeros((n, HDIM), dtype=np.float32)
    weighted = np.zeros((n, HDIM), dtype=np.float32)
    routed_sum = np.zeros(HDIM, dtype=np.float32)
    ret = lib.lko_runner_selected_expert_q4(
        layer_idx,
        x.ctypes.data,
        expert_ids.ctypes.data,
        routing_weights.ctypes.data,
        n,
        expert.ctypes.data,
        weighted.ctypes.data,
        routed_sum.ctypes.data,
    )
    assert ret == n, f"lko_runner_selected_expert_q4 failed: {ret}"
    return expert, weighted, routed_sum


def print_expert_table(expert_ids, routing_weights, hf_expert, rust_expert, hf_weighted, rust_weighted):
    print("\nexpert_id | gate_weight | raw_cos  | weighted_cos | contribution_norm")
    for i, eid in enumerate(expert_ids):
        raw_cos = cos(hf_expert[i], rust_expert[i])
        weighted_cos = cos(hf_weighted[i], rust_weighted[i])
        contrib_norm = float(np.linalg.norm(hf_weighted[i]))
        print(f"{int(eid):8d} | {float(routing_weights[i]):11.6f} | {raw_cos:7.6f} | {weighted_cos:12.6f} | {contrib_norm:16.6f}")


def print_q4_expert_table(expert_ids, routing_weights, fp_expert, q4_expert, fp_weighted, q4_weighted):
    print("\nexpert_id | gate_weight | fp_norm   | q4_norm   | raw_cos  | weighted_cos | contribution_ratio")
    for i, eid in enumerate(expert_ids):
        fp_norm = float(np.linalg.norm(fp_expert[i]))
        q4_norm = float(np.linalg.norm(q4_expert[i]))
        raw_cos = cos(fp_expert[i], q4_expert[i])
        weighted_cos = cos(fp_weighted[i], q4_weighted[i])
        contrib_ratio = float(np.linalg.norm(fp_weighted[i]) / max(np.linalg.norm(fp_weighted.reshape(-1)), 1e-12))
        print(f"{int(eid):8d} | {float(routing_weights[i]):11.6f} | {fp_norm:8.6f} | {q4_norm:8.6f} | {raw_cos:7.6f} | {weighted_cos:12.6f} | {contrib_ratio:17.6f}")


def topk_overlap_from_hidden(layer, h_norm2_a, h_norm2_b):
    router_w = layer.mlp.gate.weight.detach().float().cpu().numpy()
    logits_a = router_w @ h_norm2_a
    logits_b = router_w @ h_norm2_b
    idx_a = np.argsort(logits_a)[::-1][:TOP_K]
    idx_b = np.argsort(logits_b)[::-1][:TOP_K]
    return len(set(map(int, idx_a)) & set(map(int, idx_b))), idx_a, idx_b


def compare_layer(model, ids, target_layer, *, per_expert_router_limit=0, skip_next_layer=False):
    layer = model.model.layers[target_layer]
    lib_router = init_runner()
    rust_prefill(lib_router, ids)
    rust_router_logits, rust_idx, rust_w, rust_ent = rust_trace_router(lib_router, ids, target_layer)

    lib_comp = init_runner()
    rust_prefill(lib_comp, ids)
    rust_comp = rust_trace_components(lib_comp, ids, target_layer)

    hf_h_norm2 = hf_capture_h_norm2(model, ids, target_layer)
    hf_router_logits, hf_idx, hf_w, hf_ent = hf_router(layer, hf_h_norm2)

    print(f"\n=== Layer {target_layer} ===")
    print(f"expert_input cos (HF vs Rust): {cos(hf_h_norm2, rust_comp['h_norm2']):.6f}")
    print(f"router logits cos:            {cos(hf_router_logits, rust_router_logits):.6f}")
    print(f"router top-k overlap:         {len(set(map(int, hf_idx)) & set(map(int, rust_idx)))}/{TOP_K}")
    print(f"router entropy diff:          {abs(hf_ent - rust_ent):.6e}")

    expert_ids = hf_idx
    routing_weights = hf_w.astype(np.float32)
    gate_w, up_w, down_w = extract_selected_expert_weights(layer, expert_ids)

    hf_gate, hf_up, hf_hidden, hf_expert, hf_weighted, hf_routed = hf_selected_experts(
        gate_w, up_w, down_w, hf_h_norm2, routing_weights
    )
    rust_gate, rust_up, rust_hidden, rust_expert, rust_weighted, rust_routed = rust_dense_selected(
        lib_comp, gate_w, up_w, down_w, hf_h_norm2.astype(np.float32), routing_weights
    )

    rust_gate_on_rust, rust_up_on_rust, rust_hidden_on_rust, rust_expert_on_rust, rust_weighted_on_rust, rust_routed_on_rust = rust_dense_selected(
        lib_comp, gate_w, up_w, down_w, rust_comp["h_norm2"].astype(np.float32), routing_weights
    )
    q4_expert_on_rust, q4_weighted_on_rust, q4_routed_on_rust = rust_selected_expert_q4(
        lib_comp, target_layer, rust_comp["h_norm2"].astype(np.float32), expert_ids.astype(np.int32), routing_weights
    )

    hf_shared = shared_manual(layer, hf_h_norm2)
    rust_shared_fp = shared_manual(layer, rust_comp["h_norm2"])
    q4_final = rust_comp["shared"] + rust_comp["moe"]
    fp_final_rustinput = rust_shared_fp["shared_contrib"] + rust_routed_on_rust

    print("\n[Step 2A / common input = HF h_norm2]")
    print(f"gate_proj cos:                {cos(hf_gate.reshape(-1), rust_gate.reshape(-1)):.6f}")
    print(f"up_proj cos:                  {cos(hf_up.reshape(-1), rust_up.reshape(-1)):.6f}")
    print(f"silu(gate)*up cos:            {cos(hf_hidden.reshape(-1), rust_hidden.reshape(-1)):.6f}")
    print(f"expert raw cos:               {cos(hf_expert.reshape(-1), rust_expert.reshape(-1)):.6f}")
    print(f"weighted expert cos:          {cos(hf_weighted.reshape(-1), rust_weighted.reshape(-1)):.6f}")
    print(f"routed_sum cos:               {cos(hf_routed, rust_routed):.6f}")
    print(f"shared_expert cos:            {cos(hf_shared['shared_expert'], rust_shared_fp['shared_expert']):.6f}")
    print(f"shared_gate abs diff:         {abs(hf_shared['shared_gate'] - rust_shared_fp['shared_gate']):.6e}")
    print(f"shared contribution cos:      {cos(hf_shared['shared_contrib'], rust_shared_fp['shared_contrib']):.6f}")
    print(f"final MoE output cos:         {cos(hf_shared['shared_contrib'] + hf_routed, rust_shared_fp['shared_contrib'] + rust_routed):.6f}")

    print_expert_table(expert_ids, routing_weights, hf_expert, rust_expert, hf_weighted, rust_weighted)

    print("\n[Input drift sensitivity: same fp weights, Rust h_norm2]")
    print(f"routed_sum cos (HF input vs Rust input): {cos(hf_routed, rust_routed_on_rust):.6f}")
    print(f"final MoE cos (HF input vs Rust input):  {cos(hf_shared['shared_contrib'] + hf_routed, fp_final_rustinput):.6f}")

    print("\n[Shared vs Routed]")
    shared_norm = float(np.linalg.norm(rust_shared_fp["shared_contrib"]))
    routed_norm = float(np.linalg.norm(rust_routed_on_rust))
    final_norm = float(np.linalg.norm(rust_shared_fp["shared_contrib"] + rust_routed_on_rust))
    print(f"||shared||:                   {shared_norm:.6f}")
    print(f"||routed_sum||:               {routed_norm:.6f}")
    print(f"||shared + routed||:          {final_norm:.6f}")
    print(f"cos(shared, routed_sum):      {cos(rust_shared_fp['shared_contrib'], rust_routed_on_rust):.6f}")

    print("\n[Step 2B preview: q4 path impact]")
    print(f"Rust traced shared cos vs fp shared@Rust input: {cos(rust_comp['shared'], rust_shared_fp['shared_contrib']):.6f}")
    print(f"Rust traced routed(q4) cos vs fp routed@Rust input: {cos(rust_comp['moe'], rust_routed_on_rust):.6f}")
    print(f"Rust traced routed(q4) norm ratio:                {norm_ratio(rust_comp['moe'], rust_routed_on_rust):.6f}")
    print(f"Rust traced final(q4) cos vs fp final@Rust input: {cos(q4_final, fp_final_rustinput):.6f}")

    print_q4_expert_table(expert_ids, routing_weights, rust_expert_on_rust, q4_expert_on_rust, rust_weighted_on_rust, q4_weighted_on_rust)
    print("\nexpert_id | cumulative routed cos after adding this expert")
    cum_fp = np.zeros(HDIM, dtype=np.float32)
    cum_q4 = np.zeros(HDIM, dtype=np.float32)
    for i, eid in enumerate(expert_ids):
        cum_fp += rust_weighted_on_rust[i]
        cum_q4 += q4_weighted_on_rust[i]
        print(f"{int(eid):8d} | {cos(cum_fp, cum_q4):.6f}")

    result = {
        "layer": target_layer,
        "expert_input_cos": cos(hf_h_norm2, rust_comp["h_norm2"]),
        "common_input_routed_cos": cos(hf_routed, rust_routed),
        "common_input_shared_cos": cos(hf_shared["shared_contrib"], rust_shared_fp["shared_contrib"]),
        "common_input_final_cos": cos(hf_shared["shared_contrib"] + hf_routed, rust_shared_fp["shared_contrib"] + rust_routed),
        "drift_routed_cos": cos(hf_routed, rust_routed_on_rust),
        "drift_final_cos": cos(hf_shared["shared_contrib"] + hf_routed, fp_final_rustinput),
        "routed_q4_vs_fp": cos(rust_comp["moe"], rust_routed_on_rust),
        "final_q4_vs_fp": cos(q4_final, fp_final_rustinput),
        "shared_norm": float(np.linalg.norm(rust_shared_fp["shared_contrib"])),
        "routed_norm": float(np.linalg.norm(rust_routed_on_rust)),
        "shared_routed_cos": cos(rust_shared_fp["shared_contrib"], rust_routed_on_rust),
        "q4_final": q4_final,
        "fp_final_rustinput": fp_final_rustinput,
        "routing_weights": routing_weights,
        "expert_ids": expert_ids,
        "fp_expert": rust_expert_on_rust,
        "q4_expert": q4_expert_on_rust,
        "q4_weighted": q4_weighted_on_rust,
        "fp_weighted": rust_weighted_on_rust,
    }

    # Approximate per-expert q4 raw outputs by proportional split of the traced routed sum is not valid.
    # Instead, recover weighted q4 contributions by running each selected expert alone through the q4 path later if needed.
    # For now keep the routed/final q4 metrics exact and defer per-expert q4 decomposition to the next extension.

    if (not skip_next_layer) and target_layer < len(model.model.layers) - 1:
        next_layer = model.model.layers[target_layer + 1]
        lib_next_q4 = init_runner()
        rust_prefill(lib_next_q4, ids)
        q4_next = rust_eval_layer_from_hidden(lib_next_q4, target_layer + 1, ids, q4_final.astype(np.float32))

        lib_next_fp = init_runner()
        rust_prefill(lib_next_fp, ids)
        fp_next = rust_eval_layer_from_hidden(lib_next_fp, target_layer + 1, ids, fp_final_rustinput.astype(np.float32))

        overlap, idx_a, idx_b = topk_overlap_from_hidden(next_layer, q4_next["h_norm2"], fp_next["h_norm2"])
        result["next_layer_hidden_cos"] = cos(q4_next["after_mlp"], fp_next["after_mlp"])
        result["next_router_topk_overlap"] = overlap
        result["next_router_ids_q4"] = idx_a
        result["next_router_ids_fp"] = idx_b

        per_expert_router = []
        contrib_order = np.argsort(
            [float(np.linalg.norm(v)) for v in rust_weighted_on_rust]
        )[::-1]
        if per_expert_router_limit <= 0:
            selected_indices = []
        else:
            selected_indices = contrib_order[: min(per_expert_router_limit, len(expert_ids))]
        for i in selected_indices:
            eid = expert_ids[i]
            hybrid_final = rust_shared_fp["shared_contrib"] + rust_routed_on_rust - rust_weighted_on_rust[i] + q4_weighted_on_rust[i]
            lib_next_one = init_runner()
            rust_prefill(lib_next_one, ids)
            one_next = rust_eval_layer_from_hidden(lib_next_one, target_layer + 1, ids, hybrid_final.astype(np.float32))
            overlap_i, idx_q4_i, idx_fp_i = topk_overlap_from_hidden(next_layer, one_next["h_norm2"], fp_next["h_norm2"])
            changed = tuple(map(int, idx_q4_i)) != tuple(map(int, idx_fp_i))
            per_expert_router.append({
                "expert_id": int(eid),
                "next_router_overlap": overlap_i,
                "next_router_changed": changed,
            })
        result["per_expert_router"] = per_expert_router
    else:
        result["next_layer_hidden_cos"] = None
        result["next_router_topk_overlap"] = None
        result["next_router_ids_q4"] = None
        result["next_router_ids_fp"] = None
        result["per_expert_router"] = []

    return result


def print_summary_tables(results):
    print("\n=== Table 1: same-input MoE implementation parity ===")
    print("layer | expert_input cos | expert path common-input cos | shared cos | final_moe cos")
    for r in results:
        print(f"{r['layer']:5d} | {r['expert_input_cos']:.6f} | {r['common_input_routed_cos']:.6f} | {r['common_input_shared_cos']:.6f} | {r['common_input_final_cos']:.6f}")

    print("\n=== Table 2: input-drift amplification ===")
    print("layer | expert_input cos | routed_sum cos HFvsRustInput | final_moe cos HFvsRustInput | amplification_ratio")
    for r in results:
        denom = max(1.0 - r["expert_input_cos"], 1e-12)
        amplification_ratio = (1.0 - r["drift_final_cos"]) / denom
        print(f"{r['layer']:5d} | {r['expert_input_cos']:.6f} | {r['drift_routed_cos']:.6f} | {r['drift_final_cos']:.6f} | {amplification_ratio:.6f}")

    print("\n=== Table 3: q4 quantization drift ===")
    print("layer | routed_q4_vs_fp | final_q4_vs_fp | next_layer_hidden_cos | next_router_topk_overlap")
    for r in results:
        nh = "n/a" if r["next_layer_hidden_cos"] is None else f"{r['next_layer_hidden_cos']:.6f}"
        no = "n/a" if r["next_router_topk_overlap"] is None else f"{r['next_router_topk_overlap']}/{TOP_K}"
        print(f"{r['layer']:5d} | {r['routed_q4_vs_fp']:.6f} | {r['final_q4_vs_fp']:.6f} | {nh:>21} | {no}")

    print("\n=== Shared/Routed Layer Profile ===")
    print("layer | ||shared|| | ||routed|| | routed/shared | cos(shared, routed)")
    for r in results:
        ratio = r["routed_norm"] / max(r["shared_norm"], 1e-12)
        print(f"{r['layer']:5d} | {r['shared_norm']:.6f} | {r['routed_norm']:.6f} | {ratio:.6f} | {r['shared_routed_cos']:.6f}")

    print("\n=== Per-Expert q4 Impact ===")
    print("layer | expert_id | gate_weight | fp_norm | q4_norm | raw_cos | weighted_cos | contribution_ratio | next_router_changed? | next_router_overlap")
    for r in results:
        total_weighted_norm = max(float(np.linalg.norm(r["fp_weighted"].reshape(-1))), 1e-12)
        router_map = {x["expert_id"]: x for x in r["per_expert_router"]}
        for i, eid in enumerate(r["expert_ids"]):
            entry = router_map.get(int(eid), {})
            fp_norm = float(np.linalg.norm(r["fp_expert"][i]))
            q4_norm = float(np.linalg.norm(r["q4_expert"][i]))
            raw_cos = cos(r["fp_expert"][i], r["q4_expert"][i])
            weighted_cos = cos(r["fp_weighted"][i], r["q4_weighted"][i])
            contribution_ratio = float(np.linalg.norm(r["fp_weighted"][i]) / total_weighted_norm)
            overlap = entry.get("next_router_overlap", "n/a")
            changed = entry.get("next_router_changed", "n/a" if not entry else False)
            print(f"{r['layer']:5d} | {int(eid):8d} | {float(r['routing_weights'][i]):11.6f} | {fp_norm:.6f} | {q4_norm:.6f} | {raw_cos:.6f} | {weighted_cos:.6f} | {contribution_ratio:.6f} | {str(changed):>19} | {overlap}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", nargs="?", default="0,3,7,15,23,31,39")
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument(
        "--per-expert-router-limit",
        type=int,
        default=0,
        help="How many high-contribution experts per layer to evaluate with single-expert q4 -> next-router checks. 0 disables this heavy step.",
    )
    parser.add_argument(
        "--skip-next-layer",
        action="store_true",
        help="Skip q4 next-layer hidden/router evaluation entirely for a fast local-only pass.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shortcut for a light pass: disables per-expert next-router checks.",
    )
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    prompt = args.prompt
    per_expert_router_limit = 0 if args.quick else args.per_expert_router_limit

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = chat_ids(tok, prompt)
    print(f"Prompt: {prompt}")
    print(f"Target layers: {layers}")
    print(f"Chat length: {len(ids)} tokens")
    print(f"Per-expert next-router checks: {per_expert_router_limit}")
    print(f"Skip next-layer eval: {args.skip_next_layer}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    model.eval()

    results = []
    for layer_idx in layers:
        results.append(compare_layer(
            model,
            ids,
            layer_idx,
            per_expert_router_limit=per_expert_router_limit,
            skip_next_layer=args.skip_next_layer,
        ))
    print_summary_tables(results)


if __name__ == "__main__":
    main()
