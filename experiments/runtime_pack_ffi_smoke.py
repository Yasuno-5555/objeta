#!/usr/bin/env python3
"""Direct/env smoke for lko_runner_load_runtime_pack."""

import argparse
import ctypes
import json
import os
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

PROJECT = Path(__file__).parent.parent
BIN_DIR = PROJECT / "models" / "qwen36_bin"
LIB_PATH = PROJECT / "target" / "release" / "libobjeta_qwen36_executor.dylib"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
HDIM = 2048


def make_mock_pack() -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="objeta_runtime_pack_smoke_"))
    pack = tmpdir / "qwen36-m1-8gb.objeta"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pack_type": "objeta_runtime_pack",
                "model_family": "qwen",
                "model_name": "qwen36",
                "target": "m1-8gb",
                "files": {
                    "expert_layout": "expert_layout.json",
                    "expert_importance": "expert_importance.json",
                    "expert_coresidency": "expert_coresidency.json",
                    "residency_plan": "residency_plan.json",
                    "phase_policy": "phase_policy.json",
                    "runtime_profile": "runtime_profile.json",
                },
            },
            indent=2,
        )
    )
    (pack / "runtime_profile.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_name": "ffi-pack-smoke",
                "target": "m1-8gb",
                "backend": "legacy",
                "policy_kind": "exact",
                "moe_top_p": 1.0,
                "moe_min_experts": 8,
                "moe_max_experts": 8,
                "resident_cache_capacity_bytes": 0,
                "group_preresolve_top_n": 0,
                "group_preresolve_max_bytes": 0,
                "source_model": str(BIN_DIR),
                "source_calibration": None,
            },
            indent=2,
        )
    )
    (pack / "expert_importance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experts": [
                    {
                        "layer": 31,
                        "expert": 42,
                        "selected_count": 12,
                        "avg_gate_weight": 0.31,
                        "importance": 0.91,
                        "tier": "hot",
                        "eviction_priority": 0.09,
                    }
                ],
            },
            indent=2,
        )
    )
    (pack / "expert_coresidency.json").write_text(json.dumps({"schema_version": 1, "pairs": []}, indent=2))
    (pack / "phase_policy.json").write_text(json.dumps({"schema_version": 1, "layers": []}, indent=2))
    (pack / "residency_plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": "m1-8gb",
                "resident_cache_capacity_bytes": 0,
                "initial_hot_experts": [],
                "eviction_priority": [],
                "summary": {
                    "initial_hot_expert_count": 0,
                    "initial_hot_expert_bytes": 0,
                    "eviction_priority_count": 0,
                    "bytes_fallback_expert_count": 0,
                },
            },
            indent=2,
        )
    )
    (pack / "expert_layout.json").write_text(json.dumps({"schema_version": 1, "experts": []}, indent=2))
    return pack


def rust_step(lib, token_id: int, pos: int, seq_len: int, top_k: int = 50):
    hn = (ctypes.c_float * HDIM)()
    indices = (ctypes.c_int32 * top_k)()
    values = (ctypes.c_float * top_k)()
    entropy = ctypes.c_float(0.0)
    k = lib.lko_runner_step_with_entropy(
        token_id,
        pos,
        seq_len,
        ctypes.cast(hn, ctypes.c_void_p),
        top_k,
        ctypes.cast(indices, ctypes.c_void_p),
        ctypes.cast(values, ctypes.c_void_p),
        ctypes.byref(entropy),
    )
    out_ids = [indices[i] for i in range(k)]
    out_vals = [values[i] for i in range(k)]
    order = sorted(range(k), key=lambda i: out_vals[i], reverse=True)
    return [out_ids[i] for i in order]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["direct", "env"], default="direct")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["OBJETA_TIMING"] = "0"
    pack_path = make_mock_pack()
    if args.mode == "env":
        os.environ["OBJETA_RUNTIME_PACK_PATH"] = str(pack_path)

    lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))
    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
    lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
    lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
    lib.lko_moe_init_page_cache.argtypes = [ctypes.c_int64]
    lib.lko_moe_init_page_cache.restype = ctypes.c_int32
    lib.lko_runner_get_instance.argtypes = []
    lib.lko_runner_get_instance.restype = ctypes.c_void_p
    lib.lko_runner_step_with_entropy.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.lko_runner_step_with_entropy.restype = ctypes.c_int32
    lib.lko_runner_get_moe_stats_json.argtypes = []
    lib.lko_runner_get_moe_stats_json.restype = ctypes.c_void_p
    lib.lko_runner_load_runtime_pack.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lko_runner_load_runtime_pack.restype = ctypes.c_int32

    assert lib.lko_runner_init(str(BIN_DIR).encode(), 256), "Runner init failed"
    assert lib.lko_runner_set_fusion_ratio(0.80), "fusion_ratio set failed"
    assert lib.lko_runner_set_moe_on_deltanet(1), "moe_on_deltanet set failed"
    assert lib.lko_moe_init_page_cache(0), "page cache init failed"

    if args.mode == "direct":
        runner = lib.lko_runner_get_instance()
        assert runner, "Runner instance missing"
        assert lib.lko_runner_load_runtime_pack(runner, str(pack_path).encode()), "Pack load failed"

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    chat = tok.apply_chat_template(
        [{"role": "user", "content": "The capital of France is"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok.encode(chat, add_special_tokens=False)

    indices = None
    for i, tid in enumerate(ids):
        indices = rust_step(lib, tid, i, i + 1, 50)
    top1 = int(indices[0])
    output = tok.decode([top1])

    stats_ptr = lib.lko_runner_get_moe_stats_json()
    stats = {}
    if stats_ptr:
        stats = json.loads(ctypes.cast(stats_ptr, ctypes.c_char_p).value.decode("utf-8"))

    print(f"mode={args.mode}")
    print(f"pack_path={pack_path}")
    print(f"top1_token_id={top1}")
    print(f"output={output}")
    print(json.dumps(stats.get("runtime_pack", {}), indent=2))


if __name__ == "__main__":
    main()
