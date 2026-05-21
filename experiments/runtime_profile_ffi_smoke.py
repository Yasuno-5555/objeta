#!/usr/bin/env python3
"""Direct FFI smoke for lko_runner_load_runtime_profile."""

import argparse
import ctypes
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

PROJECT = Path(__file__).parent.parent
BIN_DIR = PROJECT / "models" / "qwen36_bin"
LIB_PATH = PROJECT / "target" / "release" / "libobjeta_qwen36_executor.dylib"
PROFILE_PATH = Path("/tmp/runtime_profile_ffi_smoke.json")
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
HDIM = 2048

os.environ["OBJETA_TIMING"] = "0"


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
    out_ids = [out_ids[i] for i in order]
    out_vals = [out_vals[i] for i in order]
    return out_ids, out_vals, float(entropy.value)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy-kind",
        choices=["exact", "top_p"],
        default="exact",
    )
    p.add_argument("--moe-top-p", type=float, default=1.0)
    p.add_argument("--moe-min-experts", type=int, default=8)
    p.add_argument("--moe-max-experts", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    PROFILE_PATH.write_text(
        json.dumps(
            {
                "name": "ffi-smoke-profile",
                "target": "m1_8gb",
                "notes": "direct ffi smoke",
                "policy_kind": args.policy_kind,
                "knobs": {
                    "backend": "legacy",
                    "moe_top_p": args.moe_top_p,
                    "moe_min_experts": args.moe_min_experts,
                    "moe_max_experts": args.moe_max_experts,
                    "resident_cache_capacity_bytes": 0,
                    "residency_group_size": 1,
                    "group_preresolve_top_n": 0,
                    "group_preresolve_max_bytes": 0,
                },
            },
            indent=2,
        )
    )

    lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))
    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    lib.lko_runner_load_runtime_profile.argtypes = [ctypes.c_char_p]
    lib.lko_runner_load_runtime_profile.restype = ctypes.c_int32
    lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
    lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32
    lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
    lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32
    lib.lko_moe_init_page_cache.argtypes = [ctypes.c_int64]
    lib.lko_moe_init_page_cache.restype = ctypes.c_int32
    lib.lko_runner_set_expert_policy_json.argtypes = [ctypes.c_char_p]
    lib.lko_runner_set_expert_policy_json.restype = ctypes.c_int32
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

    assert lib.lko_runner_init(str(BIN_DIR).encode(), 256), "Runner init failed"
    assert lib.lko_runner_set_fusion_ratio(0.80), "fusion_ratio set failed"
    assert lib.lko_runner_set_moe_on_deltanet(1), "moe_on_deltanet set failed"
    assert lib.lko_moe_init_page_cache(4096 * 1024 * 1024), "page cache init failed"
    assert lib.lko_runner_set_expert_policy_json(b'{"kind":"exact"}'), "expert policy set failed"
    assert lib.lko_runner_load_runtime_profile(str(PROFILE_PATH).encode()), "Profile load failed"

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    chat = tok.apply_chat_template(
        [{"role": "user", "content": "The capital of France is"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok.encode(chat, add_special_tokens=False)

    indices = values = None
    for i, tid in enumerate(ids):
        indices, values, _ = rust_step(lib, tid, i, i + 1, 50)
    top1 = int(indices[0])
    text = tok.decode([top1])
    print(f"policy_kind={args.policy_kind}")
    print(f"top1_token_id={top1}")
    print(f"output={text}")


if __name__ == "__main__":
    main()
