#!/usr/bin/env python3
"""Qwen3.6 forward correctness oracle check.
Runs only prefill (no generation) and validates logits / entropy / hidden norm.
"""
import argparse
import ctypes
import numpy as np
import os
import sys
import json
import hashlib
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
from experiments.oracle_registry import (
    get_git_commit,
    get_model_integrity_hashes,
    register_golden,
    lookup_golden
)
from transformers import AutoTokenizer


def build_model_input(tok, prompt: str, use_chat_template: bool):
    if use_chat_template:
        model_input_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_mode = "chat_template"
    else:
        model_input_text = prompt
        prompt_mode = "raw"
    ids = tok.encode(model_input_text)
    prompt_hash = hashlib.sha256(model_input_text.encode("utf-8")).hexdigest()[:8]
    return model_input_text, ids, prompt_hash, prompt_mode

def main():
    parser = argparse.ArgumentParser(description="Qwen3.6 no-generation correctness oracle")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt for prefill")
    parser.add_argument("--chat-template", action="store_true", help="Apply the tokenizer chat template before prefill")
    parser.add_argument("--strategy", default="safe", help="Strategy config to use")
    parser.add_argument("--save-golden", action="store_true", help="Save the current run to runs/oracles/")
    parser.add_argument("--compare", default=None, help="Compare current run against the specified golden JSON path")
    parser.add_argument("--compare-golden", action="store_true", help="Compare current run against golden registry entry")
    parser.add_argument("--golden-name", default="exact_good_prefill", help="Golden registry entry name to save or compare")
    parser.add_argument("--entropy-tol", type=float, default=0.05, help="Tolerance for entropy check")
    parser.add_argument("--norm-tol", type=float, default=0.1, help="Tolerance for hidden norm check")
    parser.add_argument("--warmup-tokens", type=int, default=100, help="Warmup tokens for page cache; use 0 for light checks")
    parser.add_argument("--trace-record-path", default=None, help="Path to save trace JSON")
    parser.add_argument("--trace-replay-path", default=None, help="Path to load trace JSON for replay")
    args = parser.parse_args()

    # Load shared library
    lib = get_lib()
    HDIM = 2048

    # Init Metal
    lib.lko_metal_init.argtypes = [ctypes.c_char_p]
    lib.lko_metal_init.restype = ctypes.c_int32
    lib.lko_metal_init(b"/nonexistent.metallib")

    # Init Rust runner
    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    BIN_DIR = str(Path(__file__).parent.parent / "models" / "qwen36_bin")
    assert lib.lko_runner_init(BIN_DIR.encode(), 256), "Runner init failed"

    # Setup FFI functions with fallback support
    if hasattr(lib, "lko_runner_reset_kv_cache"):
        lib.lko_runner_reset_kv_cache.argtypes = []
        lib.lko_runner_reset_kv_cache.restype = ctypes.c_int32
    
    if hasattr(lib, "lko_runner_set_fusion_ratio"):
        lib.lko_runner_set_fusion_ratio.argtypes = [ctypes.c_double]
        lib.lko_runner_set_fusion_ratio.restype = ctypes.c_int32

    if hasattr(lib, "lko_runner_set_moe_on_deltanet"):
        lib.lko_runner_set_moe_on_deltanet.argtypes = [ctypes.c_int32]
        lib.lko_runner_set_moe_on_deltanet.restype = ctypes.c_int32

    if hasattr(lib, "lko_runner_set_moe_top_p"):
        lib.lko_runner_set_moe_top_p.argtypes = [ctypes.c_float]
        lib.lko_runner_set_moe_top_p.restype = ctypes.c_int32

    if hasattr(lib, "lko_runner_set_moe_prune_mode"):
        lib.lko_runner_set_moe_prune_mode.argtypes = [ctypes.c_int32]
        lib.lko_runner_set_moe_prune_mode.restype = ctypes.c_int32

    if hasattr(lib, "lko_runner_set_moe_contrib_threshold"):
        lib.lko_runner_set_moe_contrib_threshold.argtypes = [ctypes.c_float]
        lib.lko_runner_set_moe_contrib_threshold.restype = ctypes.c_int32

    if hasattr(lib, "lko_runner_set_moe_min_experts"):
        lib.lko_runner_set_moe_min_experts.argtypes = [ctypes.c_int32]
        lib.lko_runner_set_moe_min_experts.restype = ctypes.c_int32

    if hasattr(lib, "lko_runner_set_moe_max_experts"):
        lib.lko_runner_set_moe_max_experts.argtypes = [ctypes.c_int32]
        lib.lko_runner_set_moe_max_experts.restype = ctypes.c_int32

    if hasattr(lib, "lko_runner_set_expert_policy_json"):
        lib.lko_runner_set_expert_policy_json.argtypes = [ctypes.c_char_p]
        lib.lko_runner_set_expert_policy_json.restype = ctypes.c_int32

    if hasattr(lib, "lko_moe_init_page_cache"):
        lib.lko_moe_init_page_cache.argtypes = [ctypes.c_int64]
        lib.lko_moe_init_page_cache.restype = ctypes.c_int32

    if hasattr(lib, "lko_runner_reset_moe_stats"):
        lib.lko_runner_reset_moe_stats.argtypes = []
        lib.lko_runner_reset_moe_stats.restype = ctypes.c_int32

    # Step APIs
    has_entropy_api = hasattr(lib, "lko_runner_step_with_entropy")
    if has_entropy_api:
        lib.lko_runner_step_with_entropy.argtypes = [
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.lko_runner_step_with_entropy.restype = ctypes.c_int32
    else:
        lib.lko_runner_step.argtypes = [
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.lko_runner_step.restype = ctypes.c_int32

    # Strategy setup
    strategy_dict = {
        "fusion_ratio": 0.33,
        "moe_on_deltanet": 0,
        "expert_policy": None,
        "moe_prune_mode": "top_p",
        "moe_top_p": 1.0,
        "moe_contrib_threshold": 1.0,
        "expert_cache_mb": 4000,
        "timing": True,
        "trace": False,
        "max_experts": 8,
        "min_experts": 2,
        "debug_switches": None
    }

    if args.strategy:
        strat_name = args.strategy.lower()
        presets = ["safe", "fast", "turbo", "debug"]
        if strat_name in presets:
            strat_path = f"configs/{strat_name}.json"
        else:
            strat_path = args.strategy
        
        if os.path.exists(strat_path):
            print(f"Loading strategy config from {strat_path}")
            with open(strat_path, "r") as f:
                strategy_dict.update(json.load(f))
        else:
            print(f"Warning: strategy config file '{strat_path}' not found. Using defaults.")
    strategy_hash = hashlib.sha256(json.dumps(strategy_dict, sort_keys=True).encode("utf-8")).hexdigest()

    # Init caches & configure runner
    if hasattr(lib, "lko_moe_init_page_cache"):
        lib.lko_moe_init_page_cache(strategy_dict["expert_cache_mb"] * 1024 * 1024)
    if hasattr(lib, "lko_runner_set_fusion_ratio"):
        lib.lko_runner_set_fusion_ratio(strategy_dict["fusion_ratio"])
    if hasattr(lib, "lko_runner_set_moe_on_deltanet"):
        lib.lko_runner_set_moe_on_deltanet(strategy_dict["moe_on_deltanet"])
    if hasattr(lib, "lko_runner_set_moe_top_p"):
        lib.lko_runner_set_moe_top_p(strategy_dict["moe_top_p"])
    if hasattr(lib, "lko_runner_set_moe_prune_mode"):
        lib.lko_runner_set_moe_prune_mode(0 if strategy_dict["moe_prune_mode"] == "top_p" else 1)
    if hasattr(lib, "lko_runner_set_moe_contrib_threshold"):
        lib.lko_runner_set_moe_contrib_threshold(strategy_dict["moe_contrib_threshold"])
    if hasattr(lib, "lko_runner_set_moe_min_experts"):
        lib.lko_runner_set_moe_min_experts(strategy_dict.get("min_experts", 2))
    if hasattr(lib, "lko_runner_set_moe_max_experts"):
        lib.lko_runner_set_moe_max_experts(strategy_dict.get("max_experts", 8))
    if strategy_dict.get("expert_policy") is not None and hasattr(lib, "lko_runner_set_expert_policy_json"):
        policy_json = json.dumps(strategy_dict["expert_policy"]).encode("utf-8")
        assert lib.lko_runner_set_expert_policy_json(policy_json), "Failed to set expert_policy JSON"

    # Setup debug switches
    debug_switches = strategy_dict.get("debug_switches")
    force_attn_full = 0
    force_moe_skip = 0
    if isinstance(debug_switches, dict):
        if debug_switches.get("force_attn_full"):
            force_attn_full = 1
        if debug_switches.get("force_moe_skip"):
            force_moe_skip = 1
    elif isinstance(debug_switches, list):
        if "force_attn_full" in debug_switches:
            force_attn_full = 1
        if "force_moe_skip" in debug_switches:
            force_moe_skip = 1

    if hasattr(lib, "lko_runner_set_force_attn_full"):
        lib.lko_runner_set_force_attn_full.argtypes = [ctypes.c_int32]
        lib.lko_runner_set_force_attn_full.restype = ctypes.c_int32
        lib.lko_runner_set_force_attn_full(force_attn_full)

    if hasattr(lib, "lko_runner_set_force_moe_skip"):
        lib.lko_runner_set_force_moe_skip.argtypes = [ctypes.c_int32]
        lib.lko_runner_set_force_moe_skip.restype = ctypes.c_int32
        lib.lko_runner_set_force_moe_skip(force_moe_skip)

    # Trace record/replay
    if hasattr(lib, "lko_runner_set_trace_record"):
        lib.lko_runner_set_trace_record.argtypes = [ctypes.c_char_p]
        lib.lko_runner_set_trace_record.restype = ctypes.c_int32
        if args.trace_record_path:
            if os.path.exists(args.trace_record_path):
                os.remove(args.trace_record_path)
            lib.lko_runner_set_trace_record(args.trace_record_path.encode("utf-8"))

    if hasattr(lib, "lko_runner_set_trace_replay"):
        lib.lko_runner_set_trace_replay.argtypes = [ctypes.c_char_p]
        lib.lko_runner_set_trace_replay.restype = ctypes.c_int32
        if args.trace_replay_path:
            assert lib.lko_runner_set_trace_replay(args.trace_replay_path.encode("utf-8")), "Failed to load trace replay file"

    # Warmup OS page cache
    if hasattr(lib, "lko_runner_warmup"):
        if hasattr(lib, "lko_runner_set_fusion_ratio"):
            lib.lko_runner_set_fusion_ratio(0.0)
        lib.lko_runner_warmup.argtypes = [ctypes.c_int32]
        lib.lko_runner_warmup.restype = ctypes.c_int32
        if args.warmup_tokens > 0:
            print(f"Warming OS page cache... ({args.warmup_tokens} tokens)")
            lib.lko_runner_warmup(args.warmup_tokens)
        else:
            print("Skipping OS page cache warmup")
        if hasattr(lib, "lko_runner_set_fusion_ratio"):
            lib.lko_runner_set_fusion_ratio(strategy_dict["fusion_ratio"])

    if hasattr(lib, "lko_runner_reset_kv_cache"):
        lib.lko_runner_reset_kv_cache()
    if hasattr(lib, "lko_runner_reset_moe_stats"):
        lib.lko_runner_reset_moe_stats()

    # Load Tokenizer & calculate integrity hashes
    snap = sorted(os.listdir("/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"))[-1]
    tok_dir = f"/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{snap}"
    tok = AutoTokenizer.from_pretrained(tok_dir)
    integrity_hashes = get_model_integrity_hashes(tok_dir, BIN_DIR)

    # Encode prompt
    model_input_text, ids, prompt_hash, prompt_mode = build_model_input(tok, args.prompt, args.chat_template)

    print(f"Prompt: \"{args.prompt}\"")
    print(f"Prompt mode: {prompt_mode}")
    if args.chat_template:
        print(f"Model input: {repr(model_input_text)}")
    print(f"Tokenizer IDs ({len(ids)} tokens): {ids}")

    # Prefill only
    hn = np.zeros(HDIM, dtype=np.float32)
    indices = np.zeros(50, dtype=np.int32)
    values = np.zeros(50, dtype=np.float32)
    entropy_val = ctypes.c_float(0.0)

    print("Running prefill...")
    t0 = time.perf_counter()
    for i, tid in enumerate(ids):
        hn.fill(0.0)
        indices.fill(0)
        values.fill(0.0)
        entropy_val.value = 0.0

        if has_entropy_api:
            k = lib.lko_runner_step_with_entropy(
                tid, i, i+1, hn.ctypes.data, 50,
                indices.ctypes.data, values.ctypes.data, ctypes.byref(entropy_val)
            )
        else:
            k = lib.lko_runner_step(
                tid, i, i+1, hn.ctypes.data, 50,
                indices.ctypes.data, values.ctypes.data
            )
            val_max = np.max(values[:k])
            probs = np.exp(values[:k] - val_max)
            probs /= np.sum(probs)
            entropy_val.value = -float(np.sum(probs * np.log(probs + 1e-10)))
    
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    order = np.argsort(values[:k])[::-1]
    final_indices = indices[:k][order]
    final_values = values[:k][order]
    final_entropy = float(entropy_val.value)
    final_hidden_norm = float(np.linalg.norm(hn))

    top10_ids = [int(x) for x in final_indices[:10]]
    top10_values = [float(x) for x in final_values[:10]]
    top10_tokens = [tok.decode([x]) for x in top10_ids]

    print("\nPrefill final token results:")
    print(f"  Entropy: {final_entropy:.6f}")
    print(f"  Hidden Norm: {final_hidden_norm:.6f}")
    print("  Top-10 tokens:")
    for idx, (tid, val, tstr) in enumerate(zip(top10_ids, top10_values, top10_tokens)):
        print(f"    [{idx+1}] ID={tid:<6} Logit={val:.4f} Token={repr(tstr)}")

    # Check for NaN / Inf
    no_nan_inf = True
    if np.isnan(final_entropy) or np.isinf(final_entropy):
        no_nan_inf = False
    if np.isnan(final_hidden_norm) or np.isinf(final_hidden_norm):
        no_nan_inf = False
    if np.any(np.isnan(hn)) or np.any(np.isinf(hn)):
        no_nan_inf = False
    if np.any(np.isnan(final_values)) or np.any(np.isinf(final_values)):
        no_nan_inf = False

    # Save mode
    if args.save_golden:
        oracle_dir = Path("runs/oracles")
        oracle_dir.mkdir(parents=True, exist_ok=True)
        golden_path = oracle_dir / "exact_good_prefill.json"
        
        golden_data = {
            "prompt": args.prompt,
            "prompt_mode": prompt_mode,
            "model_input_text": model_input_text,
            "tokenizer_ids": ids,
            "expected_top10_ids": top10_ids,
            "expected_entropy": final_entropy,
            "expected_entropy_range": [final_entropy - args.entropy_tol, final_entropy + args.entropy_tol],
            "expected_hidden_norm": final_hidden_norm,
            "expected_hidden_norm_range": [final_hidden_norm - args.norm_tol, final_hidden_norm + args.norm_tol],
            "expected_first_token_id": top10_ids[0] if top10_ids else None
        }
        
        with open(golden_path, "w") as f:
            json.dump(golden_data, f, indent=2)
        print(f"\nSaved golden oracle to {golden_path}")
        
        register_golden(
            golden_name=args.golden_name,
            model_id="Qwen3.6-35B-A3B",
            strategy_name=args.strategy,
            prompt=model_input_text,
            prompt_hash=prompt_hash,
            tokenizer_hash=integrity_hashes["tokenizer_hash"],
            weight_manifest_hash=integrity_hashes["weight_manifest_hash"],
            file_path=golden_path,
            git_commit=get_git_commit(),
            prompt_mode=prompt_mode,
            model_input_text=model_input_text,
            tokenizer_ids=ids,
            strategy_hash=strategy_hash,
            config_hash=integrity_hashes["config_hash"],
            lm_head_hash=integrity_hashes["lm_head_hash"],
            embed_hash=integrity_hashes["embed_hash"],
        )
        sys.exit(0)

    # Compare mode
    compare_path = None
    if args.compare_golden:
        entry = lookup_golden(args.golden_name)
        if not entry:
            print(f"Error: Golden registry entry '{args.golden_name}' not found.")
            sys.exit(1)
        compare_path = entry["file_path"]
        print(f"Comparing using registry entry '{args.golden_name}' -> {compare_path}")
    elif args.compare:
        compare_path = args.compare

    if compare_path:
        print(f"\nComparing current run against {compare_path}...")
        if not os.path.exists(compare_path):
            print(f"Error: Golden file '{compare_path}' not found.")
            sys.exit(1)
            
        with open(compare_path, "r") as f:
            golden = json.load(f)

        # Check prompt & tokens
        prompt_mode_ok = golden.get("prompt_mode") in (None, prompt_mode)
        if golden.get("tokenizer_ids") != ids:
            print("Warning: Current tokenizer IDs do not match golden IDs.")
            print(f"  Golden:  {golden.get('tokenizer_ids')}")
            print(f"  Current: {ids}")
        if not prompt_mode_ok:
            print(f"Prompt mode mismatch: golden={golden.get('prompt_mode')} current={prompt_mode}")

        golden_top10 = golden.get("expected_top10_ids", [])
        overlap = len(set(top10_ids) & set(golden_top10))
        overlap_ok = overlap >= 7

        ent_range = golden.get("expected_entropy_range", [0.0, 0.0])
        ent_ok = ent_range[0] <= final_entropy <= ent_range[1]

        norm_range = golden.get("expected_hidden_norm_range", [0.0, 0.0])
        norm_ok = norm_range[0] <= final_hidden_norm <= norm_range[1]

        exp_first = golden.get("expected_first_token_id")
        first_token_ok = (exp_first in top10_ids) if exp_first is not None else True

        passed = prompt_mode_ok and overlap_ok and ent_ok and norm_ok and first_token_ok and no_nan_inf

        print("\nVerification Results:")
        print(f"  0. Prompt Mode:       {'PASS' if prompt_mode_ok else 'FAIL'} (Golden: {golden.get('prompt_mode')} / Current: {prompt_mode})")
        print(f"  1. Top-10 Overlap:   {overlap}/10 (Expected >= 7) -> {'PASS' if overlap_ok else 'FAIL'}")
        print(f"     Golden top-10:    {golden_top10}")
        print(f"     Current top-10:   {top10_ids}")
        print(f"  2. Entropy:          {final_entropy:.6f} (Expected: {ent_range[0]:.6f} - {ent_range[1]:.6f}) -> {'PASS' if ent_ok else 'FAIL'}")
        print(f"  3. Hidden Norm:      {final_hidden_norm:.6f} (Expected: {norm_range[0]:.6f} - {norm_range[1]:.6f}) -> {'PASS' if norm_ok else 'FAIL'}")
        print(f"  4. Expected First ID in top-10: {'PASS' if first_token_ok else 'FAIL'} (Expected ID: {exp_first})")
        print(f"  5. NaN/Inf check:    {'PASS' if no_nan_inf else 'FAIL'}")

        # Classify failures
        failure_type = None
        suspected_area = None
        if not passed:
            if not no_nan_inf:
                failure_type = "NaN/Inf values"
                suspected_area = "Numerical stability / weight corruption"
            elif not first_token_ok or not overlap_ok:
                failure_type = "Token selection divergence"
                suspected_area = "Attention routing / lm_head calculation"
            elif not prompt_mode_ok:
                failure_type = "Prompt mode mismatch"
                suspected_area = "Oracle input encoding / chat-template selection"
            elif not norm_ok:
                failure_type = "Hidden norm mismatch"
                suspected_area = "Layer-wise calculation / precision rounding"
            elif not ent_ok:
                failure_type = "Entropy out of range"
                suspected_area = "Logits scaling / generation parameters"

        # Setup run dir to output summary.json
        import datetime
        run_dir = Path(f"runs/oracle_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Populate all METRICS_SCHEMA.md required fields
        summary = {
            "strategy_name": args.strategy,
            "git_commit": get_git_commit(),
            "model_id": "Qwen3.6-35B-A3B",
            "prompt_hash": prompt_hash,
            "prompt_mode": prompt_mode,
            "tok_s": 0.0,
            "total_wall_ms": total_time_ms,
            "forward_wall_ms_avg": total_time_ms / len(ids),
            "moe_wall_ms_avg": 0.0,
            "non_moe_wall_ms_avg": total_time_ms / len(ids),
            "moe_fraction": 0.0,
            "avg_experts_per_layer": 0.0,
            "avg_bytes_read": 0.0,
            "warm_hit_rate": 0.0,
            "cold_load_count": 0,
            "first_garbage": None,
            "first_repetition": None,
            "entropy_first": final_entropy,
            "entropy_last": final_entropy,
            "entropy_min": final_entropy,
            "entropy_max": final_entropy,
            
            # Oracle fields
            "pass": passed,
            "entropy": final_entropy,
            "hidden_norm": final_hidden_norm,
            "top10_ids": top10_ids,
            "integrity_hashes": integrity_hashes,
            "effective_debug_switches": strategy_dict.get("debug_switches"),
            "failure_type": failure_type,
            "suspected_area": suspected_area,
            
            "metrics": {
                "top10_overlap": overlap,
                "entropy_ok": ent_ok,
                "hidden_norm_ok": norm_ok,
                "first_token_ok": first_token_ok,
                "no_nan_inf": no_nan_inf
            }
        }
        
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWritten verification summary to {summary_path}")

        # Update latest runs/summary.json
        latest_summary_path = Path("runs/summary.json")
        with open(latest_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated runs/summary.json")

        if passed:
            print("\nResult: regression check PASSED")
            sys.exit(0)
        else:
            print("\nResult: regression check FAILED")
            sys.exit(1)

if __name__ == "__main__":
    main()
