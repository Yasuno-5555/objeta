#!/usr/bin/env python3
"""Qwen3.6 Layer-wise Forward Regression Oracle.
Traces intermediate layers and compares activations with golden snapshot.
"""
import os
import sys
import argparse
import ctypes
import json
import hashlib
import time
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
from experiments.oracle_registry import (
    get_git_commit,
    get_model_integrity_hashes,
    register_golden,
    lookup_golden
)

def calculate_checksum(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


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
    parser = argparse.ArgumentParser(description="Qwen3.6 Layer-wise Forward Regression Oracle")
    parser.add_argument("--strategy", type=str, default="safe", help="Path to config or preset name")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt text")
    parser.add_argument("--chat-template", action="store_true", help="Apply the tokenizer chat template before prefill")
    parser.add_argument("--save-golden", action="store_true", help="Save the current run to runs/oracles/")
    parser.add_argument("--compare", type=str, help="Path to golden snapshot to compare against")
    parser.add_argument("--compare-golden", action="store_true", help="Compare current run against golden registry entry")
    parser.add_argument("--golden-name", default="exact_good_layer_trace", help="Golden registry entry name to save or compare")
    parser.add_argument("--layers", type=str, default="0,3,7,15,23,31,39,final", help="Layers to trace (comma-separated)")
    parser.add_argument("--warmup-tokens", type=int, default=100, help="Warmup tokens for page cache; use 0 for light checks")
    parser.add_argument("--trace-record-path", default=None, help="Path to save trace JSON")
    parser.add_argument("--trace-replay-path", default=None, help="Path to load trace JSON for replay")
    args = parser.parse_args()

    lib = get_lib()
    HDIM = 2048

    # Init Metal
    lib.lko_metal_init(b"/nonexistent.metallib")

    # Init Rust runner
    lib.lko_runner_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.lko_runner_init.restype = ctypes.c_int32
    BIN_DIR = str(Path(__file__).parent.parent / "models" / "qwen36_bin")
    assert lib.lko_runner_init(BIN_DIR.encode(), 256), "Runner init failed"

    # Setup FFI
    if hasattr(lib, "lko_runner_reset_kv_cache"):
        lib.lko_runner_reset_kv_cache.argtypes = []
        lib.lko_runner_reset_kv_cache.restype = ctypes.c_int32
    
    lib.lko_runner_forward_n.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p
    ]
    lib.lko_runner_forward_n.restype = ctypes.c_int32

    lib.lko_runner_step.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.lko_runner_step.restype = ctypes.c_int32

    # Strategy FFI signatures
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

    # Load strategy
    strategy_dict = {
        "fusion_ratio": 1.0,
        "moe_on_deltanet": 1,
        "moe_prune_mode": "top_p",
        "moe_top_p": 1.0,
        "moe_contrib_threshold": 1.0,
        "expert_cache_mb": 4096,
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

    # Apply strategy parameters
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
        lib.lko_runner_set_expert_policy_json(policy_json)

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

    # Tokenizer & hashes
    snap = sorted(os.listdir("/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"))[-1]
    tok_dir = f"/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{snap}"
    tok = AutoTokenizer.from_pretrained(tok_dir)
    integrity_hashes = get_model_integrity_hashes(tok_dir, BIN_DIR)
    model_input_text, ids, prompt_hash, prompt_mode = build_model_input(tok, args.prompt, args.chat_template)
    print(f"Prompt mode: {prompt_mode}")
    if args.chat_template:
        print(f"Model input: {repr(model_input_text)}")

    # Parse layers to trace
    layer_parts = [p.strip() for p in args.layers.split(",")]
    trace_layers = []
    has_final = False
    for p in layer_parts:
        if p.lower() == "final":
            has_final = True
        else:
            trace_layers.append(int(p))
    trace_layers = sorted(list(set(trace_layers)))

    print(f"Tracing layers: {trace_layers} + final={has_final}")

    results = {}
    t0 = time.perf_counter()
    
    # 1. Trace specified hidden layers
    for lyr in trace_layers:
        print(f"Prefilling for layer {lyr} output...")
        if hasattr(lib, "lko_runner_reset_kv_cache"):
            lib.lko_runner_reset_kv_cache()
        hn = np.zeros(HDIM, dtype=np.float32)
        for i, tid in enumerate(ids):
            lib.lko_runner_forward_n(tid, i, i+1, lyr + 1, hn.ctypes.data)
        
        results[str(lyr)] = {
            "checksum": calculate_checksum(hn),
            "norm": float(np.linalg.norm(hn)),
            "values": hn.tolist()
        }

    # 2. Trace final
    top10_info = []
    if has_final:
        print("Prefilling for final layer output (with lm_head)...")
        if hasattr(lib, "lko_runner_reset_kv_cache"):
            lib.lko_runner_reset_kv_cache()
        hn = np.zeros(HDIM, dtype=np.float32)
        indices = np.zeros(50, dtype=np.int32)
        values = np.zeros(50, dtype=np.float32)
        k = 0
        for i, tid in enumerate(ids):
            k = lib.lko_runner_step(
                tid, i, i+1, hn.ctypes.data, 50,
                indices.ctypes.data, values.ctypes.data
            )
        
        order = np.argsort(values[:k])[::-1]
        final_indices = indices[:k][order]
        final_values = values[:k][order]
        
        results["final"] = {
            "checksum": calculate_checksum(hn),
            "norm": float(np.linalg.norm(hn)),
            "values": hn.tolist()
        }
        
        for r_idx in range(min(10, len(final_indices))):
            tid_out = int(final_indices[r_idx])
            logit_out = float(final_values[r_idx])
            tok_out = tok.decode([tid_out])
            top10_info.append({
                "id": tid_out,
                "logit": logit_out,
                "token": tok_out
            })

    total_time_ms = (time.perf_counter() - t0) * 1000.0

    output_data = {
        "prompt": args.prompt,
        "prompt_mode": prompt_mode,
        "model_input_text": model_input_text,
        "tokenizer_ids": ids,
        "layers": results,
        "top10_logits": top10_info
    }

    # Save golden
    if args.save_golden:
        oracle_dir = Path("runs/oracles")
        oracle_dir.mkdir(parents=True, exist_ok=True)
        golden_path = oracle_dir / "exact_good_layer_trace.json"
        
        with open(golden_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved layer trace oracle to {golden_path}")
        
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
        if not os.path.exists(compare_path):
            print(f"Error: compare target {compare_path} not found.")
            sys.exit(1)
        with open(compare_path, "r") as f:
            golden = json.load(f)
        prompt_mode_ok = golden.get("prompt_mode") in (None, prompt_mode)
        if not prompt_mode_ok:
            print(f"Prompt mode mismatch: golden={golden.get('prompt_mode')} current={prompt_mode}")
        
        # Compare layers
        print("\n--- Regression Comparison Results ---")
        print(f"{'Layer':<10} | {'Golden Norm':<12} | {'Current Norm':<12} | {'Norm Ratio':<10} | {'Cos Sim':<10} | {'Max Abs Diff':<12} | {'Status':<8}")
        print("-" * 90)
        
        first_bad_layer = None
        passed = prompt_mode_ok
        
        all_keys = trace_layers + (["final"] if has_final else [])
        for k_val in all_keys:
            key_str = str(k_val)
            if key_str not in golden["layers"]:
                print(f"Warning: layer {key_str} missing in golden snapshot.")
                continue
            
            gold_layer = golden["layers"][key_str]
            curr_layer = results[key_str]
            
            h_gold = np.array(gold_layer["values"])
            h_curr = np.array(curr_layer["values"])
            
            norm_gold = gold_layer["norm"]
            norm_curr = curr_layer["norm"]
            norm_ratio = norm_curr / (norm_gold + 1e-10)
            
            denom = np.linalg.norm(h_curr) * np.linalg.norm(h_gold)
            cos = float(np.dot(h_curr, h_gold) / (denom + 1e-10)) if denom > 1e-10 else 0.0
            max_abs = float(np.max(np.abs(h_curr - h_gold)))
            
            status = "OK"
            # Strict tolerance check
            if cos < 0.999 or max_abs > 0.05 or abs(norm_ratio - 1.0) > 0.05:
                status = "FAIL"
                passed = False
                if first_bad_layer is None:
                    first_bad_layer = key_str
            
            print(f"{key_str:<10} | {norm_gold:<12.5f} | {norm_curr:<12.5f} | {norm_ratio:<10.5f} | {cos:<10.5f} | {max_abs:<12.5f} | {status:<8}")

        print(f"\nPrompt Mode Check: {'PASS' if prompt_mode_ok else 'FAIL'} (Golden: {golden.get('prompt_mode')} / Current: {prompt_mode})")

        # Failure Classifier
        failure_type = None
        suspected_area = None
        if not passed:
            if not prompt_mode_ok:
                failure_type = "Prompt mode mismatch"
                suspected_area = "Oracle input encoding / chat-template selection"
            else:
                failure_type = "Layerwise activation divergence"
                if first_bad_layer == "final":
                    suspected_area = "final RMSNorm / lm_head calculation"
                elif first_bad_layer is not None:
                    layer_num = int(first_bad_layer)
                    if layer_num % 4 == 3:
                        suspected_area = f"GQA Attention or MoE Experts at layer {layer_num}"
                    else:
                        suspected_area = f"DeltaNet Linear Algebra at layer {layer_num}"

        # Write summary compliant with METRICS_SCHEMA.md
        import datetime
        run_dir = Path(f"runs/oracle_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        run_dir.mkdir(parents=True, exist_ok=True)

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
            "entropy_first": 0.0,
            "entropy_last": 0.0,
            "entropy_min": 0.0,
            "entropy_max": 0.0,

            # Oracle specific fields
            "pass": passed,
            "first_bad_layer": first_bad_layer,
            "integrity_hashes": integrity_hashes,
            "effective_debug_switches": strategy_dict.get("debug_switches"),
            "failure_type": failure_type,
            "suspected_area": suspected_area,
            "prompt_mode_ok": prompt_mode_ok
        }

        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWritten verification summary to {summary_path}")

        latest_summary_path = Path("runs/summary.json")
        with open(latest_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated runs/summary.json")

        if passed:
            print("\nResult: layerwise correctness check PASSED")
            sys.exit(0)
        else:
            print(f"\nResult: layerwise correctness check FAILED (First bad layer: {first_bad_layer})")
            sys.exit(1)

if __name__ == "__main__":
    main()
