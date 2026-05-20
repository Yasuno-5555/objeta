#!/usr/bin/env python3
"""Qwen3.6 generation — Rust executor (all ops in Rust: forward + lm_head + top-k)."""
import argparse
import ctypes
import numpy as np
import math
import time
import sys
import os
import json
import hashlib
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_executor import get_lib
from experiments.oracle_registry import (
    get_git_commit,
    get_model_integrity_hashes
)
from transformers import AutoTokenizer

lib = get_lib()
HDIM = 2048

# Schema and validation
REQUIRED_FIELDS = [
    "strategy_name", "git_commit", "model_id", "prompt_hash", "tok_s", "total_wall_ms",
    "forward_wall_ms_avg", "moe_wall_ms_avg", "non_moe_wall_ms_avg", "moe_fraction",
    "avg_experts_per_layer", "avg_bytes_read", "warm_hit_rate", "cold_load_count",
    "first_garbage", "first_repetition", "entropy_first", "entropy_last", "entropy_min",
    "entropy_max", "strategy_hash", "tokenizer_hash", "config_hash", "lm_head_hash",
    "embed_hash", "weight_manifest_hash", "logical_expert_bytes_requested",
    "actual_expert_bytes_loaded", "resident_cache_bytes_reused",
    "resident_cache_hit_count", "resident_cache_miss_count", "direct_cold_load_count",
    "resident_cache_enabled", "resident_cache_capacity_bytes",
    "resident_cache_resident_bytes"
]

def validate_summary_schema(summary):
    for f in REQUIRED_FIELDS:
        if f not in summary:
            raise KeyError(f"Missing required field in summary: {f}")

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

# FFI functions
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

if hasattr(lib, "lko_runner_reset_kv_cache"):
    lib.lko_runner_reset_kv_cache.argtypes = []
    lib.lko_runner_reset_kv_cache.restype = ctypes.c_int32

if hasattr(lib, "lko_runner_reset_moe_stats"):
    lib.lko_runner_reset_moe_stats.argtypes = []
    lib.lko_runner_reset_moe_stats.restype = ctypes.c_int32

if hasattr(lib, "lko_runner_get_moe_stats_json"):
    lib.lko_runner_get_moe_stats_json.argtypes = []
    lib.lko_runner_get_moe_stats_json.restype = ctypes.c_void_p

# Warmup: touch q4 pages to bring them into OS page cache
lib.lko_runner_warmup.argtypes = [ctypes.c_int32]
lib.lko_runner_warmup.restype = ctypes.c_int32

# C API: single step = forward + RMSNorm + lm_head + top-k
has_entropy_api = hasattr(lib, "lko_runner_step_with_entropy")
if has_entropy_api:
    lib.lko_runner_step_with_entropy.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # token_id, pos, seq_len
        ctypes.c_void_p,                                   # hn_out
        ctypes.c_int32,                                    # top_k
        ctypes.c_void_p, ctypes.c_void_p,                  # indices, values out
        ctypes.c_void_p                                    # entropy out
    ]
    lib.lko_runner_step_with_entropy.restype = ctypes.c_int32
else:
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
    entropy_val = ctypes.c_float(0.0)

    if has_entropy_api:
        k = lib.lko_runner_step_with_entropy(
            token_id, pos, seq_len, hn.ctypes.data, top_k,
            indices.ctypes.data, values.ctypes.data, ctypes.byref(entropy_val)
        )
        entropy = float(entropy_val.value)
    else:
        k = lib.lko_runner_step(
            token_id, pos, seq_len, hn.ctypes.data, top_k,
            indices.ctypes.data, values.ctypes.data
        )
        # Calculate manual entropy from top-k values
        val_max = np.max(values[:k])
        probs = np.exp(values[:k] - val_max)
        probs /= np.sum(probs)
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

    indices = indices[:k]
    values = values[:k]
    order = np.argsort(values)[::-1]
    return hn, indices[order], values[order], entropy

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

def has_repetition(tokens, max_l=8):
    """Checks if there is any repetition loop of length up to max_l."""
    n = len(tokens)
    for l in range(1, max_l + 1):
        if n >= 2 * l:
            if tokens[-l:] == tokens[-2*l:-l]:
                return True
    return False

def generate(prompt_ids, max_tokens=20, temperature=0.7, top_k=40, tok=None, early_abort=False):
    tokens = list(prompt_ids)
    n_prompt = len(tokens)
    print(f"Prefilling {n_prompt} tokens...")
    t0 = time.perf_counter()

    indices = values = None
    prefill_entropies = []
    for i, tid in enumerate(tokens):
        _, indices, values, ent = rust_step(tid, i, i+1, max(50, top_k))
        prefill_entropies.append(ent)
        if i % 5 == 0 or i == n_prompt-1:
            print(f"  [{i+1}/{n_prompt}] {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"  Prefill done in {time.perf_counter()-t0:.1f}s")

    if temperature == 0:
        next_token = int(indices[0])
    else:
        next_token = sample(indices, values, temperature, top_k)

    generated = []
    entropies = list(prefill_entropies)
    step_metrics = []
    t_start = time.perf_counter()
    
    aborted = False
    abort_reason = None
    abort_step = None

    for step in range(max_tokens):
        generated.append(next_token)
        pos = n_prompt + step
        if next_token == 2:
            break

        # Check early abort conditions
        if early_abort:
            token_text = tok.decode([next_token])
            # 1. Garbage token check (replacement character or junk collapse)
            if "\ufffd" in token_text:
                aborted = True
                abort_reason = "garbage token (replacement character)"
                abort_step = step
                print(f"Early Abort triggered: {abort_reason} at step {step}")
                break

            # 2. Entropy check (> 8.0 for 2 consecutive steps)
            if len(entropies) >= 2 and entropies[-1] > 8.0 and entropies[-2] > 8.0:
                aborted = True
                abort_reason = "high entropy (>8.0 for 2 consecutive steps)"
                abort_step = step
                print(f"Early Abort triggered: {abort_reason} at step {step}")
                break

            # 3. Repetition loop check
            if has_repetition(generated):
                aborted = True
                abort_reason = "repetition loop detected"
                abort_step = step
                print(f"Early Abort triggered: {abort_reason} at step {step}")
                break

        step_t0 = time.perf_counter()
        _, indices, values, ent = rust_step(next_token, pos, pos+1, max(50, top_k))
        step_time_ms = (time.perf_counter() - step_t0) * 1000.0
        entropies.append(ent)
        step_metrics.append({
            "event": "decode_step",
            "step": step + 1,
            "token_id": int(next_token),
            "token_text": tok.decode([next_token]) if tok is not None else "",
            "entropy": float(ent),
            "step_time_ms": float(step_time_ms),
            "cache_warm_hits": None,
            "cache_cold_loads": None,
        })

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
    return generated, entropies, step_metrics, aborted, abort_reason, abort_step

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.6 Rust executor smoke/full generation")
    parser.add_argument("fusion_ratio", nargs="?", type=float, default=None)
    parser.add_argument("moe_on_deltanet", nargs="?", type=int, default=None)
    parser.add_argument("--strategy", type=str, default="safe",
                        help="Path to strategy config JSON or preset name (safe, fast, turbo)")
    parser.add_argument("--warmup-tokens", type=int, default=100,
                        help="Number of warmup tokens for OS page cache. Use 0 for a light smoke test.")
    parser.add_argument("--max-tokens", type=int, default=15,
                        help="Number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. 0 = greedy.")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Sampling top-k. Ignored when temperature=0.")
    parser.add_argument("--prompt", default="The meaning of life is",
                        help="User prompt content before chat templating.")
    parser.add_argument("--early-abort", action="store_true",
                        help="Enable early abort on token/entropy degradation.")
    parser.add_argument("--trace-record-path", default=None, help="Path to save trace JSON")
    parser.add_argument("--trace-replay-path", default=None, help="Path to load trace JSON for replay")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load strategy configuration
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

    # Override with positional args if provided
    if args.fusion_ratio is not None:
        strategy_dict["fusion_ratio"] = args.fusion_ratio
    if args.moe_on_deltanet is not None:
        strategy_dict["moe_on_deltanet"] = args.moe_on_deltanet

    # Configure runner
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

    print(
        f"Strategy: ΔN={strategy_dict['fusion_ratio']:.0%} ({int(30*strategy_dict['fusion_ratio'])}/30 layers), "
        f"MoE on ΔN={'yes' if strategy_dict['moe_on_deltanet'] else 'no'}"
    )

    if args.warmup_tokens > 0:
        if hasattr(lib, "lko_runner_warmup"):
            if hasattr(lib, "lko_runner_set_fusion_ratio"):
                lib.lko_runner_set_fusion_ratio(0.0)
            print(f"Warming OS page cache... ({args.warmup_tokens} tokens)")
            lib.lko_runner_warmup(args.warmup_tokens)
            if hasattr(lib, "lko_runner_set_fusion_ratio"):
                lib.lko_runner_set_fusion_ratio(strategy_dict["fusion_ratio"])
    else:
        print("Skipping OS page cache warmup for smoke test")

    if hasattr(lib, "lko_runner_reset_kv_cache"):
        lib.lko_runner_reset_kv_cache()
    if hasattr(lib, "lko_runner_reset_moe_stats"):
        lib.lko_runner_reset_moe_stats()

    from transformers import AutoTokenizer
    snap = sorted(os.listdir(
        "/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"))[-1]
    tok_dir = f"/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/{snap}"
    tok = AutoTokenizer.from_pretrained(tok_dir)
    integrity_hashes = get_model_integrity_hashes(tok_dir, BIN_DIR)
    print(f"Vocab: {tok.vocab_size}\n")

    t_start_all = time.perf_counter()
    prompt = args.prompt
    msgs = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]

    print(f"── Prompt: \"{prompt}\" ──")
    gen, entropies, step_metrics, aborted, abort_reason, abort_step = generate(
        ids, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, tok=tok, early_abort=args.early_abort
    )
    text = tok.decode(gen, skip_special_tokens=True)
    print(f"  Output: {text}")

    total_time_ms = (time.perf_counter() - t_start_all) * 1000.0

    # Retrieve MoE stats at end of run
    avg_executed_experts = 0.0
    avg_bytes_read = 0.0
    warm_hit_rate = 0.0
    cold_load_count = 0
    logical_expert_bytes_requested = 0
    actual_expert_bytes_loaded = 0
    resident_cache_bytes_reused = 0
    resident_cache_hit_count = 0
    resident_cache_miss_count = 0
    direct_cold_load_count = 0
    resident_cache_enabled = False
    resident_cache_capacity_bytes = 0
    resident_cache_resident_bytes = 0
    resident_cache_hit_rate = 0.0
    moe_fraction = 0.0
    avg_forward_wall_ms = 0.0
    avg_moe_wall_ms = 0.0
    stats_json = None

    if hasattr(lib, "lko_runner_get_moe_stats_json"):
        stats_ptr = lib.lko_runner_get_moe_stats_json()
        if stats_ptr:
            stats_str = ctypes.cast(stats_ptr, ctypes.c_char_p).value.decode("utf-8")
            stats_json = json.loads(stats_str)
            summary_stats = stats_json.get("summary", {})
            fwd_stats = stats_json.get("forward_summary", {})
            
            avg_executed_experts = summary_stats.get("avg_executed_experts", 0.0)
            avg_bytes_read = summary_stats.get("avg_bytes_read", 0.0)
            logical_expert_bytes_requested = int(summary_stats.get("logical_expert_bytes_requested", 0))
            actual_expert_bytes_loaded = int(summary_stats.get("actual_expert_bytes_loaded", 0))
            resident_cache_bytes_reused = int(summary_stats.get("resident_cache_bytes_reused", 0))
            resident_cache_hit_count = int(summary_stats.get("resident_cache_hit_count", 0))
            resident_cache_miss_count = int(summary_stats.get("resident_cache_miss_count", 0))
            direct_cold_load_count = int(summary_stats.get("direct_cold_load_count", 0))
            resident_cache_enabled = bool(summary_stats.get("resident_cache_enabled", False))
            resident_cache_capacity_bytes = int(summary_stats.get("resident_cache_capacity_bytes", 0))
            resident_cache_resident_bytes = int(summary_stats.get("resident_cache_resident_bytes", 0))
            resident_cache_hit_rate = float(summary_stats.get("resident_cache_hit_rate", 0.0))
            
            warm_hits = summary_stats.get("avg_warm_hit_count", 0.0) * summary_stats.get("total_calls", 0.0)
            cold_hits = summary_stats.get("avg_cold_hit_count", 0.0) * summary_stats.get("total_calls", 0.0)
            if (warm_hits + cold_hits) > 0:
                warm_hit_rate = warm_hits / (warm_hits + cold_hits)
            cold_load_count = int(cold_hits)
            
            avg_forward_wall_ms = fwd_stats.get("avg_forward_wall_ms", 0.0)
            avg_moe_wall_ms = fwd_stats.get("avg_moe_wall_ms_per_token", 0.0)
            if avg_forward_wall_ms > 0:
                moe_fraction = avg_moe_wall_ms / avg_forward_wall_ms

    # Early abort stats
    first_garbage = None
    first_repetition = None
    if aborted:
        if "garbage" in abort_reason:
            first_garbage = abort_step
        elif "repetition" in abort_reason:
            first_repetition = abort_step

    # Output summary compliant with METRICS_SCHEMA.md
    import datetime
    run_dir = Path(f"runs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save effective strategy.json inside run directory
    with open(run_dir / "strategy.json", "w") as f:
        json.dump(strategy_dict, f, indent=2)
    with open(run_dir / "output.txt", "w") as f:
        f.write(text)
    with open(run_dir / "metrics.jsonl", "w") as f:
        for metric in step_metrics:
            f.write(json.dumps(metric, ensure_ascii=False) + "\n")
        if stats_json is not None:
            for event in stats_json.get("moe_io_events", []):
                f.write(json.dumps({"event": "moe_io", **event}, ensure_ascii=False) + "\n")

    summary = {
        "strategy_name": args.strategy,
        "strategy_hash": strategy_hash,
        "git_commit": get_git_commit(),
        "model_id": "Qwen3.6-35B-A3B",
        "prompt_hash": prompt_hash,
        "token_budget": args.max_tokens,
        "generated_tokens": len(gen),
        "tok_s": len(gen) / (total_time_ms / 1000.0) if total_time_ms > 0 else 0.0,
        "total_wall_ms": total_time_ms,
        "forward_wall_ms_avg": avg_forward_wall_ms,
        "moe_wall_ms_avg": avg_moe_wall_ms,
        "non_moe_wall_ms_avg": max(0.0, avg_forward_wall_ms - avg_moe_wall_ms),
        "moe_fraction": moe_fraction,
        "avg_experts_per_layer": avg_executed_experts,
        "avg_executed_experts": avg_executed_experts, # Alias for backwards compatibility in tests
        "avg_bytes_read": avg_bytes_read,
        "logical_expert_bytes_requested": logical_expert_bytes_requested,
        "actual_expert_bytes_loaded": actual_expert_bytes_loaded,
        "resident_cache_bytes_reused": resident_cache_bytes_reused,
        "resident_cache_hit_count": resident_cache_hit_count,
        "resident_cache_miss_count": resident_cache_miss_count,
        "direct_cold_load_count": direct_cold_load_count,
        "resident_cache_enabled": resident_cache_enabled,
        "resident_cache_capacity_bytes": resident_cache_capacity_bytes,
        "resident_cache_resident_bytes": resident_cache_resident_bytes,
        "resident_cache_hit_rate": resident_cache_hit_rate,
        "warm_hit_rate": warm_hit_rate,
        "cold_load_count": cold_load_count,
        "first_garbage": first_garbage,
        "first_repetition": first_repetition,
        "entropy_first": float(entropies[0]) if entropies else 0.0,
        "entropy_last": float(entropies[-1]) if entropies else 0.0,
        "entropy_min": float(min(entropies)) if entropies else 0.0,
        "entropy_max": float(max(entropies)) if entropies else 0.0,
        
        # Early abort info
        "aborted": aborted,
        "abort_reason": abort_reason,
        "abort_step": abort_step,
        "output": text,
        
        # Debug/Integrity metadata
        "tokenizer_hash": integrity_hashes["tokenizer_hash"],
        "config_hash": integrity_hashes["config_hash"],
        "lm_head_hash": integrity_hashes["lm_head_hash"],
        "embed_hash": integrity_hashes["embed_hash"],
        "weight_manifest_hash": integrity_hashes["weight_manifest_hash"],
        "integrity_hashes": integrity_hashes,
        "effective_debug_switches": strategy_dict.get("debug_switches"),
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWritten run summary to {summary_path}")

    # Update latest runs/summary.json
    latest_summary_path = Path("runs/summary.json")
    with open(latest_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated runs/summary.json")

if __name__ == "__main__":
    main()
