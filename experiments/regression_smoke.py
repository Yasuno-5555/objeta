#!/usr/bin/env python3
"""Smoke regression test for Qwen3.6 Rust executor."""
import os
import sys
import json
import math
import subprocess
from pathlib import Path

def find_latest_run():
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    # sort by modification time or directory name (YYYYMMDD_HHMMSS)
    subdirs.sort(key=lambda x: x.name)
    return subdirs[-1]

def run_strategy(strat_name):
    print(f"\n==========================================")
    print(f"Running strategy: {strat_name}")
    print(f"==========================================")
    cmd = [
        sys.executable,
        "experiments/qwen36_full_rust.py",
        "--strategy", strat_name,
        "--max-tokens", "5",
        "--warmup-tokens", "0"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:")
        print(res.stdout)
        print("STDERR:")
        print(res.stderr)
        raise RuntimeError(f"Runner failed for strategy {strat_name} with exit code {res.returncode}")
    
    latest_run = find_latest_run()
    if not latest_run:
        raise RuntimeError("No run directory created under runs/")
        
    summary_path = latest_run / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"summary.json not found in {latest_run}")
        
    with open(summary_path, "r") as f:
        summary = json.load(f)
        
    with open(latest_run / "strategy.json", "r") as f:
        strategy = json.load(f)
        
    return summary, strategy

def main():
    strategies = ["safe", "fast", "turbo"]
    errors = []
    
    for strat in strategies:
        try:
            summary, strategy = run_strategy(strat)
            
            # 1. Check first_garbage is None
            fg = summary.get("first_garbage")
            if fg is not None:
                errors.append(f"[{strat}] first_garbage is not None: {fg}")
                
            # 2. Check entropy is not NaN or inf
            entropy = summary.get("avg_entropy")
            if entropy is None or math.isnan(entropy) or math.isinf(entropy):
                errors.append(f"[{strat}] invalid entropy: {entropy}")
                
            # 3. Check avg_executed_experts/layer is consistent with strategy constraints
            avg_experts = summary.get("avg_executed_experts")
            if avg_experts is None:
                errors.append(f"[{strat}] avg_executed_experts is missing in summary.json")
            else:
                min_exp = strategy.get("min_experts", 2)
                max_exp = strategy.get("max_experts", 8)
                print(f"[{strat}] avg_executed_experts: {avg_experts} (bounds: {min_exp}..{max_exp})")
                if strat == "safe":
                    # Safe should run exactly 8 experts (no top-p/contrib pruning by default)
                    if abs(avg_experts - 8.0) > 1e-5:
                        errors.append(f"[{strat}] safe strategy should execute exactly 8.0 experts, got {avg_experts}")
                elif strat in ["fast", "turbo"]:
                    # Pruned strategy should run less than 8.0 experts, but >= min_experts
                    if avg_experts < min_exp or avg_experts > max_exp:
                        errors.append(f"[{strat}] pruned strategy experts {avg_experts} out of bounds [{min_exp}, {max_exp}]")
                    if avg_experts >= 8.0:
                        errors.append(f"[{strat}] pruned strategy should execute < 8.0 experts, got {avg_experts}")
            
        except Exception as e:
            errors.append(f"[{strat}] Exception raised: {e}")
            
    if errors:
        print("\nSmoke Regression FAILures:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\nAll smoke regression tests PASSED.")
        sys.exit(0)

if __name__ == "__main__":
    main()
