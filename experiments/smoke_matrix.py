#!/usr/bin/env python3
"""Verification matrix for all strategies with metrics schema validation."""
import os
import sys
import json
import math
import subprocess
from pathlib import Path

# Add project root to sys.path to import validate_summary_schema
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.qwen36_full_rust import validate_summary_schema, REQUIRED_FIELDS

def find_latest_run():
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda x: x.name)
    return subdirs[-1]

def run_strategy(strat_name):
    print(f"Running strategy: {strat_name}...")
    cmd = [
        sys.executable,
        "experiments/qwen36_full_rust.py",
        "--strategy", strat_name,
        "--max-tokens", "5",
        "--warmup-tokens", "0"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[{strat_name}] Execution error:")
        print(res.stderr)
        return None, f"Execution exit code {res.returncode}"
    
    latest_run = find_latest_run()
    if not latest_run:
        return None, "No run directory found"
        
    summary_path = latest_run / "summary.json"
    if not summary_path.exists():
        return None, f"summary.json missing at {latest_run}"
        
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
        return summary, None
    except Exception as e:
        return None, f"JSON load failed: {e}"

def main():
    strategies = ["safe", "fast", "turbo"]
    results = {}
    
    print("==========================================")
    print("Starting Smoke Matrix and Schema Validation")
    print("==========================================")
    
    any_failed = False
    
    for strat in strategies:
        summary, err = run_strategy(strat)
        
        if err:
            results[strat] = {"status": "FAILED", "reason": err}
            any_failed = True
            continue
            
        # Try schema validation
        try:
            validate_summary_schema(summary)
            
            # Check content correctness matching strategy config limits
            min_exp = 2
            max_exp = 8
            avg_exp = summary.get("avg_experts_per_layer", 0.0)
            
            content_errs = []
            if strat == "safe":
                if abs(avg_exp - 8.0) > 1e-5:
                    content_errs.append(f"Safe strategy executed {avg_exp} experts (expected exactly 8.0)")
            else:
                if avg_exp < min_exp or avg_exp > max_exp:
                    content_errs.append(f"Pruned strategy executed {avg_exp} experts (outside bounds [{min_exp}, {max_exp}])")
                if avg_exp >= 8.0:
                    content_errs.append(f"Pruned strategy executed {avg_exp} experts (expected < 8.0)")
                    
            if content_errs:
                results[strat] = {"status": "FAILED", "reason": ", ".join(content_errs)}
                any_failed = True
            else:
                results[strat] = {
                    "status": "PASSED",
                    "tok_s": summary.get("tok_s"),
                    "avg_experts": avg_exp,
                    "warm_hit_rate": summary.get("warm_hit_rate")
                }
                
        except ValueError as ve:
            results[strat] = {"status": "FAILED", "reason": f"Schema Validation Failure:\n{ve}"}
            any_failed = True
            
    print("\n==========================================")
    print("Smoke Matrix Results:")
    print("==========================================")
    for strat, data in results.items():
        status = data["status"]
        if status == "PASSED":
            print(f"  {strat:<8}: \033[92m{status}\033[0m (tok/s={data['tok_s']:.2f}, avg_exp={data['avg_experts']:.2f}, warm_hit={data['warm_hit_rate']:.2f}%)")
        else:
            print(f"  {strat:<8}: \033[91m{status}\033[0m - Reason: {data['reason']}")
            
    if any_failed:
        print("\nResult: Matrix run FAILED")
        sys.exit(1)
    else:
        print("\nResult: Matrix run PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()
