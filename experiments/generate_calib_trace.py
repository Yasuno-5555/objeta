#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ROOT / "calib" / "prompts" / "general.jsonl"
DEFAULT_STRATEGY = ROOT / "configs" / "safe_exact.json"
DEFAULT_TRACE_OUT = ROOT / "runs" / "calib_trace_general.jsonl"
DEFAULT_SUMMARY_OUT = ROOT / "runs" / "calib_trace_general_summary.json"


def parse_args():
    p = argparse.ArgumentParser(description="Generate calibration trace JSONL for objeta-aot specialize")
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS, help="Prompt corpus JSONL")
    p.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY, help="Strategy JSON path")
    p.add_argument("--runtime-profile", type=Path, default=None, help="Optional runtime_profile.json")
    p.add_argument("--runtime-pack", type=Path, default=None, help="Optional runtime pack dir")
    p.add_argument("--model-dir", type=Path, default=None, help="Optional Qwen model dir for logical expert count")
    p.add_argument("--out", type=Path, default=DEFAULT_TRACE_OUT, help="Output calibration JSONL")
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT, help="Output summary JSON")
    p.add_argument("--max-prompts", type=int, default=0, help="Limit number of prompts (0 = all)")
    p.add_argument("--max-tokens", type=int, default=8, help="Generated tokens per prompt")
    p.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature")
    p.add_argument("--categories", type=str, default="", help="Comma-separated category filter")
    p.add_argument("--append-trace", type=Path, default=None, help="Existing calibration trace JSONL to merge with")
    return p.parse_args()


def load_prompts(path: Path):
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt_id" not in obj or "prompt" not in obj:
                raise ValueError(f"{path}:{line_no} missing prompt_id or prompt")
            prompts.append(obj)
    return prompts


def load_existing_trace(path: Path):
    events = []
    if path is None or not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def infer_model_dir() -> Path:
    snap_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots"
    snaps = sorted(snap_root.iterdir())
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under {snap_root}")
    return snaps[-1]


def load_logical_total_experts(model_dir: Path) -> int:
    config_path = model_dir / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config", {})
    num_layers = cfg.get("num_hidden_layers", text_cfg.get("num_hidden_layers", 0))
    num_experts = cfg.get("num_experts", text_cfg.get("num_experts", cfg.get("num_local_experts", text_cfg.get("num_local_experts", 0))))
    if not num_layers or not num_experts:
        return 0
    return int(num_layers) * int(num_experts)


def newest_run_dir(before: set[Path], runs_root: Path) -> Path:
    after = {p for p in runs_root.glob("run_*") if p.is_dir()}
    created = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if created:
        return created[-1]
    newest = sorted(after, key=lambda p: p.stat().st_mtime)
    if not newest:
        raise RuntimeError("No run directories found after execution")
    return newest[-1]


def run_prompt(prompt_obj, args, runs_root: Path):
    before = {p for p in runs_root.glob("run_*") if p.is_dir()}
    env = os.environ.copy()
    env["OBJETA_GROUP_PRERESOLVE_TOP_N"] = "0"
    env["OBJETA_GROUP_PRERESOLVE_MAX_BYTES"] = "0"
    env["OBJETA_GOVERNOR_MODE"] = "disabled"
    if args.runtime_profile:
        env["OBJETA_RUNTIME_PROFILE_PATH"] = str(args.runtime_profile)
    if args.runtime_pack:
        env["OBJETA_RUNTIME_PACK_PATH"] = str(args.runtime_pack)

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "experiments" / "qwen36_full_rust.py"),
        "--strategy",
        str(args.strategy),
        "--warmup-tokens",
        "0",
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--prompt",
        prompt_obj["prompt"],
    ]
    print(f"[calib] prompt={prompt_obj['prompt_id']} category={prompt_obj.get('category','')} max_tokens={args.max_tokens}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    run_dir = newest_run_dir(before, runs_root)
    return run_dir


def extract_events(prompt_obj, run_dir: Path):
    stats_path = run_dir / "moe_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing moe_stats.json in {run_dir}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    out = []
    for event in stats.get("moe_io_events", []):
        selected_experts = event.get("selected_experts") or []
        selected_weights = event.get("selected_weights") or []
        if not selected_experts or not selected_weights:
            continue
        out.append({
            "prompt_id": prompt_obj["prompt_id"],
            "category": prompt_obj.get("category"),
            "task_profile": prompt_obj.get("task_profile", "general"),
            "phase": "prefill" if int(event.get("step", 0)) == 0 else "decode",
            "token_id": event.get("token_id"),
            "layer": event.get("layer_id"),
            "selected_experts": selected_experts,
            "selected_weights": selected_weights,
            "routing_mass_kept_pre_renorm": event.get("routing_mass_kept_pre_renorm", 0.0),
            "routing_mass_dropped_pre_renorm": event.get("routing_mass_dropped_pre_renorm", 0.0),
        })
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return out, summary


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts)
    if args.categories:
        wanted = {x.strip() for x in args.categories.split(",") if x.strip()}
        prompts = [p for p in prompts if p.get("category") in wanted]
    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    model_dir = args.model_dir or infer_model_dir()
    logical_total_experts = load_logical_total_experts(model_dir)
    runs_root = ROOT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    existing_events = load_existing_trace(args.append_trace)
    all_events = list(existing_events)
    run_summaries = []
    existing_unique_experts = set()
    unique_experts = set()
    per_layer_experts = defaultdict(set)
    per_task_experts = defaultdict(set)
    newly_discovered_experts_per_batch = []

    for ev in existing_events:
        layer = int(ev["layer"])
        task = ev.get("task_profile", "general")
        for expert in ev.get("selected_experts", []):
            exp_key = (layer, int(expert))
            existing_unique_experts.add(exp_key)
            unique_experts.add(exp_key)
            per_layer_experts[layer].add(int(expert))
            per_task_experts[task].add(exp_key)

    for prompt_obj in prompts:
        run_dir = run_prompt(prompt_obj, args, runs_root)
        events, summary = extract_events(prompt_obj, run_dir)
        all_events.extend(events)
        run_summaries.append({
            "prompt_id": prompt_obj["prompt_id"],
            "category": prompt_obj.get("category"),
            "run_dir": str(run_dir),
            "output": summary.get("output", ""),
            "finish_reason": summary.get("finish_reason"),
            "stopped_at_decode_token": summary.get("stopped_at_decode_token"),
            "tok_s": summary.get("tok_s"),
        })
        new_for_prompt = set()
        for ev in events:
            layer = int(ev["layer"])
            task = ev.get("task_profile", "general")
            for expert in ev["selected_experts"]:
                exp_key = (layer, int(expert))
                unique_experts.add(exp_key)
                per_layer_experts[layer].add(int(expert))
                per_task_experts[task].add(exp_key)
                if exp_key not in existing_unique_experts:
                    new_for_prompt.add(exp_key)
                    existing_unique_experts.add(exp_key)
        newly_discovered_experts_per_batch.append({
            "prompt_id": prompt_obj["prompt_id"],
            "category": prompt_obj.get("category"),
            "new_unique_experts": len(new_for_prompt),
        })

    with args.out.open("w", encoding="utf-8") as f:
        for ev in all_events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    num_experts_per_layer = 0
    if logical_total_experts and per_layer_experts:
        num_experts_per_layer = logical_total_experts // max(1, max(per_layer_experts.keys()) + 1)
    elif logical_total_experts:
        # Qwen3.6 known shape, but keep it derived if possible.
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        text_cfg = cfg.get("text_config", {})
        num_experts_per_layer = int(cfg.get("num_experts", text_cfg.get("num_experts", 0)) or 0)

    per_layer_coverage = {}
    for layer, experts in sorted(per_layer_experts.items()):
        denom = num_experts_per_layer if num_experts_per_layer else 0
        per_layer_coverage[str(layer)] = {
            "unique_experts": len(experts),
            "logical_total_experts_per_layer": denom,
            "coverage": (len(experts) / denom) if denom else 0.0,
        }

    per_task_profile_coverage = {}
    for task, experts in sorted(per_task_experts.items()):
        per_task_profile_coverage[task] = {
            "unique_experts": len(experts),
            "logical_total_experts": logical_total_experts,
            "coverage": (len(experts) / logical_total_experts) if logical_total_experts else 0.0,
        }

    coverage = (len(unique_experts) / logical_total_experts) if logical_total_experts else 0.0
    summary = {
        "prompt_count": len(run_summaries),
        "event_count": len(all_events),
        "unique_experts": len(unique_experts),
        "logical_total_experts": logical_total_experts,
        "coverage": coverage,
        "per_layer_coverage": per_layer_coverage,
        "per_task_profile_coverage": per_task_profile_coverage,
        "newly_discovered_experts_per_batch": newly_discovered_experts_per_batch,
        "existing_event_count": len(existing_events),
        "trace_path": str(args.out),
        "strategy": str(args.strategy),
        "runtime_profile": str(args.runtime_profile) if args.runtime_profile else None,
        "runtime_pack": str(args.runtime_pack) if args.runtime_pack else None,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "runs": run_summaries,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[calib] wrote trace: {args.out}")
    print(f"[calib] wrote summary: {args.summary_out}")
    print(f"[calib] prompts={len(prompts)} events={len(all_events)} unique_experts={len(unique_experts)} coverage={coverage*100:.2f}%")


if __name__ == "__main__":
    main()
