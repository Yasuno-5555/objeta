#!/usr/bin/env python3
"""Bundle a run directory with adjacent metadata for handoff to another agent."""
import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def maybe_copy(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def run_text(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except Exception as exc:
        return f"<unavailable: {exc}>"


def latest_oracle_summaries():
    oracle_runs = sorted(Path("runs").glob("oracle_run_*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for path in oracle_runs[:3]:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        result.append({
            "path": str(path),
            "pass": data.get("pass"),
            "failure_type": data.get("failure_type"),
            "suspected_area": data.get("suspected_area"),
            "prompt_hash": data.get("prompt_hash"),
            "prompt_mode": data.get("prompt_mode"),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="Package a run directory into a handoff bundle")
    parser.add_argument("run_dir", help="Path like runs/run_YYYYMMDD_HHMMSS")
    parser.add_argument("--output-dir", default="runs/bundles", help="Parent directory for bundles")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    bundle_root = Path(args.output_dir).resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = bundle_root / f"{run_dir.name}_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    copied = []
    for name in ("summary.json", "strategy.json", "metrics.jsonl", "output.txt"):
        if maybe_copy(run_dir / name, bundle_dir / name):
            copied.append(name)

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    oracle_result = {
        "source_run": str(run_dir),
        "latest_oracles": latest_oracle_summaries(),
    }
    (bundle_dir / "oracle_result.json").write_text(json.dumps(oracle_result, indent=2))
    copied.append("oracle_result.json")

    (bundle_dir / "git_commit.txt").write_text(summary.get("git_commit", run_text(["git", "rev-parse", "HEAD"])).strip() + "\n")
    copied.append("git_commit.txt")

    git_diff = run_text(["git", "diff", "--", "."])
    (bundle_dir / "git_diff.patch").write_text(git_diff)
    copied.append("git_diff.patch")

    system_info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "uname": run_text(["uname", "-a"]).strip(),
        "sw_vers": run_text(["sw_vers"]),
        "git_status_short": run_text(["git", "status", "--short"]),
    }
    (bundle_dir / "system_info.json").write_text(json.dumps(system_info, indent=2))
    copied.append("system_info.json")

    manifest = {
        "bundle_dir": str(bundle_dir),
        "source_run": str(run_dir),
        "copied_files": copied,
        "summary_present": summary_path.exists(),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
