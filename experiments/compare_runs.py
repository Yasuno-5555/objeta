#!/usr/bin/env python3
"""Compare two run summary.json files and print a compact verdict table."""
import argparse
import json
from pathlib import Path


METRICS = [
    ("tok_s", "higher"),
    ("total_wall_ms", "lower"),
    ("forward_wall_ms_avg", "lower"),
    ("moe_wall_ms_avg", "lower"),
    ("non_moe_wall_ms_avg", "lower"),
    ("moe_fraction", "lower"),
    ("avg_experts_per_layer", "lower"),
    ("avg_bytes_read", "lower"),
    ("logical_expert_bytes_requested", "lower"),
    ("actual_expert_bytes_loaded", "lower"),
    ("resident_cache_bytes_reused", "higher"),
    ("resident_cache_hit_rate", "higher"),
    ("resident_cache_resident_bytes", "lower"),
    ("warm_hit_rate", "higher"),
    ("cold_load_count", "lower"),
    ("direct_cold_load_count", "lower"),
    ("entropy_first", "lower"),
    ("entropy_last", "lower"),
    ("entropy_min", "lower"),
    ("entropy_max", "lower"),
]


def load_summary(path: str):
    with open(path, "r") as f:
        return json.load(f)


def fmt_value(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.4g}"
    return str(value)


def delta_str(a, b, mode):
    if a is None or b is None:
        return "-"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff = b - a
        if a != 0:
            pct = (diff / a) * 100.0
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.1f}%"
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.4g}"
    return "changed" if a != b else "same"


def verdict(metric, a, b, mode):
    if metric == "first_garbage":
        if a is None and b is None:
            return "OK"
        if a is None and b is not None:
            return "FAIL"
        return "CHECK"
    if metric == "first_repetition":
        if a is None and b is None:
            return "OK"
        if a is None and b is not None:
            return "FAIL"
        return "CHECK"
    if a is None or b is None:
        return "CHECK"
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "OK" if a == b else "CHECK"

    if metric.startswith("entropy_"):
        if b > a + 1.0:
            return "FAIL"
        if b > a + 0.2:
            return "CHECK"
        return "OK"
    if metric == "tok_s":
        if b < a * 0.9:
            return "FAIL"
        if b < a:
            return "CHECK"
        return "OK"
    if mode == "lower":
        if b > a * 1.25:
            return "FAIL"
        if b > a * 1.05:
            return "CHECK"
        return "OK"
    if mode == "higher":
        if a != 0 and b < a * 0.75:
            return "FAIL"
        if b < a:
            return "CHECK"
        return "OK"
    return "CHECK"


def main():
    parser = argparse.ArgumentParser(description="Compare two summary.json run artifacts")
    parser.add_argument("summary_a")
    parser.add_argument("summary_b")
    args = parser.parse_args()

    path_a = Path(args.summary_a)
    path_b = Path(args.summary_b)
    a = load_summary(str(path_a))
    b = load_summary(str(path_b))

    print(f"A: {path_a}")
    print(f"B: {path_b}")
    print()

    warnings = []
    for field in ("prompt_hash", "strategy_hash", "token_budget", "aborted"):
        av = a.get(field)
        bv = b.get(field)
        if av != bv:
            warnings.append(f"{field} mismatch: A={av!r} B={bv!r}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    rows = []
    for metric, mode in METRICS:
        rows.append((metric, fmt_value(a.get(metric)), fmt_value(b.get(metric)), delta_str(a.get(metric), b.get(metric), mode), verdict(metric, a.get(metric), b.get(metric), mode)))
    for metric in ("first_garbage", "first_repetition", "aborted", "abort_reason", "strategy_hash", "git_commit"):
        rows.append((metric, fmt_value(a.get(metric)), fmt_value(b.get(metric)), delta_str(a.get(metric), b.get(metric), "same"), verdict(metric, a.get(metric), b.get(metric), "same")))

    widths = [
        max(len("metric"), max(len(r[0]) for r in rows)),
        max(len("A"), max(len(r[1]) for r in rows)),
        max(len("B"), max(len(r[2]) for r in rows)),
        max(len("delta"), max(len(r[3]) for r in rows)),
        max(len("verdict"), max(len(r[4]) for r in rows)),
    ]
    header = ["metric", "A", "B", "delta", "verdict"]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(val.ljust(w) for val, w in zip(row, widths)))


if __name__ == "__main__":
    main()
