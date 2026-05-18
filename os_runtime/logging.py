"""Structured logging for the objeta OS runtime.

Every token generates a complete research data point:
  token, entropy, steering, precision, skip_rate, layer_actions, collapse_score

Output: JSON-lines file (one object per token) + run summary.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


@dataclass
class LayerAction:
    """Per-layer execution record."""
    layer: int
    attn_ran: bool
    ffn_ran: bool
    precision_used: int
    phase: str = ""


@dataclass
class TokenLog:
    """Complete per-token structured log — research data point."""

    # Identity
    token_idx: int = 0
    token_id: int = -1
    token_text: str = ""

    # Observation
    entropy: float = 0.0
    steering: float = 0.0
    top1_logit: float = 0.0
    is_repeat: bool = False

    # Classification
    token_class: str = "default"
    collapse_score: float = 0.0       # 0=healthy, 1=critical
    collapse_status: str = "healthy"

    # Allocation
    precision: int = 16               # target precision bits
    skip_rate: float = 0.0            # fraction of layers skipped
    layers_run: int = 0
    layers_skipped: int = 0
    layers_low_precision: int = 0

    # Per-layer detail (for DEBUG)
    layer_actions: list[LayerAction] = field(default_factory=list)

    # Timing
    elapsed_ms: float = 0.0
    forward_ms: float = 0.0
    sample_ms: float = 0.0

    # Fault injection (if active)
    fault_active: str = ""
    fault_type: str = ""


@dataclass
class RuntimeLogger:
    """Structured runtime logger — research data collector.

    Two output modes:
      - Console: human-readable per-token summary (INFO)
      - File: JSON-lines, one object per token (for analysis)
    """

    level: LogLevel = LogLevel.INFO
    output_file: Path | None = None
    token_logs: list[TokenLog] = field(default_factory=list)
    run_start: float = 0.0
    _warnings: list[str] = field(default_factory=list)
    _errors: list[str] = field(default_factory=list)
    _file_handle: Any = None
    _seq_t0: float = 0.0

    def start_run(self):
        self.run_start = time.perf_counter()
        self._seq_t0 = self.run_start
        self.token_logs.clear()
        self._warnings.clear()
        self._errors.clear()

        if self.output_file:
            self._file_handle = open(self.output_file, "w")

    def end_run(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def log_token(self, entry: TokenLog):
        """Record one token's complete execution data."""
        entry.elapsed_ms = (time.perf_counter() - self.run_start) * 1000
        self.token_logs.append(entry)

        # File output: full JSON (research data)
        if self._file_handle:
            d = {
                "token_idx": entry.token_idx,
                "token_id": entry.token_id,
                "token_text": entry.token_text,
                "entropy": round(entry.entropy, 6),
                "steering": round(entry.steering, 6),
                "top1_logit": round(entry.top1_logit, 3),
                "is_repeat": entry.is_repeat,
                "token_class": entry.token_class,
                "collapse_score": round(entry.collapse_score, 4),
                "collapse_status": entry.collapse_status,
                "precision": entry.precision,
                "skip_rate": round(entry.skip_rate, 4),
                "layers_run": entry.layers_run,
                "layers_skipped": entry.layers_skipped,
                "layers_low_precision": entry.layers_low_precision,
                "forward_ms": round(entry.forward_ms, 2),
                "sample_ms": round(entry.sample_ms, 2),
                "elapsed_ms": round(entry.elapsed_ms, 1),
                "fault_active": entry.fault_active,
                "fault_type": entry.fault_type,
            }
            if entry.layer_actions:
                d["layer_actions"] = [
                    {"layer": a.layer, "attn": a.attn_ran, "ffn": a.ffn_ran,
                     "prec": a.precision_used}
                    for a in entry.layer_actions
                ]
            self._file_handle.write(json.dumps(d) + "\n")
            self._file_handle.flush()

        # Console output
        if self.level.value <= LogLevel.INFO.value:
            collapse_flag = ""
            if entry.collapse_status == "warning":
                collapse_flag = " ⚠"
            elif entry.collapse_status == "critical":
                collapse_flag = " 🔴"
            fault_flag = f" [{entry.fault_active}]" if entry.fault_active else ""
            print(
                f"  tok={entry.token_idx:3d} id={entry.token_id:5d} "
                f"class={entry.token_class:<12s} "
                f"ent={entry.entropy:.3f} steer={entry.steering:.3f} "
                f"skip={entry.skip_rate*100:3.0f}% prec={entry.precision}bit "
                f"cs={entry.collapse_score:.2f}{collapse_flag}{fault_flag}"
            )

    def warn(self, message: str, context: dict[str, Any] | None = None):
        self._warnings.append(message)
        if self.level.value <= LogLevel.WARNING.value:
            ctx = context or {}
            print(f"  ⚠ WARNING: {message} {json.dumps(ctx, default=str)}")

    def error(self, message: str, context: dict[str, Any] | None = None):
        self._errors.append(message)
        if self.level.value <= LogLevel.ERROR.value:
            ctx = context or {}
            print(f"  🔴 ERROR: {message} {json.dumps(ctx, default=str)}")

    def run_summary(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.run_start
        n = len(self.token_logs)
        if n == 0:
            return {"tokens": 0, "elapsed_s": elapsed, "tok_per_s": 0}

        classes = {}
        for tl in self.token_logs:
            classes[tl.token_class] = classes.get(tl.token_class, 0) + 1

        avg_skip = sum(tl.skip_rate for tl in self.token_logs) / n
        avg_entropy = sum(tl.entropy for tl in self.token_logs) / n
        avg_steering = sum(tl.steering for tl in self.token_logs) / n
        avg_precision = sum(tl.precision for tl in self.token_logs) / n
        collapse_events = sum(
            1 for tl in self.token_logs if tl.collapse_status != "healthy"
        )
        avg_collapse_score = sum(tl.collapse_score for tl in self.token_logs) / n

        return {
            "tokens": n,
            "elapsed_s": round(elapsed, 2),
            "tok_per_s": round(n / elapsed, 2) if elapsed > 0 else 0,
            "token_classes": classes,
            "avg_skip_rate": round(avg_skip, 3),
            "avg_entropy": round(avg_entropy, 4),
            "avg_steering": round(avg_steering, 4),
            "avg_precision": round(avg_precision, 1),
            "avg_collapse_score": round(avg_collapse_score, 3),
            "collapse_events": collapse_events,
            "warnings": len(self._warnings),
            "errors": len(self._errors),
        }

    def print_summary(self):
        s = self.run_summary()
        print()
        print("═" * 60)
        print("  Run Summary")
        print("═" * 60)
        print(f"  Tokens:       {s.get('tokens', 0)}")
        print(f"  Time:         {s.get('elapsed_s', 0):.1f}s")
        print(f"  Throughput:   {s.get('tok_per_s', 0):.1f} tok/s")
        print(f"  Token classes: {s.get('token_classes', {})}")
        print(f"  Avg entropy:  {s.get('avg_entropy', 0):.3f}")
        print(f"  Avg steering: {s.get('avg_steering', 0):.3f}")
        print(f"  Avg precision:{s.get('avg_precision', 0):.1f} bits")
        print(f"  Avg skip:     {s.get('avg_skip_rate', 0)*100:.1f}%")
        print(f"  Collapse ev:  {s.get('collapse_events', 0)}")
        print(f"  Warnings:     {s.get('warnings', 0)}")
        print(f"  Errors:       {s.get('errors', 0)}")
