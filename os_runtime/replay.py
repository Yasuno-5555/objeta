"""Runtime trace record → replay for deterministic debugging.

Two modes:
  1. Record: intercept all observations and scheduler decisions → JSON-lines trace
  2. Replay: feed recorded observations back through the scheduler,
     comparing decisions against the recorded trace.

The trace format is the same as the logging.TokenLog JSON-lines format,
so any run with file logging enabled is automatically replayable.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .logging import TokenLog, LayerAction


@dataclass
class TraceReplay:
    """Replay engine — loads a recorded trace and replays it through the scheduler.

    Usage:
        trace = TraceReplay.load("run_2026-05-18.jsonl")
        for i, recorded in enumerate(trace.tokens):
            # Feed recorded.entropy, recorded.steering to scheduler
            tc = scheduler.begin_token(recorded.entropy, recorded.steering, ...)
            # Compare scheduler's decisions against recorded.layer_actions
            match_result = trace.compare(i, scheduler)
    """

    tokens: list[TokenLog] = field(default_factory=list)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "TraceReplay":
        """Load a recorded trace from JSON-lines file."""
        trace = cls(source_path=Path(path))
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                tl = TokenLog(
                    token_idx=d.get("token_idx", 0),
                    token_id=d.get("token_id", -1),
                    token_text=d.get("token_text", ""),
                    entropy=d.get("entropy", 0.0),
                    steering=d.get("steering", 0.0),
                    top1_logit=d.get("top1_logit", 0.0),
                    is_repeat=d.get("is_repeat", False),
                    token_class=d.get("token_class", "default"),
                    collapse_score=d.get("collapse_score", 0.0),
                    collapse_status=d.get("collapse_status", "healthy"),
                    precision=d.get("precision", 16),
                    skip_rate=d.get("skip_rate", 0.0),
                    layers_run=d.get("layers_run", 0),
                    layers_skipped=d.get("layers_skipped", 0),
                    layers_low_precision=d.get("layers_low_precision", 0),
                    forward_ms=d.get("forward_ms", 0.0),
                    sample_ms=d.get("sample_ms", 0.0),
                    elapsed_ms=d.get("elapsed_ms", 0.0),
                )
                if "layer_actions" in d:
                    tl.layer_actions = [
                        LayerAction(
                            layer=a["layer"],
                            attn_ran=a["attn"],
                            ffn_ran=a["ffn"],
                            precision_used=a["prec"],
                        )
                        for a in d["layer_actions"]
                    ]
                trace.tokens.append(tl)
        return trace

    def save(self, path: str | Path):
        """Save trace to JSON-lines file."""
        with open(path, "w") as f:
            for tl in self.tokens:
                d = {
                    "token_idx": tl.token_idx,
                    "token_id": tl.token_id,
                    "token_text": tl.token_text,
                    "entropy": tl.entropy,
                    "steering": tl.steering,
                    "top1_logit": tl.top1_logit,
                    "is_repeat": tl.is_repeat,
                    "token_class": tl.token_class,
                    "collapse_score": tl.collapse_score,
                    "collapse_status": tl.collapse_status,
                    "precision": tl.precision,
                    "skip_rate": tl.skip_rate,
                    "layers_run": tl.layers_run,
                    "layers_skipped": tl.layers_skipped,
                    "layers_low_precision": tl.layers_low_precision,
                    "forward_ms": tl.forward_ms,
                    "sample_ms": tl.sample_ms,
                    "elapsed_ms": tl.elapsed_ms,
                }
                if tl.layer_actions:
                    d["layer_actions"] = [
                        {"layer": a.layer, "attn": a.attn_ran,
                         "ffn": a.ffn_ran, "prec": a.precision_used}
                        for a in tl.layer_actions
                    ]
                f.write(json.dumps(d) + "\n")

    def compare(self, token_idx: int,
                pred_class: str, pred_collapse: str,
                pred_skip_rate: float, pred_precision: int) -> dict:
        """Compare a replayed scheduler decision against the recorded trace.

        Returns a diff dict with any mismatches.
        """
        if token_idx >= len(self.tokens):
            return {"match": False, "error": "token_idx out of range"}

        recorded = self.tokens[token_idx]
        diffs = {}

        if recorded.token_class != pred_class:
            diffs["token_class"] = {
                "recorded": recorded.token_class,
                "replayed": pred_class,
            }
        if recorded.collapse_status != pred_collapse:
            diffs["collapse_status"] = {
                "recorded": recorded.collapse_status,
                "replayed": pred_collapse,
            }
        if abs(recorded.skip_rate - pred_skip_rate) > 0.01:
            diffs["skip_rate"] = {
                "recorded": recorded.skip_rate,
                "replayed": pred_skip_rate,
            }
        if recorded.precision != pred_precision:
            diffs["precision"] = {
                "recorded": recorded.precision,
                "replayed": pred_precision,
            }

        return {
            "match": len(diffs) == 0,
            "token_idx": token_idx,
            "diffs": diffs,
        }

    def compare_layer_actions(self, token_idx: int,
                              actions: list[LayerAction]) -> dict:
        """Compare per-layer execution decisions."""
        if token_idx >= len(self.tokens):
            return {"match": False, "error": "token_idx out of range"}

        recorded = self.tokens[token_idx]
        if not recorded.layer_actions:
            return {"match": True, "note": "no layer actions in trace"}

        if len(actions) != len(recorded.layer_actions):
            return {"match": False,
                    "error": f"layer count mismatch: {len(actions)} vs {len(recorded.layer_actions)}"}

        mismatches = []
        for i, (a, r) in enumerate(zip(actions, recorded.layer_actions)):
            if (a.attn_ran != r.attn_ran or
                a.ffn_ran != r.ffn_ran or
                a.precision_used != r.precision_used):
                mismatches.append({
                    "layer": i,
                    "replayed": {"attn": a.attn_ran, "ffn": a.ffn_ran,
                                 "prec": a.precision_used},
                    "recorded": {"attn": r.attn_ran, "ffn": r.ffn_ran,
                                 "prec": r.precision_used},
                })

        return {
            "match": len(mismatches) == 0,
            "token_idx": token_idx,
            "n_mismatches": len(mismatches),
            "mismatches": mismatches[:10],  # cap for readability
        }

    def stats(self) -> dict:
        """Compute aggregate statistics from the trace."""
        if not self.tokens:
            return {}

        n = len(self.tokens)
        classes = {}
        collapse_events = 0
        for tl in self.tokens:
            classes[tl.token_class] = classes.get(tl.token_class, 0) + 1
            if tl.collapse_status != "healthy":
                collapse_events += 1

        return {
            "tokens": n,
            "token_classes": classes,
            "collapse_events": collapse_events,
            "avg_entropy": sum(tl.entropy for tl in self.tokens) / n,
            "avg_steering": sum(tl.steering for tl in self.tokens) / n,
            "avg_skip_rate": sum(tl.skip_rate for tl in self.tokens) / n,
            "avg_precision": sum(tl.precision for tl in self.tokens) / n,
        }
