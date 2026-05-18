"""Fault injection harness — tests collapse detector resilience.

Deliberately inject:
  - ForceQ3: all layers to q3 precision
  - ExcessiveSkip: skip all non-sacred attention
  - ExpertDrop: drop half of experts (MoE only)
  - HiddenNoise: inject Gaussian noise into hidden states
  - RandomClass: force random token classification

Each fault is injected at a specific token index for a specified duration.
The harness measures detection latency and recovery time.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FaultType(Enum):
    FORCE_Q3 = "force_q3"
    EXCESSIVE_SKIP = "excessive_skip"
    EXPERT_DROP = "expert_drop"
    HIDDEN_NOISE = "hidden_noise"
    RANDOM_CLASS = "random_class"


@dataclass
class FaultInjection:
    """A single fault injection event."""
    fault_type: FaultType
    token_idx: int | None = None        # None = all tokens
    duration: int | None = None         # None = until recovery, 0 = permanent
    intensity: float = 1.0              # 0.0-1.0

    def is_active(self, token_idx: int, tokens_since_fault: int) -> bool:
        """Check if this fault is active at the given token."""
        if self.token_idx is not None and token_idx < self.token_idx:
            return False
        if self.duration is not None and self.duration > 0:
            if tokens_since_fault >= self.duration:
                return False
        return True


@dataclass
class FaultTestResult:
    """Result of a single fault injection test."""
    fault_type: str
    token_idx: int
    detected: bool
    detection_latency: int             # tokens from injection to detection
    recovery_latency: int              # tokens from fault removal to recovery
    collapse_sequence: list[str]       # collapse status at each token during test
    notes: str = ""


@dataclass
class FaultHarness:
    """Fault injection test harness.

    Usage:
        harness = FaultHarness()
        harness.add(FaultInjection(FaultType.FORCE_Q3, token_idx=5, duration=10))

        for token_idx in range(max_tokens):
            active_faults = harness.active_faults(token_idx)
            # Apply faults to scheduler/observation...
            harness.record_status(token_idx, collapse_status)
            harness.check_detection(token_idx, collapse_status)

        results = harness.results()
    """

    injections: list[FaultInjection] = field(default_factory=list)
    _fault_start: dict[str, int] = field(default_factory=dict)  # fault_type → start token
    _fault_end: dict[str, int] = field(default_factory=dict)
    _statuses: list[tuple[int, str]] = field(default_factory=list)  # (token_idx, status)
    _results: list[FaultTestResult] = field(default_factory=list)

    def add(self, fault: FaultInjection):
        self.injections.append(fault)

    def active_faults(self, token_idx: int) -> list[FaultType]:
        """Return list of active fault types at this token."""
        active = []
        for inj in self.injections:
            start = inj.token_idx or 0
            tokens_since = token_idx - start
            if inj.is_active(token_idx, tokens_since):
                active.append(inj.fault_type)
        return active

    def apply_fault(self, fault_type: FaultType,
                    precision: int,
                    should_skip: bool) -> tuple[int, bool]:
        """Apply fault to precision and skip decision.

        Returns (modified_precision, modified_skip).
        """
        if fault_type == FaultType.FORCE_Q3:
            return 3, should_skip
        elif fault_type == FaultType.EXCESSIVE_SKIP:
            return precision, True
        else:
            return precision, should_skip

    def apply_hidden_noise(self, hidden: "np.ndarray",
                           intensity: float = 0.1) -> "np.ndarray":
        """Inject Gaussian noise into hidden state."""
        import numpy as np
        noise = np.random.randn(*hidden.shape).astype(hidden.dtype) * intensity
        return hidden + noise

    def start_fault(self, fault_type: FaultType, token_idx: int):
        key = fault_type.value
        if key not in self._fault_start:
            self._fault_start[key] = token_idx

    def end_fault(self, fault_type: FaultType, token_idx: int):
        key = fault_type.value
        if key not in self._fault_end:
            self._fault_end[key] = token_idx

    def record_status(self, token_idx: int, collapse_status: str):
        self._statuses.append((token_idx, collapse_status))

    def check_detection(self, token_idx: int, collapse_status: str):
        """Check if any fault has been detected (collapse status changed)."""
        for inj in self.injections:
            key = inj.fault_type.value
            # Auto-start the fault when first observed
            if key not in self._fault_start and self.active_faults(token_idx):
                if self.active_faults(token_idx):
                    self.start_fault(inj.fault_type, token_idx)
            if key in self._fault_start and key not in self._fault_end:
                if collapse_status in ("warning", "critical"):
                    self.end_fault(inj.fault_type, token_idx)

    def results(self) -> list[FaultTestResult]:
        results = []
        for inj in self.injections:
            key = inj.fault_type.value
            start = self._fault_start.get(key)
            end = self._fault_end.get(key)

            detected = end is not None
            detection_latency = (end - start) if detected and start is not None else -1

            # Recovery: after fault ends, how long until healthy?
            recovery_latency = -1
            if end is not None:
                for ti, status in self._statuses:
                    if ti > end and status == "healthy":
                        recovery_latency = ti - end
                        break

            # Collapse sequence during fault window
            seq = []
            if start is not None:
                fault_end = end or (start + (inj.duration or 100))
                for ti, status in self._statuses:
                    if start <= ti <= fault_end:
                        seq.append(status)

            results.append(FaultTestResult(
                fault_type=key,
                token_idx=start or 0,
                detected=detected,
                detection_latency=detection_latency,
                recovery_latency=recovery_latency,
                collapse_sequence=seq,
                notes=f"intensity={inj.intensity}, duration={inj.duration}",
            ))

        return results

    def save_results(self, path: str | Path):
        with open(path, "w") as f:
            json.dump([
                {
                    "fault_type": r.fault_type,
                    "token_idx": r.token_idx,
                    "detected": r.detected,
                    "detection_latency": r.detection_latency,
                    "recovery_latency": r.recovery_latency,
                    "collapse_sequence": r.collapse_sequence,
                    "notes": r.notes,
                }
                for r in self.results()
            ], f, indent=2)
        print(f"Fault test results saved to {path}")

    def print_results(self):
        print()
        print("═" * 60)
        print("  Fault Injection Results")
        print("═" * 60)
        for r in self.results():
            status = "✓ DETECTED" if r.detected else "✗ MISSED"
            det_str = f"{r.detection_latency} tok" if r.detected else "-"
            rec_str = f"{r.recovery_latency} tok" if r.recovery_latency >= 0 else "-"
            print(f"  {r.fault_type:<20s} {status:<12s} "
                  f"detect={det_str:<8s} recovery={rec_str:<8s} "
                  f"seq={r.collapse_sequence}")
