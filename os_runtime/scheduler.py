"""LKO Reflexive Runtime OS — Scheduler Core.

The scheduler is the kernel. It replaces the static 'for layer in layers' loop
with state-dependent compute allocation:

    observe_state() → classify_token() → build_policy() → dispatch_execution()

Architecture:
  - Phase-aware static policy table (compiled at init)
  - Token-class-based dynamic override (runtime classification)
  - Precision governor (DVFS for LLM)
  - Collapse detector (trajectory stabilization)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class ExecMode(Enum):
    FULL = auto()
    COLLAPSE = auto()      # Koopman identity (J≈I)
    SKIP = auto()
    LOW_PRECISION = auto()


class Phase(Enum):
    SYNC = "sync"
    UNFOLD = "unfold"
    ISOMETRIC = "isometric"
    DIVERGENT = "divergent"
    OUTPUT = "output"


class TokenClass(Enum):
    REPETITIVE = "repetitive"
    STABLE = "stable"
    DEFAULT = "default"
    STEERING = "steering"
    TRANSITION = "transition"


class CollapseStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayerPolicy:
    """Per-layer execution policy — compiled at init, invariant per layer."""
    layer_idx: int = 0
    mode: ExecMode = ExecMode.FULL
    precision_bits: int = 16
    phase: Phase = Phase.ISOMETRIC
    is_sacred: bool = False       # Never skip (UNFOLD, output)
    is_steering: bool = False     # Course-correction layer (GQA, DIVERGENT)
    recompute: bool = True        # Cannot cache


@dataclass
class TokenState:
    """Per-token trajectory state — measured at runtime."""
    entropy: float = 0.0
    steering: float = 0.0
    token_class: TokenClass = TokenClass.DEFAULT
    precision: int = 16
    collapse_status: CollapseStatus = CollapseStatus.HEALTHY
    budget_used: int = 0


@dataclass
class SchedulerConfig:
    """Global OS configuration."""
    family: str = "residual_transport"
    backbone: str = "attention"
    safe_skip_ceiling: float = 0.30
    fusion_ratio: float = 0.50
    temporal_stride: int = 0
    # Thresholds
    entropy_stable_max: float = 0.05
    entropy_transition_min: float = 0.2
    entropy_collapse_warn: float = 0.1
    entropy_collapse_critical: float = 0.03
    steering_stable_max: float = 0.4
    steering_active_min: float = 0.5
    steering_transition_min: float = 0.6
    # Collapse detection
    collapse_entropy_window: int = 8
    collapse_repetition_threshold: int = 5
    collapse_steering_spike: float = 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# Hysteresis — prevents scheduler thrashing (OS scheduler's oldest problem)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HysteresisState:
    """Tracks current classification mode with enter/leave thresholds.

    Without hysteresis, token classes oscillate at boundary conditions.
    Enter thresholds are higher than leave thresholds, creating a dead zone
    that prevents mode flapping.

    Pattern (CPU scheduler since 1970s):
      enter STEERING:  steering > 0.6
      leave STEERING:  steering < 0.45
    """
    current_class: TokenClass = TokenClass.DEFAULT
    consecutive_stable: int = 0
    consecutive_unstable: int = 0
    precision_level: int = 8
    precision_stable_count: int = 0

    def classify(self, entropy: float, steering: float,
                 is_repeat: bool) -> TokenClass:
        """Classify with hysteresis — prevents boundary oscillation."""

        # Repetition has highest priority (no hysteresis needed — absolute signal)
        if is_repeat:
            self.consecutive_unstable += 1
            self.consecutive_stable = 0
            self.current_class = TokenClass.REPETITIVE
            return self.current_class

        # Enter TRANSITION: both entropy AND steering spike
        if (entropy > 0.22 and steering > 0.7 and
            self.current_class != TokenClass.TRANSITION):
            self.current_class = TokenClass.TRANSITION
            self.consecutive_unstable += 1
            self.consecutive_stable = 0
            return self.current_class

        # Stay in TRANSITION: lower leave threshold
        if self.current_class == TokenClass.TRANSITION:
            if steering > 0.5 or entropy > 0.15:
                self.consecutive_unstable += 1
                return self.current_class
            # Leave: both drop below leave thresholds
            self.consecutive_unstable = 0

        # Enter STEERING: high steering
        if steering > 0.6 and self.current_class != TokenClass.STEERING:
            self.current_class = TokenClass.STEERING
            self.consecutive_unstable += 1
            self.consecutive_stable = 0
            return self.current_class

        # Stay in STEERING: lower leave threshold
        if self.current_class == TokenClass.STEERING:
            if steering > 0.45:
                self.consecutive_unstable += 1
                return self.current_class
            self.consecutive_unstable = 0

        # Enter STABLE: sustained low entropy + low steering
        if (entropy < 0.04 and steering < 0.35 and
            self.current_class != TokenClass.STABLE):
            self.consecutive_stable += 1
            if self.consecutive_stable >= 2:  # need 2 consecutive
                self.current_class = TokenClass.STABLE
                self.consecutive_unstable = 0
                return self.current_class
        elif entropy > 0.06 or steering > 0.4:
            self.consecutive_stable = 0

        # Stay in STABLE: wider leave threshold
        if self.current_class == TokenClass.STABLE:
            if entropy < 0.08 and steering < 0.5:
                return self.current_class
            self.consecutive_stable = 0

        # Default
        self.current_class = TokenClass.DEFAULT
        self.consecutive_stable = 0
        return self.current_class

    def get_precision(self, base_precision: int,
                      collapse_status) -> int:
        """Rate-limit precision changes.

        Precision can only change by ±1 level per token
        to prevent thrashing.
        """
        target = base_precision

        # Allow immediate upgrade (never delay safety)
        if target > self.precision_level:
            self.precision_level = target
            self.precision_stable_count = 0
            return target

        # Downgrade requires sustained stability
        if target < self.precision_level:
            self.precision_stable_count += 1
            if self.precision_stable_count >= 3:  # 3 tokens stable before downgrade
                self.precision_level = max(target, self.precision_level - 1)
                self.precision_stable_count = 0
            return self.precision_level

        self.precision_stable_count = 0
        return self.precision_level


@dataclass
class CollapseHysteresis:
    """Prevents collapse status flapping.

    Enter WARNING:  requires signal above threshold
    Leave WARNING:  requires N consecutive healthy tokens

    Enter CRITICAL: immediate (safety)
    Leave CRITICAL: requires M consecutive non-critical tokens
    """
    current_status: CollapseStatus = CollapseStatus.HEALTHY
    healthy_count: int = 0
    warning_count: int = 0

    def update(self, raw_status: CollapseStatus,
               entropy_history_len: int) -> CollapseStatus:
        """Apply hysteresis to collapse detection."""

        # CRITICAL: enter immediately, leave slowly
        if raw_status == CollapseStatus.CRITICAL:
            self.current_status = CollapseStatus.CRITICAL
            self.healthy_count = 0
            return self.current_status

        if self.current_status == CollapseStatus.CRITICAL:
            self.healthy_count += 1
            if self.healthy_count >= 5:  # 5 healthy tokens to clear critical
                self.current_status = CollapseStatus.HEALTHY
                self.healthy_count = 0
            return self.current_status

        # WARNING: enter/leave with debounce
        if raw_status == CollapseStatus.WARNING:
            self.warning_count += 1
            if self.warning_count >= 2:  # 2 consecutive warnings to enter
                self.current_status = CollapseStatus.WARNING
                self.healthy_count = 0
            return self.current_status

        self.warning_count = 0
        self.healthy_count += 1
        if self.current_status == CollapseStatus.WARNING:
            if self.healthy_count >= 3:  # 3 healthy to clear warning
                self.current_status = CollapseStatus.HEALTHY
        return self.current_status


# ═══════════════════════════════════════════════════════════════════════════════
# Collapse Memory — persistent degradation tracking for long-context
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CollapseMemory:
    """Tracks collapse history over long sequences.

    In long-context generation (512+ tokens), single-token collapse detection
    is insufficient. Degradation propagates:
      token N:    steering misclassify
      token N+5:  precision drop
      token N+10: entropy collapse
      token N+20: repetition attractor locked
      token N+100: entire output destroyed

    CollapseMemory accumulates risk and can force conservative mode
    before the cascade becomes unrecoverable.
    """
    window_size: int = 128

    # Sliding windows
    collapse_history: list[float] = field(default_factory=list)
    steering_history: list[float] = field(default_factory=list)
    entropy_history: list[float] = field(default_factory=list)
    repetition_history: list[int] = field(default_factory=list)

    # Accumulators
    risk_score: float = 0.0
    conservative_mode: bool = False
    conservative_mode_entered_at: int = -1
    total_collapse_tokens: int = 0
    total_warning_tokens: int = 0

    def update(self, collapse_status, steering: float, entropy: float,
               is_repeat: bool, token_idx: int):
        """Feed one token's observation into memory."""

        # Status score: 0=healthy, 0.5=warning, 1.0=critical
        status_score = {
            CollapseStatus.HEALTHY: 0.0,
            CollapseStatus.WARNING: 0.5,
            CollapseStatus.CRITICAL: 1.0,
        }.get(collapse_status, 0.0)

        self.collapse_history.append(status_score)
        self.steering_history.append(steering)
        self.entropy_history.append(entropy)
        self.repetition_history.append(1 if is_repeat else 0)

        # Trim
        if len(self.collapse_history) > self.window_size:
            self.collapse_history.pop(0)
            self.steering_history.pop(0)
            self.entropy_history.pop(0)
            self.repetition_history.pop(0)

        # Count
        if collapse_status == CollapseStatus.CRITICAL:
            self.total_collapse_tokens += 1
        elif collapse_status == CollapseStatus.WARNING:
            self.total_warning_tokens += 1

        # Risk score: exponential moving average of collapse signals
        # with steering and repetition as amplifiers
        recent_window = min(32, len(self.collapse_history))
        recent_collapse = self.collapse_history[-recent_window:]
        recent_repeats = self.repetition_history[-recent_window:]
        recent_steering = self.steering_history[-recent_window:]

        mean_collapse = sum(recent_collapse) / recent_window
        repeat_rate = sum(recent_repeats) / recent_window
        mean_steering = sum(recent_steering) / recent_window

        # Risk formula: collapse severity × steering amplification × repetition penalty
        new_risk = mean_collapse * (1.0 + mean_steering) * (1.0 + repeat_rate * 3.0)
        self.risk_score = 0.8 * self.risk_score + 0.2 * new_risk  # EMA

        # Force conservative mode if risk sustained above threshold
        if self.risk_score > 0.4 and not self.conservative_mode:
            self.conservative_mode = True
            self.conservative_mode_entered_at = token_idx
        elif self.risk_score < 0.15 and self.conservative_mode:
            self.conservative_mode = False

    def should_force_conservative(self) -> bool:
        """Should the scheduler force conservative (no skip, fp16) mode?"""
        return self.conservative_mode

    def stats(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 4),
            "conservative_mode": self.conservative_mode,
            "conservative_since_token": self.conservative_mode_entered_at,
            "total_collapse_tokens": self.total_collapse_tokens,
            "total_warning_tokens": self.total_warning_tokens,
            "recent_collapse_rate": round(
                sum(self.collapse_history[-32:]) / max(1, len(self.collapse_history[-32:])), 3
            ) if self.collapse_history else 0,
            "recent_repeat_rate": round(
                sum(self.repetition_history[-32:]) / max(1, len(self.repetition_history[-32:])), 3
            ) if self.repetition_history else 0,
        }

    def reset(self):
        self.collapse_history.clear()
        self.steering_history.clear()
        self.entropy_history.clear()
        self.repetition_history.clear()
        self.risk_score = 0.0
        self.conservative_mode = False
        self.conservative_mode_entered_at = -1
        self.total_collapse_tokens = 0
        self.total_warning_tokens = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase Policy Table
# ═══════════════════════════════════════════════════════════════════════════════

def build_tinyllama_policy(n_layers: int = 22,
                           fusion_ratio: float = 0.50) -> list[LayerPolicy]:
    """Build static policy table for TinyLlama-1.1B from LKO phase structure.

    Phase map:
        L0-L1:   SYNC — sacred, fp16
        L2:      UNFOLD — sacred, J≠I, fp16
        L3-L13:  ISOMETRIC — J≈I, cacheable, q4-q5
        L14-L20: DIVERGENT — λ>0, steering, q8
        L21:     OUTPUT — sacred, fp16
    """
    stride = max(1, round(1.0 / max(fusion_ratio, 0.01)))
    delta_count = 0
    table = []

    for l in range(n_layers):
        if l <= 1:
            phase = Phase.SYNC
            sacred = True
            steering = False
        elif l == 2:
            phase = Phase.UNFOLD
            sacred = True
            steering = False
        elif l <= 13:
            phase = Phase.ISOMETRIC
            sacred = False
            steering = False
        elif l <= 20:
            phase = Phase.DIVERGENT
            sacred = False
            steering = True
        else:
            phase = Phase.OUTPUT
            sacred = True
            steering = False

        if sacred:
            mode = ExecMode.FULL
            precision = 16
        elif phase == Phase.ISOMETRIC:
            delta_count += 1
            if delta_count % stride == 0:
                mode = ExecMode.FULL
                precision = 8
            else:
                mode = ExecMode.COLLAPSE
                precision = 4
        else:  # DIVERGENT
            mode = ExecMode.FULL
            precision = 8

        table.append(LayerPolicy(
            layer_idx=l,
            mode=mode,
            precision_bits=precision,
            phase=phase,
            is_sacred=sacred,
            is_steering=steering,
            recompute=sacred or steering,
        ))

    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Precision Governor (DVFS for LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class PrecisionGovernor:
    """Maps token state → precision budget.

    Like CPU DVFS, but for numerical precision bits.
    """

    def get_precision(self, state: TokenState, policy: LayerPolicy) -> int:
        if policy.is_sacred or policy.is_steering:
            return 16
        if state.collapse_status == CollapseStatus.CRITICAL:
            return 16
        if state.collapse_status == CollapseStatus.WARNING:
            return 8

        tc = state.token_class
        if tc == TokenClass.TRANSITION:
            return 16
        if tc == TokenClass.STEERING:
            return 8
        if tc == TokenClass.STABLE:
            return 4
        if tc == TokenClass.REPETITIVE:
            return 3
        if state.entropy < 0.05 and state.steering < 0.4:
            return 4
        return 8


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Budget
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicBudget:
    """Per-token-class compute budget allocation."""

    BUDGET = {
        TokenClass.REPETITIVE:  (0.30, True,  0.80),
        TokenClass.STABLE:      (0.40, True,  0.50),
        TokenClass.DEFAULT:     (0.50, False, 0.27),
        TokenClass.STEERING:    (1.0,  False, 0.0),
        TokenClass.TRANSITION:  (1.0,  False, 0.0),
    }

    @classmethod
    def get_budget(cls, tc: TokenClass) -> tuple[float, bool, float]:
        return cls.BUDGET.get(tc, (0.50, False, 0.27))


# ═══════════════════════════════════════════════════════════════════════════════
# Collapse Detector
# ═══════════════════════════════════════════════════════════════════════════════

class CollapseDetector:
    """Detects trajectory collapse and triggers recovery."""

    def __init__(self, config: SchedulerConfig):
        self.cfg = config
        self.repetition_count: int = 0
        self.prev_token_id: int | None = None

    def update(self, token_id: int):
        if self.prev_token_id == token_id:
            self.repetition_count += 1
        else:
            self.repetition_count = 0
        self.prev_token_id = token_id

    def detect(self, entropy_history: list[float],
               steering: float) -> CollapseStatus:
        # Repetition lock
        if self.repetition_count >= self.cfg.collapse_repetition_threshold:
            return CollapseStatus.CRITICAL

        # Entropy collapse
        if len(entropy_history) >= self.cfg.collapse_entropy_window:
            recent = entropy_history[-self.cfg.collapse_entropy_window:]
            mean_ent = sum(recent) / len(recent)
            if mean_ent < self.cfg.entropy_collapse_critical:
                return CollapseStatus.CRITICAL
            if mean_ent < self.cfg.entropy_collapse_warn:
                return CollapseStatus.WARNING

        # Steering spike
        if steering > self.cfg.collapse_steering_spike:
            return CollapseStatus.WARNING

        return CollapseStatus.HEALTHY

    def reset(self):
        self.repetition_count = 0
        self.prev_token_id = None


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduler (Kernel)
# ═══════════════════════════════════════════════════════════════════════════════

class Scheduler:
    """Phase-aware trajectory controller — the OS kernel.

    Replaces 'for layer in layers' with state-dependent compute allocation:
    what to run, at what precision, for how long.
    """

    def __init__(self, config: SchedulerConfig | None = None,
                 n_layers: int = 22):
        self.config = config or SchedulerConfig()
        self.n_layers = n_layers
        self.policy_table = build_tinyllama_policy(
            n_layers, self.config.fusion_ratio)
        self.governor = PrecisionGovernor()
        self.collapse_detector = CollapseDetector(self.config)

        # State
        self.state = TokenState()
        self.entropy_history: list[float] = []

        # Hysteresis (prevents thrashing)
        self.token_hysteresis = HysteresisState()
        self.collapse_hysteresis = CollapseHysteresis()

        # Collapse memory (long-context degradation tracking)
        self.collapse_memory = CollapseMemory()

        # Counters
        self.token_count = 0
        self.layers_run = 0
        self.layers_skipped = 0
        self.layers_low_precision = 0
        self.temporal_skips = 0
        self.class_oscillations = 0
        self._last_class: TokenClass | None = None

    # ── Token lifecycle ──

    def begin_token(self, entropy: float, steering: float,
                    prev_token_id: int | None = None,
                    predicted_token_id: int | None = None) -> TokenClass:
        """Called at start of each token. Classifies with hysteresis."""
        self.token_count += 1

        # Update collapse detector
        if prev_token_id is not None and predicted_token_id is not None:
            self.collapse_detector.update(predicted_token_id)

        # Update history
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > max(64, self.config.collapse_entropy_window * 2):
            self.entropy_history.pop(0)

        # Classify WITH hysteresis — prevents boundary oscillation
        is_repeat = (
            predicted_token_id == prev_token_id
            if prev_token_id is not None and predicted_token_id is not None
            else False
        )
        tc = self.token_hysteresis.classify(entropy, steering, is_repeat)

        # Track class oscillations
        if self._last_class is not None and tc != self._last_class:
            self.class_oscillations += 1
        self._last_class = tc

        # Collapse detection WITH hysteresis — prevents false positives
        raw_cs = self.collapse_detector.detect(self.entropy_history, steering)
        cs = self.collapse_hysteresis.update(
            raw_cs, len(self.entropy_history))

        # Precision WITH hysteresis — rate-limited
        raw_prec = self.governor.get_precision(
            TokenState(entropy=entropy, steering=steering,
                      token_class=tc, collapse_status=cs),
            LayerPolicy())
        prec = self.token_hysteresis.get_precision(raw_prec, cs)

        # Feed collapse memory (long-context degradation tracking)
        self.collapse_memory.update(
            cs, steering, entropy, is_repeat, self.token_count)

        self.state = TokenState(
            entropy=entropy,
            steering=steering,
            token_class=tc,
            precision=prec,
            collapse_status=cs,
        )

        return tc

    # ── Dispatch ──

    def should_run_attn(self, layer_idx: int) -> bool:
        """Should attention run at this layer?"""
        policy = self.policy_table[layer_idx]
        tc = self.state.token_class

        # Sacred and steering layers always run
        if policy.is_sacred or policy.is_steering:
            self.layers_run += 1
            return True

        # Collapse forces full compute
        if self.state.collapse_status == CollapseStatus.CRITICAL:
            self.layers_run += 1
            return True

        # Long-context: conservative mode overrides all skip
        if self.collapse_memory.should_force_conservative():
            self.layers_run += 1
            return True

        # Temporal stride
        if self.config.temporal_stride > 1 and \
           self.token_count % self.config.temporal_stride != 0 and \
           self.state.steering < 0.5:
            self.temporal_skips += 1
            self.layers_skipped += 1
            return False

        # Token-class-based skip
        _, _, skip_frac = DynamicBudget.get_budget(tc)

        if tc == TokenClass.REPETITIVE:
            if layer_idx % 4 != 0:
                self.layers_skipped += 1
                return False
        elif tc == TokenClass.STABLE:
            if layer_idx % 2 != 0:
                self.layers_skipped += 1
                return False
        elif tc in (TokenClass.STEERING, TokenClass.TRANSITION):
            pass  # never skip
        else:  # DEFAULT
            if policy.mode == ExecMode.COLLAPSE:
                self.layers_skipped += 1
                return False

        self.layers_run += 1
        return True

    def should_run_ffn(self, layer_idx: int) -> bool:
        """Should FFN run at this layer?"""
        policy = self.policy_table[layer_idx]
        if policy.is_sacred or policy.is_steering:
            return True
        if self.state.collapse_status == CollapseStatus.CRITICAL:
            return True
        tc = self.state.token_class
        if tc == TokenClass.REPETITIVE and policy.phase == Phase.ISOMETRIC:
            self.layers_low_precision += 1
            return False
        return True

    def get_precision(self, layer_idx: int) -> int:
        """Get target precision for this (layer, state) pair."""
        policy = self.policy_table[layer_idx]
        return self.governor.get_precision(self.state, policy)

    # ── Stats ──

    def stats(self) -> dict:
        total = max(self.layers_run + self.layers_skipped +
                    self.layers_low_precision, 1)
        return {
            "token_class": self.state.token_class.value,
            "collapse": self.state.collapse_status.value,
            "entropy": round(self.state.entropy, 4),
            "steering": round(self.state.steering, 4),
            "precision": self.state.precision,
            "layers_run": self.layers_run,
            "layers_skipped": self.layers_skipped,
            "layers_low_prec": self.layers_low_precision,
            "temporal_skips": self.temporal_skips,
            "skip_rate": round(self.layers_skipped / total, 3),
            "class_oscillations": self.class_oscillations,
            "hysteresis": {
                "current_class": self.token_hysteresis.current_class.value,
                "consecutive_stable": self.token_hysteresis.consecutive_stable,
                "precision_level": self.token_hysteresis.precision_level,
                "collapse_healthy_count": self.collapse_hysteresis.healthy_count,
            },
            "collapse_memory": self.collapse_memory.stats(),
            "config": {
                "family": self.config.family,
                "backbone": self.config.backbone,
                "fusion_ratio": self.config.fusion_ratio,
                "temporal_stride": self.config.temporal_stride,
            },
        }

    def reset(self):
        self.state = TokenState()
        self.entropy_history.clear()
        self.collapse_detector.reset()
        self.token_hysteresis = HysteresisState()
        self.collapse_hysteresis = CollapseHysteresis()
        self.collapse_memory.reset()
        self.layers_run = 0
        self.layers_skipped = 0
        self.layers_low_precision = 0
        self.temporal_skips = 0
        self.token_count = 0
        self.class_oscillations = 0
        self._last_class = None
