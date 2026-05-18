"""Runtime Routing Thermodynamics — closed-loop VM + routing control.

Three subsystems in one Governor:
  1. ThrashDetector     — VM pressure → auto λ/top-k/conservative trigger
  2. SemanticCollapse    — diversity floor, drift detection
  3. DynamicLambda       — token-class-aware locality strength

This is not a static configuration. It is a runtime control system that
adjusts routing thermodynamics based on observed system state.
"""

from dataclasses import dataclass, field
from collections import deque

import numpy as np


# ═══════════════════════════════════════════════════════════
# Thrash Detector — VM pressure governor
# ═══════════════════════════════════════════════════════════

@dataclass
class ThrashDetector:
    """Detects and responds to VM thrashing.

    Thrashing = working_set > RAM_budget → constant page faults.
    Response: increase locality λ, shrink top-k, force conservative mode,
              evict KV cache, suspend prefetch.

    Like Linux's OOM killer, but for expert residency.
    """

    # Configuration
    ram_budget_mb: float = 4000              # available RAM for experts
    page_fault_threshold: float = 0.3        # >30% accesses are faults → warning
    page_fault_critical: float = 0.5         # >50% → critical
    fault_window: int = 32                   # sliding window for fault rate

    # State
    fault_history: list[float] = field(default_factory=list)
    working_set_size: int = 0
    working_set_history: list[int] = field(default_factory=list)
    current_level: str = "low"

    # Control outputs
    lambda_modifier: float = 0.0             # added to base λ
    top_k_modifier: int = 0                  # subtracted from base top-k
    conservative_forced: bool = False
    kv_eviction_triggered: bool = False
    prefetch_suspended: bool = False

    def update(self, page_fault: bool, unique_experts: int,
               total_accesses: int, ram_used_mb: float):
        """Feed one token's VM statistics."""

        self.fault_history.append(1.0 if page_fault else 0.0)
        if len(self.fault_history) > self.fault_window:
            self.fault_history.pop(0)

        self.working_set_size = unique_experts
        self.working_set_history.append(unique_experts)
        if len(self.working_set_history) > 64:
            self.working_set_history.pop(0)

        # Compute fault rate
        fault_rate = (sum(self.fault_history) /
                      max(1, len(self.fault_history)))

        # Compute pressure
        ram_pressure = ram_used_mb / max(1, self.ram_budget_mb)
        ws_pressure = unique_experts * 10.5 / max(1, self.ram_budget_mb)

        # Level determination
        if fault_rate > self.page_fault_critical or ram_pressure > 0.95:
            self.current_level = "critical"
        elif fault_rate > self.page_fault_threshold or ram_pressure > 0.85:
            self.current_level = "high"
        elif ram_pressure > 0.5:
            self.current_level = "medium"
        else:
            self.current_level = "low"

        # Control responses
        if self.current_level == "critical":
            self.lambda_modifier = 3.0       # strongly increase locality
            self.top_k_modifier = 4          # reduce from 8→4
            self.conservative_forced = True
            self.kv_eviction_triggered = True
            self.prefetch_suspended = True
        elif self.current_level == "high":
            self.lambda_modifier = 1.5
            self.top_k_modifier = 2          # reduce from 8→6
            self.conservative_forced = True
            self.kv_eviction_triggered = False
            self.prefetch_suspended = False
        elif self.current_level == "medium":
            self.lambda_modifier = 0.5
            self.top_k_modifier = 0
            self.conservative_forced = False
            self.kv_eviction_triggered = False
            self.prefetch_suspended = False
        else:  # low
            self.lambda_modifier = 0.0
            self.top_k_modifier = -1         # allow +1 top-k (more diversity)
            self.conservative_forced = False

    def stats(self) -> dict:
        fault_rate = (sum(self.fault_history) /
                      max(1, len(self.fault_history)))
        return {
            "level": self.current_level,
            "fault_rate": round(fault_rate, 3),
            "working_set": self.working_set_size,
            "lambda_mod": round(self.lambda_modifier, 1),
            "top_k_mod": self.top_k_modifier,
            "conservative": self.conservative_forced,
            "kv_eviction": self.kv_eviction_triggered,
            "prefetch_off": self.prefetch_suspended,
        }


# ═══════════════════════════════════════════════════════════
# Semantic Collapse Monitor
# ═══════════════════════════════════════════════════════════

@dataclass
class SemanticCollapse:
    """Detects intelligence narrowing from excessive locality.

    When locality bias is too strong, the model reuses the same
    experts → thinking narrows → outputs become repetitive,
    entropic collapse, semantic drift.

    Monitors:
      - Entropy floor (too low → peaked distribution → collapse)
      - Expert diversity (too few unique experts → narrow thinking)
      - Repetition rate (same token repeating → attractor lock)
      - Semantic drift (output distribution shifts from baseline)
    """

    # Thresholds
    entropy_floor: float = 0.01            # below this = collapse risk
    diversity_floor: int = 3               # min unique experts per token
    repetition_critical: float = 0.3       # >30% repeat rate = critical
    drift_window: int = 64                 # tokens for baseline comparison

    # State
    entropy_history: list[float] = field(default_factory=list)
    expert_count_history: list[int] = field(default_factory=list)
    repetition_history: list[float] = field(default_factory=list)
    token_history: list[int] = field(default_factory=list)

    # Baseline (first N tokens establish normality)
    baseline_entropy: float = 0.1
    baseline_expert_count: float = 8.0
    baseline_established: bool = False

    # Detection
    collapse_risk: float = 0.0             # 0=healthy, 1=collapsed
    diversity_warning: bool = False
    entropy_warning: bool = False
    drift_warning: bool = False

    def update(self, entropy: float, unique_experts: int,
               token_id: int, is_repeat: bool):
        """Feed one token's semantic metrics."""

        self.entropy_history.append(entropy)
        self.expert_count_history.append(unique_experts)
        self.repetition_history.append(1.0 if is_repeat else 0.0)
        self.token_history.append(token_id)

        # Trim
        for h in [self.entropy_history, self.expert_count_history,
                  self.repetition_history, self.token_history]:
            if len(h) > self.drift_window * 2:
                h.pop(0)

        # Establish baseline after drift_window tokens
        if (len(self.entropy_history) >= self.drift_window
                and not self.baseline_established):
            self.baseline_entropy = np.mean(
                self.entropy_history[-self.drift_window:])
            self.baseline_expert_count = np.mean(
                self.expert_count_history[-self.drift_window:])
            self.baseline_established = True

        # Current metrics
        recent_ent = (np.mean(self.entropy_history[-16:])
                      if len(self.entropy_history) >= 16
                      else entropy)
        recent_div = (np.mean(self.expert_count_history[-16:])
                      if len(self.expert_count_history) >= 16
                      else unique_experts)
        recent_rep = (np.mean(self.repetition_history[-16:])
                      if len(self.repetition_history) >= 16
                      else 0.0)

        # Entropy warning
        self.entropy_warning = recent_ent < self.entropy_floor

        # Diversity warning
        self.diversity_warning = recent_div < self.diversity_floor

        # Drift warning
        if self.baseline_established:
            ent_drift = abs(recent_ent - self.baseline_entropy)
            self.drift_warning = (ent_drift > 0.1 and
                                 recent_ent < self.baseline_entropy)

        # Repetition critical
        rep_critical = recent_rep > self.repetition_critical

        # Collapse risk: weighted combination
        risk = 0.0
        if self.entropy_warning: risk += 0.3
        if self.diversity_warning: risk += 0.3
        if self.drift_warning: risk += 0.2
        if rep_critical: risk += 0.4
        self.collapse_risk = min(1.0, risk)

    def should_reduce_locality(self) -> bool:
        """Should we reduce locality λ to restore diversity?"""
        return self.collapse_risk > 0.5

    def should_force_diversity(self) -> bool:
        """Should we force expert diversity (disable all bias)?"""
        return self.collapse_risk > 0.8

    def stats(self) -> dict:
        return {
            "collapse_risk": round(self.collapse_risk, 3),
            "entropy_warning": self.entropy_warning,
            "diversity_warning": self.diversity_warning,
            "drift_warning": self.drift_warning,
            "baseline_entropy": round(self.baseline_entropy, 4),
            "recent_entropy": round(
                np.mean(self.entropy_history[-16:]), 4
            ) if self.entropy_history else 0,
            "recent_diversity": round(
                np.mean(self.expert_count_history[-16:]), 1
            ) if self.expert_count_history else 0,
        }


# ═══════════════════════════════════════════════════════════
# Dynamic Lambda — token-class-aware locality
# ═══════════════════════════════════════════════════════════

@dataclass
class DynamicLambda:
    """Adjusts locality strength based on token class and cognitive mode.

    Different modes need different expert diversity:
      - Easy continuation ("the", "is", "a"): high λ (safe to reuse)
      - Reasoning (math, logic): low λ (need diverse thinking)
      - Code generation: medium λ
      - Hallucination risk: low λ (need fact-checking diversity)
      - Repetition detected: zero λ (break the loop)
    """

    base_lambda: float = 3.0

    # Per-class multipliers
    class_multipliers: dict[str, float] = field(default_factory=lambda: {
        "repetitive": 0.0,      # zero: must diversify
        "stable": 1.5,           # high: safe to reuse
        "default": 1.0,          # base
        "steering": 0.5,         # low: need flexibility
        "transition": 0.0,       # zero: maximum diversity
    })

    # Mode detection
    reasoning_keywords: list[str] = field(default_factory=lambda: [
        "therefore", "because", "if", "then", "=", "+", "compute",
        "calculate", "solve", "proof", "theorem",
    ])
    code_keywords: list[str] = field(default_factory=lambda: [
        "def ", "class ", "import ", "return", "function",
        "{", "}", "var ", "const ",
    ])

    def compute(self, token_class: str, recent_text: str = "",
                thrash_level: str = "low",
                collapse_risk: float = 0.0) -> float:
        """Compute dynamic λ from current state.

        λ = base × class_multiplier × thrash_modifier × collapse_modifier × mode_modifier
        """
        lam = self.base_lambda

        # 1. Token class modifier
        lam *= self.class_multipliers.get(token_class, 1.0)

        # 2. Thrash modifier: increase λ under memory pressure
        if thrash_level == "critical":
            lam += 3.0
        elif thrash_level == "high":
            lam += 1.5
        elif thrash_level == "medium":
            lam += 0.5

        # 3. Collapse modifier: REDUCE λ if intelligence is narrowing
        if collapse_risk > 0.8:
            lam = 0.0  # force full diversity
        elif collapse_risk > 0.5:
            lam *= 0.3

        # 4. Cognitive mode modifier
        text_lower = recent_text.lower()
        if any(kw in text_lower for kw in self.reasoning_keywords):
            lam *= 0.5  # reasoning needs diversity
        if any(kw in text_lower for kw in self.code_keywords):
            lam *= 0.6  # code needs moderate diversity

        return max(0.0, lam)

    def set_base(self, lam: float):
        self.base_lambda = lam


# ═══════════════════════════════════════════════════════════
# Governor — unified runtime routing thermodynamics
# ═══════════════════════════════════════════════════════════

@dataclass
class Governor:
    """Unified runtime controller for VM + routing thermodynamics.

    Closed-loop control:
      ThrashDetector ──→ λ↑, top-k↓, conservative
      SemanticCollapse ──→ λ↓, diversity↑
      DynamicLambda ──→ token-adaptive λ
                          ↓
                   effective λ = f(thrash, collapse, class, mode)
    """

    thrash: ThrashDetector = field(default_factory=ThrashDetector)
    collapse: SemanticCollapse = field(default_factory=SemanticCollapse)
    dyn_lambda: DynamicLambda = field(default_factory=DynamicLambda)

    # Outputs
    effective_lambda: float = 3.0
    effective_top_k: int = 8
    conservative_mode: bool = False
    force_diversity: bool = False

    def update(self,
               # VM metrics
               page_fault: bool,
               unique_experts: int,
               ram_used_mb: float,
               # Semantic metrics
               entropy: float,
               token_id: int,
               is_repeat: bool,
               token_class: str = "default",
               recent_text: str = "",
               ) -> tuple[float, int]:
        """One tick of the governor. Returns (effective_λ, effective_top_k)."""

        total_accesses = self.thrash.fault_window
        self.thrash.update(page_fault, unique_experts,
                          total_accesses, ram_used_mb)
        self.collapse.update(entropy, unique_experts, token_id, is_repeat)

        # Compute dynamic λ from all three subsystems
        self.effective_lambda = self.dyn_lambda.compute(
            token_class, recent_text,
            self.thrash.current_level,
            self.collapse.collapse_risk,
        )

        # Compute effective top-k
        base_k = 8
        k_mod = self.thrash.top_k_modifier
        if self.collapse.should_force_diversity():
            k_mod = -4  # force wider expert set
        self.effective_top_k = max(2, min(8, base_k - k_mod))

        # Conservative mode
        self.conservative_mode = (
            self.thrash.conservative_forced or
            self.collapse.collapse_risk > 0.5
        )

        # Force diversity
        self.force_diversity = self.collapse.should_force_diversity()

        return self.effective_lambda, self.effective_top_k

    def stats(self) -> dict:
        return {
            "effective_lambda": round(self.effective_lambda, 1),
            "effective_top_k": self.effective_top_k,
            "conservative": self.conservative_mode,
            "force_diversity": self.force_diversity,
            "thrash": self.thrash.stats(),
            "collapse": self.collapse.stats(),
        }
