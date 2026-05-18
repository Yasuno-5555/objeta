"""MoE Router Rewriter — runtime-aware routing modification.

Techniques:
  1. Temperature scaling (T < 1 sharpens routing, reduces entropy)
  2. Locality bias (prefer previous expert, improve cache hit rate)
  3. Soft expert pinning (restrict experts by token class)

All operate at inference time. No model weight modification needed.

Key insight from OLMoE measurement:
  Load-balanced routing has H ≈ log(N) with zero temporal locality.
  Cache hit rate = 10% (static 16/64), prefetch impossible.

Goal:
  Reduce routing entropy from log(64)=4.16 → 1.5-2.5 nat
  Improve temporal locality from 2% → 25%+
  Boost cache hit rate from 10% → 50-80%
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RoutingConfig:
    """Configuration for runtime routing modification."""
    temperature: float = 1.0         # T < 1 sharpens, T > 1 flattens
    locality_bias: float = 0.0       # weight added to previous expert logit
    locality_decay: float = 0.9      # exponential decay of locality memory
    pinning_enabled: bool = False    # enable soft expert pinning
    pinning_ratio: float = 0.5       # fraction of experts available per class
    min_experts: int = 2             # minimum experts regardless of pinning


class RouterRewriter:
    """Runtime router modifier — shapes routing distribution for OS friendliness.

    Usage:
        rewriter = RouterRewriter(RoutingConfig(temperature=0.7, locality_bias=2.0))
        modified_weights = rewriter.rewrite(logits, prev_expert=42, token_class="stable")
        # modified_weights is ready for top-k selection
    """

    def __init__(self, config: RoutingConfig, n_experts: int = 64):
        self.config = config
        self.n_experts = n_experts
        self.prev_experts: dict[int, int] = {}  # layer → last expert
        self.locality_memory: dict[int, np.ndarray] = {}  # layer → EMA weights
        self.pinned_experts: dict[str, list[int]] = {}
        self.stats = RewriterStats()

    def rewrite(self, logits: np.ndarray, layer_idx: int = 0,
                prev_expert: int | None = None,
                token_class: str = "default") -> np.ndarray:
        """Rewrite routing weights for OS-friendly inference.

        Args:
            logits: Raw router logits shape (n_experts,)
            layer_idx: Which MoE layer
            prev_expert: Expert selected at previous token (for locality bias)
            token_class: Current token class (for soft pinning)

        Returns:
            Modified softmax weights (n_experts,)
        """
        n = len(logits)
        modified = logits.astype(np.float64).copy()

        # 1. Temperature scaling
        if self.config.temperature != 1.0:
            modified /= self.config.temperature

        # 2. Locality bias: boost previous expert
        if prev_expert is not None and self.config.locality_bias > 0:
            # Initialize or decay locality memory
            if layer_idx not in self.locality_memory:
                self.locality_memory[layer_idx] = np.zeros(n)
            self.locality_memory[layer_idx] *= self.config.locality_decay
            self.locality_memory[layer_idx][prev_expert] += self.config.locality_bias

            modified += self.locality_memory[layer_idx]

        # 3. Soft expert pinning: restrict by token class
        if self.config.pinning_enabled:
            mask = self._pinning_mask(token_class, n)
            # Penalize non-pinned experts (additive penalty, not multiplicative)
            modified[~mask] -= 10.0  # strong penalty, still possible if needed

        # Track previous expert
        if prev_expert is not None:
            self.prev_experts[layer_idx] = prev_expert

        # Softmax
        modified_stable = modified - modified.max()
        probs = np.exp(modified_stable)
        probs /= probs.sum()

        # Update stats
        self.stats.update(probs)

        return probs

    def _pinning_mask(self, token_class: str, n_experts: int) -> np.ndarray:
        """Return boolean mask of allowed experts for this token class."""
        if token_class not in self.pinned_experts:
            # Build pinning set on first use
            allowed = self._compute_pinned_set(token_class, n_experts)
            self.pinned_experts[token_class] = allowed

        mask = np.zeros(n_experts, dtype=bool)
        for eid in self.pinned_experts[token_class]:
            mask[eid] = True
        return mask

    def _compute_pinned_set(self, token_class: str,
                            n_experts: int) -> list[int]:
        """Determine which experts are available for a token class.

        Strategy:
          - REPETITIVE: fewest experts (max entropy reduction)
          - STABLE: moderate restriction
          - DEFAULT: normal
          - STEERING: most experts (need flexibility)
          - TRANSITION: all experts (full access)
        """
        n_pinned = max(
            self.config.min_experts,
            int(n_experts * self.config.pinning_ratio)
        )

        if token_class == "repetitive":
            n_pinned = max(2, n_pinned // 2)
        elif token_class == "stable":
            n_pinned = max(4, n_pinned // 2)
        elif token_class == "steering":
            n_pinned = max(8, n_pinned * 2)
        elif token_class == "transition":
            return list(range(n_experts))  # all experts

        # If no history, allow all (cold start)
        if not self.prev_experts:
            return list(range(n_experts))

        # Use recently used experts across all layers as the pinned set
        recent = list(set(self.prev_experts.values()))
        # Pad with random experts if not enough
        if len(recent) < n_pinned:
            all_experts = list(range(n_experts))
            rng = np.random.RandomState(42)
            extra = [e for e in all_experts if e not in recent]
            rng.shuffle(extra)
            recent.extend(extra[:n_pinned - len(recent)])

        return recent[:n_pinned]

    def get_routing_entropy(self, probs: np.ndarray) -> float:
        """Compute normalized routing entropy for modified distribution."""
        ent = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_ent = np.log(len(probs))
        return ent / max_ent if max_ent > 0 else 0.0

    def get_effective_k(self, probs: np.ndarray,
                        cumulative: float = 0.9) -> int:
        """Number of experts needed to reach cumulative probability."""
        sorted_idx = np.argsort(-probs)
        cumsum = np.cumsum(probs[sorted_idx])
        return int(np.searchsorted(cumsum, cumulative) + 1)

    def reset(self):
        self.prev_experts.clear()
        self.locality_memory.clear()
        self.pinned_experts.clear()
        self.stats = RewriterStats()


@dataclass
class RewriterStats:
    """Accumulated statistics for the rewriter."""
    total_calls: int = 0
    entropy_sum: float = 0.0
    effective_k_sum: float = 0.0
    locality_hits: int = 0
    entropy_samples: list[float] = field(default_factory=list)

    def update(self, probs: np.ndarray):
        self.total_calls += 1
        ent = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_ent = np.log(len(probs))
        self.entropy_sum += ent / max_ent if max_ent > 0 else 0.0
        sorted_idx = np.argsort(-probs)
        cumsum = np.cumsum(probs[sorted_idx])
        self.effective_k_sum += float(np.searchsorted(cumsum, 0.9) + 1)

        if len(self.entropy_samples) < 1000:
            self.entropy_samples.append(float(ent))

    def summary(self) -> dict:
        n = max(self.total_calls, 1)
        return {
            "calls": self.total_calls,
            "avg_entropy_normalized": round(self.entropy_sum / n, 4),
            "avg_entropy_nat": round(self.entropy_sum / n * np.log(64), 2),
            "avg_effective_k": round(self.effective_k_sum / n, 1),
            "locality_hits": self.locality_hits,
        }


# ── Convenience: measure entropy reduction ──

# ═══════════════════════════════════════════════════════════════════════════════
# Expert Residency Manager — virtual memory for MoE experts
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExpertResidencyManager:
    """Manages which experts are in RAM vs SSD.

    Three tiers (like CPU page cache):
      LOADED  — in RAM, immediate access
      WARM    — mmap'd, lazy access
      COLD    — SSD, needs I/O

    The router is biased toward LOADED experts to minimize I/O.
    """

    n_experts: int = 64
    n_loaded: int = 16      # RAM budget (like L1 cache)
    n_warm: int = 16        # mmap budget (like L2 cache)

    # Per-layer residency state
    loaded: dict[int, set[int]] = field(default_factory=dict)
    warm: dict[int, set[int]] = field(default_factory=dict)

    # Access tracking
    access_count: dict[int, np.ndarray] = field(default_factory=dict)
    last_access: dict[int, np.ndarray] = field(default_factory=dict)
    current_token: int = 0

    # Residency bias weight (how strongly to prefer loaded experts)
    residency_bias: float = 3.0

    def __post_init__(self):
        for l in range(32):  # max layers
            self.access_count[l] = np.zeros(self.n_experts)
            self.last_access[l] = np.zeros(self.n_experts) - 1
            # Initially: first n_loaded are loaded, next n_warm are warm
            self.loaded[l] = set(range(self.n_loaded))
            self.warm[l] = set(range(self.n_loaded, self.n_loaded + self.n_warm))

    def record_access(self, layer_idx: int, expert_id: int):
        """Record that an expert was accessed."""
        self.access_count[layer_idx][expert_id] += 1
        self.last_access[layer_idx][expert_id] = self.current_token

    def advance_token(self):
        """Move to next token. Triggers residency rebalancing."""
        self.current_token += 1

        # Every N tokens, rebalance residency
        if self.current_token % 16 == 0:
            self._rebalance()

    def _rebalance(self):
        """Promote hot experts to LOADED, demote cold ones to COLD."""
        for l in self.access_count:
            if len(self.access_count[l]) == 0:
                continue

            # Score each expert: access frequency × recency
            scores = self.access_count[l].copy()
            # Recency bonus: recently accessed experts get boost
            recency = np.exp(
                -(self.current_token - self.last_access[l]) / 32.0
            )
            scores += recency * 2.0

            # Top N → LOADED
            top_idx = np.argsort(-scores)
            self.loaded[l] = set(top_idx[:self.n_loaded].tolist())
            # Next M → WARM
            self.warm[l] = set(top_idx[self.n_loaded:self.n_loaded + self.n_warm].tolist())

    def is_loaded(self, layer_idx: int, expert_id: int) -> bool:
        return expert_id in self.loaded.get(layer_idx, set())

    def is_warm(self, layer_idx: int, expert_id: int) -> bool:
        return expert_id in self.warm.get(layer_idx, set())

    def residency_score(self, layer_idx: int, expert_id: int) -> float:
        """Get residency priority score for routing bias.

        Returns 1.0 for loaded, 0.3 for warm, 0.0 for cold.
        """
        if self.is_loaded(layer_idx, expert_id):
            return 1.0
        elif self.is_warm(layer_idx, expert_id):
            return 0.3
        return 0.0

    def stats(self) -> dict:
        total_accessed = sum(
            int(np.sum(self.access_count[l] > 0))
            for l in self.access_count
        )
        return {
            "n_loaded": self.n_loaded,
            "n_warm": self.n_warm,
            "total_experts_ever_accessed": total_accessed,
            "current_token": self.current_token,
        }


# ═══════════════════════════════════════════════════════════
# Sticky Router — multi-factor routing bias
# ═══════════════════════════════════════════════════════════

@dataclass
class StickyRouterConfig:
    """Configuration for sticky expert routing."""
    temporal_weight: float = 3.0     # λ1: previous expert continuity
    residency_weight: float = 2.0    # λ2: prefer RAM-resident experts
    affinity_weight: float = 1.0     # λ3: co-occurrence affinity
    temporal_decay: float = 0.85     # EMA decay for temporal memory
    min_experts: int = 2             # minimum experts regardless of bias


class StickyRouter:
    """Multi-factor routing bias for OS-friendly expert selection.

    Adds three bias terms to router logits:
      1. Temporal: prefer expert from previous token
      2. Residency: prefer experts currently in RAM
      3. Affinity: prefer experts that co-occur with recent selections
    """

    def __init__(self, config: StickyRouterConfig,
                 residency: ExpertResidencyManager | None = None,
                 n_experts: int = 64):
        self.config = config
        self.residency = residency or ExpertResidencyManager(n_experts)
        self.n_experts = n_experts

        # Temporal memory: EMA of recent selections per layer
        self.temporal_memory: dict[int, np.ndarray] = {}
        # Affinity graph: P(e_j | e_i) learned online
        self.affinity_graph: np.ndarray = np.eye(n_experts) * 0.1
        self.affinity_counts: np.ndarray = np.zeros((n_experts, n_experts))
        self._prev_expert: dict[int, int] = {}

        self.stats = StickyRouterStats()

    def rewrite(self, logits: np.ndarray, layer_idx: int = 0,
                prev_expert: int | None = None) -> np.ndarray:
        """Rewrite router logits with multi-factor OS bias.

        g'_i = g_i + λ1·1[e_i=e_{t-1}] + λ2·residency(e_i) + λ3·affinity(e_i)
        """
        n = len(logits)
        modified = logits.astype(np.float64).copy()

        # 1. Temporal bias: boost previous expert
        if prev_expert is not None and self.config.temporal_weight > 0:
            if layer_idx not in self.temporal_memory:
                self.temporal_memory[layer_idx] = np.zeros(n)
            self.temporal_memory[layer_idx] *= self.config.temporal_decay
            self.temporal_memory[layer_idx][prev_expert] += self.config.temporal_weight
            modified += self.temporal_memory[layer_idx]

        # 2. Residency bias: prefer loaded/warm experts
        if self.config.residency_weight > 0:
            for eid in range(n):
                score = self.residency.residency_score(layer_idx, eid)
                modified[eid] += score * self.config.residency_weight

        # 3. Affinity bias: prefer co-occurring experts
        if prev_expert is not None and self.config.affinity_weight > 0:
            modified += self.affinity_graph[prev_expert] * self.config.affinity_weight

        # Update affinity graph
        if prev_expert is not None:
            if layer_idx in self._prev_expert:
                prev_prev = self._prev_expert[layer_idx]
                self.affinity_counts[prev_prev, prev_expert] += 1
                # Normalize row periodically
                if self.affinity_counts[prev_prev].sum() > 100:
                    row = self.affinity_counts[prev_prev]
                    self.affinity_graph[prev_prev] = row / row.sum()
            self._prev_expert[layer_idx] = prev_expert

        # Softmax
        modified_stable = modified - modified.max()
        probs = np.exp(modified_stable)
        probs /= probs.sum()

        # Update residency tracking
        top_expert = int(np.argmax(probs))
        self.residency.record_access(layer_idx, top_expert)
        self.residency.advance_token()

        self.stats.update(probs, modified, logits)
        return probs

    def reset(self):
        self.temporal_memory.clear()
        self.affinity_graph = np.eye(self.n_experts) * 0.1
        self.affinity_counts = np.zeros((self.n_experts, self.n_experts))
        self._prev_expert.clear()
        self.stats = StickyRouterStats()


@dataclass
class StickyRouterStats:
    total_calls: int = 0
    entropy_sum: float = 0.0
    baseline_entropy_sum: float = 0.0
    effective_k_sum: float = 0.0
    loaded_hits: int = 0
    warm_hits: int = 0
    cold_misses: int = 0

    def update(self, probs: np.ndarray, modified_logits: np.ndarray,
               original_logits: np.ndarray):
        self.total_calls += 1
        # Modified entropy
        ent = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_ent = np.log(len(probs))
        self.entropy_sum += ent / max_ent if max_ent > 0 else 0.0
        # Baseline entropy
        base = np.exp(original_logits.astype(np.float64) - original_logits.max())
        base /= base.sum()
        base_ent = -float(np.sum(base * np.log(base + 1e-12)))
        self.baseline_entropy_sum += base_ent / max_ent if max_ent > 0 else 0.0
        # Eff k
        sorted_idx = np.argsort(-probs)
        cumsum = np.cumsum(probs[sorted_idx])
        self.effective_k_sum += float(np.searchsorted(cumsum, 0.9) + 1)

    def summary(self, residency: ExpertResidencyManager | None = None) -> dict:
        n = max(self.total_calls, 1)
        result = {
            "calls": self.total_calls,
            "baseline_entropy_nat": round(
                self.baseline_entropy_sum / n * np.log(64), 2),
            "modified_entropy_nat": round(
                self.entropy_sum / n * np.log(64), 2),
            "entropy_reduction_pct": round(
                (1.0 - self.entropy_sum / max(1e-12, self.baseline_entropy_sum)) * 100, 1),
            "avg_effective_k": round(self.effective_k_sum / n, 1),
        }
        if residency:
            result["residency"] = residency.stats()
        return result


def measure_entropy_reduction(
    rewriter: RouterRewriter,
    logits_batch: list[np.ndarray],
    prev_experts: list[int | None] | None = None,
) -> dict:
    """Measure how much the rewriter reduces routing entropy.

    Returns before/after comparison.
    """
    before_entropies = []
    after_entropies = []
    before_k = []
    after_k = []

    for i, logits in enumerate(logits_batch):
        # Before
        before = np.exp(logits.astype(np.float64) - logits.max())
        before /= before.sum()
        before_entropies.append(
            -float(np.sum(before * np.log(before + 1e-12))) / np.log(len(before))
        )
        sorted_before = np.argsort(-before)
        before_k.append(
            float(np.searchsorted(np.cumsum(before[sorted_before]), 0.9) + 1)
        )

        # After
        prev = prev_experts[i] if prev_experts and i < len(prev_experts) else None
        after = rewriter.rewrite(logits, layer_idx=0, prev_expert=prev)
        after_entropies.append(
            -float(np.sum(after * np.log(after + 1e-12))) / np.log(len(after))
        )
        sorted_after = np.argsort(-after)
        after_k.append(
            float(np.searchsorted(np.cumsum(after[sorted_after]), 0.9) + 1)
        )

    return {
        "n_samples": len(logits_batch),
        "before_entropy_nat": round(float(np.mean(before_entropies)) * np.log(len(logits_batch[0])), 2),
        "after_entropy_nat": round(float(np.mean(after_entropies)) * np.log(len(logits_batch[0])), 2),
        "entropy_reduction": round(
            (1.0 - np.mean(after_entropies) / max(1e-12, np.mean(before_entropies))) * 100, 1
        ),
        "before_effective_k": round(float(np.mean(before_k)), 1),
        "after_effective_k": round(float(np.mean(after_k)), 1),
        "k_reduction": round(
            (1.0 - np.mean(after_k) / max(1e-12, np.mean(before_k))) * 100, 1
        ),
    }
