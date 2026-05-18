"""MoE-aware scheduler extensions.

Adds MoE-specific observation and control:
  - Routing entropy → adaptive expert count
  - Expert frequency tracking → cache policy
  - Adaptive top-k → MoeMode::Adaptive implementation
"""

from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class RoutingObservation:
    """MoE routing observation at one layer, one token."""
    layer_idx: int
    routing_entropy: float          # normalized entropy of router probs
    active_experts: int             # number of experts with significant weight
    top1_expert: int                # most-weighted expert
    top1_weight: float              # weight of top expert
    expert_weights: list[float] = field(default_factory=list)


@dataclass
class ExpertCachePolicy:
    """Per-layer expert cache management policy.

    Two-tier:
      Static tier — per-layer frequency bias (pre-computed from warmup)
      Dynamic tier — global LRU (updated during inference)
    """

    # Static: top-N experts by historical frequency per layer
    static_per_layer: dict[int, list[int]] = field(default_factory=dict)
    static_size: int = 16

    # Dynamic: global LRU cache
    dynamic_size: int = 8
    dynamic_cache: dict[int, deque] = field(default_factory=dict)  # layer → deque of expert ids

    # Hit/miss tracking
    hits: int = 0
    misses: int = 0

    def update_static(self, layer_idx: int, expert_frequencies: dict[int, int]):
        """Set static cache from warmup frequency data."""
        sorted_experts = sorted(expert_frequencies.items(),
                               key=lambda x: x[1], reverse=True)
        self.static_per_layer[layer_idx] = [
            eid for eid, _ in sorted_experts[:self.static_size]
        ]

    def update_dynamic(self, layer_idx: int, expert_id: int):
        """Record an expert access, update LRU."""
        if layer_idx not in self.dynamic_cache:
            self.dynamic_cache[layer_idx] = deque(maxlen=self.dynamic_size)

        cache = self.dynamic_cache[layer_idx]
        if expert_id in cache:
            cache.remove(expert_id)  # move to front
            self.hits += 1
        else:
            self.misses += 1
        cache.appendleft(expert_id)

    def is_cached(self, layer_idx: int, expert_id: int) -> bool:
        """Check if expert is in cache (static + dynamic)."""
        if layer_idx in self.static_per_layer:
            if expert_id in self.static_per_layer[layer_idx]:
                return True
        if layer_idx in self.dynamic_cache:
            if expert_id in self.dynamic_cache[layer_idx]:
                return True
        return False

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def total_size(self) -> int:
        """Total cached experts across all layers."""
        return self.static_size + self.dynamic_size


class AdaptiveTopK:
    """Entropy-based adaptive expert count.

    High routing entropy → near-uniform distribution → need more experts
    Low routing entropy → peaked distribution → fewer experts sufficient
    """

    def __init__(self, min_k: int = 2, max_k: int = 8,
                 entropy_threshold: float = 0.7):
        self.min_k = min_k
        self.max_k = max_k
        self.entropy_threshold = entropy_threshold

    def compute_k(self, routing_entropy: float) -> int:
        """Map routing entropy → expert count.

        entropy near 1.0 (uniform) → max_k
        entropy near 0.0 (peaked) → min_k
        """
        if routing_entropy > self.entropy_threshold:
            return self.max_k
        # Linear interpolation between min_k and max_k
        frac = routing_entropy / self.entropy_threshold
        k = int(self.min_k + frac * (self.max_k - self.min_k))
        return max(self.min_k, min(self.max_k, k))

    def truncate_weights(self, weights: np.ndarray,
                         routing_entropy: float) -> tuple[np.ndarray, np.ndarray]:
        """Truncate expert weights to adaptive top-k, renormalize.

        Returns (indices, renormalized_weights).
        """
        k = self.compute_k(routing_entropy)
        # Top-k by cumulative probability
        sorted_idx = np.argsort(-weights)
        cumsum = np.cumsum(weights[sorted_idx])
        k_cumsum = min(k, int(np.searchsorted(cumsum, 0.9) + 1))

        top_idx = sorted_idx[:k_cumsum]
        top_weights = weights[top_idx]
        top_weights /= top_weights.sum()
        return top_idx, top_weights


@dataclass
class MoeSchedulerExtension:
    """MoE-specific scheduler state and logic.

    Extends the base Scheduler for MoE-aware execution:
      - Routing entropy observation
      - Adaptive expert count
      - Expert cache policy
    """

    n_layers: int
    n_experts: int = 64
    default_top_k: int = 8
    adaptive_top_k: AdaptiveTopK = field(default_factory=AdaptiveTopK)
    cache_policy: ExpertCachePolicy = field(default_factory=ExpertCachePolicy)
    routing_history: list[list[RoutingObservation]] = field(default_factory=list)

    def observe_routing(self, layer_idx: int,
                        router_weights: np.ndarray) -> RoutingObservation:
        """Observe routing distribution at one layer.

        router_weights: (n_experts,) — post-softmax expert weights.
        """
        # Entropy
        w = router_weights + 1e-12
        max_ent = np.log(len(w))
        ent = -float(np.sum(w * np.log(w)))
        norm_entropy = ent / max_ent if max_ent > 0 else 0.0

        # Active experts (weight > 1/n_experts = significant)
        threshold = 1.0 / len(w)
        active = int(np.sum(w > threshold))

        # Top expert
        top1 = int(np.argmax(w))

        obs = RoutingObservation(
            layer_idx=layer_idx,
            routing_entropy=norm_entropy,
            active_experts=active,
            top1_expert=top1,
            top1_weight=float(w[top1]),
            expert_weights=w.tolist(),
        )

        # Update cache
        self.cache_policy.update_dynamic(layer_idx, top1)

        return obs

    def get_expert_count(self, routing_entropy: float,
                         collapse_active: bool = False) -> int:
        """Get number of experts to use based on routing entropy."""
        if collapse_active:
            return self.default_top_k  # conservative during collapse
        return self.adaptive_top_k.compute_k(routing_entropy)

    def should_prefetch(self, layer_idx: int, expert_id: int) -> bool:
        """Should this expert be prefetched?"""
        return not self.cache_policy.is_cached(layer_idx, expert_id)

    def prefetch_candidates(self, layer_idx: int,
                            router_weights: np.ndarray,
                            n_prefetch: int = 3) -> list[int]:
        """Get expert IDs to prefetch for the next layer."""
        # Top-N experts by weight, minus already cached
        sorted_idx = np.argsort(-router_weights)
        candidates = []
        for eid in sorted_idx:
            if not self.cache_policy.is_cached(layer_idx, int(eid)):
                candidates.append(int(eid))
            if len(candidates) >= n_prefetch:
                break
        return candidates

    def build_static_cache_from_warmup(self,
                                        warmup_routing: list[list[RoutingObservation]]):
        """Build per-layer static cache from warmup routing data.

        warmup_routing: [token][layer] RoutingObservation list.
        """
        # Aggregate expert frequencies per layer
        from collections import Counter
        per_layer_freq: dict[int, Counter] = {}

        for token_obs in warmup_routing:
            for obs in token_obs:
                if obs.layer_idx not in per_layer_freq:
                    per_layer_freq[obs.layer_idx] = Counter()
                per_layer_freq[obs.layer_idx][obs.top1_expert] += 1

        for layer_idx, counter in per_layer_freq.items():
            self.cache_policy.update_static(layer_idx, dict(counter))

    def stats(self) -> dict:
        return {
            "cache_hit_rate": self.cache_policy.hit_rate(),
            "cache_size": self.cache_policy.total_size(),
            "avg_expert_count": self.adaptive_top_k.compute_k(0.5),
        }
