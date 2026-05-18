"""objeta-vm — LLM Virtual Memory Manager v2.

5-tier residency + speculative prefetch + layer-overlap scheduling.

Tiers:
  HOT     — GPU-visible (Metal buffer), zero-copy access
  WARM    — unified RAM, CPU-accessible, <1µs access
  COOL    — mmap cached, OS page cache, ~200µs access
  COLD    — SSD, requires I/O, ~2600µs access
  FROZEN  — compressed SSD, requires decompress + I/O

Components:
  PageTable           — 5-tier residency tracking
  SpeculativePrefetch — transition-matrix predictive, multi-token lookahead
  LayerOverlapScheduler — compute layer N while prefetching layer N+1
  EvictionPolicy       — LRU + affinity-aware, 5-tier transitions
  MemoryPressure       — RAM + GPU buffer monitoring
  AffinityClustering   — expert page coloring
"""

from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import numpy as np


# ═══════════════════════════════════════════════════════════
# 5-Tier Residency
# ═══════════════════════════════════════════════════════════

class Tier(Enum):
    HOT = 0      # GPU-visible (Metal buffer), ~1µs
    WARM = 1     # Unified RAM, CPU-accessible, ~1µs
    COOL = 2     # mmap cached (OS page cache), ~200µs
    COLD = 3     # SSD, requires I/O, ~2600µs
    FROZEN = 4   # Compressed SSD, requires decompress + I/O, ~5000µs

# Latency model (µs) — from wall-clock measurement 2026-05-18
TIER_LATENCY_US = {
    Tier.HOT: 1,
    Tier.WARM: 1,
    Tier.COOL: 198,
    Tier.COLD: 2627,
    Tier.FROZEN: 5000,
}

# Promotion path: FROZEN → COLD → COOL → WARM → HOT
# Each step requires different operations
PROMOTION_COST = {
    (Tier.FROZEN, Tier.COLD): ("decompress", 2000),   # decompress from SSD
    (Tier.COLD, Tier.COOL):   ("mmap", 2600),          # mmap into page cache
    (Tier.COOL, Tier.WARM):   ("load", 200),            # page cache → RAM
    (Tier.WARM, Tier.HOT):    ("upload", 100),          # RAM → GPU buffer
}

TIER_ORDER = [Tier.FROZEN, Tier.COLD, Tier.COOL, Tier.WARM, Tier.HOT]


@dataclass
class PageTableEntry:
    expert_id: int
    tier: Tier = Tier.FROZEN
    last_access: int = -1
    access_count: int = 0
    load_time: int = -1
    size_bytes: int = 0
    size_compressed_bytes: int = 0  # FROZEN tier only
    layer_idx: int = 0
    pinned: bool = False
    gpu_buffer_id: int = -1  # HOT tier only


@dataclass
class PageTable:
    n_experts: int = 64
    n_layers: int = 16
    expert_size_bytes: int = 6_300_000

    entries: dict[int, dict[int, PageTableEntry]] = field(default_factory=dict)

    # Per-tier budgets
    hot_budget_bytes: int = 50_000_000    # GPU buffer (50MB)
    warm_budget_bytes: int = 150_000_000  # RAM (150MB)
    cool_budget_bytes: int = 300_000_000  # mmap (300MB)

    hot_used_bytes: int = 0
    warm_used_bytes: int = 0
    cool_used_bytes: int = 0

    page_faults: int = 0
    page_promotions: int = 0
    page_demotions: int = 0
    current_token: int = 0

    def __post_init__(self):
        for l in range(self.n_layers):
            self.entries[l] = {}
            for e in range(self.n_experts):
                self.entries[l][e] = PageTableEntry(
                    expert_id=e, layer_idx=l,
                    size_bytes=self.expert_size_bytes,
                    size_compressed_bytes=self.expert_size_bytes // 3,
                    tier=Tier.FROZEN,
                )

    def access(self, layer_idx: int, expert_id: int) -> int:
        """Record access. Returns estimated latency in µs."""
        entry = self.entries[layer_idx][expert_id]
        entry.last_access = self.current_token
        entry.access_count += 1

        latency = TIER_LATENCY_US[entry.tier]

        # Auto-promote: FROZEN→COLD on access, COLD→COOL on access
        if entry.tier == Tier.FROZEN:
            self._set_tier(entry, Tier.COLD)
            self.page_faults += 1
            latency = TIER_LATENCY_US[Tier.COLD]  # pay decompress + I/O
        elif entry.tier == Tier.COLD:
            self._set_tier(entry, Tier.COOL)
            self.page_faults += 1
            latency = TIER_LATENCY_US[Tier.COLD]
        elif entry.tier == Tier.COOL:
            self._set_tier(entry, Tier.WARM)
            self.page_promotions += 1

        return latency

    def promote_to_hot(self, layer_idx: int, expert_id: int):
        """Explicitly promote to HOT (GPU buffer)."""
        entry = self.entries[layer_idx][expert_id]
        while entry.tier != Tier.HOT:
            next_tier = TIER_ORDER[min(TIER_ORDER.index(entry.tier) + 1, len(TIER_ORDER) - 1)]
            self._set_tier(entry, next_tier)
            self.page_promotions += 1
            if next_tier == Tier.HOT:
                break

    def evict_one_step(self, layer_idx: int, expert_id: int):
        """Demote one tier (HOT→WARM→COOL→COLD→FROZEN)."""
        entry = self.entries[layer_idx][expert_id]
        if entry.pinned:
            return
        idx = TIER_ORDER.index(entry.tier)
        if idx > 0:
            self._set_tier(entry, TIER_ORDER[idx - 1])
            self.page_demotions += 1

    def _set_tier(self, entry: PageTableEntry, new_tier: Tier):
        old = entry.tier
        if old == new_tier:
            return
        # Remove from old tier budget
        self._dec_budget(old, entry.size_bytes)
        # Add to new tier budget
        self._inc_budget(new_tier, entry.size_bytes)
        entry.tier = new_tier

    def _inc_budget(self, tier: Tier, size: int):
        if tier == Tier.HOT: self.hot_used_bytes += size
        elif tier == Tier.WARM: self.warm_used_bytes += size
        elif tier == Tier.COOL: self.cool_used_bytes += size

    def _dec_budget(self, tier: Tier, size: int):
        if tier == Tier.HOT: self.hot_used_bytes = max(0, self.hot_used_bytes - size)
        elif tier == Tier.WARM: self.warm_used_bytes = max(0, self.warm_used_bytes - size)
        elif tier == Tier.COOL: self.cool_used_bytes = max(0, self.cool_used_bytes - size)

    def advance_token(self):
        self.current_token += 1

    def stats(self) -> dict:
        def count(tier): return sum(
            1 for l in self.entries for e in self.entries[l].values()
            if e.tier == tier)
        return {
            "hot": count(Tier.HOT), "warm": count(Tier.WARM),
            "cool": count(Tier.COOL), "cold": count(Tier.COLD),
            "frozen": count(Tier.FROZEN),
            "hot_mb": round(self.hot_used_bytes / 1e6, 1),
            "warm_mb": round(self.warm_used_bytes / 1e6, 1),
            "cool_mb": round(self.cool_used_bytes / 1e6, 1),
            "faults": self.page_faults,
            "promotions": self.page_promotions,
            "demotions": self.page_demotions,
        }


# ═══════════════════════════════════════════════════════════
# Speculative Prefetch Engine
# ═══════════════════════════════════════════════════════════

class SpeculativePrefetch:
    """Transition-matrix predictive prefetch with multi-token lookahead.

    Uses P(e_j | e_i) to predict: e_{t+1}, e_{t+2}, ..., e_{t+lookahead}
    and prefetches them COLD→COOL→WARM before they're needed.
    """

    def __init__(self, page_table: PageTable,
                 transition_matrix: np.ndarray | None = None,
                 lookahead: int = 3,
                 budget_per_layer: int = 4):
        self.pt = page_table
        self.trans = transition_matrix
        self.lookahead = lookahead
        self.budget_per_layer = budget_per_layer

        self.prefetched: int = 0
        self.useful: int = 0
        self.wasted: int = 0

    def set_transition_matrix(self, trans: np.ndarray):
        """Set transition matrix from observed data."""
        self.trans = trans

    def predict_chain(self, current_expert: int,
                      steps: int | None = None) -> list[list[int]]:
        """Predict expert chains for multiple steps ahead.

        Returns list of [step1_experts, step2_experts, ...]
        where step1 is the most likely next experts.
        """
        steps = steps or self.lookahead
        if self.trans is None or current_expert >= len(self.trans):
            return [[] for _ in range(steps)]

        chains = []
        # Step 1: P(e_{t+1} | e_t)
        row = self.trans[current_expert]
        top = [int(e) for e in np.argsort(-row)[:self.budget_per_layer]
               if e != current_expert]
        chains.append(top)

        # Step 2+: use stationary distribution * transition
        dist = row.copy()
        for s in range(1, steps):
            dist = dist @ self.trans  # P(e_{t+s} | e_t)
            top = [int(e) for e in np.argsort(-dist)[:self.budget_per_layer]]
            chains.append(top)

        return chains

    def prefetch(self, layer_idx: int, current_expert: int):
        """Execute speculative prefetch for predicted next experts."""
        chains = self.predict_chain(current_expert)

        for step, experts in enumerate(chains):
            for eid in experts:
                entry = self.pt.entries[layer_idx][eid]
                # Promote as far as budget allows
                # Step 0 (next token): promote to COOL
                # Step 1+: promote to COLD (just decompress)
                if step == 0 and entry.tier in (Tier.COLD, Tier.FROZEN):
                    self.pt.access(layer_idx, eid)  # triggers promotion
                    # Further promote to WARM if budget allows
                    if self.pt.warm_used_bytes + entry.size_bytes <= self.pt.warm_budget_bytes:
                        self.pt._set_tier(entry, Tier.WARM)
                    self.prefetched += 1
                elif step >= 1 and entry.tier == Tier.FROZEN:
                    self.pt._set_tier(entry, Tier.COLD)
                    self.prefetched += 1

    def record_hit(self, layer_idx: int, expert_id: int):
        """Record that a prefetched expert was actually used."""
        entry = self.pt.entries[layer_idx][expert_id]
        if entry.tier in (Tier.COOL, Tier.WARM, Tier.HOT) and entry.access_count <= 1:
            self.useful += 1

    def hit_rate(self) -> float:
        total = self.useful + self.wasted
        return self.useful / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════
# Layer Overlap Scheduler
# ═══════════════════════════════════════════════════════════

class LayerOverlapScheduler:
    """Compute layer N while prefetching layer N+1.

    On Apple Silicon unified memory:
      - GPU computes layer N expert GEMVs
      - CPU concurrently issues mmap/Metal upload for layer N+1 experts
      - Metal command queue allows overlapping compute + data transfer

    Timeline (ideal):
      Layer 0 compute |----GPU----|
      Layer 1 prefetch    |--CPU--|
      Layer 1 compute          |----GPU----|
      Layer 2 prefetch             |--CPU--|
    """

    def __init__(self, n_layers: int = 40):
        self.n_layers = n_layers
        self.overlap_achieved: float = 0.0  # fraction of I/O hidden
        self.total_compute_us: float = 0.0
        self.total_io_us: float = 0.0
        self.hidden_io_us: float = 0.0

    def schedule(self, layer_idx: int,
                 compute_time_us: float,
                 io_time_us: float) -> tuple[float, float]:
        """Schedule one layer. Returns (actual_wall_time, io_hidden).

        If the previous layer's compute is longer than this layer's I/O,
        the I/O is fully hidden.
        """
        # IO for this layer can be hidden by previous layer's compute
        hidden = min(io_time_us, compute_time_us)
        actual = compute_time_us + max(0, io_time_us - compute_time_us)

        self.total_compute_us += compute_time_us
        self.total_io_us += io_time_us
        self.hidden_io_us += hidden

        return actual, hidden

    def total_wall_time(self) -> float:
        """Estimate total wall time with overlap."""
        # Sequential: sum(compute + io)
        sequential = self.total_compute_us + self.total_io_us
        # Overlapped: sum(compute) + residual io
        overlapped = self.total_compute_us + (self.total_io_us - self.hidden_io_us)
        self.overlap_achieved = self.hidden_io_us / max(1, self.total_io_us)
        return overlapped

    def stats(self) -> dict:
        return {
            "sequential_ms": round((self.total_compute_us + self.total_io_us) / 1000, 1),
            "overlapped_ms": round(self.total_wall_time() / 1000, 1),
            "io_hidden_pct": round(self.overlap_achieved * 100, 1),
            "speedup": round(
                (self.total_compute_us + self.total_io_us) / max(1, self.total_wall_time()), 1
            ),
        }


# ═══════════════════════════════════════════════════════════
# Eviction Policy (5-tier)
# ═══════════════════════════════════════════════════════════

class EvictionPolicy:
    """LRU + affinity-aware eviction across 5 tiers."""

    def __init__(self, page_table: PageTable, affinity: np.ndarray | None = None):
        self.pt = page_table
        self.affinity = affinity

    def select_victim(self, layer_idx: int,
                      active_experts: set[int] | None = None,
                      from_tier: Tier = Tier.WARM) -> int | None:
        """Select best expert to demote from given tier."""
        active = active_experts or set()
        candidates = []

        for eid, entry in self.pt.entries[layer_idx].items():
            if entry.tier == from_tier and not entry.pinned:
                score = self._score(entry, active)
                candidates.append((score, eid))

        if not candidates:
            return None

        candidates.sort()  # lowest score = best to evict
        return candidates[0][1]

    def _score(self, entry: PageTableEntry, active: set[int]) -> float:
        age = max(0, self.pt.current_token - entry.last_access)
        recency = np.exp(-age / 32.0)
        freq = min(1.0, entry.access_count / 50.0)
        aff = 0.0
        if self.affinity is not None and active:
            aff = max(self.affinity[int(a), entry.expert_id] for a in active if int(a) < len(self.affinity))
        return 1.0 - (recency * 0.4 + freq * 0.3 + aff * 0.3)


# ═══════════════════════════════════════════════════════════
# Memory Pressure
# ═══════════════════════════════════════════════════════════

class MemoryPressure:
    def __init__(self, page_table: PageTable):
        self.pt = page_table
        self.pressure_history: list[float] = []

    def warm_pressure(self) -> float:
        if self.pt.warm_budget_bytes == 0: return 0.0
        return self.pt.warm_used_bytes / self.pt.warm_budget_bytes

    def hot_pressure(self) -> float:
        if self.pt.hot_budget_bytes == 0: return 0.0
        return self.pt.hot_used_bytes / self.pt.hot_budget_bytes

    def level(self) -> str:
        p = max(self.warm_pressure(), self.hot_pressure())
        if p >= 0.95: return "critical"
        if p >= 0.85: return "high"
        if p >= 0.5: return "medium"
        return "low"

    def update(self):
        self.pressure_history.append(max(self.warm_pressure(), self.hot_pressure()))
        if len(self.pressure_history) > 100:
            self.pressure_history.pop(0)

    def stats(self) -> dict:
        return {
            "warm_pressure": round(self.warm_pressure(), 3),
            "hot_pressure": round(self.hot_pressure(), 3),
            "level": self.level(),
        }


# ═══════════════════════════════════════════════════════════
# Affinity Clustering
# ═══════════════════════════════════════════════════════════

class AffinityClustering:
    def __init__(self, n_experts: int = 64):
        self.n_experts = n_experts
        self.cooccurrence = np.zeros((n_experts, n_experts))
        self.transition = np.zeros((n_experts, n_experts))
        self.clusters: dict[int, list[int]] = {}
        self._last: dict[int, int] = {}

    def observe(self, layer_idx: int, expert_id: int):
        if layer_idx in self._last:
            prev = self._last[layer_idx]
            self.cooccurrence[prev, expert_id] += 1
        self._last[layer_idx] = expert_id

    def build(self, n_clusters: int = 4) -> np.ndarray:
        """Build clusters and return normalized transition matrix."""
        trans = self.cooccurrence.copy()
        for i in range(self.n_experts):
            s = trans[i].sum()
            if s > 0:
                trans[i] /= s
        self.transition = trans

        remaining = set(range(self.n_experts))
        for cid in range(n_clusters):
            if not remaining: break
            seed = max(remaining, key=lambda e: trans[e].sum())
            remaining.remove(seed)
            cluster = [seed]
            neighbors = np.argsort(-trans[seed])
            for nbr in neighbors:
                if nbr in remaining and len(cluster) < self.n_experts // n_clusters:
                    cluster.append(int(nbr))
                    remaining.remove(int(nbr))
            self.clusters[cid] = cluster

        return self.transition

    def stats(self) -> dict:
        return {
            "n_clusters": len(self.clusters),
            "sizes": [len(c) for c in self.clusters.values()],
        }


# ═══════════════════════════════════════════════════════════
# VirtualMemoryManager v2
# ═══════════════════════════════════════════════════════════

class VirtualMemoryManager:
    """LLM Virtual Memory Manager v2.

    5-tier residency + speculative prefetch + layer overlap.
    """

    def __init__(self, n_experts: int = 64, n_layers: int = 16,
                 ram_budget_mb: float = 150, gpu_budget_mb: float = 50,
                 expert_size_mb: float = 6.0):
        expert_bytes = int(expert_size_mb * 1e6)
        self.pt = PageTable(
            n_experts=n_experts, n_layers=n_layers,
            expert_size_bytes=expert_bytes,
            warm_budget_bytes=int(ram_budget_mb * 1e6),
            hot_budget_bytes=int(gpu_budget_mb * 1e6),
        )
        self.affinity = AffinityClustering(n_experts)
        self.prefetch = SpeculativePrefetch(self.pt)
        self.overlap = LayerOverlapScheduler(n_layers)
        self.eviction = EvictionPolicy(self.pt)
        self.pressure = MemoryPressure(self.pt)
        self.active_experts: dict[int, set[int]] = field(default_factory=lambda: {l: set() for l in range(n_layers)})

    def start_token(self):
        self.active_experts = {l: set() for l in range(self.pt.n_layers)}

    def access_expert(self, layer_idx: int, expert_id: int,
                      compute_us: float = 400) -> tuple[int, float]:
        """Access expert. Returns (latency_us, io_hidden_us)."""
        latency = self.pt.access(layer_idx, expert_id)
        self.affinity.observe(layer_idx, expert_id)
        self.prefetch.record_hit(layer_idx, expert_id)
        self.active_experts[layer_idx].add(expert_id)

        # Promote to HOT if budget allows
        entry = self.pt.entries[layer_idx][expert_id]
        if (entry.tier == Tier.WARM and
            self.pt.hot_used_bytes + entry.size_bytes <= self.pt.hot_budget_bytes):
            self.pt.promote_to_hot(layer_idx, expert_id)

        # Evict if WARM over budget
        if self.pt.warm_used_bytes > self.pt.warm_budget_bytes:
            victim = self.eviction.select_victim(
                layer_idx, self.active_experts[layer_idx], Tier.WARM)
            if victim is not None:
                self.pt.evict_one_step(layer_idx, victim)

        # Layer overlap scheduling
        io_us = TIER_LATENCY_US[entry.tier]
        actual, hidden = self.overlap.schedule(layer_idx, compute_us, io_us)

        return latency, hidden

    def end_token(self):
        self.pt.advance_token()
        self.pressure.update()

    def build_affinity(self, n_clusters: int = 4):
        trans = self.affinity.build(n_clusters)
        self.prefetch.set_transition_matrix(trans)
        self.eviction.affinity = trans

    def stats(self) -> dict:
        return {
            "page_table": self.pt.stats(),
            "prefetch_hit_rate": round(self.prefetch.hit_rate(), 3),
            "overlap": self.overlap.stats(),
            "pressure": self.pressure.stats(),
            "affinity": self.affinity.stats(),
        }
