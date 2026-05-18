"""Cross-Request Expert Residency — shared cache across sessions.

Like a CPU's shared L3 cache, this manages expert residency
across multiple concurrent users and requests.

Key mechanisms:
  1. Global Expert Cache — shared across all sessions
  2. Conversational Locality — same user returns → same experts
  3. Workload Clustering — similar prompts → similar expert sets
  4. Semantic Affinity — topic-based expert grouping
  5. Cross-User Prefetch — user A's experts pre-loaded for user B

This is not a metaphor. It IS a shared cache controller.
"""

from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import hashlib

import numpy as np


@dataclass
class CacheLine:
    """One cache line = one expert in the shared residency pool."""
    expert_id: int
    layer_idx: int
    last_access: float = 0.0       # wall-clock timestamp
    access_count: int = 0
    access_count_window: int = 0   # recent window
    pinned: bool = False
    # Affinity
    co_accessed: dict[int, int] = field(default_factory=dict)  # expert → count
    # Semantic tags
    prompt_hashes: set[str] = field(default_factory=set)
    topic_tags: set[str] = field(default_factory=set)


@dataclass
class SessionState:
    """Per-session (per-user/conversation) expert affinity."""
    session_id: str
    created: float = 0.0
    last_active: float = 0.0
    expert_history: list[tuple[int, int]] = field(default_factory=list)  # [(layer, expert), ...]
    prompt_hashes: list[str] = field(default_factory=list)
    topic: str = ""
    # Affinity score per expert
    expert_affinity: dict[int, float] = field(default_factory=dict)


class CrossRequestResidency:
    """Shared expert cache across concurrent sessions.

    Like a CPU's L3 cache:
      - L1 = per-token working set
      - L2 = per-session affinity cache
      - L3 = global shared residency pool (this)
    """

    def __init__(self, n_experts: int = 256, n_layers: int = 40,
                 max_loaded: int = 50, max_warm: int = 100):
        self.n_experts = n_experts
        self.n_layers = n_layers
        self.max_loaded = max_loaded
        self.max_warm = max_warm

        # Global cache
        self.cache: dict[tuple[int, int], CacheLine] = {}  # (layer, expert) → CacheLine

        # Sessions
        self.sessions: dict[str, SessionState] = {}

        # Topic clusters
        self.topic_clusters: dict[str, set[int]] = defaultdict(set)

        # Statistics
        self.cross_user_hits: int = 0
        self.same_user_hits: int = 0
        self.cold_misses: int = 0
        self.evictions: int = 0

    # ── Session management ──

    def create_session(self, session_id: str | None = None) -> str:
        """Create a new session. Returns session_id."""
        sid = session_id or hashlib.sha256(
            f"{time.time()}-{len(self.sessions)}".encode()).hexdigest()[:12]
        self.sessions[sid] = SessionState(
            session_id=sid, created=time.time(), last_active=time.time())
        return sid

    def close_session(self, session_id: str):
        """Close a session. Keep its affinity data for future sessions."""
        if session_id in self.sessions:
            # Decay expert affinities (keep for future sessions)
            session = self.sessions[session_id]
            # Don't delete — let GC handle
            pass

    # ── Access tracking ──

    def access(self, session_id: str, layer_idx: int, expert_id: int,
               prompt_hash: str = "", topic: str = ""):
        """Record an expert access from a specific session.

        Updates:
          - Global cache line
          - Session affinity
          - Topic clustering
        """
        now = time.time()
        key = (layer_idx, expert_id)

        # Update session
        if session_id not in self.sessions:
            self.create_session(session_id)
        session = self.sessions[session_id]
        session.last_active = now
        session.expert_history.append((layer_idx, expert_id))
        if prompt_hash:
            session.prompt_hashes.append(prompt_hash)
        if topic:
            session.topic = topic

        # Session affinity: exponential moving average
        decay = 0.9
        for eid in session.expert_affinity:
            session.expert_affinity[eid] *= decay
        session.expert_affinity[expert_id] = (
            session.expert_affinity.get(expert_id, 0) * decay + 1.0
        )

        # Update global cache
        if key not in self.cache:
            self.cache[key] = CacheLine(expert_id=expert_id, layer_idx=layer_idx)

        line = self.cache[key]
        line.last_access = now
        line.access_count += 1
        line.access_count_window += 1
        if prompt_hash:
            line.prompt_hashes.add(prompt_hash)
        if topic:
            line.topic_tags.add(topic)

        # Topic clustering
        if topic:
            self.topic_clusters[topic].add(expert_id)

    # ── Lookup ──

    def is_cached(self, layer_idx: int, expert_id: int) -> bool:
        """Is this expert in the global residency pool?"""
        return (layer_idx, expert_id) in self.cache

    def get_affinity(self, session_id: str, expert_id: int) -> float:
        """Get a session's affinity score for an expert (0-1)."""
        if session_id not in self.sessions:
            return 0.0
        return self.sessions[session_id].expert_affinity.get(expert_id, 0.0)

    def predict_experts(self, session_id: str, layer_idx: int,
                         n: int = 4) -> list[int]:
        """Predict which experts this session will need next.

        Uses:
          1. Session affinity (personal history)
          2. Topic clustering (similar users)
          3. Global frequency (everyone)
        """
        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]
        scores = {}

        # 1. Session affinity (weight: 0.5)
        for eid, aff in session.expert_affinity.items():
            scores[eid] = scores.get(eid, 0) + aff * 0.5

        # 2. Topic clustering (weight: 0.3)
        if session.topic and session.topic in self.topic_clusters:
            for eid in self.topic_clusters[session.topic]:
                scores[eid] = scores.get(eid, 0) + 0.3

        # 3. Global recency (weight: 0.2)
        for (l, eid), line in self.cache.items():
            if l == layer_idx:
                recency = np.exp(-(time.time() - line.last_access) / 60.0)
                scores[eid] = scores.get(eid, 0) + recency * 0.2

        # Sort and return top N
        sorted_experts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in sorted_experts[:n]]

    def find_similar_sessions(self, session_id: str, n: int = 3) -> list[str]:
        """Find other sessions with similar expert usage patterns."""
        if session_id not in self.sessions:
            return []

        current = self.sessions[session_id]
        current_experts = set(eid for _, eid in current.expert_history[-32:])

        similarities = []
        for sid, other in self.sessions.items():
            if sid == session_id:
                continue
            other_experts = set(eid for _, eid in other.expert_history[-32:])
            if not current_experts or not other_experts:
                continue
            overlap = len(current_experts & other_experts)
            jaccard = overlap / len(current_experts | other_experts)
            similarities.append((sid, jaccard))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in similarities[:n]]

    # ── Cache management ──

    def evict_lru(self, n: int = 1):
        """Evict least recently used experts from global cache."""
        now = time.time()
        candidates = []
        for key, line in self.cache.items():
            if not line.pinned:
                age = now - line.last_access
                score = age / (1 + line.access_count_window * 0.1)
                candidates.append((score, key))

        candidates.sort(reverse=True)
        for _, key in candidates[:n]:
            del self.cache[key]
            self.evictions += 1

    def decay_window(self):
        """Decay the access count window (called periodically)."""
        for line in self.cache.values():
            line.access_count_window = int(line.access_count_window * 0.8)

    def stats(self) -> dict:
        total = self.cross_user_hits + self.same_user_hits + self.cold_misses
        return {
            "cached_experts": len(self.cache),
            "active_sessions": len(self.sessions),
            "topic_clusters": len(self.topic_clusters),
            "hit_rate": round((self.cross_user_hits + self.same_user_hits) / max(1, total), 3),
            "cross_user_hits": self.cross_user_hits,
            "same_user_hits": self.same_user_hits,
            "cold_misses": self.cold_misses,
            "evictions": self.evictions,
        }


# ═══════════════════════════════════════════════════════════
# Workload Clustering
# ═══════════════════════════════════════════════════════════

class WorkloadClusterer:
    """Groups similar prompts/users by expert usage patterns.

    Detects:
      - Same topic → similar expert sets
      - Same user returning → affinity reuse
      - Burst detection → prefetch all burst experts
    """

    def __init__(self, residency: CrossRequestResidency):
        self.res = residency
        self.prompt_to_experts: dict[str, list[int]] = {}
        self.user_profiles: dict[str, dict] = {}

    def hash_prompt(self, text: str) -> str:
        """Simple semantic hash of prompt text."""
        return hashlib.sha256(text.lower().encode()).hexdigest()[:16]

    def record_query(self, session_id: str, prompt: str,
                     experts_used: list[int], topic: str = ""):
        """Record a query and its expert usage."""
        phash = self.hash_prompt(prompt)
        self.prompt_to_experts[phash] = list(set(experts_used))

        # Update user profile
        if session_id not in self.user_profiles:
            self.user_profiles[session_id] = {
                "topics": defaultdict(int),
                "expert_freq": defaultdict(int),
            }
        profile = self.user_profiles[session_id]
        if topic:
            profile["topics"][topic] += 1
        for eid in experts_used:
            profile["expert_freq"][eid] += 1

        # Record in residency
        for eid in set(experts_used):
            self.res.access(session_id, 0, eid, prompt_hash=phash, topic=topic)

    def find_similar_prompts(self, prompt: str, n: int = 3) -> list[str]:
        """Find similar past prompts by expert overlap."""
        phash = self.hash_prompt(prompt)
        current_experts = set(self.prompt_to_experts.get(phash, []))

        if not current_experts:
            return []

        similarities = []
        for other_hash, other_experts in self.prompt_to_experts.items():
            if other_hash == phash:
                continue
            other_set = set(other_experts)
            overlap = len(current_experts & other_set)
            jaccard = overlap / max(1, len(current_experts | other_set))
            if jaccard > 0.3:
                similarities.append((other_hash, jaccard))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in similarities[:n]]

    def preload_for_prompt(self, prompt: str, n_experts: int = 8) -> list[int]:
        """Preload experts based on similar past prompts."""
        similar = self.find_similar_prompts(prompt, n=3)
        experts = set()
        for phash in similar:
            experts.update(self.prompt_to_experts.get(phash, [])[:n_experts])
        return list(experts)[:n_experts]
