"""Observation pipeline — runtime signal measurement.

All measurements are O(dim) or O(vocab) — negligible vs layer compute (<0.1%).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    """Runtime observation signals — measured every token."""

    # Token softmax entropy: 0=peaked (certain), 1=uniform (uncertain)
    entropy: float = 0.0

    # Steering magnitude: 1 - cos(h_t, h_{t-1})
    steering: float = 0.0

    # MoE routing entropy (if applicable)
    routing_entropy: float | None = None

    # Attention map change: 1 - mean(cos(A_heads_t, A_heads_{t-1}))
    attention_divergence: float | None = None

    # Top-1 logit value
    top1_logit: float = 0.0

    # Whether predicted token repeats the previous
    is_repeat: bool = False

    # Token position in sequence
    token_index: int = 0
    seq_len: int = 0

    # Per-layer measurements (optional, for detailed analysis)
    layer_entropies: list[float] = field(default_factory=list)
    layer_hidden_cos: list[float] = field(default_factory=list)


class ObservationPipeline:
    """Measures runtime signals from model outputs.

    Three primary signals:
      1. entropy — from logits (softmax distribution shape)
      2. steering — from hidden states (trajectory change)
      3. attention_divergence — from attention weights (transport stability)
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.prev_hidden: np.ndarray | None = None
        self.prev_attn_weights: dict[int, np.ndarray] = {}

    def reset(self):
        self.prev_hidden = None
        self.prev_attn_weights.clear()

    def observe_logits(self, logits: np.ndarray) -> tuple[float, float, bool]:
        """Compute entropy, top-1 logit, and repeat flag from logits.

        Returns (entropy, top1_logit, is_repeat).
        """
        logits_stable = logits - logits.max()
        probs = np.exp(logits_stable.astype(np.float64))
        probs /= probs.sum()

        max_ent = np.log(len(probs))
        shannon = -float(np.sum(probs * np.log(probs + 1e-12)))
        entropy = shannon / max_ent if max_ent > 0 else 0.0

        top1 = int(np.argmax(logits))
        top1_logit = float(logits[top1])

        return entropy, top1_logit, top1

    def observe_hidden(self, hidden: np.ndarray) -> float:
        """Compute steering magnitude: 1 - cos(h_t, h_{t-1}).

        Returns steering in [0, 2]. 0 = identical, 2 = opposite.
        """
        h = hidden.flatten().astype(np.float64)
        if self.prev_hidden is None:
            self.prev_hidden = h.copy()
            return 0.0

        cos = float(np.dot(h, self.prev_hidden) /
                    (np.linalg.norm(h) * np.linalg.norm(self.prev_hidden) + 1e-12))
        steering = 1.0 - cos
        self.prev_hidden = h.copy()
        return steering

    def observe_attention(self, layer_idx: int,
                          attn_weights: np.ndarray) -> float | None:
        """Compute attention divergence from previous step.

        attn_weights shape: (n_heads, seq_len).
        Returns mean(1 - cos(A_head_t, A_head_{t-1})) across heads.
        """
        if layer_idx not in self.prev_attn_weights:
            self.prev_attn_weights[layer_idx] = attn_weights.copy()
            return None

        prev = self.prev_attn_weights[layer_idx]
        if prev.shape != attn_weights.shape:
            self.prev_attn_weights[layer_idx] = attn_weights.copy()
            return None

        n_heads = attn_weights.shape[0]
        divergences = []
        for h in range(n_heads):
            cs = float(np.dot(attn_weights[h], prev[h]) /
                      (np.linalg.norm(attn_weights[h]) * np.linalg.norm(prev[h]) + 1e-12))
            divergences.append(1.0 - cs)

        self.prev_attn_weights[layer_idx] = attn_weights.copy()
        return float(np.mean(divergences)) if divergences else None


# Standalone convenience functions

def compute_entropy(logits: np.ndarray) -> float:
    """Normalized Shannon entropy from logits."""
    stable = logits - logits.max()
    probs = np.exp(stable.astype(np.float64))
    probs /= probs.sum()
    max_ent = np.log(len(probs))
    ent = -float(np.sum(probs * np.log(probs + 1e-12)))
    return ent / max_ent if max_ent > 0 else 0.0


def compute_steering(h_curr: np.ndarray, h_prev: np.ndarray) -> float:
    """1 - cos(h_curr, h_prev)."""
    c = h_curr.flatten().astype(np.float64)
    p = h_prev.flatten().astype(np.float64)
    cos = float(np.dot(c, p) / (np.linalg.norm(c) * np.linalg.norm(p) + 1e-12))
    return 1.0 - cos


def compute_attention_divergence(a_curr: np.ndarray,
                                  a_prev: np.ndarray) -> float | None:
    """Mean per-head attention divergence."""
    if a_curr.shape != a_prev.shape:
        return None
    n_heads = a_curr.shape[0]
    divs = []
    for h in range(n_heads):
        cs = float(np.dot(a_curr[h], a_prev[h]) /
                  (np.linalg.norm(a_curr[h]) * np.linalg.norm(a_prev[h]) + 1e-12))
        divs.append(1.0 - cs)
    return float(np.mean(divs))
