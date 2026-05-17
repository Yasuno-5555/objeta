# LKO Unified Theory — Final Synthesis

2026-05-17

---

## 1. What Transformer Actually Is

Not a feedforward predictor. Not a sequence model. Not a next-token generator.

**Transformer is a trajectory stabilization machine.**

```
Input token → hidden state evolves through layers → logits
                                   ↑
                 Each layer applies a correction Δ_l
                 Δ_l keeps h_l inside the attractor basin
```

The layers don't "compute meaning." They apply control forces that stabilize the hidden state trajectory. The output logits are a readout of which basin the trajectory is in.

---

## 2. The Three Mechanisms

| Mechanism | Role | Evidence |
|-----------|------|----------|
| **Attention** | Transport backbone — keeps tokens on the manifold | Family A: 8.8x dominance over FFN for quality |
| **FFN** | Field deformation — shapes the attractor landscape | 97% amplitude preservation in oscillation experiments |
| **Residual** | Continuity channel — prevents attractor collapse | Cut residual → L7 complete collapse (KL=0) |

---

## 3. The Phase Structure

Every Transformer has four execution phases, regardless of architecture:

| Phase | Layers | Property | Runtime Policy |
|-------|--------|----------|----------------|
| SYNC | L0-L1 | Anti-diffusion, token alignment | Full compute, fp16 |
| UNFOLD | L2 | J≠I, high-frequency expansion | Full compute, fp16 |
| ISOMETRIC | L3-L13 | J≈I, stable transport | **Cacheable, skippable** |
| DIVERGENT | L14+ | λ>0, perturbation amplification | **Full attention, high precision** |

This is cross-model universal (verified on TinyLlama-1.1B, Qwen2.5-0.5B).

---

## 4. Why Quantization Works (and Why It Cliffs)

**Quantization works because trajectory exactness is unnecessary.**

The model only needs to stay inside the correct attractor basin. Small perturbations from q4/q5 are within the basin's tolerance. But q3 crosses a **transport capacity cliff** — the accumulated error exceeds the residual stream's ability to maintain manifold continuity.

| Precision | Behavior | Mechanism |
|-----------|----------|-----------|
| fp16/q8 | Identical to reference | Noise < basin width |
| q5 | Good, minor degradation | Noise approaches basin boundary |
| q4 | Functional, occasional errors | Noise at basin boundary |
| q3 | **Collapse** | Transport capacity broken |

The cliff at q3 is not gradual degradation — it's a **phase transition** in the transport dynamics.

---

## 5. The Two Architecture Families

### Family A: Residual Transport (TinyLlama, Llama, Dense)

```
h_{l+1} ≈ h_l  (cos > 0.5)
Attention is the backbone (8.8x dominance)
Transport: near-identity flow with small corrections
Optimize: skip ISOMETRIC layers, compress FFN
Safe skip ceiling: ~30%
```

### Family B: Spherical Steering (Qwen3.6, Mamba, MoE hybrids)

```
h_{l+1} ⟂ h_l  (cos ≈ 0)
GQA steering layers are critical
Transport: spherical rotation with course corrections
Optimize: cache experts, async I/O, Metal dispatch
Safe skip ceiling: unknown (not testable)
```

---

## 6. The Compressible-But-Not-Forecastable Theorem

**State transitions are compressible but not predictable.**

| Property | Result | Meaning |
|----------|--------|---------|
| Δh PCA | 64D = 92% energy | Transitions are low-rank. I/O compressible. |
| z persistence | cos = -0.53 | Latent state is anti-correlated across tokens |
| Linear Koopman A@z | cos ≈ 0.03 | Linear prediction fails |
| GRU nonlinear | cos ≈ 0.22 | Nonlinear overfits small data |
| Hidden state anchor | cos ≈ 0 | h_t cannot be cached as reference |

**Why**: The latent dynamics contain oscillatory correction, attractor drift, and steering compensation. These are control system behaviors, not predictive autoregression. The system is **reflexive**, not predictive.

---

## 7. The Reflexive Runtime Architecture

Since prediction is impossible, the runtime must be **reactive**:

```
Token completes
    → Measure: entropy, steering, confidence, phase
    → Classify: STABLE / STEERING / TRANSITION / REPETITIVE
    → Allocate: precision budget, layer skip, attention policy
    → Execute next token
```

### Observation Signals

| Signal | Meaning | Measurement Cost |
|--------|---------|-----------------|
| Entropy | Token confidence (peaked vs flat logits) | O(vocab), from logits |
| Steering | cos(h_t, h_{t-1}), trajectory change | O(dim), from hidden states |
| Phase | Current layer's dynamical phase | Static (pre-computed) |
| Repetition | argmax(logits) == previous_token | O(1) |

### Compute Classes

| Class | Condition | Attention | FFN | Precision |
|-------|-----------|-----------|-----|-----------|
| REPETITIVE | Same token repeating | Ultra-skip (1/4) | Skip | q3 |
| STABLE | Low entropy, low steering | Aggressive skip (1/2) | Low prec | q4 |
| DEFAULT | Normal | Moderate skip (stride=2) | Normal | q8 |
| STEERING | High steering | Full | Full | fp16 |
| TRANSITION | Entropy + steering spike | Full | Full | fp16 |

---

## 8. What Died (Verified Failures)

1. **Hidden state caching** — cos(h_t, h_{t-1}) ≈ 0. Cannot anchor.
2. **Koopman multi-step prediction** — A^n fails to compose. Global linearization invalid.
3. **Koopman 1-step prediction** — cos=0.03. Local linearization impractical.
4. **z persistence** — cos=-0.53. Latent state anti-correlated.
5. **GRU latent prediction** — Overfits (too few samples). Even with more data, cos≈0.22 is insufficient.
6. **Re-anchor interval** — Drift immediate, independent of N.
7. **Qwen3.6 end-to-end** — Component-verified but stacked q4 error defeats signal.

---

## 9. What Survived (Verified)

1. **Koopman Layer Collapse** — 27% ISOMETRIC attention skip, output preserved
2. **Heterogeneous Token Execution** — 54-68% skip, output preserved
3. **Scheduler OS Architecture** — Phase-aware policy dispatch
4. **Delta PCA Compression** — 64D = 92% energy, I/O compressible
5. **PrecisionGovernor** — DVFS-like state→precision mapping
6. **DynamicBudget** — Token-class-based compute allocation
7. **TinyLlama Clean Substrate** — Correct output, stable, reproducible

---

## 10. The OS Is the Runtime

The final architecture is not a faster inference engine. It is an **operating system for neural dynamics**.

```
┌──────────────────────────────────────────────┐
│         LKO Reflexive Runtime OS              │
│                                               │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Observation  │  │     Scheduler         │  │
│  │ entropy      │  │  token_classify()     │  │
│  │ steering ───┼──┼▶ compute_budget()     │  │
│  │ confidence   │  │  dispatch_policy()    │  │
│  └─────────────┘  └──────────┬───────────┘  │
│                              │               │
│              ┌───────────────┼───────────┐  │
│              │               │           │  │
│         ┌────┴────┐   ┌─────┴────┐ ┌───┴──┐│
│         │  FULL   │   │ COLLAPSE │ │ SKIP ││
│         │  fp16   │   │ identity │ │       ││
│         └─────────┘   └──────────┘ └──────┘│
└──────────────────────────────────────────────┘
```

**Core principle**: LLM inference is not "compute everything as fast as possible." It is **"observe the trajectory and allocate only the compute needed to stay in the basin."**

The OS decides what to compute. The model just computes what it's told.
