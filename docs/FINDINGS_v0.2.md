# objeta Findings v0.2 — 2026-05-16

## 3 failed approaches, 1 confirmed direction

### M2: Rotation Kernel — FAILED

Goal: replace FFN with low-rank rotation Δ ≈ U_k Σ_k V_k^T x.

Single-layer result:
- cos(Δ_full, Δ_rot) at k=64: 0.87
- cos(Δ_full, Δ_rot) at k=192: 0.99
- FLOPs saved: 98%

22-layer autoregressive rollout result:
- token agreement: **0.167** (16.7%)
- hidden cos after first generated token: **0.17**
- repetition rate: **0.829**
- output text: **empty** (complete degeneration)

Root cause:
- Distribution shift: bases computed from random inputs, real hidden states differ
- Error accumulation: 3% per-layer error compounds exponentially across 22 layers
- DIVERGENT phase (L16+): λ>0 amplifies any error

### M4: Latent MoE — FAILED

Goal: extract latent experts from dense FFN via spectral clustering.

Result:
- active_neurons: **100%** across all 22 layers
- sparsity: **≈0**
- expert coverage: 13-17% per cluster (uniform distribution)
- **No natural expert structure exists**

Root cause:
- Dense FFNs don't contain hidden sparse conditional compute
- All neurons contribute slightly to every input
- Neuron specialization is absent

### Rotation kernel (real-distribution bases) — not run, structurally predicted to fail

Even with real-distribution bases, the autoregressive rollout causes p(h_t) → p'(h_t) drift,
breaking any fixed-basis approximation. The tangent approximation is local, but the flow is global.

## What we learned

### The critical inversion

| | Old assumption | Actual finding |
|---|---|---|
| FFN operator | Low-rank | **Full-rank** (down eff_rank=1846/2048) |
| FFN activations | Sparse, clustered | **100% active, uniform** |
| Trajectory manifold | Consequence of operator | **Generator of operator usage** |
| What to compress | Operator | **Trajectory** |

### Core theoretical insight

```
Transformer:
  Operator is HIGH-DIMENSIONAL (full rank, dense)
  Trajectory is LOW-DIMENSIONAL (d_95=1, thin transport tube)

→ Compress the trajectory, not the operator.
```

This is the opposite of conventional model compression.

### Why rotation kernel failed (mathematical)

The FFN operator F: R^d → R^d has full rank. The approximation F̂_k = U_k Σ_k V_k^T has rank k.
For the autoregressive map T(h) = h + F(h) + Attn(h):

```
||T(h) - T̂_k(h)|| ≤ ||F(h) - F̂_k(h)||  (per-step error ~3%)
```

After N steps:

```
||T^N(h₀) - T̂_k^N(h₀)|| ≈ O(N · ε · exp(λ_max N))
```

where λ_max > 0 in the DIVERGENT phase. The Lyapunov exponent dominates.
This is not an approximation accuracy problem — it's a **dynamical stability problem**.

## The new direction: Trajectory Compiler

### Architecture

```
Dense Transformer
    ↓
Static Geometry Analysis (phase, inversion, coupling)
    ↓
Trajectory Collection (real prompts → hidden states, Δ, entropy)
    ↓
Trajectory Archetype Extraction (cluster full 22-layer paths)
    ↓
Runtime State Machine (archetype → compute policy)
    ↓
Stability-Controlled Execution
```

### Runtime nodes (new IR)

```rust
enum RuntimeNode {
    FullCompute,                              // L0-2, L16-21: sacred + divergent
    ArchetypeLookup { archetype_id, delta },  // L3-15: lookup Δ for this archetype
    RefreshPoint,                             // L3 (Type I), L8 (Type II): forced full compute
    Skip,                                     // identity when cos > 0.97
}
```

### What we track at runtime

```
Position in sequence
Current phase zone
Hidden state archetype
Entropy trend
Lyapunov estimate (||δh_{t+1}|| / ||δh_t||)
```

### Key go/no-go criterion

```
var(Δ_l | archetype) / var(Δ_l) < 0.3
```

If intra-archetype Δ variance is <30% of total Δ variance, archetype lookup replaces FFN.
Currently being measured (experiment: trajectory_archetypes.py).

## Experimental results summary

| Experiment | Metric | Value | Verdict |
|---|---|---|---|
| Single-layer rotation cos (k=192) | cos(Δ, Δ̂) | 0.992 | ✓ local fidelity ok |
| Single-layer rotation cos (k=64) | cos(Δ, Δ̂) | 0.874 | △ marginal |
| 22-layer rollout (k=192) | token agreement | 0.167 | ✗ catastrophic |
| 22-layer rollout (k=192) | hidden cos | 0.177 | ✗ catastrophic |
| 22-layer rollout (k=192) | repetition rate | 0.829 | ✗ degeneration |
| Neuron sparsity | active fraction | 1.000 | ✗ no expert structure |
| Neuron clustering | expert coverage | 0.13-0.17 | ✗ uniform, no clusters |
| Down matrix eff_rank | rank | 1846/2048 | ✗ near full rank |
| Δ eff_rank | rank | 80-188 | △ moderate compressibility |
| Per-layer hidden cos (L0) | cos | 0.734 | ✗ 26% drift at first layer |
| Per-layer hidden cos (L17) | cos | -0.022 | ✗ anti-correlation |
| Trajectory archetype viability | (running) | TBD | go/no-go for trajectory VM |

## Phase-specific design

| Phase | Layers | Lyapunov | Policy |
|---|---|---|---|
| SYNC | L0-L1 | λ≈0, D<0 (anti-diffusion) | Full compute |
| UNFOLD | L2 | J≠I, σ_max≈48 | Full compute |
| ISOMETRIC-LOCAL | L3-L6 | λ≈0, J≈I | Archetype lookup candidate |
| ISOMETRIC-GLOBAL | L7-L13 | λ≈0, cos<0 (inversion) | Archetype + full attention |
| DIVERGENT | L14-L21 | λ>0 (amplification) | **Full compute mandatory** |

## SSD streaming viability

SSD streaming requires trajectory predictability. If archetypes form a low-entropy Markov chain,
speculative prefetch becomes viable:

```
if archetype == A7:
    prefetch experts for archetypes [A3, A7, A12]  // top-3 transitions
```

This is only viable if:
1. Archetype count is small (< 20)
2. Transition entropy is low (diagonal-dominant transition matrix)
3. Prefetch latency is hidden by current-layer compute

## Next steps

1. **Complete trajectory archetype experiment** — measure var(Δ|archetype) / var(Δ)
2. **If go**: implement archetype lookup runtime, measure end-to-end perplexity
3. **If no-go**: investigate hybrid approaches (archetype-aware refresh, partial compute)
4. **Irrespective**: DIVERGENT phase requires full compute — this is a hard constraint from Lyapunov theory
