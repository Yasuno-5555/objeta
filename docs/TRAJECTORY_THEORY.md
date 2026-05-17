# Trajectory Theory — Why compressing the operator fails

2026-05-16

## The fundamental asymmetry

```
Transformer = high-dimensional operator + low-dimensional trajectory

  Operator:    F: R^d → R^d    eff_rank(F) ≈ d       (full rank)
  Trajectory:  T(h) = h + Attn(h) + F(h)   dim(orbit) ≈ 1  (thin tube)
```

The operator CAN compute in all directions. The trajectory DOES NOT use all directions.

This is not a bug. It's the defining property of overparameterized neural computation.

## Why local approximation fails globally

For a single step with approximated FFN F̂_k:

```
||T(h) - T̂_k(h)|| = ||F(h) - F̂_k(h)|| ≤ ε · ||F(h)||
```

where ε ≈ 0.01 at k=192. Per-step error is small.

For N autoregressive steps:

```
δ_N = ||T^N(h₀) - T̂^N(h₀)||
    ≤ Σ_{t=0}^{N-1} ||J_T(h_t)||^t · ε · ||F(h_t)||
    ≈ O(N · ε · exp(λ_max · N))
```

where λ_max is the maximal Lyapunov exponent of the true dynamics.

If λ_max > 0 (DIVERGENT phase), the error grows exponentially with sequence length.
If λ_max ≈ 0 (ISOMETRIC phase), the error grows only linearly.

**This is why rotation kernel works for single layers but fails for rollouts.**

## The Lyapunov phase map

Measured on TinyLlama-1.1B:

| Phase | Layers | λ_max | Error growth | Approximation viability |
|---|---|---|---|---|
| SYNC | L0-1 | ≈0, D<0 | Anti-damped | Full compute only (short, cheap) |
| UNFOLD | L2 | >0 (J≠I) | Exponential | Full compute mandatory |
| ISOMETRIC-LOCAL | L3-6 | ≈0 | Linear | **Archetype lookup viable** |
| ISOMETRIC-GLOBAL | L7-13 | ≈0 | Linear | **Archetype lookup viable** |
| DIVERGENT | L14-21 | >0 | Exponential | **Full compute mandatory** |

## The thin transport tube

LKO theory established: d_95 = 1 (95% of trajectory variance in 1 dimension).

Visual: the 2048-dimensional hidden state doesn't explore R^2048. It moves through
a narrow tube whose cross-section is ~1-dimensional.

This means:
- The trajectory has **geometry** (direction, curvature, torsion)
- The trajectory has **topology** (branching, merging, cycles)
- The trajectory is **predictable** (low entropy in archetype space)

## The archetype hypothesis

If trajectories cluster into a small number of archetypes:
- Each archetype has a characteristic Δ signature per layer
- Within an archetype, Δ variance is small: var(Δ|A) << var(Δ)
- Archetype transitions are Markov-predictable: P(A_{t+1}|A_t) has low entropy

Then:
- runtime = archetype classifier + Δ lookup table + stability controller
- compute = O(d · n_archetypes) instead of O(d · ffn_dim)

## What we proved DOESN'T work

1. **Global low-rank FFN** — trajectory drift breaks the tangent approximation
2. **Neuron-level expert extraction** — no sparse structure in dense FFN activations
3. **Fixed-basis approximation** — p(h_t) shifts during autoregression

## What might work

1. **Archetype-based Δ lookup** — if var(Δ|archetype) is small enough
2. **Phase-aware refresh scheduling** — Type I (L3) and Type II (L8) forced full compute
3. **Stability-controlled execution** — monitor Lyapunov estimate, fall back to full compute
4. **Hybrid sparse execution** — archetype lookup in ISOMETRIC, full compute in DIVERGENT

## The final form

Objeta is not a model compressor. It is a **trajectory compiler**.

Input: model weights + real prompt distribution
Output: state machine where each state maps to a compute policy

```python
if phase == DIVERGENT:
    return full_compute(x)
elif phase == ISOMETRIC and var_ratio[archetype][layer] < 0.3:
    return x + delta_cache[archetype][layer]
else:
    return x + ffn(x)  # fallback
```
