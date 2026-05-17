# objeta — Final Synthesis 2026-05-16

## What died (independent, verified failures)

| Approach | Experiment | Verdict |
|----------|-----------|---------|
| FFN low-rank rotation | 22-layer rollout | ✗ cos→0.17, degeneration |
| Latent MoE extraction | Neuron graph | ✗ 100% active, no clusters |
| Trajectory archetype lookup | Δ variance per archetype | ✗ ratio≈0.6-1.2, no reduction |

These are **independent** failures. Together they form strong evidence:
**Dense Transformer operators cannot be compressed post-hoc.**

## What survived (theoretical value)

### The core insight

```
observable geometry ≠ generative dynamics
```

What is low-dimensional:
- Δ direction structure
- Phase oscillation
- Inversion schedule
- Cosine geometry between layers
- d₉₅≈1 (macroscopic transport axis)

What is NOT low-dimensional:
- Exact next-state update
- Local velocity field
- Rollout dynamics

### Transformer is a stiff dense system

Not a flexible sparse routing machine. Not a sleeping MoE. It's a highly constrained
dense dynamical system where:
- Every neuron matters
- Small errors amplify (DIVERGENT: λ>0)
- Trajectory is narrow but dynamics are stiff

### The Lyapunov phase map (concrete, reusable)

| Phase | Layers | λ_local | Error behavior | Policy |
|---|---|---|---|---|
| SYNC | L0-1 | ≈0, D<0 | Anti-damped | q3-q4 safe |
| UNFOLD | L2 | J≠I | Exponential | fp16 mandatory |
| ISOMETRIC | L3-13 | ≈0 | Linear | q4-q5 safe |
| DIVERGENT | L14-21 | >0 | Exponential | fp16/q8 mandatory |

## What objeta becomes

### From: Transformer decomposer (dead)
- FFN → low-rank rotation
- Dense → latent MoE
- Operator replacement

### To: Stability orchestrator (viable)
- Phase-aware quantization
- Stability-guided precision control
- Instability-aware refresh scheduling
- Phase-profile-driven adaptive runtime

## Two concrete paths

### Path A: MoE Compiler (medium risk)

Target: Qwen3.6-35B-A3B (real expert sparsity exists)

Use objeta's analysis pipeline to:
- Detect bridge layers (L2 circuit breaker)
- Determine per-layer precision topology
- Generate static expert tier assignments (hot/warm/cold)
- Output phase_profile.json → Rust dispatch config

### Path B: Adaptive Runtime Enhancement (lower risk)

Target: v8 adaptive runtime (TinyLlama already working)

Use objeta's phase profile to:
- Bake per-zone compute policies into the runtime
- Replace runtime entropy detection with static phase boundaries
- Generate refresh schedules from inversion zone detection
- Output phase_profile.json → adaptive policy config

## What to keep in the codebase

```
objeta/
├── crates/
│   ├── objeta-core       ✓ keep: Phase, Family, LayerProfile, PhaseProfile
│   ├── objeta-parser     ✓ keep: mmap weight loader
│   ├── objeta-analysis   ✓ keep: static geometry analysis
│   ├── objeta-phase      ✓ keep: phase/family classification
│   ├── objeta-cli        ✓ keep: analyze command
│   ├── objeta-routing    △ repurpose: MoE routing analysis
│   ├── objeta-runtime    △ repurpose: stability-aware policy generation
│   ├── objeta-expert     ✗ remove: latent expert extraction (proven non-viable)
│   ├── objeta-metal      △ keep: rotation kernel for reference, shared attn for future
│   └── objeta-ssd        △ repurpose: expert storage layout (for MoE path)
├── experiments/
│   ├── verify_rotation           experimental record (negative: cos→0.17)
│   ├── rollout_divergence        experimental record (negative: degeneration)
│   ├── build_neuron_graph        experimental record (negative: no expert structure)
│   ├── trajectory_archetypes     experimental record (negative: Δ variance unexplained)
│   ├── collect_activations       dataset collection (100 tokens, 5 prompts)
│   └── moe_routing_analyzer      ✓ MoE routing compiler (Qwen3.6 → execution_plan.json)
└── docs/
    ├── FINAL_SYNTHESIS.md        this document
    ├── FINDINGS_v0.2.md          detailed experimental findings
    ├── TRAJECTORY_THEORY.md      theoretical framework
    └── DESIGN.md                 architecture overview
```

## MoE Routing Analysis (Qwen3.6-35B-A3B, 2026-05-16)

### Verified

| Metric | Value | Implication |
|--------|-------|-------------|
| Routing entropy | 5.545 = log(256) | Near-uniform across all 40 layers |
| Occupancy skew | 3-5x | Some experts used more, but distribution is flat |
| Transition P(same_expert) | 0.001 | Experts change every layer — near random |
| Bridge layers detected | 0 | No structural routing phase transitions |
| Hot tier viability | ✓ | Top-8 by occupancy → 3-5x hit rate advantage |
| Prefetch viability | ✗ | Transition-based prediction impossible (0.1% = random) |

### Load-balanced router constraint

Qwen3.6 uses a load-balanced router trained for maximum entropy. This is an explicit
design choice, not a bug. The router is optimized to use ALL experts equally, which:
- Eliminates periodic routing structure (LKO v8: oscillation prediction = 12.5% = random)
- Makes transition-based speculative prefetch non-viable
- Makes static frequency-based tiering the only viable optimization

### What objeta can compile for MoE

```
execution_plan.json:
  hot_experts[layer]   = top-8 by occupancy → always in RAM (144MB for 24 experts × 40 layers × q4)
  warm_experts[layer]  = next-16 → mmap cached
  cold_experts[layer]  = rest → SSD lazy load
  occupancy_skew       = confidence metric for tier assignment
```

### What objeta cannot compile for MoE

- Dynamic prefetch schedules (transition entropy too high)
- Bridge layer policies (no structural routing transitions)
- Expert co-activation clusters (no natural grouping in load-balanced routing)

## Key design rules (battle-tested)

1. **DIVERGENT phase is non-negotiable**: λ>0 → full precision mandatory
2. **Expert sparsity only exists in trained MoE** (Qwen3.6), not in dense models
3. **Phase boundaries are structural**, not detected per-token — bake them statically
4. **Refresh points are known**: L3 (Type I, semantic sharpening), L8 (Type II, structural coordination)
5. **Observable compression is real but insufficient**: you can measure low-dimensional structure without being able to exploit it for compute reduction
6. **Load-balanced MoE routers have near-zero predictability**: static tiering works; dynamic prefetch does not
7. **objeta's value is static analysis, not dynamic optimization**: phase_profile.json + execution_plan.json are the output artifacts
