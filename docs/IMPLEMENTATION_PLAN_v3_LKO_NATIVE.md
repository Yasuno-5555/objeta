# objeta — LKO-Native Runtime Architecture v3

2026-05-17

## Foundational Insight

LKO proved: **LLM is not a homogeneous stack. It is a phase-structured nonlinear dynamical system.**

Therefore the runtime must be not a **static graph executor** but a **trajectory controller**.

```
Wrong:  Layer = universal operator, same compute policy ∀ token, ∀ layer
Right:  Layer = phase-specific role (basin compiler / transport / steering)
```

---

## 1. Layer-as-Controller

**Insight**: LKO's 4-phase structure means different layers have fundamentally different roles.

| LKO Phase | Layers | Role | Precision | Execution |
|-----------|--------|------|-----------|-----------|
| UNFOLD | L0-L2 | Basin compiler | fp16 mandatory | Full |
| ISOMETRIC | L3-L13 | Transport | q4-q5 | Cacheable, skippable |
| DIVERGENT | L14+ | Steering | q8+ | Triggered correction |

**Runtime Architecture**:

```rust
enum LayerRole {
    Controller,   // high precision, full attention, trajectory-determining
    Transport,    // cheap, low precision, cacheable, skippable
    Correction,   // sparse activation, trigger-based execution
}

struct LKOLayer {
    role: LayerRole,
    phase: Phase,
    steering_basis: Option<Matrix>,  // for tangent-space execution
    basin_signature: Option<Vec<f32>>, // for basin-stable memory
}
```

**Decision logic**: Assign `LayerRole` at compile time via `objeta analyze`. Layer role determines precision, attention policy, cache strategy, and skip eligibility.

---

## 2. Continuous Depth Execution

**Insight**: Δ is small across long intervals (ISOMETRIC transport). The full 40-layer path is unnecessary for most tokens.

**LKO data**: ||Δ_l|| = 2-12% of ||h_l||. Most layers contribute micro-corrections. Only GQA layers (every 4th) make significant course corrections.

**Architecture**:

```rust
fn forward_continuous(token, depth_budget) -> HiddenState {
    let h = embed(token);
    let (phase, drift_pred) = predict_trajectory(h);

    match drift_pred {
        Drift::Small(_)  => skip_n_layers(4),       // 4-layer jump
        Drift::Medium(_) => run_compressed_transport(), // reduced precision
        Drift::Large(_)  => run_full_correction(),   // full GQA + MoE
    }
}
```

**This is Neural ODE for LLM**: adaptive step size based on trajectory curvature. The CFL condition from LKO theory gives the stability bound.

**Expected**: 60-80% of tokens skip 50%+ of layers. Token throughput: 2-4x.

---

## 3. Tangent-Space Execution

**Insight**: LKO proved trajectory clustering ❌ but low-dimensional steering ✅. The state *itself* cannot be compressed, but the *direction of change* can.

**The inversion**:

```
Old:  compute h_{l+1} = h_l + W @ h_l         (full matmul)
New:  compute δh = U @ α,   h_{l+1} = h_l + δh  (steering projection)
```

Where:
- `U` = steering basis (19D for Qwen3.6, 8D for TinyLlama — measured)
- `α` = tiny coefficient vector (19 floats instead of 2048×2048 matmul)

**Architecture**:

```rust
// Offline: objeta analyze extracts steering basis U from trajectory data
let U = svd(trajectory_deltas).top_k(19);  // per-model steering basis

// Runtime: replace large matmuls with steering projection
fn tangent_forward(h: &[f32], U: &[f32], alpha: &[f32]) -> Vec<f32> {
    let dh = project(U, alpha);  // 2048 × 19 @ 19 = 2048 ops (not 2048×2048!)
    h + dh
}
```

**This transforms Transformer from a state machine into a flow field simulator.**

**Expected**: FFN computation: 2048×2048 → 2048×19. ~100x FLOP reduction for ISOMETRIC layers.

**Risk**: U basis drifts at phase boundaries. Needs periodic recalibration at refresh points (L3, L8, L11).

---

## 4. Basin-Stable Memory

**Insight**: LKO's residual ablation showed sharp phase transition (L7 → complete attractor collapse). Generation is **basin hopping**, not continuous generation.

**The inversion**:

```
Old:  KV cache = exact token history (every token's K,V stored)
New:  KV cache = basin-stable summary (which attractor, distance to boundary)
```

**Architecture**:

```rust
struct BasinStableKV {
    current_basin: BasinId,           // which attractor we're in
    boundary_distance: f32,           // how close to basin edge
    basin_history: RingBuffer<BasinTransition>,  // last N basin hops
    compressed_kv: Option<CompressedKV>, // only for boundary-proximal tokens
}
```

**Decision logic**:
- Stable basin interior → discard KV, extrapolate from basin signature
- Basin boundary → retain KV, run full attention
- Basin transition → trigger full correction

**Expected**: KV cache memory: O(seq_len) → O(n_basins). Long context without memory explosion.

---

## 5. Entropy-Conditioned Precision

**Insight**: LKO already measures entropy, phase, steering norm, and trajectory drift *cheaply* (O(dim), <0.1% overhead).

**The inversion**:

```
Old:  static precision schedule (per-layer q3/q4/q5)
New:  runtime precision field: bit = f(entropy, phase, steering_norm, drift)
```

**Architecture**:

```rust
fn compute_precision(token_entropy, layer_phase, steering_norm, drift_rate) -> u8 {
    if layer_phase == Phase::UNFOLD { return 16; }    // sacred
    if steering_norm > threshold { return 8; }         // correction needed
    if drift_rate > critical { return 5; }              // approaching boundary
    if token_entropy < 0.3 { return 3; }               // easy token
    return 4;                                           // baseline transport
}
```

**This is not static quantization — it's a runtime precision field.** GPU kernels receive per-token precision parameters.

**Expected**: Average precision drops below q4 without quality loss. Memory bandwidth proportional reduction.

---

## 6. Topology-Preserving Compute

**Insight**: LKO showed `cos similarity is a lie` — topology collapse (neighborhood graph destruction) is the real degeneration mechanism, not vector distance.

**The inversion**:

```
Old:  min ||h - ĥ||  (pointwise fidelity)
New:  preserve adjacency, connectivity, manifold continuity  (topological fidelity)
```

**Architecture**:

```rust
fn topology_loss(h_true, h_approx, neighbors) -> f32 {
    // Not: ||h_true - h_approx||
    // But: do the same K nearest neighbors exist in both spaces?
    let true_neighbors = knn(h_true, neighbors);
    let approx_neighbors = knn(h_approx, neighbors);
    overlap_score(true_neighbors, approx_neighbors)
}
```

**Runtime use**: Monitor topology preservation online. If neighborhood graph diverges → trigger refresh.

---

## 7. Event-Driven Transformer

**Insight**: Large steering changes are SPARSE. Most tokens ride the ISOMETRIC flow. Only occasional tokens trigger major corrections.

**The inversion**:

```
Old:  clock-driven — every token runs full computation
New:  event-driven — computation triggered by steering events
```

**Event types**:

| Event | Trigger | Response |
|-------|---------|----------|
| `SteeringJump` | \|\|Δ\|\| > threshold | Run expensive attention |
| `BasinTransition` | entropy spike, norm change | Refresh memory, KV, cache |
| `BoundaryApproach` | Lyapunov λ → positive | Increase precision |
| `StableFlow` | default | Extrapolate trajectory, skip |

**This is essentially an interrupt-based LLM.** Like a CPU, most cycles are idle; only events trigger expensive operations.

**Expected**: 80-90% of tokens use the `StableFlow` cheap path. 3-5x throughput.

---

## 8. Trajectory Extrapolation Decoding

**Insight**: ISOMETRIC transport is smooth and predictable. The tangent flow `v_t = h_t - h_{t-1}` is persistent.

**Architecture**:

```rust
fn extrapolate_next_token(h_current, v_current, steps) -> HiddenState {
    // Linear extrapolation in tangent space
    let mut h = h_current;
    for _ in 0..steps {
        h = h + v_current;  // cheap: 2048-element vector add
    }
    h
}

fn speculative_generate():
    loop {
        let h_fast = extrapolate(h, v, 4);  // predict 4 tokens ahead
        let (tokens, h_real) = full_forward_verify(h);
        if diverged(h_fast, h_real):  // rare: ~10% of the time
            rollback, run full correction
    }
```

**This is speculative hidden-state decoding** — much cheaper than speculative token decoding because hidden state extrapolation is O(dim), not O(vocab).

**Expected**: 4-8x effective throughput in ISOMETRIC regions.

---

## 9. Layer Collapse — Macro Transport Operator

**Insight**: If Δ_l vectors are near-parallel across consecutive layers (LKO: ISOMETRIC phase, cos(Δ_l, Δ_{l+1}) > 0), multiple layers can be collapsed into a single macro-step.

**Architecture**:

```rust
// Offline: group layers with cos(Δ_l, Δ_{l+1}) > 0.95
let macro_groups = group_parallel_deltas(trajectory_data);
// Example: L3-L8 form one macro transport operator, L9-L13 another

// Runtime: execute macro operators instead of individual layers
fn macro_transport(h, operator) -> HiddenState {
    // Single integrated step replaces 6 individual layers
    operator.apply(h)
}
```

**Expected**: 40 layers → effective 10-15 macro-steps. 3-4x layer count reduction.

---

## 10. Runtime as Dynamical Scheduler

**Final form**: The runtime is no longer a layer executor. It is a **trajectory manager**.

**Scheduler responsibilities**:

```
┌─────────────────────────────────────────────┐
│           Trajectory Scheduler               │
│                                              │
│  ┌─ Phase Detection ──────────────────────┐ │
│  │ entropy, cos, λ, ||Δ||, basin_id       │ │
│  └────────────────────────────────────────┘ │
│                    ↓                         │
│  ┌─ Policy Decision ──────────────────────┐ │
│  │ • precision: q3 / q5 / fp16             │ │
│  │ • attention: full / skip / refresh       │ │
│  │ • expert: load / cache / prefetch        │ │
│  │ • KV: retain / compress / discard        │ │
│  │ • transport: extrapolate / step / full   │ │
│  │ • steering: tangent / correction / base  │ │
│  └────────────────────────────────────────┘ │
│                    ↓                         │
│  ┌─ Execution ───────────────────────────┐  │
│  │ layer fusion, precision cast, dispatch │  │
│  └────────────────────────────────────────┘ │
│                    ↓                         │
│  ┌─ Monitoring ──────────────────────────┐  │
│  │ topology drift, basin stability, λ est │  │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**The scheduler decides EVERYTHING**:
- Precision (per token, per layer)
- Attention refresh (per token, per layer)
- Expert loading (per token, per layer)
- KV retention (per token)
- Transport extrapolation (per token)
- Steering correction (per token)

---

## Priority Roadmap

| Priority | Name | LKO Basis | Risk | Impact |
|----------|------|-----------|------|--------|
| **P0** | Entropy-Conditioned Precision (#5) | entropy/phase measurable | Low | Medium |
| **P0** | Continuous Depth Execution (#2) | Δ small in ISOMETRIC | Low | High |
| **P1** | Layer Collapse (#9) | cos(Δ_l, Δ_{l+1}) > 0 | Medium | High |
| **P1** | Event-Driven Transformer (#7) | steering jumps sparse | Medium | High |
| **P2** | Tangent-Space Execution (#3) | steering subspace 19D | High | Very High |
| **P2** | Basin-Stable Memory (#4) | basin bifurcation | High | High |
| **P3** | Trajectory Extrapolation (#8) | smooth tangent flow | High | Very High |
| **P3** | Topology-Preserving Compute (#6) | topology collapse | High | Medium |
| **P4** | Layer-as-Controller (#1) | phase-specific roles | Medium | High |
| **P4** | Dynamical Scheduler (#10) | unifies all | High | Very High |

---

## Core Principle

> **Uniform quantization, uniform execution, uniform caching, uniform scheduling — all of these are wrong because LLM is not a homogeneous stack.**
>
> The runtime must be heterogeneous because the dynamics are heterogeneous.
>
> This is what LKO actually proved.

---

# Part II: LKO-Native OS Architecture

2026-05-17

## Why LKO Demands an Operating System

LKO's measurements imply the LLM is not a static compute graph. It is a **stateful execution environment**.

| LKO Observation | OS Interpretation |
|----------------|------------------|
| Phase structure (SYNC→UNFOLD→ISOMETRIC→DIVERGENT) | Execution mode |
| Basin transition (residual ablation: L7 attractor collapse) | Process state transition |
| Steering layer (GQA, \|\|Δ\|\| = 10-12%) | Interrupt point |
| Precision cliff (q3→q3.25: 13.2x PPL jump in 0.25 bit) | Bandwidth saturation |
| Trajectory continuity (cos(h, h_next) ≈ 0.97 in ISOMETRIC) | Process continuity |
| Sparse correction (GQA every 4th layer) | Event-driven compute |
| Family difference (Residual Transport vs Spherical Steering) | ISA difference |

**Conclusion**: LLM = stateful execution environment. The runtime must become an OS.

---

## 1. LLM Scheduler (Kernel)

**Current**:
```python
for layer in layers:
    execute(layer)
```

**LKO OS**:
```rust
scheduler.step(token_state)  // scheduler decides everything
```

The scheduler owns the decision for every token, every layer:
- Which layer to execute (or skip, or collapse, or extrapolate)
- What precision to use (q3 → fp16, dynamic)
- Which experts to load (prefetch, cache, evict)
- Whether to fire attention (interrupt-driven)
- Whether to extrapolate trajectory

**Scheduler state**:
```rust
struct RuntimeState {
    trajectory_phase: Phase,       // SYNC/UNFOLD/ISOMETRIC/DIVERGENT
    steering_energy: f32,          // ||Δ||, trigger for correction
    entropy: f32,                  // router entropy → precision
    active_experts: Vec<usize>,    // currently loaded expert set
    kv_pressure: f32,              // KV cache memory stress
    basin_id: Option<BasinId>,     // current attractor
    lyapunov_estimate: f32,        // stability monitor
}
```

---

## 2. Virtual Memory System (KV Cache)

**Current**: KV cache = exact token history. O(seq_len) memory.

**LKO OS**: KV cache = virtual memory. Only the working set is in RAM.

```
hot KV (RAM):     current basin — full precision, immediate access
warm KV (mmap):   neighboring basins — compressed, lazy load
cold KV (SSD):    distant basins — heavily compressed summary
```

This is **paging** — exactly like a CPU OS. The page table maps basin_id → storage tier.

**Mechanism**: Basin transition → TLB flush → page fault → load new basin's KV from warm/cold storage.

---

## 3. Interrupt-Driven Attention

**Current**: Polling — every token runs attention.

**LKO OS**: Interrupt-driven.

```rust
if trajectory_drift > threshold {
    interrupt();        // kernel trap
    run_attention();    // service the interrupt
}
```

Attention is not "always-on compute." It is an **interrupt service routine** triggered by steering events:
- `SteeringJump` interrupt → full GQA
- `BasinBoundary` interrupt → refresh attention + KV
- `EntropySpike` interrupt → precision escalation

Between interrupts: cheap transport (identity, extrapolation, low-precision).

---

## 4. Expert Paging (MoE Virtual Memory)

**Current**: Expert miss → SSD read → CPU stall. Primitive.

**LKO OS**: Expert daemon (background thread).

```rust
// Daemon loop (runs continuously, background)
loop {
    predict_next_experts();  // Markov routing: P(E_j | E_i)
    prefetch(ssd_fd, offset, len);  // async SSD → RAM
    evict_lru();             // working set management
    compress_cold();         // cold experts → tighter encoding
}
```

The routing graph `P(E_j | E_i)` drives **speculative paging** — exactly like a CPU's prefetch engine using branch prediction.

---

## 5. Precision Governor (DVFS for LLM)

Linux CPU governor → LKO precision governor.

```rust
fn precision_governor(state: &RuntimeState) -> u8 {
    if state.entropy < 0.3       { return 3; }  // q3: idle
    if state.steering_energy > 0.1 { return 6; }  // q6: active compute
    if state.lyapunov_estimate > 0.5 { return 16; } // fp16: instability
    return 4;  // q4: baseline
}
```

**Dynamic Precision Scaling** — same concept as DVFS (Dynamic Voltage/Frequency Scaling), applied to numerical precision instead of power states.

---

## 6. Process Classes (Token Priority)

Tokens have different priorities, like Unix nice levels.

| Token Type | Priority | Precision | Attention | Example |
|-----------|----------|-----------|-----------|---------|
| System prompt | `nice -20` (critical) | fp16 | Always | `<|system|>` |
| Reasoning | `nice -10` | q8 | Steering-triggered | Math, code |
| Content | `nice 0` | q5 | Phase-dependent | Narrative |
| Filler | `nice +10` | q3 | Skip | "the", "is", "a" |
| Structural | `nice -5` | q6 | Phase-dependent | Punctuation, formatting |

Scheduler policy varies by token class — just like a real OS scheduler.

---

## 7. Context Switching (Basin Transition)

LLM topic shifts are **context switches**.

```
save_trajectory_state():    // store current basin signature
    steering_basis
    active_experts
    kv_working_set
    entropy_profile

load_new_basin():           // restore target basin
    compressed_kv → decompress
    expert_set → prefetch
    precision_profile → apply
```

A topic shift is a basin jump. The scheduler saves the old basin state and loads the new one — exactly like a CPU context switch saves/restores registers.

---

## 8. User-Space vs Kernel-Space

High-cost operations run in "kernel mode." Everything else in "user mode."

```
Kernel-space (high precision, high cost):
    - Steering computation (attention, MoE)
    - Basin transition detection
    - Precision governor decisions
    - Expert paging (I/O)

User-space (low precision, low cost):
    - Transport (identity/linear interpolation)
    - Extrapolation (cheap prediction)
    - Entropy monitoring (cheap, O(dim))
    - Cache lookup (hashmap, no I/O)
```

---

## 9. Runtime IPC (Expert Communication)

MoE experts are not isolated MLPs. They are **communicating services** on a shared bus.

```
math expert ←→ code expert ←→ memory expert
        ↑           ↑            ↑
        └───────────┴────────────┘
              shared residual bus
```

This is **microkernel architecture**: experts are user-space services communicating via the residual stream (the kernel's message bus).

---

## 10. LLM Hypervisor (Final Form)

The ultimate architecture: one model → multiple virtualized sub-models.

```
┌──────────────────────────────────────────┐
│              LLM Hypervisor              │
│                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Transport │ │ Steering │ │ Memory   │ │
│  │ Model    │ │ Model    │ │ Model    │ │
│  │ (cheap)  │ │ (precise)│ │ (paging) │ │
│  └──────────┘ └──────────┘ └──────────┘ │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │        Scheduler / Governor       │   │
│  │  (precision, attention, experts)  │   │
│  └──────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

Each sub-model is a virtualized instance with its own precision budget, memory allocation, and compute policy. The hypervisor schedules them, allocates resources, and handles faults.

**This is an LLM virtualization layer.**

---

## The Fundamental Shift

The research moved from:

> "How do we matmul faster?"

to:

> "What do we NOT need to compute right now?"

This is exactly the question an OS answers. An OS does not make individual operations faster. It decides **which operations to skip** given limited resources.

LKO proved that the LLM's internal structure supports this skipping — it has phases, basins, sparse corrections, and smooth transport. The physics says skipping is safe; the OS makes it systematic.

**objeta is becoming an LLM operating system.**
