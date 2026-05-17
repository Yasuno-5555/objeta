# objeta Implementation Plan v2 — 2026-05-17

## Phase Complete: FFN/Expert Compiler

The FFN side is structurally solved:
- MoE dispatch: Rust SIMD + Metal GPU (4.3ms/layer for 8 experts)
- Expert tiering: hot/warm/cold via static occupancy analysis
- weight loading: mmap, zero-copy, 850MB RSS for 35B model
- GEMV: NEON+rayon, 23 GFLOPS, 1.9x NumPy
- SSD streaming: layout designed, ready for integration

## Phase Current: Attention Transport Engine

The bottleneck has shifted decisively to attention.

### Why attention is different

| | FFN | Attention |
|---|---|---|
| Nature | Static steering | Dynamic modulation |
| Compressibility | Expert sparse (MoE) | Not compressible |
| Optimize at | Compile time | Runtime |
| Theory | Phase structure, inversion zones | Position gradient, trajectory stability |
| objeta role | Static Compiler | Dynamic Runtime |

### Architecture (final form)

```
safetensors
    │
    ▼
┌──────────────────────┐
│ Static Compiler      │
│ (objeta analyze)     │
│ ├── phase profile    │
│ ├── expert tiering   │
│ ├── SSD layout       │
│ └── bridge detection │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Dynamic Runtime      │
│ (attention engine)   │
│ ├── fused GQA kernel │
│ ├── KV cache manager │
│ ├── speculative dec  │
│ └── OpenAI API       │
└──────────────────────┘
```

FFN → compile-time. Attention → runtime-time. Complete separation.

## Priority Tasks

### Tier 1: Attention Transport

1. **Fused GQA Metal kernel**
   - QKV projection + RoPE + online softmax + value mix + output projection
   - Single dispatch, zero intermediate buffers
   - Expected: 3-5x attention speedup

2. **KV cache layout optimization**
   - Reorder from [layer][head][token][dim] to [token][kv_group][dim]
   - GQA locality: 2 KV heads vs 16 Q heads → KV cache is the access pattern bottleneck
   - Expected: 1.5-2x memory bandwidth improvement

3. **Paged KV cache**
   - 4KB pages, vLLM-style
   - Natural fit with SSD streaming architecture
   - Enables long-context without memory explosion

### Tier 2: Phase-Aware Speculation

4. **Speculative decoding using phase profiles**
   - Low-entropy zones (detected by objeta analyze) → aggressive speculation
   - High-entropy zones → conservative
   - Phase boundaries → verification points
   - Expected: 1.5-3x throughput in low-entropy regions

### Tier 3: Ecosystem

5. **OpenAI API compatibility**
   - `POST /v1/chat/completions`
   - Drop-in replacement for any OpenAI client
   - Enables standard benchmarking and UI integration

## Performance Roadmap

| Stage | tok/s | Key enabler |
|-------|-------|-------------|
| Current | 0.03 | Baseline (Python+MLX) |
| Rust executor | 0.05 | All ops in Rust |
| + fused attention | 0.15-0.25 | Single-dispatch GQA kernel |
| + KV layout | 0.2-0.5 | Cache-optimized memory access |
| + speculative | 0.3-1.5 | Phase-aware multi-token prediction |
| Target | 1-3 | M1 8GB, Qwen3.6-35B-A3B |

## Design Rules

1. **Attention = transport, not compute.** Implement as fused data movement, not sequential matmuls.
2. **Apple GPU is latency-bound, not FLOP-bound.** Kernel fusion beats algorithmic optimization.
3. **KV locality > Q locality.** GQA has 8:1 Q:KV ratio. Optimize for KV access pattern.
4. **Phase profiles feed the runtime.** Static analysis output drives dynamic decisions.

---

## v3 Breaking the Walls (2026-05-17)

The 8GB M1 is structurally constrained by three walls:
1. **MoE I/O Wall**: 5GB q4 data, random-access, SSD-bound — 500ms/token
2. **DeltaNet Recurrence Wall**: 524K f32 scalar triple-loop state update — 480ms/token
3. **Attention Scaling Wall**: O(seq_len) scalar attention loop, Metal kernel exists but underutilized

Current measured steady-state: **~1.2s/token (0.8 tok/s)**. Physical limit without structural changes: **~1.3 tok/s**.

The following 9 breakthrough approaches target these walls directly.

### 1. Metal 3 MTLIO — Bypass OS Page Cache (MoE I/O Wall)

**Problem**: OS page cache (kernel space) → CPU RAM swap-in → GPU reads. This indirection is the bottleneck, not raw SSD bandwidth.

**Solution**: Apple Metal 3 `MTLIOCommandBuffer` (Fast Resource Loading API). Apple's equivalent of DirectStorage. Bypasses OS virtual memory paging entirely. DMA transfer from SSD → Unified Memory (GPU-accessible) asynchronously, without CPU blocking.

**Expected**: Eliminates SWAP thrashing. Approaches raw hardware SSD bandwidth (~2.5-3 GB/s). MoE 500ms → ~250ms.

**Risk**: MTLIO API is iOS/macOS 15+. Requires Metal shader pipeline integration for expert GEMV dispatch. Complex implementation.

### 2. Early Routing — Async Prefetch via Router Prediction (MoE I/O Wall)

**Problem**: Router is computed at the MoE layer itself. Expert selection + SSD read are synchronous.

**Solution**: Compute router (lightweight W_router @ hidden_state) several layers ahead. In the ISOMETRIC zone (Phase III of LKO theory), hidden state cosine between adjacent layers is high — the previous layer's hidden state is a good predictor of the next MoE layer's routing.

```
Layer L (current): compute router for layer L+4
                   → initiate async prefetch (madvise or MTLIO)
Layer L+4 (MoE):   expert weights already in RAM → zero-wait
```

**Expected**: Hides SSD latency behind compute. Effective MoE time → ~100ms (remaining SSD reads fully pipelined).

**Risk**: Prediction accuracy degrades at phase boundaries (UNFOLD, DIVERGENT). Needs confidence-gated fallback.

### 3. Dynamic Cumulative Pruning (Top-P Expert) — Reduce Total I/O (MoE Capacity Wall)

**Problem**: Top-8 with uniform routing (entropy=log 256) means every token reads 15.6MB/layer from SSD regardless of token difficulty.

**Solution**: After softmax, sort expert weights descending. Truncate at cumulative probability threshold (e.g., 0.85). Easy tokens ("the", "is") use 2-3 experts. Hard tokens use full 8.

```
cumsum(sorted_weights) > 0.85 → stop
min 2 experts, max 8
renormalize remaining weights
```

**Expected**: Average expert count drops from 8 → 3-4. Total SSD I/O reduces by 50-60%. MoE 500ms → 250ms (compounds with MTLIO/Early Routing).

**Risk**: Quality impact on hard tokens (threshold too aggressive). Mitigation: adaptive threshold based on router entropy.

### 4. DeltaNet State Quantization (q8/q4) — Memory Bandwidth (DeltaNet Wall)

**Problem**: delta_state_update reads/writes 524K f32 (2MB) per layer per token. The state matrix S (32×128×128 f32) is the bottleneck — every element is read and written on every token.

**Solution**: Store DeltaNet recurrent state S in q8 or q4 format. Decode on-the-fly during state update, re-quantize before storing.

```
S_q4 (load, 512KB)
  → decode to f32 (on-the-fly)
  → compute S_new (f32)
  → quantize back to q4
  → store S_q4 (512KB)
```

**Expected**: Memory bandwidth for state I/O drops from 2MB → 0.5MB (q8) or 0.25MB (q4). DeltaNet state update 480ms → ~200ms.

**Risk**: Quantization error accumulation over recurrent steps. Needs error feedback (delta coding) to prevent drift.

### 5. Expert State Persistence — MoE as Recurrent Cache Module (MoE Architecture)

**Problem**: Every expert is a stateless MLP: E_i(h_t). Every token, full GEMV from scratch.

**Solution**: Give each expert a persistent state s_i. Expert becomes E_i(h_t, s_i) where s_i captures the expert's recent activation patterns.

```
s_i ← update(s_i, h_t)          // cheap recurrent step
output = if ||h_t - h_{t-1}|| < ε:
    cached_E_i(s_i)              // most tokens: cheap lookup
else:
    W_i @ h_t                    // few tokens: full GEMV
```

**Expected**: For tokens with small steering (||Δ|| ≪ 1, which is most tokens in ISOMETRIC phase), expert computation reduces to recurrent state update + cached output lookup. MoE effective compute → near-zero for 80%+ tokens.

**Risk**: State dimension trade-off. 512-dim state per expert × 256 experts × 10 layers = 5MB — acceptable.

### 6. Temporal Expert Locality Cache — Markov Routing Prefetch (MoE I/O Wall)

**Problem**: LRU cache doesn't work for bursty router patterns. Expert 18 → expert 29 → expert 18 has temporal structure that LRU misses.

**Solution**: Learn transition probabilities P(E_j | E_i) from warmup. When expert 18 is selected, prefetch expert 29 (most likely successor). Window-pin recently used experts (last N tokens) — delay eviction even if LRU says evict.

```
expert_window: [(layer, eid, last_used_token)]
markov_table: P(E_j | E_i)  // learned from warmup
on router result:
    pin expert_window experts  // bursty locality
    prefetch markov_table[selected_expert]  // predictive
```

**Expected**: Cache hit rate improvement beyond simple LRU. Effective MoE I/O reduction compound with Top-P pruning.

**Risk**: Markov table needs periodic refresh. Cold start requires warmup tokens.

### 7. DeltaNet ODE Approximation — Latent State Compression (DeltaNet Wall)

**Problem**: 128×128 recurrence per head × 32 heads. But steering subspace is only 19D.

**Solution**: Compress DeltaNet state via fixed basis projection. S ≈ U @ z where U is a fixed low-rank basis and z is the latent state (19-dim instead of 128×128).

The insight: trajectory manifold is low-dimensional (19D steering subspace). The state CAN be compressed along the transport direction even though individual points cannot.

```
z = U^T @ flatten(S)             // 128×128 → 19
z_new = f(z, h_t)                // cheap 19-dim recurrence
S_new = unflatten(U @ z_new)     // 19 → 128×128
```

**Expected**: DeltaNet recurrence from 524K ops → ~10K ops. State update near-zero cost.

**Risk**: Basis U must be computed offline from trajectory data. Approximation error needs monitoring. This is research-grade.

### 8. Steering-Triggered Attention — Sparse GQA (Attention Wall)

**Problem**: GQA runs on all 10 layers, all tokens. But LKO theory shows DeltaNet layers produce ||Δ|| ≈ 0 (fine steering), while GQA layers produce ||Δ|| = 1.5-3.2 (course correction).

**Solution**: Invert the architecture. DeltaNet becomes the backbone (always runs). GQA becomes sparse correction — only runs when steering magnitude exceeds threshold.

```
if ||Δ_pred|| > threshold:
    run GQA   // course correction, ~10-20% of tokens
else:
    skip GQA  // DeltaNet-only, ~80-90% of tokens
```

**Expected**: GQA compute drops from 10 layers × every token → 1-2 layers × 20% tokens. GQA 240ms → ~30ms.

**Risk**: Skipping GQA too aggressively causes trajectory drift. Needs Lyapunov-based stability monitor. This is the most dangerous but highest-impact optimization.

### 9. Speculative Expert Execution — GPU Occupancy (MoE Compute)

**Problem**: Router → select → load → execute is sequential. GPU idle during SSD reads.

**Solution**: Launch all top-16 (or top-32) experts in parallel on GPU. Gate-weight the results. GPU occupancy first, precision second.

```
top-16 experts → parallel Metal GEMV launch → gate-weight sum
SSD pre-load for top-16 in background (MTLIO)
```

**Expected**: When combined with MTLIO, MoE layer becomes GPU-bound rather than I/O-bound. Effective throughput approaches GPU theoretical.

**Risk**: Top-16 consumes more GPU memory and bandwidth. Needs Metal kernel tuning.

### Priority & Expected Compound Effect

| Order | Approach | Target | Expected Saving | Risk |
|-------|----------|--------|----------------|------|
| 1 | Top-P Expert Pruning (#3) | MoE I/O | -50% data volume | Low |
| 2 | Early Routing + Prefetch (#2) | MoE I/O | Hides latency | Medium |
| 3 | MTLIO Direct SSD (#1) | MoE I/O | -50% access time | High (complex) |
| 4 | DeltaNet State q8 (#4) | DeltaNet BW | -75% state BW | Low |
| 5 | Temporal Expert Cache (#6) | MoE I/O | +hit rate | Low |
| 6 | Metal Fused GQA | GQA | -80% GQA time | Low |
| 7 | Steering-Triggered Attn (#8) | GQA | -90% GQA invoke | High |
| 8 | DeltaNet ODE (#7) | DeltaNet | -95% recurrence | High (research) |
| 9 | Expert State Persistence (#5) | MoE | -80% compute | Medium |
| 10 | Speculative Expert (#9) | MoE | GPU occupancy | Medium |

### Physical Limit After All Breakthroughs

| Component | Current | After | Mechanism |
|-----------|---------|-------|-----------|
| MoE | 500ms | ~80ms | Top-P + Early Routing + MTLIO + State Persistence |
| DeltaNet | 480ms | ~60ms | State q8 + ODE approximation |
| GQA | 240ms | ~20ms | Metal fused + Steering-triggered |
| Shared | 55ms | ~10ms | q8 quantize |
| **Total** | **1275ms** | **~170ms** | **~6 tok/s** |

> **The 8GB wall is real but not absolute.** The combination of MTLIO (bypassing kernel paging) + Top-P pruning (reducing data volume) + Early Routing (hiding latency) + State quantization (reducing bandwidth) is a multi-axis attack on the memory bottleneck. Each axis alone is insufficient; together they change the physics.
5. **FFN is a compiler problem. Attention is a runtime problem.** Don't mix the two.
