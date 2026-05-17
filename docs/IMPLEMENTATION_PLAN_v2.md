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
5. **FFN is a compiler problem. Attention is a runtime problem.** Don't mix the two.
