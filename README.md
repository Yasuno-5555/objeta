# objeta — LLM Inference Operating System

**Qwen3.6-35B on M1 8GB — from impossible to 3.0 tok/s via runtime surgery.**

objeta is not a model compressor. It is an **operating system for LLM inference** that replaces static `for layer in layers` with state-dependent compute allocation: **observe → classify → allocate → execute**.

## Quick Start

```bash
# Start OpenAI-compatible server
python server.py --port 8000

# Start daemon with system monitoring
python daemon.py --port 8000 --duration 1800

# Run benchmark
python bench.py --quick

# Run all verification tests
python experiments/verify_determinism.py
```

## Architecture

13 OS modules. Rust kernel. 5-tier virtual memory. Cross-request residency.

```
observe → classify → allocate → execute
    ↑                        ↓
    └── TokenTrace ← replay ─┘
```

See `docs/STATUS_v1.0.md` for the complete architecture.

## Key Results

| Metric | Value |
|--------|-------|
| Qwen3.6-35B projection (M1 8GB, locality) | **3.0 tok/s** |
| MoE effective expert reduction (OLMoE) | **58 → 9.5 (83.6%)** |
| Cache hit improvement (locality + residency) | **10% → 68%** |
| Scheduler overhead | **12.7 µs (0.02%)** |
| Replay determinism | **100%** |
| Fault recovery | **100%** |

## Verifications

```bash
# Scheduler determinism + fault + latency
python experiments/verify_determinism.py

# Real Qwen3.6 I/O measurement
python experiments/measure_qwen36_io.py

# Quality frontier (λ sweep)
python experiments/quality_frontier.py --model stories-moe --quick

# Stability phase diagram
python experiments/stability_phase.py

# Expert locality visualization
python experiments/expert_locality.py
```

## Docs

- `docs/STATUS_v1.0.md` — **complete architecture and results**
- `docs/TRAJECTORY_THEORY.md` — why compressing the operator fails
- `docs/LKO_UNIFIED_THEORY_FINAL.md` — Transformer as trajectory stabilization
- `docs/FINAL_SYNTHESIS.md` — what died, what survived

## Crate Map

| Crate | Purpose |
|-------|---------|
| `objeta-os` | Rust kernel (ExecutionPlan, Scheduler, FaultInjection) |
| `objeta-core` | Shared types (Phase, Family, LayerZone) |
| `objeta-analysis` | Static geometry analysis (SVD, intra_cos, Lyapunov) |
| `objeta-phase` | Phase/family classification |
| `objeta-routing` | Per-layer precision assignment |
| `objeta-quantize` | Phase-adaptive quantization |
| `objeta-parser` | Safetensors mmap loader |
| `objeta-cli` | CLI (analyze, strategy, quantize) |

## Core Principle

> LLM inference is not static computation. It is adaptive dynamical resource allocation.
>
> runtime = OS. inference = control problem. quantization = transport capacity allocation.
