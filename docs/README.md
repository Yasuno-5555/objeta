# objeta Documentation Index

## Primary (current)

| Document | Description |
|----------|-------------|
| **[CURRENT_STATUS_2026_05_21.md](CURRENT_STATUS_2026_05_21.md)** | ★ Latest — packed Qwen3.6 layout parsing fixed, real calibration coverage up to 66.22%, target-aware specialization results |
| **[CURRENT_STATUS_2026_05_20.md](CURRENT_STATUS_2026_05_20.md)** | Unified MoE pipeline, runtime pack loader, importance-aware eviction, byte telemetry, oracle PASS |
| **[ARCHITECTURE_BOUNDARIES.md](ARCHITECTURE_BOUNDARIES.md)** | ★ Crate/module boundaries, FFI table, execution flow, hard prohibitions |
| **[METRICS_SCHEMA.md](METRICS_SCHEMA.md)** | ★ Full schema for `summary.json`, `moe_stats.json`, `metrics.jsonl` |
| **[AOT_RUNTIME_PACK.md](AOT_RUNTIME_PACK.md)** | ★ AOT runtime metadata compiler and runtime pack layout (`objeta-aot`) |

## Operational

| Document | Description |
|----------|-------------|
| `STATUS_v1.0.md` | v2.0 OS runtime reference — architecture, 5-tier memory, KPIs, verified results |
| `EXECUTOR_BOTTLENECKS_2026_05_19.md` | Detailed bottleneck analysis — MoE vs non-MoE split, pruning experiments, warm-hit issue |

## Theory (foundational)

| Document | Description |
|----------|-------------|
| `TRAJECTORY_THEORY.md` | Why compressing the operator fails — Lyapunov phase map, thin transport tube |
| `LKO_UNIFIED_THEORY_FINAL.md` | Transformer as trajectory stabilization machine — 3 mechanisms, reflexive runtime |
| `FINAL_SYNTHESIS.md` | What died (FFN rotation, latent MoE, archetypes) vs what survived (stability orchestration) |

## Historical (superseded)

| Document | Date | Content |
|----------|------|---------|
| `CURRENT_STATUS_2026_05_18.md` | 2026-05-20 | Fused MoE v0 status; 1-token E2E regression under investigation |
| `FINDINGS_2026_05_17.md` | 2026-05-17 | Koopman prediction, Delta PCA, TinyLlama OS early results |
| `FINDINGS_v0.2.md` | 2026-05-16 | 3 failed approaches, trajectory compiler direction |
| `IMPLEMENTATION_PLAN_v2.md` | 2026-05-17 | Attention transport engine, DeltaNet/Expert walls, 9 breakthroughs |
| `IMPLEMENTATION_PLAN_v3_LKO_NATIVE.md` | 2026-05-17 | Layer-as-Controller, continuous depth, tangent-space execution, OS architecture vision |
| `DESIGN.md` | 2026-05-17 | Pipeline, memory layout, Qwen3.6 executor design |
| `QUANTIZATION_EXPERIMENT_PLAN.md` | 2026-05-17 | Family-aware quantization, precision cliff, transport asymmetry |
| `OS_RUNTIME_RUST_PORT.md` | — | Rust port design for OS runtime |
