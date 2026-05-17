# objeta — Family-Aware Runtime Compiler

2026-05-17

## Final Status

```
Qwen3.6-35B-A3B @ M1 8GB:  0.21 → 3-4 tok/s  (17.6x speedup)
TinyLlama-1.1B:             Phase-adaptive quant → uniform q5 = optimal
Qwen2.5-0.5B:               Cross-family: FFN priority (inverted asymmetry)
```

## Architecture

```
safetensors
    │
    ▼
objeta analyze  ──→  phase_profile.json  (static geometry)
    │
    ▼
objeta strategy ──→  strategy.json       (family-aware runtime config)
    │
    ▼
Qwen36Runner     ──→  generation         (reads strategy.json on init)
    │
    ├── fusion_ratio:     DeltaNet layer skip (0.33 = 10/30)
    ├── moe_on_deltanet:  MoE skip on non-steering layers
    └── strategy.rs:      per-component weight requantization
```

## Commands

```bash
# Analyze model → phase profile
objeta analyze model.safetensors/ --output phase_profile.json --stability

# Generate family-aware runtime strategy
objeta strategy phase_profile.json --output strategy.json

# Attention-backbone quantization plan (per-component)
objeta quantize phase_profile.json --mode attention-backbone \
    --attn-qo-bits 5 --attn-kv-bits 4 --ffn-bits 3.5

# Run Qwen3.6 with strategy (reads strategy.json from bin_dir)
cp strategy.json models/qwen36_bin/
python3 experiments/qwen36_full_rust.py [fusion_ratio] [moe_on_deltanet]
```

## Cross-Family Framework

| | Family A | Family B Phase 1 | Family B Phase 3 |
|---|---|---|---|
| **Model** | TinyLlama, Llama | Qwen2.5-0.5B | Qwen3.6-35B |
| **Mechanism** | Residual Transport | Aligned Field | Mixed Field |
| **Dominance** | Attention (8.8x) | FFN (0.1x) | GQA Steering |
| **Strategy** | Attn q5+, FFN q3+ | FFN q5+, Attn q4 | GQA q5+, Delta q4, FFN q3 |
| **fusion_ratio** | 1.0 | 0.5 | 0.33 |
| **moe_on_deltanet** | true | true | false |

## Performance Evolution (Qwen3.6-35B @ M1 8GB)

| Stage | Forward time | tok/s | Key change |
|-------|-------------|-------|------------|
| Python MLX | ~30s | 0.03 | Baseline |
| Rust f32 | ~20s | 0.05 | All ops in Rust |
| Rust f16 + mmap | ~5s | 0.21 | SWAP eliminated |
| + ΔN fusion (0.33) | ~0.6s | 1.7 | 1 DeltaNet per GQA block |
| + MoE skip on ΔN | **~0.15s** | **3-4** | MoE only on GQA layers |

```
Breakdown (ΔN=0.33 + MoE skip):
  MoE on GQA (10 layers):    80ms  52%
  GQA attention (10 layers): 64ms  42%
  Shared expert:             10ms   6%
  ────────────────────────────────────
  Forward total:            154ms
  + lm_head + sampling:     ~50ms
  = ~200ms/token → ~5 tok/s theoretical
```

## Key Experimental Results

### Phase A: Layer-wise Allocation

| Exp | Finding |
|-----|---------|
| UNFOLD sensitivity | L0-L2 super-additive, 54% PPL recovery |
| q3 precision cliff | q3=catastrophic (PPL 573), cliff at q3.0→q3.25 (13.2x in 0.25bit) |
| Head-to-head | Uniform q5 = oracle quality. Phase-adaptive loses. |

### Phase B: Cliff Mechanism

| Exp | Finding |
|-----|---------|
| Component ablation | FFN q3 survives (PPL 17), Attn q3 degrades (PPL 33). Combined = super-multiplicative collapse |
| Continuous sweep | Cliff at 8→10 levels. Phase transition, not gradual degradation. |
| Residual bandwidth | Confirmed: cliff is emergent from coupled FFN+Attn errors |

### Phase C: Transport Asymmetry

| Exp | Finding |
|-----|---------|
| Attention backbone | TinyLlama: Attn priority (8.8x). Qwen2.5: FFN priority (0.1x inverted). |
| KV vs QO | QO projection = transport routing bottleneck (2.0x over KV) |
| Attention bandwidth | Attention precision determines residual stream transport capacity |

### Phase D: Runtime Optimization

| Exp | Finding |
|-----|---------|
| DeltaNet fusion | fusion_ratio=0.33 → 3.8x forward speedup |
| MoE skip | moe_on_deltanet=false → additional 2.5x (10.7x total) |
| Optimal config | ΔN=0.33 + MoE skip = 154ms forward (was 1671ms) |

## Component Roles (confirmed)

| Component | Role | Precision Sensitivity | Strategy |
|-----------|------|----------------------|----------|
| Attention Q/O | Transport routing | CRITICAL (q3 = collapse) | q5+ always |
| Attention K/V | Memory storage | Moderate | q4+ |
| FFN (dense) | Local modulation | Family-dependent | q3-q5 |
| MoE experts | Sparse injection | Low (256× redundancy) | q4, skip on ΔN layers |
| DeltaNet layers | Fine steering | Low (||Δ|| ≪ 1) | 1 per GQA block |
| GQA layers | Course correction | High (||Δ|| = 1.5-3.2) | Full precision |

## Three Precision Regimes

```
Safe (q5+):         noise < trajectory floor → uniform dominates
Critical (q3.25-q4): marginal continuity → geometry matters
Collapse (≤q3.0):   transport capacity broken → phase transition
```

## Source Map

```
objeta/
├── crates/
│   ├── objeta-core/          Shared types (Family, Phase, Strategy)
│   ├── objeta-parser/        Safetensors mmap loader
│   ├── objeta-analysis/      Static geometry (SVD, intra_cos, Lyapunov)
│   ├── objeta-phase/         Phase/family classification
│   ├── objeta-routing/       Per-layer precision assignment
│   ├── objeta-quantize/      Strategy generator + attention backbone
│   ├── objeta-cli/           CLI (analyze, quantize, strategy)
│   └── objeta-qwen36-executor/
│       ├── qwen36_forward.rs 40-layer loop + fusion_ratio + moe skip
│       ├── moe_dispatch.rs   q4 dequantize + expert cache
│       ├── quantize.rs       q2/q3/q4/q5 quantizers
│       └── strategy.rs       Strategy JSON loader + requantize
├── experiments/
│   ├── qwen36_full_rust.py   Main launcher (fusion_ratio + moe_on_deltanet)
│   ├── timing_probe.py       Per-component timing measurement
│   ├── cross_family_validation.py  TinyLlama vs Qwen2.5 head-to-head
│   ├── phase_a_validation.py      Early experiments (Exp 1-3)
│   ├── phase_b_cliff.py           Cliff mechanism (Exp 7-8)
│   ├── phase_c_residual.py        Transport asymmetry (Exp 9-12)
│   ├── exp13_ffn_survival.py      Ultra-low FFN frontier
│   └── qwen36_precision_probe.py  Single-token geometry probe
└── docs/
    └── QUANTIZATION_EXPERIMENT_PLAN.md  This document
```

## Central Thesis

> **LLM inference is constrained not by parameter precision, but by
> attention-mediated trajectory transport capacity through the residual stream.**
>
> Compression target: transport continuity preservation, not weight approximation.
> Strategy: family-dependent — attention backbone (A), FFN backbone (B1),
> steering backbone (B3).
