# Current Status — 2026-05-21

## Summary

The project now has a working bridge from:

1. real Qwen3.6 checkpoint metadata
2. real executor MoE routing traces
3. AOT specialization plans
4. executor runtime-pack loading

The biggest new milestone is that **packed Qwen3.6 expert layout parsing is fixed**, and calibration coverage has grown from a tiny recent-trace slice to a much more useful partial corpus.

## High-Level State

### Executor

- `safe_exact` remains the correctness baseline.
- Runtime pack loading is integrated.
- Importance-aware eviction is active when pack priorities are non-empty.
- Governor has disabled / observe-only / safety-only / offensive-v0 modes.
- `moe_io_events` now include `selected_weights`, making executor traces reusable for AOT calibration.

### AOT / Specialization

- Real Qwen3.6 packed expert layers are parsed correctly.
- `logical_routed_expert_count = 10240`
- Specialization packs can be emitted for:
  - `m1-8gb`
  - `rtx3070-8gb-vram-32gb-ram`
- Coverage reporting is now sane.

## Calibration Coverage

### Earlier trace

- coverage: `53.57%`
- calibrated experts: `5486`

### Current fuller trace

Artifacts:

- `/tmp/qwen36_calib_trace_general_full.jsonl`
- `/tmp/qwen36_calib_trace_general_full_summary.json`

Metrics:

- prompt_count: `5` additional prompts in the extension batch
- event_count: `9560`
- unique experts: `6781`
- logical total experts: `10240`
- coverage: **`66.22%`**

Per-layer coverage:

- min: `54.69%`
- median: `66.02%`
- max: `86.72%`

Batch discovery:

- summarization: `378`
- japanese_chat: `349`
- english_chat: `171`
- instruction: `110`
- story: `287`

## Real Specialization Results

### M1 8GB conservative

Pack:

- `/tmp/qwen36-specialize-m1-conservative-full.objeta`

Results:

- calibrated experts: `6781`
- coverage: `66.22%`
- pruning enabled: `false`
- protect: `610`
- keep: `5035`
- cold_tier: `1136`
- compress: `0`
- prune_candidate: `0`
- estimated routing mass loss: `0.0`
- backend: `fused_row_parallel`
- resident cache capacity: `3221225472`

Quant counts:

- `q8: 40`
- `q5: 1131`
- `q4: 5650`
- `iq3: 0`

### RTX3070 8GB VRAM / 32GB RAM balanced

Pack:

- `/tmp/qwen36-specialize-rtx3070-balanced-full.objeta`

Results:

- calibrated experts: `6781`
- coverage: `66.22%`
- pruning enabled: `false`
- protect: `610`
- keep: `5035`
- cold_tier: `1136`
- compress: `0`
- prune_candidate: `0`
- estimated routing mass loss: `0.0`
- backend: `cuda_fused`
- resident cache capacity: `8589934592`

Quant counts:

- `q8: 40`
- `q5: 1131`
- `q4: 3095`
- `iq3: 2555`

## Important Fixes Landed

### 1. Packed expert layout parsing

Real tensors like:

```text
model.language_model.layers.{L}.mlp.experts.gate_up_proj
model.language_model.layers.{L}.mlp.experts.down_proj
```

are now handled as packed expert layers instead of parser failures.

### 2. Coverage accounting

Coverage now uses:

```text
logical_routed_expert_count = num_layers * num_experts
```

instead of broken routed-layout entry counts.

### 3. Trace generator

New:

- [calib/prompts/general.jsonl](../calib/prompts/general.jsonl)
- [experiments/generate_calib_trace.py](../experiments/generate_calib_trace.py)

The generator produces AOT-friendly calibration JSONL directly from executor runs.

### 4. Target-aware quant behavior

`precision_pass` now respects target format preferences.

Meaning:

- M1 does **not** emit `iq3`
- RTX may emit `iq3`

## Current Constraints

- Coverage is improved, but still below the pruning gate.
- `compress` and `prune_candidate` remain zero.
- Reports are still advisory:
  - `estimated_only = yes`
  - `requires_verification = yes`

This is expected and healthy.

## Recommended Next Step

Add more routing-diverse prompts, especially:

- factual QA
- coding
- Japanese chat

The next milestone is not “more plans,” but **crossing the calibration coverage gate cleanly** so specialization can start proposing non-zero compress/prune candidates.
