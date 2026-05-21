# objeta Handoff — 2026-05-21

## Current Truth

- `safe_exact` chat-template path is still the executor correctness baseline.
- Heavy validation is **sequential only**.
- Do **not** run `scripts/check_all.sh`.
- Full oracle sweeps are reserved for semantic changes.
- `objeta-aot` now supports:
  - real SafeTensors index parsing
  - packed Qwen3.6 expert layout parsing
  - calibration trace analysis
  - residency planning
  - runtime pack generation
  - specialization pack generation
- `objeta-qwen36-executor` now supports:
  - runtime pack loading
  - importance-aware eviction
  - runtime profile loading
  - governor disabled / observe-only / safety-only / offensive-v0 modes

## Most Important Recent Changes

### 1. Real Qwen3.6 packed expert layout parsing works

Real Qwen3.6 checkpoint tensors such as:

```text
model.language_model.layers.X.mlp.experts.gate_up_proj
model.language_model.layers.X.mlp.experts.down_proj
```

are now treated as **packed expert layers**, not parser failures.

Current AOT layout truth:

- `layout_kind = packed_experts`
- `packed_expert_layers = 40`
- `logical_routed_expert_count = 10240`

This fixed the old bogus state:

- routed experts = `0`
- coverage = `58900%`

### 2. Real calibration trace generation exists

New files:

- [calib/prompts/general.jsonl](/Users/yasuno/projects/objeta/calib/prompts/general.jsonl)
- [experiments/generate_calib_trace.py](/Users/yasuno/projects/objeta/experiments/generate_calib_trace.py)

The generator:

- runs prompts sequentially through `qwen36_full_rust.py`
- reads `moe_stats.json`
- emits AOT-friendly calibration JSONL with:
  - `prompt_id`
  - `task_profile`
  - `phase`
  - `token_id`
  - `layer`
  - `selected_experts`
  - `selected_weights`
  - `routing_mass_kept_pre_renorm`
  - `routing_mass_dropped_pre_renorm`

### 3. `moe_io_events` now include `selected_weights`

This was a real telemetry gap that blocked calibration reuse.

Updated:

- [crates/objeta-qwen36-executor/src/moe_stats.rs](/Users/yasuno/projects/objeta/crates/objeta-qwen36-executor/src/moe_stats.rs)
- [crates/objeta-qwen36-executor/src/qwen36_forward.rs](/Users/yasuno/projects/objeta/crates/objeta-qwen36-executor/src/qwen36_forward.rs)

This is **telemetry-only**, not math-changing.

### 4. M1 quant planning no longer emits `iq3`

`precision_pass` now respects target quant preferences.

Result:

- `m1-8gb` falls back cold transport experts to `q4`
- `rtx3070-8gb-vram-32gb-ram` may still emit `iq3`

So:

- `iq3` is now a **real RTX candidate**
- not a Metal / M1 recommendation

## Real Calibration Coverage Status

### Earlier short trace

- calibrated experts: `5486`
- logical total experts: `10240`
- coverage: `53.57%`

### Current fuller trace

Generated from remaining categories:

- `summarization`
- `japanese_chat`
- `english_chat`
- `instruction`
- `story`

Current full trace artifacts:

- `/tmp/qwen36_calib_trace_general_full.jsonl`
- `/tmp/qwen36_calib_trace_general_full_summary.json`

Current coverage:

- prompt_count: `5` newly added in the extension batch
- event_count: `9560`
- unique experts: `6781`
- logical total experts: `10240`
- overall coverage: **`66.22%`**

Per-layer coverage:

- min: `54.69%`
- median: `66.02%`
- max: `86.72%`

Newly discovered experts per batch:

- `summarization`: `378`
- `japanese_chat`: `349`
- `english_chat`: `171`
- `instruction`: `110`
- `story`: `287`

## Current Specialization Outputs

### M1 8GB conservative

Pack:

- `/tmp/qwen36-specialize-m1-conservative-full.objeta`

Key results:

- calibrated experts: `6781`
- coverage: `66.22%`
- pruning: **disabled**
- `protect = 610`
- `keep = 5035`
- `cold_tier = 1136`
- `compress = 0`
- `prune_candidate = 0`
- estimated routing mass loss: `0.0`
- backend: `fused_row_parallel`
- resident cache capacity: `3GB`

Quant counts:

- `q8 = 40`
- `q5 = 1131`
- `q4 = 5650`
- `iq3 = 0`

### RTX3070 8GB VRAM / 32GB RAM balanced

Pack:

- `/tmp/qwen36-specialize-rtx3070-balanced-full.objeta`

Key results:

- calibrated experts: `6781`
- coverage: `66.22%`
- pruning: **disabled**
- `protect = 610`
- `keep = 5035`
- `cold_tier = 1136`
- `compress = 0`
- `prune_candidate = 0`
- estimated routing mass loss: `0.0`
- backend: `cuda_fused`
- resident cache capacity: `8GB`

Quant counts:

- `q8 = 40`
- `q5 = 1131`
- `q4 = 3095`
- `iq3 = 2555`

## Why pruning is still disabled

This is still healthy behavior.

Current gating picture:

- coverage is better, but not yet at the pruning gate
- routing mass loss estimates are still conservative
- reports still mark:
  - `estimated_only = yes`
  - `requires_verification = yes`

So the compiler is doing the right thing by refusing to become overly bold.

## Runtime Pack Loader / Executor Status

Runtime pack loading is live in executor:

- env: `OBJETA_RUNTIME_PACK_PATH=/path/to/pack`
- FFI: `lko_runner_load_runtime_pack(runner, pack_path)`

Applied in v0:

- `runtime_profile.json`
- `expert_importance.json`
- `residency_plan.json`

Read-only metadata in v0:

- `phase_policy.json`
- `expert_coresidency.json`

Importance-aware eviction is enabled when loaded importance is non-empty.

Eviction order with priorities:

1. `Cold`
2. `Unknown`
3. `Warm`
4. `Hot`

Tie-break:

1. lower importance first
2. older `last_used_token` first

## Governor Status

Governor modes currently exist:

- `Disabled`
- `ObserveOnly`
- `ApplyAtTokenBoundary`
- offensive mode v0 behind `OBJETA_GOVERNOR_OFFENSIVE=1`

Important caveat:

- offensive mode is implemented
- but current real runs are still mostly blocked by memory/IO hard risk
- so offensive actions have often stayed at `0`

This is not a bug by itself; it means the governor is still mostly defensive on M1-class memory pressure.

## Operational Rules

- Never run `scripts/check_all.sh`
- Avoid parallel heavy runs
- For runtime / executor changes:
  1. `cargo build -p objeta-qwen36-executor`
  2. `cargo test -p objeta-qwen36-executor`
  3. light smoke only
  4. full oracles only for semantic changes
- For AOT changes:
  1. `cargo test -p objeta-aot`
  2. `cargo build -p objeta-aot`
  3. specialize smoke on real checkpoint if parser/report semantics changed

## Next Recommended Steps

1. Extend calibration corpus again with more routing-diverse prompts
   - especially `factual_qa`, `coding`, `japanese_chat`
2. Push coverage from `66.22%` toward `80%+`
3. Re-run specialize once coverage crosses the pruning gate
4. Only then evaluate whether `compress` / `prune_candidate` become meaningful

## Caveats to Remember

- `qwen36_ffi.rs` is active in the current tree. Do not rely on older notes claiming it was removed.
- AOT reports are now structurally trustworthy for real Qwen3.6 packed layout.
- But specialization outputs are still **advisory** until verification runner work lands.
