# AOT Runtime Pack

`objeta-aot` is the ahead-of-time runtime metadata compiler for Objeta.

The goal is not to table-lookup generation. The goal is to precompile the
runtime execution map that the executor needs for MoE routing, residency, and
initial runtime tuning.

## Scope

v0 is intentionally narrow:

- parse checkpoint metadata
- recognize expert tensor layout
- read optional calibration traces
- compute runtime metadata
- emit a runtime pack directory

v0 does **not**:

- replace generation with lookup tables
- replace router math
- modify DeltaNet, GQA, or lm_head equations
- re-enable group pre-resolve by default

## Runtime Pack Layout

```text
packs/qwen36-m1-8gb.objeta/
  manifest.json
  expert_layout.json
  expert_importance.json
  expert_coresidency.json
  residency_plan.json
  phase_policy.json
  runtime_profile.json
```

## File Roles

### `manifest.json`

Top-level index for the pack.

### `expert_layout.json`

Maps `(layer, expert)` to source tensor metadata such as:

- tensor name
- source file
- shape
- dtype
- byte length

v0 allows missing byte offsets if the source format does not expose them
cleanly.

### `expert_importance.json`

Stores frequency / gate-weight-derived importance scores and hot-warm-cold
tiering.

### `expert_coresidency.json`

Stores layer-local co-selection relationships between experts. v0 is metadata
only; the executor does not consume this file yet.

### `residency_plan.json`

Stores initial hot expert set and eviction-priority guidance under a capacity
budget.

### `phase_policy.json`

Stores LKO-aware metadata about phase and recommended policy per layer. v0 does
not force these recommendations into execution.

### `runtime_profile.json`

Stores executor-ready initial knobs such as:

- backend
- policy kind
- `moe_top_p`
- `moe_min_experts`
- resident cache capacity

## Initial CLI

```bash
objeta-aot compile \
  --model /path/to/model \
  --calib traces/calib_moe_trace.jsonl \
  --target m1-8gb \
  --out packs/qwen36-m1-8gb.objeta
```

## v0 Status

Implemented through Phase 5:

- Phase 0/1: schema + compiler skeleton
- Phase 2: SafeTensors index + config parsing, real `expert_layout.json`
- Phase 3: calibration trace analysis, real `expert_importance.json` and `expert_coresidency.json`
- Phase 4: residency planner and planner-derived `runtime_profile.json`
- Phase 5: executor runtime-pack loader

Current executor integration:

- `OBJETA_RUNTIME_PACK_PATH=/path/to/pack`
- FFI: `lko_runner_load_runtime_pack(runner, pack_path)`
- applied in v0:
  - `runtime_profile.json`
  - `expert_importance.json`
  - `residency_plan.json`
- loaded-but-not-applied yet:
  - `phase_policy.json`
  - `expert_coresidency.json`

## Real Qwen3.6 Status

The compiler now supports the real Qwen3.6 packed-expert checkpoint layout.

Real packed tensors look like:

```text
model.language_model.layers.{L}.mlp.experts.gate_up_proj
model.language_model.layers.{L}.mlp.experts.down_proj
```

These do **not** include expert IDs in the tensor name. They represent packed
per-layer expert tensors.

Current emitted layout fields reflect that:

- `layout_kind = packed_experts`
- `packed_expert_layers = 40`
- `logical_routed_expert_count = 10240`

This means coverage is computed against logical routed experts, not the number
of per-expert layout entries.

## Calibration Trace Generation

Real calibration traces can now be generated from executor runs using:

- [calib/prompts/general.jsonl](../calib/prompts/general.jsonl)
- [experiments/generate_calib_trace.py](../experiments/generate_calib_trace.py)

Generated event format:

```json
{
  "prompt_id": "coding_001",
  "task_profile": "general",
  "phase": "decode",
  "token_id": 1234,
  "layer": 31,
  "selected_experts": [42, 7, 103],
  "selected_weights": [0.31, 0.22, 0.14],
  "routing_mass_kept_pre_renorm": 0.91,
  "routing_mass_dropped_pre_renorm": 0.09
}
```

This is now the preferred input for `objeta-aot specialize`.

## Current Real Coverage

Recent fuller calibration trace status:

- logical total experts: `10240`
- calibrated experts: `6781`
- coverage: `66.22%`

This is a real improvement, but still below the current pruning gate.

So current specialization behavior is expected to be conservative:

- non-zero `protect / keep / cold_tier`
- but `compress = 0`
- and `prune_candidate = 0`

## Target-Aware Quantization

The precision pass now respects target quant preferences.

### M1 / Metal-like CPU path

- preferred formats: `q5`, `q4`, `q4_k`
- `iq3` is **not** emitted

### RTX3070 CUDA path

- `iq3` remains a real cold-expert candidate

This keeps report labels honest:

- `iq3` on RTX = actionable candidate
- `iq3` on M1 = not emitted

### Importance-Aware Eviction

When a loaded pack contains non-empty `expert_importance.json`, the executor
enables `importance_lru` eviction:

1. tier retention priority
2. importance ascending
3. `last_used_token` ascending

Tier eviction order:

- `Cold`
- `Unknown`
- `Warm`
- `Hot`

Hot experts are **not pinned**. They are only less likely to be evicted.
Capacity remains hard-enforced.

### Validation Notes

- A pack can be structurally valid but still have empty `expert_importance.json`
  (for example, early mock packs).
- In that case:
  - `runtime_pack_loaded=true`
  - but `importance_eviction_enabled=false`
- To validate importance-aware eviction, use a calibration-derived pack with
  real expert entries.

Later phases will extend the pack with:

- importance-aware eviction behavior
- real model pack smoke comparisons
- optional binary pack formats
