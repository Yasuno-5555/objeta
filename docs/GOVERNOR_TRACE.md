# Governor Trace Schema

`governor_trace.jsonl` is the observe-only runtime governor trace.

Each line is a standalone JSON object with:

- `observation`
- `decision`

In v1, decisions are hypothetical unless a future `ApplyAtTokenBoundary` implementation explicitly activates them.

## Status

This schema is frozen as the v1 observe-only baseline.

- source: `crates/objeta-qwen36-executor/src/runtime_governor.rs`
- mode env: `OBJETA_GOVERNOR_MODE`
- trace path env: `OBJETA_GOVERNOR_TRACE_PATH`

## One line example

```json
{
  "observation": {
    "step": 0,
    "token_id": 248045,
    "token_position": 0,
    "prev_decode_entropy": 1.18,
    "repetition_risk": false,
    "collapse_risk": false,
    "resident_capacity_bytes": 4294967296,
    "resident_bytes": 629145600,
    "resident_hit_delta": 0,
    "resident_miss_delta": 320,
    "actual_bytes_loaded_delta": 629145600,
    "resident_bytes_reused_delta": 0,
    "avg_selected_experts": 0.0,
    "avg_routing_mass_kept": 0.0,
    "avg_routing_mass_dropped": 0.0
  },
  "decision": {
    "mode": "observe_only",
    "memory_pressure": "low",
    "io_thrash": "thrashing",
    "quality_risk": "low",
    "suggested_top_p": 0.9,
    "suggested_min_experts": 4,
    "suggested_resident_cache_capacity_bytes": null,
    "suggested_group_preresolve_top_n": 1,
    "suggested_group_preresolve_max_bytes": 134217728,
    "rationale": "pressure-or-io-thrash"
  }
}
```

## `observation` fields

### Token identity

- `step`: executor step counter
- `token_id`: token being processed at this boundary
- `token_position`: currently mirrors `step`

### Quality-side signals

- `prev_decode_entropy`: entropy from the previous decode distribution
- `repetition_risk`: boolean repetition heuristic
- `collapse_risk`: boolean collapse heuristic

Important:

- this is intentionally named `prev_decode_entropy`
- it is not the entropy of a future token
- the intended control direction is:
  - previous token observation
  - hypothetical next-token decision

### Resident cache / I/O deltas

- `resident_capacity_bytes`
- `resident_bytes`
- `resident_hit_delta`
- `resident_miss_delta`
- `actual_bytes_loaded_delta`
- `resident_bytes_reused_delta`

These are token-boundary deltas, not lifetime totals.

### MoE selection summary

- `avg_selected_experts`
- `avg_routing_mass_kept`
- `avg_routing_mass_dropped`

These summarize MoE calls observed for the token boundary window.

## `decision` fields

### `mode`

Allowed values:

- `disabled`
- `observe_only`
- `apply_at_token_boundary`

In current v1 smoke usage, only `observe_only` is active.

### Pressure/risk classes

- `memory_pressure`
  - `low`
  - `high`
  - `critical`
- `io_thrash`
  - `stable`
  - `thrashing`
- `quality_risk`
  - `low`
  - `elevated`
  - `high`

### Suggested knobs

These are hypothetical in observe-only mode:

- `suggested_top_p`
- `suggested_min_experts`
- `suggested_resident_cache_capacity_bytes`
- `suggested_group_preresolve_top_n`
- `suggested_group_preresolve_max_bytes`

### `rationale`

- type: `string`
- purpose: one short explanation of why the hypothetical decision was chosen

## Important v1 guarantee

Observe-only mode must not mutate runtime knobs.

That means:

- trace lines may contain suggested actions
- those suggestions are diagnostic only
- no top-p, cache, or group-pre-resolve knob is changed because of the governor in observe-only mode
