# Runtime Profile Schema

`runtime_profile.json` is the opt-in, Rust-native runtime tuning profile loaded by the executor.

The profile is intentionally narrow in v1:

- it may change runtime knobs
- it must not change model math
- it must not alter tokenizer/chat-template behavior
- it is disabled unless explicitly loaded

## Status

This schema is now treated as the v1 baseline.

- loader: `crates/objeta-qwen36-executor/src/runtime_profile.rs`
- FFI entrypoint: `lko_runner_load_runtime_profile`
- env convenience path: `OBJETA_RUNTIME_PROFILE_PATH`

## Top-level shape

```json
{
  "name": "smoke-profile",
  "target": "m1_8gb",
  "notes": "optional free-form notes",
  "policy_kind": "exact",
  "knobs": {
    "backend": "legacy",
    "moe_top_p": 1.0,
    "moe_min_experts": 8,
    "moe_max_experts": 8,
    "resident_cache_capacity_bytes": 0,
    "residency_group_size": 1,
    "group_preresolve_top_n": 0,
    "group_preresolve_max_bytes": 0
  }
}
```

## Fields

### `name`

- type: `string`
- required: no
- default: `""`
- purpose: human-readable profile label

### `target`

- type: `string`
- required: no
- default: `""`
- purpose: machine/profile target such as `m1_8gb`

### `notes`

- type: `string`
- required: no
- default: `""`
- purpose: free-form comments

### `policy_kind`

- type: `string`
- required: no
- default: `exact`
- allowed:
  - `exact`
  - `top_p`
  - `lko_aware`

This field freezes the *semantic identity* of the runtime policy.

- `exact` means fixed top-8 execution. No pruning semantics are active. Effective `moe_top_p = 1.0`, `moe_min_experts = 8`, `moe_max_experts = 8`.
- `top_p` means pruning semantics are active, even if `moe_top_p = 1.0`.
- `lko_aware` is reserved for explicit opt-in experimentation.

### `knobs`

- type: `object`
- required: no
- default: empty object

All knobs are optional. Missing knobs leave the current runtime unchanged.

## `knobs` fields

### `backend`

- type: `string`
- allowed:
  - `legacy`
  - `fused_row_parallel`

Behavior:

- `legacy` sets `use_fused_moe = false`
- `fused_row_parallel` sets `use_fused_moe = true` and `fused_moe_variant = row_parallel`

This does not change model math; it only selects the MoE execution backend.

### `moe_top_p`

- type: `number`
- expected range: `0.0..=1.0`

Applied only when `policy_kind = "top_p"`.

### `moe_min_experts`

- type: `integer`
- minimum practical value: `1`

Applied only when `policy_kind = "top_p"`.

### `moe_max_experts`

- type: `integer`
- minimum practical value: `1`

### `resident_cache_capacity_bytes`

- type: `integer`
- units: bytes

Replaces the current resident cache manager with a new manager sized to this capacity.

### `residency_group_size`

- type: `integer`
- minimum practical value: `1`

Mapped to `OBJETA_RESIDENCY_GROUP_SIZE`.

### `group_preresolve_top_n`

- type: `integer`

Mapped to `OBJETA_GROUP_PRERESOLVE_TOP_N`.

### `group_preresolve_max_bytes`

- type: `integer`
- units: bytes

Mapped to `OBJETA_GROUP_PRERESOLVE_MAX_BYTES`.

## Non-goals in v1

The runtime profile must not be used to:

- change DeltaNet/GQA/lm_head math
- switch tokenizer or prompt formatting
- alter sampling semantics
- perform live backend switching during a token

## Loading paths

### Direct FFI

Use:

- `lko_runner_init(...)`
- `lko_runner_load_runtime_profile(path)`

### Env convenience

Set:

```bash
OBJETA_RUNTIME_PROFILE_PATH=/path/to/runtime_profile.json
```

The runner will attempt to load it on init.

## Precedence

Runtime configuration precedence is fixed as:

1. explicit runtime profile
2. env debug override
3. strategy config
4. defaults

This precedence is reported in run summaries as `runtime_config_source`.

## Compatibility note

v1 currently applies profile knobs at runner load time only.

`ApplyAtTokenBoundary` governor mode is intentionally not wired to mutate these knobs yet.
Applied only when `policy_kind = "top_p"`.
