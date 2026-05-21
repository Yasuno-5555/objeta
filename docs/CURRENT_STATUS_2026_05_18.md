# Current Status — 2026-05-20

## Summary

The executor is now in a good debugging state:

- `safe_exact` chat-template remains the correctness baseline
- chat-template prefill / one-token / layer oracles exist and were green before the latest fused-MoE timing work
- Fused MoE v0 is mathematically correct and faster in isolated routed-exec benchmarks
- but fused MoE is **not yet an end-to-end win**

The current question is no longer "is routed MoE numerically broken?"

It is now:

> Why does routed execution get faster in microbench, while 1-token E2E still gets slower?

## Current Truth

### Correctness baseline

- strategy: `configs/safe_exact.json`
- prompt mode: chat-template
- oracle goldens:
  - `safe_exact_chat_prefill`
  - `safe_exact_chat_1token`
  - `safe_exact_chat_layer_trace`

### Operational rule

- Do **not** run `scripts/check_all.sh`
- Avoid parallel heavy runs
- Full oracle sweeps are for semantic changes, not routine infra work

## Fused MoE v0

### What is implemented

- fused routed MoE dispatch in Rust
- runtime toggle:
  - `OBJETA_USE_FUSED_MOE=1`
- variant selector:
  - `OBJETA_FUSED_MOE_VARIANT=row_parallel|chunked32|chunked64|chunked128|serial`
- current default variant when fused is enabled:
  - `chunked128`

### What is confirmed

Local checks:

- `cargo test -p objeta-qwen36-executor`: PASS
- `cargo build --release -p objeta-qwen36-executor`: PASS
- `experiments/test_fused_moe.py`: PASS

Selected-expert parity:

- layers `0, 7, 31`
- expert counts `1..8`
- cosine `1.0`
- max abs diff `0`

## Benchmarks

### 1. `call_moe` microbench using actual runner path

This is the important new result.

With direct selected-expert execution through the real runner path, fused is clearly faster than legacy.

Examples:

- Layer 0, `N=8`: `15.160 ms -> 4.735 ms`
- Layer 7, `N=8`: `10.533 ms -> 4.192 ms`
- Layer 31, `N=8`: `11.338 ms -> 5.058 ms`

So:

> routed expert execution itself is not the bottleneck anymore

### 2. Variant sweep inside fused routed exec

Release microbench on Layer 31 / `N=8`:

- `serial`: `10.290 ms`
- `row_parallel`: `4.494 ms`
- `chunked32`: `5.111 ms`
- `chunked64`: `5.434 ms`
- `chunked128`: `5.183 ms`

Current best candidate from release microbench:

- `row_parallel`

### 3. Replay microbench using actual E2E hidden states

New script:

- `experiments/call_moe_replay_microbench.py`

It uses real prompt-prefill hidden states, then replays selected routed experts through legacy vs fused.

Observed:

- Layer 0: `11.652 ms -> 4.674 ms`
- Layer 7: `11.354 ms -> 5.171 ms`
- Layer 31: `11.496 ms -> 4.963 ms`

This matters a lot:

> even with actual E2E hidden states, routed exec still wins

So the current slowdown is probably not caused by "unrealistic synthetic microbench inputs."

## 1-token E2E Status

### `chunked128`

Single-token E2E with `safe_exact`:

- output stays correct: `Here`
- but fused `chunked128` is slower than baseline

Example comparison:

- baseline:
  - `forward_wall_ms_avg ≈ 2206`
  - `moe_wall_ms_avg ≈ 1011`
- fused `chunked128`:
  - `forward_wall_ms_avg ≈ 2408`
  - `moe_wall_ms_avg ≈ 1168`

### `row_parallel`

Baseline / fused / baseline sequence was run to reduce noise.

All runs produced:

- output: `Here`

But fused `row_parallel` still lost:

- baseline 1:
  - `forward_wall_ms_avg ≈ 2427`
  - `moe_wall_ms_avg ≈ 1105`
- fused `row_parallel`:
  - `forward_wall_ms_avg ≈ 2480`
  - `moe_wall_ms_avg ≈ 1224`
- baseline 2:
  - `forward_wall_ms_avg ≈ 2251`
  - `moe_wall_ms_avg ≈ 1047`

So the present state is:

> routed exec wins in both direct and replay microbench, but `moe_wall_ms_avg` still gets worse in 1-token E2E

## New Timing / Telemetry Work

The executor now has more detailed routed-MoE timing fields:

- `call_moe_total_wall`
- `router_wall`
- `candidate_build_wall`
- `policy_select_wall`
- `cache_lookup_wall`
- `routed_exec_wall`
- `stats_wall`

And run artifacts now include:

- `summary.json`
  - `use_fused_moe`
  - `fused_moe_variant`
- `moe_stats.json`
  - full per-layer timing and I/O breakdown

`OBJETA_TIMING=0` now suppresses the verbose Rust-side per-token timing prints.

## Current Interpretation

The evidence now strongly suggests:

1. Fused routed execution is real and useful
2. The current end-to-end loss is probably in integration overhead or timing definition
3. The main suspect is no longer the fused kernel itself

More concretely:

- if `routed_exec_wall` is lower while `moe_wall_ms_avg` is higher, the problem is probably:
  - surrounding `call_moe` overhead
  - shared path accounting
  - stats / bookkeeping
  - or wall-time definition mismatch

## Next Recommended Step

Use the new `moe_stats.json` and `call_moe_replay_microbench.py` to compare:

- `call_moe_total_wall`
- `router_wall`
- `candidate_build_wall`
- `policy_select_wall`
- `cache_lookup_wall`
- `routed_exec_wall`
- `stats_wall`

before touching:

- `lm_head`
- Metal backends
- persistent runner

That is the current highest-signal path.
