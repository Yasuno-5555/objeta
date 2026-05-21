# Current Status — 2026-05-20

## Summary

The executor has undergone a significant architectural refactor:
**Unified MoE Execution Pipeline** ("One Pipeline, Multiple Backends")

- All routed-MoE execution paths now converge on a single canonical function `call_moe_pipeline()`.
- `lko_moe_forward_layer` (legacy FFI) is now a compatibility wrapper that delegates to `call_moe_pipeline`.
- `ExpertResidencyManager` (Q4-compressed resident cache) is integrated into every MoE forward call.
- Byte telemetry is now fully instrumented end-to-end: `logical → actual → reused → scratch`.
- Oracle correctness is confirmed for all 3 golden baselines.
- AOT runtime pack loading is integrated.
- Importance-aware eviction is available when a loaded pack provides expert priorities.

---

## Current Truth

### Correctness Baseline

| Oracle | Result |
|--------|--------|
| `safe_exact_chat_prefill` | ✅ PASS (10/10 top-10 overlap, entropy + norm match) |
| `safe_exact_chat_1token` | ✅ PASS (decoded token 579, 40/40 expert selection match) |
| `safe_exact_chat_layer_trace` | ✅ PASS |

- Strategy: `configs/safe_exact.json`
- Prompt mode: chat-template
- `cargo test -p objeta-qwen36-executor`: **28/28 PASS**

### Operational Rules

- Do **not** run `scripts/check_all.sh` (risk of machine crash).
- Avoid parallel heavy runs.
- Full oracle sweeps are reserved for **semantic changes** only.

---

## Unified MoE Execution Pipeline

### What Was Implemented

#### `call_moe_pipeline()` — canonical entry point

Defined in [`moe_dispatch.rs`](../crates/objeta-qwen36-executor/src/moe_dispatch.rs).

Signature:

```rust
pub fn call_moe_pipeline(
    layer_idx: usize,
    hidden: &[f32],
    router_w: Option<&[f32]>,
    selected_experts: Option<Vec<SelectedExpert>>,
    backend: MoeExecutionBackend,
    fused_down_mode: FusedDownMode,
    residency_mgr: Option<&mut ExpertResidencyManager>,
    mmap_source: Option<(&[u8], &[u8])>,
    scratch: &mut FusedMoeScratch,
    token_id: usize,
) -> MoeExecutionResult
```

Responsibilities:
1. **Router** — `router_topk_cpu` (if `selected_experts` is not pre-provided)
2. **Candidate Build** — wrap into `Vec<SelectedExpert>`
3. **Adaptive Policy Hook** — passive `RuntimeDecision` stub (no active scheduling)
4. **Residency Cache Lookup** — hit: `warm_hit_count++`, miss: `evict_until_fit()` + load
5. **Backend Execute** — `execute_backend()` dispatches to `FusedQ4` or `LegacyDequantF32`
6. **Telemetry Finalization** — populate full `MoeTelemetry` and `GLOBAL_FREQ`

#### `MoeExecutionBackend` — backend selector

| Variant | Description |
|---------|-------------|
| `FusedQ4` | Invoke `fused_moe_q4_selected_cached` directly on Q4 `Arc<[u8]>` pages |
| `LegacyDequantF32` | Dequantize pages to F32, run parallel GEMV via rayon |

#### `execute_selected_moe()` — runner-level dispatch

Now delegates fully to `call_moe_pipeline()`. No independent mmap/GEMV branches remain.

#### `lko_moe_forward_layer` — deprecated compatibility wrapper

Builds an ad-hoc `MoeExecutionRequest` and routes through `call_moe_pipeline`. Contains no independent execution logic.

### ExpertResidencyManager

Defined in [`expert_cache.rs`](../crates/objeta-qwen36-executor/src/expert_cache.rs).

Holds Q4-compressed resident pages as `Arc<[u8]>` (not dequantized F32).

Eviction behavior:

- no priorities loaded: pure LRU fallback
- priorities loaded from runtime pack: `tier -> importance asc -> last_used_token asc`

Tier eviction order:

- `Cold`
- `Unknown`
- `Warm`
- `Hot`

| Method | Behavior |
|--------|----------|
| `ensure_resident()` | Hit: return cached page. Miss: call `evict_until_fit()` then `load_fn()`. |
| `evict_until_fit()` | Evict lowest-priority page until `resident_bytes + needed ≤ capacity_bytes`. |
| `is_bypass()` | `capacity_bytes == 0` → bypass mode (no caching). |
| `resident_bytes()` | Sum of all resident pages' compressed byte counts. |

FFI control:

```c
// Preferred: pass runner pointer directly
lko_runner_init_page_cache(runner: *mut Qwen36Runner, capacity_bytes: i64) -> i32

// Deprecated legacy (singleton): use only for single-runner compatibility
lko_moe_init_page_cache(capacity_bytes: i64) -> i32

// Temporary compatibility accessor
lko_runner_get_instance() -> *mut Qwen36Runner  // 暫定互換用: multi-runner/daemonでは廃止予定
```

### Runtime Pack Loader

The executor now supports loading AOT runtime packs via:

- env: `OBJETA_RUNTIME_PACK_PATH=/path/to/pack`
- FFI: `lko_runner_load_runtime_pack(runner, pack_path)`

v0 applies:

- `runtime_profile.json`
- `expert_importance.json`
- `residency_plan.json`

v0 reads but does not yet apply:

- `phase_policy.json`
- `expert_coresidency.json`

The page-cache init path preserves loaded priorities, so runtime-pack-backed
importance tables survive resident-cache capacity reinitialization.

---

## Byte Telemetry

### Invariants

```
logical_expert_bytes_requested = actual_expert_bytes_loaded + resident_cache_bytes_reused
resident_cache_resident_bytes ≤ capacity_bytes
dequantized_scratch_bytes = 0  (if FusedQ4 backend)
```

### Fields

| Field | Location | Description |
|-------|----------|-------------|
| `logical_expert_bytes_requested` | `summary.json`, `moe_stats.json[layers]`, `moe_io_events` | Bytes logically needed |
| `actual_expert_bytes_loaded` | same | Bytes loaded from mmap (cache miss) |
| `resident_cache_bytes_reused` | same | Bytes served from resident cache (hit) |
| `resident_cache_resident_bytes` | same | Current resident cache size |
| `dequantized_scratch_bytes` | same | F32 scratch alloc (0 for FusedQ4) |

Additional runtime-pack / eviction fields in `summary.json`:

| Field | Description |
|-------|-------------|
| `runtime_pack_loaded` | Pack successfully loaded |
| `runtime_profile_loaded` | `runtime_profile.json` applied |
| `expert_importance_loaded` | `expert_importance.json` parsed |
| `residency_plan_loaded` | `residency_plan.json` parsed |
| `phase_policy_loaded` | Metadata file present and loaded |
| `expert_coresidency_loaded` | Metadata file present and loaded |
| `importance_eviction_enabled` | Importance-aware eviction active |
| `evicted_hot_count` / `evicted_warm_count` / `evicted_cold_count` / `evicted_unknown_count` | Per-tier eviction counters |
| `expert_eviction_policy` | `lru` or `importance_lru` |

### Timing Fields (per call, in `moe_stats.json[summary]`)

| Field | Meaning |
|-------|---------|
| `avg_call_moe_wall_ms` | Total wall time of `call_moe_pipeline` |
| `avg_router_wall_ms` | Router top-k |
| `avg_candidate_build_wall_ms` | Building `Vec<SelectedExpert>` |
| `avg_policy_select_wall_ms` | Adaptive policy hook (currently 0) |
| `avg_cache_lookup_wall_ms` | Residency resolution |
| `avg_routed_exec_wall_ms` | Backend execution |
| `avg_stats_wall_ms` | Telemetry finalization |

---

## Execution Backends — Status

### FusedQ4 (default when `OBJETA_USE_FUSED_MOE=1`)

- Fastest in microbench (3.6–3.8x over legacy for N=8 experts)
- Output numerically identical to LegacyDequantF32 (cosine = 1.000000)
- Current 1-token E2E still slightly slower than legacy — under investigation

Speedups (from microbench):

| Layer | N=8: Legacy → Fused |
|-------|---------------------|
| 0     | 15.2 ms → 4.7 ms   |
| 7     | 10.5 ms → 4.2 ms   |
| 31    | 11.3 ms → 5.1 ms   |

### LegacyDequantF32

- Default when `OBJETA_USE_FUSED_MOE` is not set
- Dequantizes Q4 → F32 per expert per call (8× memory expansion)
- `dequantized_scratch_bytes > 0`

---

## Pending / Open Items

| Item | Status |
|------|--------|
| E2E 1-token fused vs legacy perf delta attribution | 🔍 In progress |
| Metal GQA | ❌ Disabled (CPU fallback active) |
| Real-model AOT runtime pack compile | 📌 Next useful validation step |
| Multi-runner / daemon support | 📌 `get_instance` is singleton — future work |
| `dequantized_scratch_bytes` for LegacyDequantF32 uses `ffn_dim=512` hardcode | ⚠️ Review needed |

---

## Latest Pack Validation (2026-05-21)

- `cargo test -p objeta-qwen36-executor` ✅ 43/43 pass
- `cargo build --release -p objeta-qwen36-executor` ✅ pass
- runtime pack smoke via env/FFI ✅ output `Here`
- pack-loaded 5-token smoke:
  - run: `runs/run_20260521_140635_589664/summary.json`
  - output: `Here's a thinking process`
  - `importance_eviction_enabled=true`
  - `expert_eviction_policy=importance_lru`
  - `logical = loaded + reused` ✅

---

## Validation Ladder

Use in this order:

1. `cargo build -p objeta-qwen36-executor`
2. `cargo test -p objeta-qwen36-executor`
3. `python3 -m py_compile experiments/qwen36_full_rust.py`
4. `bash scripts/check_oracles.sh` (only for semantic changes)
