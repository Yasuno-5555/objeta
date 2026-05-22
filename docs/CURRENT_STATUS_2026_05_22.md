# Current Status — 2026-05-22 (end of day)

## TL;DR

**DeepSeek V4 Flash E2E one-token canary completed.** Official HC + attention (seq=1, pos=0) + CUDA MoE.forward connected end-to-end. The model emits finite logits and one greedy token for input_id=42. CUDA-accelerated attention path achieves 3x speedup (25s → 9s).

| Track | Status | Key Artifact |
|---|---|---|
| **DeepSeek V4 Flash E2E** | **Complete** | `experiments/src/deepseek_e2e.rs`, `experiments/src/deepseek_e2e_fast.rs` |
| **CUDA MoE.forward** | Sealed (parity cos=1.0) | `crates/objeta-cuda/src/moe.rs` |
| **Shared expert pinning** | Infrastructure ready | `CudaExpertCache`, `insert_pinned` |
| **Qwen3.6 Executor** (M1) | Calibration 66.22% | `objeta-qwen36-executor` |

---

## DeepSeek V4 Flash — Completed Milestones

### M9: Real single-layer MoE proof ✅
- Routed FP4 + shared FP8 official arithmetic
- CUDA vs CPU parity: cosine=1.0, rel_l2=0.0
- Layers 0, 27, 42 validated
- Shared expert NaN root cause identified and fixed (`f8e4m3_to_f32` exp=15 → ±448.0 clamp)

### E2E One-Token Canary ✅
- 43 decoder layers: HC pre/post + MLA attention (seq=1, pos=0) + CUDA MoE.forward
- Token embedding → HC expand [4,4096] → layers 0..42 → HC head → RMSNorm → LM head
- **Output (input_id=42)**: token **5**
- All outputs finite, `official_moe_forward=true`
- Deterministic (3 runs identical)

### CUDA-Accelerated Fast Path ✅
- All attention FP8 linears moved to CUDA (`cuda_act_quant_device` + `cuda_fp8_act_fp8_weight_gemv_device`)
- **3x faster**: 25s → 9s per token
- Per-layer breakdown (ms):
  - WqA [1024,4096]: ~4ms
  - WqB [32768,1024]: ~23ms (largest)
  - Wkv [512,4096]: ~1.5ms
  - WoA grouped [8,1024,4096]: ~19ms
  - WoB [4096,8192]: ~18ms
  - MoE kernel: ~12ms
  - MoE tensor load: ~48ms (H2D dominated)

### E2E Causal Intervention Experiments ✅
- 4 global ablations: official_full, no_moe, routed_only, shared_only
- 24 single-layer interventions (8 layers × 3 kinds)
- **Key finding**: Layer 1 is the only causal critical layer — removing shared MoE there changes token 5→680
- Deep layers (10-42) are robust to single-layer MoE removal

### Corrected Metrics
- NaN suppression removed: exp=15 in fp8 weights now clamps to ±448.0
- `parity_status: valid_finite` only when all outputs are finite
- Logit comparisons use same raw arrays (`compare_outputs`)
- MoE geometry validated per-layer (merge_residual, norm_identity)
- `max_abs_error` no longer reports 0.0 for large rel_l2 (was serde NaN→null issue)

---

## Project Structure

```
crates/
  objeta-core/          Shared types, errors
  objeta-parser/        Safetensors mmap, FP8 decode, DeepSeek metadata
  objeta-cuda/          CUDA backend, MoE executor, quant kernels
    src/bin/
      deepseek_single_layer_moe.rs  Single-layer MoE benchmark
  objeta-cli/           CLI entry point
  objeta-analysis/      SVD geometry analysis
  objeta-aot/           Ahead-of-time compilation
  objeta-routing/       Precision routing
  objeta-quantize/      Phase-adaptive quantization
  objeta-moe/           MoE routing compiler
  objeta-os/            Reflexive runtime OS
  objeta-runtime/       Runtime profiling
  objeta-metal/         Metal GPU kernels (M1/M2)
  objeta-ssd/           SSD-resident paging
  objeta-qwen36-executor/  Qwen3.6 full forward (C ABI)
experiments/
  src/
    deepseek_e2e.rs         E2E canary + intervention experiments (CPU attention)
    deepseek_e2e_fast.rs    E2E canary (CUDA-accelerated attention, 3x faster)
    deepseek_e2e_ablated.rs (archived ablation variants)
```

---

## Key Binaries

| Binary | Crate | Purpose |
|---|---|---|
| `deepseek_single_layer_moe` | objeta-cuda | Single-layer MoE benchmark + real checkpoint canary |
| `deepseek_e2e` | experiments | E2E one-token + intervention experiments (--intervention-layer, --intervention-kind) |
| `deepseek_e2e_fast` | experiments | CUDA-accelerated E2E (3x faster, attention CUDA linears) |

## Running E2E

```powershell
# Baseline (official one-token canary)
cargo run --release -p objeta-experiments --bin deepseek_e2e

# Fast CUDA-accelerated
cargo run --release -p objeta-experiments --bin deepseek_e2e_fast

# Global ablation
cargo run --release -p objeta-experiments --bin deepseek_e2e -- --global-variant no_moe_global

# Single-layer intervention
cargo run --release -p objeta-experiments --bin deepseek_e2e -- --intervention-layer 27 --intervention-kind remove_shared
```

---

## Test Coverage

| Crate | Tests |
|---|---|
| objeta-parser | 18 |
| objeta-cuda (lib) | 31 |
| objeta-cuda (deepseek_single_layer_moe) | 22 |
| objeta-cuda (bench_qgemv) | 0 |
| objeta-cuda (bench_selected_moe) | 4 |
| **Total objeta-cuda** | **57** |

All 57 tests pass as of 2026-05-22.

---

## Remaining Work

1. **Shared pinned residency**: Wiring `CudaExpertCache::insert_pinned` into execution path (infrastructure ready, not connected)
2. **Multi-token decode**: KV cache for position > 0, RoPE implementation
3. **LM head CUDA**: 374ms CPU dense GEMV → CUDA BF16 GEMV (~5ms)
4. **MoE load optimization**: Cache preload for routed experts, shared pinning
5. **WoA/WoB upload**: Single bulk upload instead of per-group sub-buffer copies

---

## Key Design Rules

1. **Never guess packed slicing** — refuse if metadata missing
2. **Load only required tensors** — no full checkpoint RAM expansion
3. **Validate everything** — shapes, dtypes, layer bounds, expert IDs
4. **CPU reference always** — every CUDA path has a CPU comparison
5. **Finite-only parity** — `parity_status` must be `valid_finite` before reporting cosine
