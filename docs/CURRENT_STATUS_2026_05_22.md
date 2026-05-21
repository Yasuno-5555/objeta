# Current Status — 2026-05-22

## TL;DR

The project has two active tracks converging:

| Track | Status | Key Artifact |
|---|---|---|
| **Qwen3.6 Executor** (Apple M1) | Calibration at 66.22%, specialization packs emitting, no pruning yet | `objeta-qwen36-executor` |
| **DeepSeek V4 Flash** (NVIDIA CUDA) | Single-layer MoE proof working, FP4 decode implemented, real checkpoint run possible | `deepseek_single_layer_moe` |

---

## Project Structure

```
crates/
  objeta-core/          Shared types, errors, Result alias
  objeta-parser/        Safetensors mmap loader + DeepSeek metadata parser
    src/
      lib.rs            ModelWeights (lazy mmap), Dtype, ModelConfig
      deepseek.rs       DeepSeek V4 Flash layout parsing, FP4 decode
      sanity.rs         Sanity report generation
  objeta-cuda/          CUDA backend + MoE executor
    src/
      lib.rs            CudaBackend, CudaExpertCache, quant API
      moe.rs            selected_moe_cpu_fp32, ExpertWeights, comparison
      quant.rs          Q4_0/Q5_0/IQ3 quant + CUDA GEMV kernels
      context.rs        CUDA context init
      stream.rs         Stream pool
      memory.rs         Pinned host buffers
      telemetry.rs      JSON event timing
      ffi.rs            Raw CUDA driver bindings
      attention.rs      (stub)
    src/bin/
      bench_qgemv.rs    Quantized GEMV benchmark harness
      bench_selected_moe.rs  Synthetic MoE benchmark
      deepseek_single_layer_moe.rs  ★ Real DeepSeek single-layer MoE proof
  objeta-cli/           CLI entry point (clap)
  objeta-analysis/      SVD-based geometry analysis
  objeta-aot/           AOT specialization pipeline (calibration→pack→verify)
  objeta-quantize/      Quantization plan generation
  objeta-routing/       Routing zone classification
  objeta-runtime/       Runtime strategy config generation
  objeta-moe/           MoE routing analysis
  objeta-qwen36-executor/ Qwen3.6 full executor (M1 Metal)
  objeta-os/            OS-level profile structures
  objeta-phase/         Phase classification
  objeta-ssd/           SSD offload path (stub)
  objeta-metal/         Metal backend (stub)
```

### CLI Commands

```
objeta analyze <path>              Static geometry analysis
objeta moe-analyze <path>          MoE routing analysis for Qwen3.6
objeta strategy <profile>          Family-aware runtime strategy
objeta quantize <profile>          Phase-adaptive quantization plan
objeta parse-deepseek <model_dir>  Parse DeepSeek V4 Flash metadata
objeta sanity-report <input_dir>   Inventory sanity report
```

Binaries:
```
deepseek_single_layer_moe  Real DeepSeek single-layer MoE proof
bench_qgemv                Quantized GEMV benchmark
bench_selected_moe         Synthetic MoE benchmark
```

---

## DeepSeek V4 Flash Integration (m9 Milestones)

### Completed

| Milestone | Description | Tests |
|---|---|---|
| **m9.0** | Metadata parser: layout JSON, expert layout, router layout, tensor index, inventory summary | 16 |
| **m9.1** | FFN naming fix: `mlp.experts.{N}.{gate,up,down}_proj` pattern resolved. `shared_experts` split from routed experts. | — |
| **m9.2** | FP4 metadata inventory: `storage_dtype`, `logical_dtype`, `scale_tensor_name`, `scale_dtype`, `logical_shape`, `block_size`, `packed_values_per_byte` fields added to expert layout. | — |
| **m9.3** | FP4 decode: E2M1FN lookup table, F8_E8M0 scale decode, `decode_deepseek_fp4_to_f32()` CPU function, synthetic fixture test, sanity report updated. | 2 |
| **m9.4** | Manual-expert mode: `--expert-ids`/`--expert-weights` CLI, FP4 I8+scale loading via `get_raw()`, decode-to-FP32 → Q4_0 → CUDA MoE pipeline, JSON report, synthetic lifecycle tests. | 6 |
| **m9.5+** | (Not started) Hash routing, multi-layer, attention, generation. | 0 |

### m9 Test Coverage (14 bin tests)

| Test | What it validates |
|---|---|
| `test_explicit_single_layer_works` | Router mode end-to-end with synthetic tensors |
| `test_cpu_router_top_k_selection` | Router top-k selection correctness |
| `test_manual_expert_ids_skip_router` | CLI parsing: `--expert-ids` |
| `test_uniform_expert_weights_sum_to_one` | Default weight normalization |
| `test_explicit_expert_weights_validation` | Explicit `--expert-weights` validation |
| `test_invalid_expert_id_fails` | Out-of-range expert rejection |
| `test_missing_scale_tensor_fails` | FP4 metadata gap detection |
| `test_fp4_decode_flow_with_synthetic_tensors` | Full I8→FP32 decode pipeline |
| `test_router_shape_mismatch_fails` | Shape validation |
| `test_missing_router_fails_clearly` | Missing router tensor |
| `test_missing_expert_tensor_fails_clearly` | Missing expert tensor |
| `test_unsupported_dtype_fails` | Unsupported dtype rejection |
| `test_packed_experts_refuses_to_run` | Packed layout rejection |
| `test_layer_out_of_range_fails` | Layer bounds check |

### FP4 Decode Semantics

Confirmed from DeepSeek V4 Flash inference source code:
- **Format**: E2M1FN (1 sign, 2 exponent, 1 mantissa, bias=1)
- **Packing**: 2 FP4 values per I8 byte, low nibble first
- **Scale**: F8_E8M0 (unsigned 8-bit exponent, `value = 2^(raw - 127)`)
- **Block size**: 32 logical FP4 elements per scale value
- **Matrix layout**: Physical `[out, in//2]` I8 → Logical `[out, in]` FP32

Full spec: [DEEPSEEK_FP4_DECODE.md](DEEPSEEK_FP4_DECODE.md)

### Real Checkpoint Run

Possible now with:
```bash
objeta-cli deepseek-single-layer-moe \
  --model ../DeepSeek-V4-Flash \
  --layer-id 27 \
  --expert-ids 0,3,7 \
  --expert-weights 0.2,0.5,0.3
```

The binary auto-detects FP4 vs BF16 storage and branches accordingly.

---

## CUDA Backend

### Quantized GEMV

| Format | Block | Bytes/Block | CUDA Kernel | CPU Reference | Tests |
|---|---|---|---|---|---|
| Q4_0 | 32 | 18 | Yes | Yes | 5+ |
| Q5_0 | 32 | 22 | Yes | Yes | 3+ |
| IQ3_0 | 32 | 14 | Yes | Yes | 1+ |

### Validated
- CUDA device discovery, context init, stream pool
- Pinned host buffers, async H2D/D2H
- CUDA event timing with JSON telemetry
- Correctness: cosine similarity, relative L2, max abs error
- Shapes: 16×32, 128×256, 1024×4096

### Not Yet
- Attention kernels
- Fused MoE kernel
- Large model residency
- Multi-GPU

Full status: [CUDA_BACKEND_STATUS_2026_05_21.md](CUDA_BACKEND_STATUS_2026_05_21.md)

---

## Qwen3.6 Executor (M1)

### State
- `safe_exact` correctness baseline
- Runtime pack loading from AOT specialization packs
- Importance-aware eviction in expert cache
- Governor modes: disabled / observe-only / safety-only / offensive-v0
- `moe_io_events` include `selected_weights` for AOT calibration

### Calibration Coverage
- **66.22%** (6,781 / 10,240 logical experts)
- 9,560 events from 5 prompts
- Per-layer: 54.69% (min) → 86.72% (max), median 66.02%

### Specialization Packs
- **M1 8GB conservative**: 6,781 experts, q8:40 q5:1,131 q4:5,650, 0% routing loss
- **RTX3070 8GB/32GB**: 6,781 experts, q8:40 q5:1,131 q4:3,095 iq3:2,555, 0% routing loss

### Constraints
- Coverage below pruning gate → `compress=0`, `prune_candidate=0`
- Reports still advisory (`estimated_only=yes`, `requires_verification=yes`)

---

## Test Coverage Summary

| Crate | Tests |
|---|---|
| objeta-parser | 18 (16 lib + 2 fp4) |
| objeta-cuda (lib) | 15 |
| objeta-cuda (deepseek_single_layer_moe) | 14 |
| objeta-cuda (bench_qgemv) | 4 |
| objeta-cuda (bench_selected_moe) | 1 |
| objeta-aot | 4 |
| objeta-quantize | 3 |
| objeta-analysis | 2 |
| objeta-qwen36-executor | 5 |
| **Total** | **~66** |

---

## Key Design Rules (from M9 spec)

1. **Never guess packed slicing** — refuse if metadata missing
2. **Load only required tensors** — no full checkpoint RAM expansion
3. **Validate everything** — shapes, dtypes, layer bounds, expert IDs
4. **CPU reference always** — every CUDA path has a CPU comparison
5. **Deterministic hidden input** — seeded synthetic, no real activations needed for proof

---

## Next Steps

1. **Run real checkpoint** — `deepseek_single_layer_moe` against actual DeepSeek V4 Flash model
2. **Hash routing** (m9.5) — deterministic expert selection from token hash
3. **Attention integration** (m9.6+) — full single-layer forward
4. **Calibration expansion** — more routing-diverse prompts for Qwen3.6 to cross pruning gate
5. **CUDA kernel optimization** — fused MoE kernel, shared quant API

---

## Superseded Docs

The following docs are snapshots from earlier dates and are superseded by this document:
- `CURRENT_STATUS_2026_05_18.md`
- `CURRENT_STATUS_2026_05_20.md`
- `CURRENT_STATUS_2026_05_21.md`

Keep: `CUDA_BACKEND_STATUS_2026_05_21.md`, `DEEPSEEK_FP4_DECODE.md`, `M9_DEEPSEEK_FLASH_SINGLE_LAYER_MOE.md`, `DESIGN.md`, architecture docs.
