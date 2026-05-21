# CUDA Backend — objeta-cuda

Updated: 2026-05-21

---

## Overview

`objeta-cuda` is the CUDA execution backend for the objeta MoE runtime.
All kernels are compiled at runtime via NVRTC — no pre-built PTX, no static `.cu` files.

### What is implemented

| Component | Status | Notes |
|-----------|--------|-------|
| CUDA context + device query | ✓ | |
| Stream pool (async H2D/D2H) | ✓ | |
| Pinned host buffers | ✓ | |
| CUDA event timing | ✓ | |
| fp32 smoke GEMV | ✓ | |
| Q4\_0 quantized GEMV | ✓ | CPU + NVRTC |
| Q5\_0 quantized GEMV | ✓ | CPU + NVRTC |
| IQ3\_0 quantized GEMV | ✓ | CPU + NVRTC |
| Benchmark hygiene (warmup/iters/p50/p95) | ✓ | `bench_qgemv` binary |
| quant\_vs\_fp32 quality reporting | ✓ | cosine, L2, max-abs |
| Selected-expert MoE executor | ✓ | Q4\_0 experts |
| silu\_mul kernel | ✓ | NVRTC |
| weighted\_accum kernel | ✓ | NVRTC |
| MoE benchmark | ✓ | `bench_selected_moe` binary |
| VRAM expert cache (LRU) | ✓ | `CudaExpertCache` |
| Cache admission policy | ✓ | oversized bypass, self-eviction diagnostics |

### What is intentionally out of scope

- Attention kernels
- DeepSeek end-to-end execution
- Fused gate+up+down expert kernels
- GGUF / K-quant format support
- Full model loading

---

## Quantization Formats

All formats use `block_size = 32`.
Both CPU and CUDA implementations dequantize inside the block loop and accumulate in `fp32`.
The full matrix is **never** expanded to dense `fp32`.

### Q4\_0

```
block_bytes = 18
[2B fp16 scale][16B packed 4-bit values]

dequant: w = (q - 8) * scale
```

### Q5\_0

```
block_bytes = 22
[2B fp16 scale][4B high-bit mask][16B packed low nibbles]

dequant: w = (q - 16) * scale
```

### IQ3\_0

```
block_bytes = 14
[2B fp16 scale][8B packed 2-bit lower values][4B packed 1-bit higher values]

dequant: w = (q - 4) * scale
```

---

## Selected-Expert MoE

The MoE executor follows the standard selected-expert equation:

```
for each selected expert e:
  gate = W_gate[e] @ x
  up   = W_up[e] @ x
  act  = silu(gate) * up
  down = W_down[e] @ act
  out += selected_weight[e] * down
```

Each of gate/up/down uses the existing `QuantBackend::q4_gemv()` path.
`silu_mul` and `weighted_accum` are separate NVRTC kernels.
The gate and up projections are **not** fused yet.

### VRAM Expert Cache

`CudaExpertCache` provides LRU-evicted VRAM residency for quantized expert tensors.

```
Cache key: (layer_id, expert_id, tensor_kind, quant_format)
```

**Admission policy:**
- If a single tensor exceeds `capacity_bytes`: skip insertion (oversized tensor bypass)
- If gate+up+down total for one expert exceeds `capacity_bytes`: configurable bypass (oversized expert bypass)

**Tracked counters:**

| Counter | Description |
|---------|-------------|
| `hit_count` | Cache hits (tensor found in VRAM) |
| `miss_count` | Cache misses (H2D copy needed) |
| `eviction_count` | LRU evictions |
| `cache_insert_attempt_count` | Total insertion attempts |
| `cache_insert_accept_count` | Accepted insertions |
| `cache_insert_bypass_count` | All bypasses combined |
| `oversized_tensor_bypass_count` | Single-tensor oversized bypasses |
| `oversized_expert_bypass_count` | Whole-expert oversized bypasses |
| `self_eviction_risk_count` | Cases where expert set ≈ capacity |

**Tracked bytes:**

| Field | Description |
|-------|-------------|
| `capacity_bytes` | Hard VRAM limit |
| `resident_bytes` | Current resident bytes |
| `actual_expert_bytes_loaded` | Total H2D traffic |
| `resident_cache_bytes_reused` | Bytes served from cache (hits) |
| `bytes_by_tensor_kind.{gate,up,down}` | Breakdown by tensor kind |

---

## Benchmark Reference

### GEMV Benchmark (`bench_qgemv`)

```powershell
# Single run
cargo run -p objeta-cuda --bin bench_qgemv -- \
    --format q4 --rows 1024 --cols 4096 --seed 42

# Matrix mode (fixed shapes, JSONL output)
cargo run -p objeta-cuda --bin bench_qgemv -- \
    --format q5 --matrix --warmup 10 --iters 50

# With warmup and iteration control
cargo run -p objeta-cuda --bin bench_qgemv -- \
    --format iq3 --rows 4096 --cols 14336 --warmup 5 --iters 20
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `q4` | `q4` / `q5` / `iq3` |
| `--rows` | 128 | Matrix rows |
| `--cols` | 256 | Matrix columns |
| `--seed` | 42 | RNG seed |
| `--warmup` | 3 | Warmup iterations before timing |
| `--iters` | 10 | Timed iterations |
| `--matrix` | off | Fixed-shape matrix mode (JSONL) |

**Matrix mode shapes:**

| Shape | Use case |
|-------|---------|
| 4096 × 4096 | Standard hidden dim |
| 11008 × 4096 | Llama FFN up |
| 14336 × 4096 | DeepSeek FFN up |
| 4096 × 6144 | Compressed hidden |
| 14336 × 6144 | Large FFN |
| 4096 × 8192 | Wide hidden |
| 14336 × 8192 | Large FFN × wide hidden |

**Output fields:**

| Field | Description |
|-------|-------------|
| `backend` | `cuda` / `cpu` |
| `format` | `q4` / `q5` / `iq3` |
| `rows`, `cols` | Shape |
| `block_size`, `block_bytes` | Format metadata |
| `telemetry.h2d_ms` | avg/min/max/p50/p95 |
| `telemetry.kernel_ms` | avg/min/max/p50/p95 |
| `telemetry.d2h_ms` | avg/min/max/p50/p95 |
| `telemetry.total_ms` | avg/min/max/p50/p95 |
| `telemetry.unaccounted_ms` | total − h2d − kernel − d2h |
| `telemetry.bytes_read` | Quantized bytes transferred |
| `telemetry.effective_gbps` | Effective memory bandwidth |
| `cuda_vs_cpu_quant.cosine_similarity` | CUDA vs CPU-quant correctness |
| `cuda_vs_cpu_quant.relative_l2_error` | |
| `cuda_vs_cpu_quant.max_abs_error` | |
| `quant_vs_fp32.cosine_similarity` | Quantization quality vs dense fp32 |
| `quant_vs_fp32.relative_l2_error` | |
| `quant_vs_fp32.max_abs_error` | |

### MoE Benchmark (`bench_selected_moe`)

```powershell
cargo run -p objeta-cuda --bin bench_selected_moe -- \
    --format q4 --num-layers 4 --num-experts 8 --top-k 2 \
    --hidden 2048 --intermediate 1024 \
    --warmup 3 --iters 10 --seed 42
```

**Output fields:**

| Field | Description |
|-------|-------------|
| `timing.h2d_ms` | avg/min/max/p50/p95 |
| `timing.gate_up_qgemv_ms` | |
| `timing.activation_ms` | silu\_mul |
| `timing.down_qgemv_ms` | |
| `timing.accum_ms` | weighted\_accum |
| `timing.unaccounted_ms` | |
| `timing.total_ms` | |
| `bytes.expert_quantized` | Per-expert quantized bytes |
| `bytes.total_h2d` | Total H2D traffic |
| `cuda_vs_cpu_quant` | cosine / L2 / max-abs |
| `quant_vs_fp32` | cosine / L2 / max-abs |
| `cache_stats` | hit/miss/eviction counts (if cache enabled) |

---

## Running Tests

```powershell
cargo test -p objeta-cuda
```

**Test coverage:**

| Test | Description |
|------|-------------|
| `test_q4_gemv_small` | rows=16, cols=32 |
| `test_q4_gemv_medium` | rows=128, cols=256 |
| `test_q4_gemv_large` | rows=1024, cols=4096 |
| `test_q4_seed_sweep` | Deterministic multi-seed sweep |
| `test_q4_invalid_cols_rejection` | Non-multiple-of-32 rejection |
| `test_q5_gemv_*` | Same shapes for Q5\_0 |
| `test_q5_quality_vs_q4` | Q5 cosine ≥ Q4 cosine |
| `test_iq3_gemv_*` | Same shapes for IQ3\_0 |
| `test_selected_moe_cuda_vs_cpu` | MoE CUDA vs CPU-quant correctness |
| `test_vram_cache_basic` | Hit/miss/eviction byte invariants |
| `test_cache_admission_oversized` | Oversized tensor bypass |

---

## Next Steps

The intended development order is:

1. Keep `QuantBackend` / telemetry / test harness stable
2. Extend MoE executor to Q5\_0 and IQ3\_0 experts
3. Profile and optimize common kernel bottlenecks across formats
4. Add fused gate+up expert kernel
5. Integrate real DeepSeek V4 Flash weights via `objeta-parser`
