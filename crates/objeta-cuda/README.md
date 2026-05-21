# objeta-cuda

`objeta-cuda` is the CUDA backend scaffold for Objeta runtime execution.

Current scope:

- CUDA device discovery and context initialization
- stream pool and async H2D / D2H copy
- pinned host buffers and device buffers
- CUDA event timing
- smoke `fp32` GEMV
- quantized `GEMV` correctness paths for `Q4_0` and `Q5_0`

Not in scope yet:

- DeepSeek integration
- attention kernels
- selected-expert MoE execution
- large model loading
- `q5_k` or fused expert kernels

## Quant Formats

The current formats are explicit and backend-local.

### `Q4_0`

- `block_size = 32`
- `block_bytes = 18`
- layout:
  - `2` bytes: `fp16` scale
  - `16` bytes: packed 4-bit values
- dequant:
  - `w = (q - 8) * scale`

### `Q5_0`

- `block_size = 32`
- `block_bytes = 22`
- layout:
  - `2` bytes: `fp16` scale
  - `4` bytes: per-value high-bit mask
  - `16` bytes: packed low nibbles
- dequant:
  - `w = (q - 16) * scale`

### `IQ3_0`

- `block_size = 32`
- `block_bytes = 14`
- layout:
  - `2` bytes: `fp16` scale
  - `8` bytes: packed 2-bit lower values
  - `4` bytes: packed 1-bit higher values
- dequant:
  - `w = (q - 4) * scale`

Both CPU and CUDA implementations dequantize inside block loops and accumulate in `fp32`.
Neither path expands the full matrix to dense `fp32`.

## API

Core types live in [src/quant.rs](./src/quant.rs):

- `QuantFormat`
- `QGemvShape`
- `QGemvTelemetry`
- `QGemvNumerics`
- `QuantBackend`

Useful entry points:

- `quantize_matrix_cpu()`
- `gemv_cpu()`
- `q4_quantize_matrix_cpu()`
- `q5_quantize_matrix_cpu()`
- `iq3_quantize_matrix_cpu()`
- `q4_gemv_cpu()`
- `q5_gemv_cpu()`
- `iq3_gemv_cpu()`
- `QuantBackend::gemv()`
- `QuantBackend::q4_gemv()`
- `QuantBackend::q5_gemv()`

## Tests

Run all backend tests:

```powershell
cargo test -p objeta-cuda
```

The quant correctness suite currently covers:

- `rows=16, cols=32`
- `rows=128, cols=256`
- `rows=1024, cols=4096`
- seed sweep for `q4` and `q5`
- `q4` non-multiple-of-32 rejection
- `q5` quality no worse than `q4` on the same random case

Reported numerical metrics:

- cosine similarity
- relative L2 error
- max absolute error

## Benchmark Helper

Use the benchmark helper to emit JSON telemetry and numerical metrics:

```powershell
cargo run -p objeta-cuda --bin bench_qgemv -- --format q4 --rows 128 --cols 256 --seed 123
cargo run -p objeta-cuda --bin bench_qgemv -- --format q5 --rows 128 --cols 256 --seed 123
cargo run -p objeta-cuda --bin bench_qgemv -- --format iq3 --rows 128 --cols 256 --seed 123
```

JSON fields:

- `backend`
- `kernel`
- `format`
- `rows`
- `cols`
- `block_size`
- `block_bytes`
- `telemetry.h2d_ms`
- `telemetry.kernel_ms`
- `telemetry.d2h_ms`
- `telemetry.total_ms`
- `telemetry.bytes_read`
- `telemetry.effective_gbps`
- `numerics.cosine_similarity`
- `numerics.relative_l2_error`
- `numerics.max_abs_error`

## Next Step

The intended follow-up order is:

1. keep the shared `QuantBackend` / telemetry / test harness stable
2. add `Q5_K` or `IQ3` as separate explicit formats
3. only then optimize common bottlenecks across `q4` / `q5` / `iq3`
