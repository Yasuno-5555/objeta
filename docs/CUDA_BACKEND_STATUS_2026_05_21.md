# CUDA Backend Status

2026-05-21

## Summary

`objeta-cuda` now has a working CUDA backend skeleton plus quantized GEMV correctness paths for `Q4_0` and `Q5_0`.

This is still intentionally narrow:

- no DeepSeek integration
- no attention kernels
- no selected-expert MoE executor
- no large model residency path

The current goal is backend bring-up and quant format validation, not end-to-end model execution.

## Landed

- CUDA device discovery and context initialization
- stream pool and async H2D / D2H copy
- pinned host buffers
- CUDA event timing
- smoke `fp32` GEMV
- `Q4_0` quantized GEMV v0
- `Q5_0` quantized GEMV v0
- shared benchmark helper with JSON telemetry

## Current Quant Formats

### `Q4_0`

- `block_size = 32`
- `block_bytes = 18`
- layout:
  - `2` bytes scale (`fp16`)
  - `16` bytes packed 4-bit values

### `Q5_0`

- `block_size = 32`
- `block_bytes = 22`
- layout:
  - `2` bytes scale (`fp16`)
  - `4` bytes high-bit mask
  - `16` bytes packed low nibbles

### `IQ3_0`

- `block_size = 32`
- `block_bytes = 14`
- layout:
  - `2` bytes scale (`fp16`)
  - `8` bytes packed 2-bit lower values
  - `4` bytes packed 1-bit higher values

Both CPU and CUDA paths:

- dequantize inside block loops
- accumulate in `fp32`
- avoid full dense `fp32` matrix expansion

## Validation

Correctness coverage currently includes:

- `rows=16, cols=32`
- `rows=128, cols=256`
- `rows=1024, cols=4096`
- deterministic seed sweeps for `q4` and `q5`
- `q4` invalid-column rejection
- `q5` quality check against `q4` on the same random case

Reported metrics:

- cosine similarity
- relative L2 error
- max absolute error
- `H2D` time
- kernel time
- `D2H` time
- total time
- bytes read
- effective GB/s

## Commands

```powershell
cargo test -p objeta-cuda
cargo run -p objeta-cuda --bin bench_qgemv -- --format q4 --rows 128 --cols 256 --seed 123
cargo run -p objeta-cuda --bin bench_qgemv -- --format q5 --rows 128 --cols 256 --seed 123
cargo run -p objeta-cuda --bin bench_qgemv -- --format iq3 --rows 128 --cols 256 --seed 123
```

## Next Step

Do not optimize `q4` in isolation yet.

The intended order is:

1. keep the shared quant API / telemetry / harness stable
2. only then optimize common kernel bottlenecks across `q4` / `q5` / `iq3`
