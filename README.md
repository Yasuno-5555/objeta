# objeta — MoE Runtime OS for LLM Inference

**Quantized GEMV, selected-expert MoE, FP4 decode, VRAM expert cache, and DeepSeek V4 Flash single-layer proof — pure Rust + CUDA NVRTC.**

---

## Overview

`objeta` is a Rust workspace that implements the execution backend for Mixture-of-Experts (MoE) LLM inference.
The design principle is **observe → classify → allocate → execute**: the runtime decides per-layer, per-token what precision, which experts, and how much VRAM to allocate, rather than running the model statically.

Current implementation focus is the CUDA backend (`objeta-cuda`) and the DeepSeek V4 Flash metadata pipeline (`objeta-parser` + `objeta-cli`).

---

## Quick Start

```powershell
# Build all crates
cargo build

# Run the CUDA quantized GEMV benchmark
cargo run -p objeta-cuda --bin bench_qgemv -- --format q4 --rows 1024 --cols 4096

# Run the selected-expert MoE benchmark
cargo run -p objeta-cuda --bin bench_selected_moe -- --format q4 --num-layers 4 --num-experts 8 --top-k 2

# Run real DeepSeek V4 Flash single-layer MoE proof
cargo run -p objeta-cuda --bin deepseek_single_layer_moe -- \
  --model <MODEL_DIR> --layer-id 27 --expert-ids 0,3,7

# Parse a DeepSeek V4 Flash checkpoint directory
cargo run -p objeta-cli -- parse-deepseek <MODEL_DIR> -o <OUTPUT_DIR>

# Run the inventory sanity report
cargo run -p objeta-cli -- sanity-report <OUTPUT_DIR> -o sanity_report.json

# Run all tests
cargo test
```

---

## Crate Map

| Crate | Purpose |
|-------|---------|
| [`objeta-core`](crates/objeta-core) | Shared types: `Phase`, `LayerZone`, `ObjetaError`, `Result` |
| [`objeta-cuda`](crates/objeta-cuda) | CUDA GEMV kernels (Q4\_0, Q5\_0, IQ3\_0), selected-expert MoE, VRAM expert cache |
| [`objeta-parser`](crates/objeta-parser) | Safetensors mmap loader, DeepSeek V4 Flash metadata parser, FP4 E2M1FN decode, sanity reporter |
| [`objeta-analysis`](crates/objeta-analysis) | Static geometry analysis: SVD rank, Lyapunov, intra-cos, phase classification |
| [`objeta-phase`](crates/objeta-phase) | Phase / family classification from geometry metrics |
| [`objeta-routing`](crates/objeta-routing) | Per-layer precision routing from phase profiles |
| [`objeta-quantize`](crates/objeta-quantize) | Phase-adaptive quantization plan generation |
| [`objeta-moe`](crates/objeta-moe) | MoE routing compiler: occupancy histograms, transition matrices, execution plans |
| [`objeta-os`](crates/objeta-os) | Reflexive runtime OS: token-class observation, allocation, fault recovery |
| [`objeta-runtime`](crates/objeta-runtime) | Runtime profiling and phase-aware execution policy |
| [`objeta-aot`](crates/objeta-aot) | Ahead-of-time compilation pack for runtime plans |
| [`objeta-metal`](crates/objeta-metal) | Metal GPU kernels for Apple Silicon (M1/M2) |
| [`objeta-ssd`](crates/objeta-ssd) | SSD-resident expert weight paging (mmap, async I/O) |
| [`objeta-qwen36-executor`](crates/objeta-qwen36-executor) | Qwen3.6-35B-A3B full forward pass executor (C ABI, Python ctypes) |
| [`objeta-cli`](crates/objeta-cli) | CLI: `analyze`, `moe-analyze`, `strategy`, `quantize`, `parse-deepseek`, `sanity-report` |

---

## CUDA Backend (`objeta-cuda`)

The CUDA backend is the primary active development target.

### Quantization Formats

All three formats dequantize inside block loops and accumulate in `fp32`.
No path expands the full matrix to dense `fp32`.

| Format | block\_size | block\_bytes | Layout |
|--------|-------------|--------------|--------|
| `Q4_0` | 32 | 18 | 2B scale (fp16) + 16B packed 4-bit |
| `Q5_0` | 32 | 22 | 2B scale (fp16) + 4B high-bit mask + 16B packed nibbles |
| `IQ3_0` | 32 | 14 | 2B scale (fp16) + 8B packed 2-bit + 4B packed 1-bit |

### Components Implemented

| Component | Status |
|-----------|--------|
| CUDA context + stream pool | ✓ |
| H2D / D2H async copy | ✓ |
| Pinned host buffers | ✓ |
| CUDA event timing | ✓ |
| fp32 smoke GEMV | ✓ |
| Q4\_0 GEMV (CPU + CUDA NVRTC) | ✓ |
| Q5\_0 GEMV (CPU + CUDA NVRTC) | ✓ |
| IQ3\_0 GEMV (CPU + CUDA NVRTC) | ✓ |
| Benchmark hygiene (warmup, iters, p50/p95) | ✓ |
| Selected-expert MoE executor (Q4\_0) | ✓ |
| VRAM expert cache (LRU + admission policy) | ✓ |
| MoE benchmark + quant\_vs\_fp32 quality | ✓ |
| DeepSeek single-layer MoE proof (real checkpoint) | ✓ |
| DeepSeek FP4 E2M1FN → FP32 CPU decode | ✓ |
| Manual-expert mode (bypass router) | ✓ |

### Key API

```rust
// Per-format metadata
QuantFormat::Q4_0.block_bytes()        // 18
QuantFormat::Q5_0.kernel_name()        // "q5_gemv"

// CPU reference paths
q4_quantize_matrix_cpu(&weights, rows, cols) -> Vec<u8>
q4_gemv_cpu(&quantized, &vector, rows, cols) -> Vec<f32>

// CUDA NVRTC backend
let backend = QuantBackend::new(CudaContext::new(0)?)?;
backend.compile_format(QuantFormat::Q4_0)?;
let result = backend.gemv(&quantized, &x, shape, &stream)?;

// Selected-expert MoE
let executor = MoeExecutor::new(backend, num_experts, hidden_size, intermediate_size)?;
executor.execute_selected_moe_cuda(&hidden, &expert_weights, &selected_experts)?;

// VRAM cache
let cache = CudaExpertCache::new(capacity_bytes);
cache.get_or_insert(layer_id, expert_id, TensorKind::Gate, format, load_fn)?;
```

Full documentation: [`crates/objeta-cuda/README.md`](crates/objeta-cuda/README.md)

---

## DeepSeek V4 Flash Metadata Pipeline

Parse checkpoint metadata without loading tensor payloads, then run a VRAM-planning sanity report.

```powershell
# Step 1: parse metadata headers only
cargo run -p objeta-cli -- parse-deepseek <MODEL_DIR> -o output/

# Step 2: sanity report + VRAM estimates
cargo run -p objeta-cli -- sanity-report output/ -o sanity_report.json
```

**`parse-deepseek`** emits five JSON files:

| File | Contents |
|------|----------|
| `deepseek_v4_flash_layout.json` | num\_layers, num\_experts, top\_k, hidden\_size, dtype |
| `deepseek_v4_flash_tensor_index.json` | All tensor names, shapes, dtypes, byte offsets |
| `deepseek_v4_flash_expert_layout.json` | Expert tensor classification (gate / up / down / gate\_up / shared) |
| `deepseek_v4_flash_router_layout.json` | Router tensor classification per layer |
| `deepseek_v4_flash_inventory_summary.json` | Total bytes, per-layer bytes, per-expert bytes, cache-fit flags |

**`sanity-report`** checks:

- Layout kind compatible with objeta-cuda MoE (`explicit_experts` / `packed_experts` / `unknown`)
- Working set estimates: single expert, current layer, 2-layer prefetch, 4-layer prefetch, full forward pass
- Cache fit: 1 GB / 2 GB / 4 GB / 8 GB
- Warnings: missing routers, unknown layout, packed without slicing, oversized experts, missing top\_k

Full documentation: [`crates/objeta-parser/README.md`](crates/objeta-parser/README.md)

---

## Analysis Pipeline (`objeta-cli analyze`)

For weight geometry analysis and phase-adaptive quantization:

```powershell
# 1. Analyze weight geometry
cargo run -p objeta-cli -- analyze <MODEL_DIR> -o phase_profile.json --stability

# 2. Generate quantization plan
cargo run -p objeta-cli -- quantize phase_profile.json -o quantization_plan.json --target-avg-bits 4.0

# 3. Generate MoE routing plan
cargo run -p objeta-cli -- moe-analyze <MODEL_DIR> -o execution_plan.json
```

---

## Architecture

```
observe → classify → allocate → execute
    ↑                        ↓
    └──── TokenTrace ← replay ┘

objeta-analysis  →  phase_profile.json
                  ↓
objeta-phase     →  LayerZone (STABLE / STEERING / TRANSITION)
                  ↓
objeta-routing   →  per-layer precision assignment
                  ↓
objeta-quantize  →  quantization_plan.json
                  ↓
objeta-cuda      →  CUDA GEMV / MoE executor / VRAM cache
                  ↓
objeta-parser    →  checkpoint metadata + VRAM planning
```

Core principle: **LLM inference is not static computation. It is adaptive dynamical resource allocation.**

---

## Docs

| Document | Purpose |
|----------|---------|
| [`docs/CURRENT_STATUS_2026_05_22.md`](docs/CURRENT_STATUS_2026_05_22.md) | **Current project status, all tracks, test coverage** |
| [`docs/M9_DEEPSEEK_FLASH_SINGLE_LAYER_MOE.md`](docs/M9_DEEPSEEK_FLASH_SINGLE_LAYER_MOE.md) | DeepSeek single-layer MoE proof design spec |
| [`docs/DEEPSEEK_FP4_DECODE.md`](docs/DEEPSEEK_FP4_DECODE.md) | DeepSeek FP4 E2M1FN + F8\_E8M0 decode semantics |
| [`docs/CUDA_BACKEND.md`](docs/CUDA_BACKEND.md) | CUDA backend: quant formats, MoE, VRAM cache |
| [`docs/CUDA_BACKEND_STATUS_2026_05_21.md`](docs/CUDA_BACKEND_STATUS_2026_05_21.md) | CUDA backend bring-up status snapshot |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture design: memory layout, performance breakdown |
| [`docs/ARCHITECTURE_BOUNDARIES.md`](docs/ARCHITECTURE_BOUNDARIES.md) | Module boundary spec for qwen36-executor |

---

## Requirements

- Rust 1.75+
- CUDA 12.x (for `objeta-cuda`)
- Windows with MSVC toolchain (primary dev platform)
- `nvcc` / NVRTC on PATH

---

## License

MIT
