# objeta — MoE Runtime Compiler

**Qwen3.6-35B-A3B on M1 8GB — 0.21 tok/s, pure Rust.**

Rust-native executor with NEON SIMD, f16 weights, mmap embedding, Metal GPU kernels.
DeltaNet verified against HuggingFace reference (cos=1.000000).

## Quick Start

```bash
# Build
cargo build --release

# Build Metal kernels
bash experiments/build_metal.sh

# Run Qwen3.6 generation
python3 experiments/qwen36_full_rust.py
```

## Performance

| Stage | tok/s | Key change |
|-------|-------|------------|
| Python MLX | 0.03 | Baseline |
| Rust f32 weights | 0.04 | Python eliminated, but SWAP |
| **Rust f16 weights** | **0.21** | SWAP eliminated (3.5GB RAM) |

## Architecture

```
Qwen36Runner (Rust, ~3.5GB RAM)
├── embed: mmap (2GB, zero-copy)
├── weights: Vec<u16> f16 (2.9GB)
├── MoE: mmap q4 (SSD, pre-loaded per layer)
│
├── forward 40 layers
│   ├── 30× DeltaNet (verified cos=1.0 vs HF)
│   ├── 10× GQA (fused QKV+RoPE+attention)
│   ├── 40× shared expert
│   └── 40× MoE dispatch
│
└── lm_head: NEON+rayon, 509M FLOPs in ~50ms
```

## Crate Map

| Crate | Purpose |
|-------|---------|
| `objeta-qwen36-executor` | Rust executor (NEON GEMV, DeltaNet, GQA, MoE, Metal) |
| `objeta-core` | Shared types |
| `objeta-parser` | Safetensors mmap loader |
| `objeta-analysis` | Static geometry analysis |
| `objeta-moe` | MoE routing analysis (Rust-native) |
| `objeta-cli` | CLI (analyze, moe-analyze) |

## Key Findings

- **DeltaNet conv1d**: PyTorch uses cross-correlation. `weight[:,3]` = newest input. Fixed.
- **q_gate**: 4096 elements (1 per attention dim), not 256.
- **SWAP**: f32 weights (5.8GB) + embed (2GB) = 7.8GB > 8GB. f16 solves it.
- **NEON GEMV**: 23 GFLOPS f32, ~10 GFLOPS f16. 1.9x NumPy.
- **Metal fused GQA**: cos=0.9999. Right for seq_len > 32.
