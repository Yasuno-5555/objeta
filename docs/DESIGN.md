# objeta — Design Document v1.0

## Status: Qwen3.6-35B-A3B running on M1 8GB at 0.21 tok/s

## Pipeline

```
safetensors / .bin weights
    │
    ├── objeta analyze ──► phase_profile.json + stability_map.json
    ├── objeta moe-analyze ──► execution_plan.json
    │
    └── Rust executor (objeta-qwen36-executor)
         ├── NEON+rayon GEMV (23 GFLOPS f32, 10 GFLOPS f16)
         ├── 40-layer forward pass (DeltaNet + GQA + shared + MoE)
         ├── lm_head + top-k (all in Rust)
         ├── Metal GPU kernels (8 kernels)
         └── C API → Python ctypes
```

## Memory Layout (3.5GB on M1 8GB)

| Component | Size | Format | Storage |
|-----------|------|--------|---------|
| Embedding | 2.0GB | f32 | mmap (OS page cache) |
| Attention weights | 2.9GB | f16 (Vec<u16>) | RAM |
| MoE experts | 20GB | q4 | mmap (SSD) |
| KV caches | ~64KB | f32 | RAM |
| DeltaNet states | ~5MB | f32 | RAM |
| Scratch buffers | ~40KB | f32 | RAM |

## Verified Correctness (2026-05-19)

| Component | Method | Result | Status |
|-----------|--------|--------|--------|
| DeltaNet L0 attention | HF `Qwen3_5MoeGatedDeltaNet` | cos=0.99999 | ✓ |
| L0 DecoderLayer | HF `Qwen3_5MoeDecoderLayer` | cos=0.99975 | ✓ |
| All bin weights | HF safetensors | cos≈1.0 | ✓ |
| Rust GEMV | NumPy reference | cos=1.000000 | ✓ |
| MoE dispatch (Q4) | HF full-precision (L0) | cos≈0.989 | ⚠ |
| GQA attention (CPU) | HF reference | not yet verified | ? |
| 40-layer generation | HF expected output | **broken** | ✗ |

## Key Fixes

1. **RMSNorm `1+w` convention** (2026-05-19): `Qwen3_5MoeRMSNorm` uses `output * (1.0 + weight)`. Rust was using `output * weight` directly, causing input/post/final norm to be wrong by up to 33x. Also fixed in GQA q_norm/k_norm and Metal kernel.
2. **DeltaNet conv1d order**: PyTorch cross-correlation → `weight[3]` = newest input
2. **q_gate dimension**: 4096 (1 per dim), not 256
3. **Metal dispatchThreads → dispatchThreadgroups**: grid was in threads, needed threadgroups
4. **SWAP elimination**: f16 weights (2.9GB) + mmap embed instead of f32 (7.8GB)

## Performance Breakdown (per token, ~5s)

| Component | Time | % |
|-----------|------|---|
| DeltaNet (30 layers) | ~2.5s | 50% |
| GQA (10 layers) | ~0.5s | 10% |
| Shared expert (40 layers) | ~0.5s | 10% |
| MoE dispatch (40 layers) | ~1.0s | 20% |
| lm_head + sampling | ~0.5s | 10% |

## Next Milestones

1. **Output quality**: fix Q4 MoE quantization / GQA attention to match HF → working generation
2. **Speed to 1 tok/s**: attention q4 quantization, fused Metal for long seq, paged KV
3. **Speed to 3 tok/s**: speculative decoding (phase-aware), persistent Metal graphs
4. **Ecosystem**: OpenAI API, standard benchmarks
