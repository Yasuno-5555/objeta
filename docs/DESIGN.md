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

## Verified Correctness

| Component | Method | Result |
|-----------|--------|--------|
| DeltaNet | HF reference comparison | cos=1.000000 (all intermediates) |
| GQA attention | Python reference | cos=0.9999 (Metal fused kernel) |
| Rust GEMV | NumPy reference | cos=1.000000 |
| MoE dispatch | Python reference | cos=1.000000 (weight test) |
| Python vs Rust executor | Full forward pass | **identical output** |

## Key Fixes

1. **DeltaNet conv1d order**: PyTorch cross-correlation → `weight[3]` = newest input
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

1. **Output quality**: system prompt for English, tokenizer verification
2. **Speed to 1 tok/s**: attention q4 quantization, fused Metal for long seq, paged KV
3. **Speed to 3 tok/s**: speculative decoding (phase-aware), persistent Metal graphs
4. **Ecosystem**: OpenAI API, standard benchmarks
