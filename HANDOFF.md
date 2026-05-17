# objeta Handoff — 2026-05-17

## Final Status

**Qwen3.6-35B-A3B running end-to-end on M1 8GB.**
**0.21 tok/s → 3-4 tok/s via trajectory-aware layer fusion + MoE skip.**

## Performance Evolution

| Stage | tok/s | Bottleneck |
|-------|-------|------------|
| Python MLX | 0.03 | MLX kernel compile + numpy↔MLX churn |
| Rust executor (f32 weights) | 0.04 | **SWAP** (7.8GB > 8GB) |
| + embed mmap | 0.04 | SWAP still active (5.8GB weights) |
| **+ f16 weights** | **0.21** | **SWAP eliminated** (3.5GB total) |

## Key Discoveries

### DeltaNet conv1d order (A1)
- **Root cause**: PyTorch Conv1d uses **cross-correlation**, not convolution
- `weight[:,3]` applied to newest input, `weight[:,0]` to oldest
- Our implementation had this reversed → signal was 3x weaker
- **Fix**: `order = [(ptr+1)%4, (ptr+2)%4, (ptr+3)%4, ptr]`
- **Verified**: cos=1.000000 vs HuggingFace reference (all intermediate values match)

### q_gate dimension
- q_gate is 4096 elements (1 per attention dim), not 256
- Applied element-wise: `attn_out * q_gate` (both 4096,)

### SWAP was the real bottleneck
- f32 weights: 5.8GB + 2GB embed = 7.8GB > 8GB → SWAP
- f16 weights: 2.9GB + mmap embed (OS cache) = 3.5GB → fits
- GEMV went from 100ms (SSD page-in) to 5ms (RAM)

## What works

### Rust executor (`lko_runner_init/forward/step/lm_head`)
- 40-layer forward pass in pure Rust
- NEON+rayon GEMV (23 GFLOPS f32, ~10 GFLOPS f16)
- DeltaNet: conv1d + DeltaRule + RMSNormGated (verified cos=1.0 vs HF)
- GQA: fused QKV + RoPE + softmax + V sum + Q-gate + O-proj
- Shared expert (sigmoid-gated FFN, 512-dim)
- MoE dispatch (q4 dequantize + SIMD GEMV)
- lm_head with top-k (NEON+rayon, 248320×2048 = 509M FLOPs in ~50ms)

### Metal GPU kernels (8 kernels)
- `q4_expert_gemv` — 160B/block, 8 sub-blocks, fp16 scales/mins
- `multi_expert_gemv` — parallel multi-expert dispatch
- `fused_gqa` — QKV+RoPE+online softmax+V sum+Q-gate (cos=0.9999, right for seq_len>32)
- `fp16_gemv` — general-purpose fp16 GEMV
- `router_forward` — softmax + top-k
- `fused_ops` — fused residual + RMSNorm

### Static analysis
- `objeta analyze` — phase profile, stability map
- `objeta moe-analyze` — Qwen3.6 routing analysis (Rust-native)

## Architecture

```
Qwen36Runner (Rust)
├── embed: mmap (2GB, zero-copy)
├── attention weights: Vec<u16> (f16, 2.9GB total)
├── routers + MoE mmaps: pre-loaded
├── KV caches: [kv_head][token][dim], per GQA layer
├── DeltaNet states: conv_state (8192×4) + S (32×128×128)
├── scratch buffers: pre-allocated, reused
│
├── forward(token_id, pos, seq_len) → hidden state
│   ├── 30× DeltaNet (gemv_f16 ×5 + conv1d + delta_state_update + RMSNormGated)
│   ├── 10× GQA (fused QKV + RoPE + attention + O-proj)
│   ├── 40× shared expert (gemv_f16 ×3)
│   └── 40× MoE dispatch (lko_moe_forward_layer)
│
├── lm_head_topk(hn, top_k) → (indices, values)
└── lko_runner_step() → forward + RMSNorm + lm_head + top-k (1 FFI call)
```

## Key constants (Qwen3.6-35B-A3B)

| Param | Value |
|-------|-------|
| Layers | 40 |
| Hidden dim | 2048 |
| Q heads | 16 |
| KV heads | 2 |
| Head dim | 256 |
| GQA ratio | 8:1 |
| Q-proj | (8192, 2048) = Q(4096) + Q-gate(4096) |
| Experts | 256, top-8 |
| Expert gate_up | (1024, 2048) q4 |
| Expert down | (2048, 512) q4 |
| Shared expert | ffn_dim=512, sigmoid-gated |
| DeltaNet layers | 30 (all except every 4th) |
| GQA layers | 10 (L3,7,11,15,19,23,27,31,35,39) |
| Q4_K_APPL block | 160 bytes, 8 sub-blocks of 32 |

## Next Steps

### Priority 1: Output quality
- Apply system prompt "You are a helpful assistant. Always respond in English."
- Verify tokenizer chat template
- Test with diverse prompts

### Priority 2: Speed (target: 1-3 tok/s)
1. **Attention q4/q8 quantization** — f16 weights → q4 saves 75% memory bandwidth
2. **Fused attention Metal kernel** — for seq_len > 32 (already cos=0.9999)
3. **Paged KV cache** — vLLM-style, enables long context
4. **Persistent command buffer** — reduce Metal dispatch overhead
5. **Speculative decoding** — phase-aware, low-entropy zones

### Priority 3: Ecosystem
- OpenAI API compatibility (`POST /v1/chat/completions`)
- Standard benchmarks (perplexity, HumanEval)
