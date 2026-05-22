# DeepSeek V4 Flash — E2E One-Token Blockers

**Status: ALL RESOLVED (2026-05-22)**

## Original Blockers → Resolution

| # | Blocker | Status | Resolution |
|---|---|---|---|
| 1 | Token embedding loading | ✅ RESOLVED | `embed.weight` loaded via `ModelWeights::get_f32` |
| 2 | HC state initialization [4,4096] | ✅ RESOLVED | Embedding repeated across 4 HC copies |
| 3 | HC pre/post Sinkhorn split | ✅ RESOLVED | 20-iteration row/col normalization, sigmoid gating |
| 4 | Attn norm (RMS) | ✅ RESOLVED | CPU RMSNorm with eps=1e-6 |
| 5 | WqA [1024,4096] FP8 projection | ✅ RESOLVED | CUDA-accelerated: `cuda_act_quant_device` + `cuda_fp8_act_fp8_weight_gemv_device` |
| 6 | Q norm (per-head) | ✅ RESOLVED | CPU RMSNorm on per-head Q vectors |
| 7 | WqB [32768,1024] FP8 projection | ✅ RESOLVED | CUDA-accelerated (largest linear, ~23ms) |
| 8 | Wkv [512,4096] FP8 projection | ✅ RESOLVED | CUDA-accelerated (~1.5ms) |
| 9 | KV norm | ✅ RESOLVED | CPU RMSNorm |
| 10 | Attn sink + softmax alpha | ✅ RESOLVED | Per-head score with sink bias, seq=1 degenerate case |
| 11 | WoA grouped projection [8,1024,4096] | ✅ RESOLVED | CUDA per-group GEMV with host-side weight slicing |
| 12 | WoB [4096,8192] FP8 projection | ✅ RESOLVED | CUDA-accelerated (~18ms) |
| 13 | HC post-attention blending | ✅ RESOLVED | post * output + comb @ residual |
| 14 | HC pre-FFN blending | ✅ RESOLVED | Same Sinkhorn, different tensors (hc_ffn_*) |
| 15 | FFN norm (RMS) | ✅ RESOLVED | CPU RMSNorm |
| 16 | Router (gate) selection | ✅ RESOLVED | CPU dense GEMV + top-k selection |
| 17 | Expert FP4 tensor loading | ✅ RESOLVED | `ModelWeights::get_raw` by expert ID |
| 18 | Shared FP8 tensor loading | ✅ RESOLVED | Uploaded to device via `copy_from_slice` |
| 19 | CUDA MoE.forward | ✅ RESOLVED | `execute_selected_moe_official_routed_fp4_cuda` (sealed, cos=1.0) |
| 20 | HC post-FFN blending | ✅ RESOLVED | post * moe_out + comb @ residual |
| 21 | HC head | ✅ RESOLVED | Sigmoid gating + weighted sum |
| 22 | Final RMSNorm | ✅ RESOLVED | CPU RMSNorm |
| 23 | LM head [129280,4096] | ✅ RESOLVED | CPU dense GEMV (374ms; CUDA path planned) |
| 24 | Position encoding (RoPE) | ⬜ N/A | seq=1, pos=0 → identity rotation |
| 25 | KV cache (multi-token) | ⬜ OUT OF SCOPE | Future work |

## Current Canary

**Input**: token 42
**Position**: 0, seq_len=1
**Output**: token **5**
**Top 5**: [5:15.9, 3398:13.2, 7519:13.1, 110704:12.9, 372:12.8]
**All finite**: ✅
**Official MoE**: ✅
**Deterministic**: ✅ (3 runs identical)

## Performance

| Path | Time |
|---|---|
| CPU attention (deepseek_e2e) | ~25s |
| CUDA attention (deepseek_e2e_fast) | ~9s |

## Intervention Findings

- Layer 1 is the only causal critical layer: removing shared MoE there changes token 5→680
- All other layers (0, 2, 10, 21, 27, 35, 42) are robust to single-layer MoE removal
- Global MoE removal completely changes output (cos=0.002)
