# DeepSeek V4 Flash — E2E One-Token Forward Blockers

## Status: Implemented

All operators required for single-token greedy decoding have been implemented:

| Operator | Supported | File |
|----------|-----------|------|
| token embedding | yes | e2e binary |
| RMSNorm | yes | e2e binary |
| MLA attention (seq=1, pos=0) | yes | e2e binary |
| Hyper-Connection (FFN path) | yes | e2e binary |
| MoE.forward (validated) | yes | `objeta-cuda::moe` |
| final norm | yes | e2e binary |
| lm_head | yes | e2e binary |
| top-10 logits extraction | yes | e2e binary |

### Required tensors

| Tensor | Shape | Dtype | Required Op | Status |
|--------|-------|-------|-------------|--------|
| `embed.weight` | [129280, 4096] | BF16 | gather (token→hidden) | done |
| `layers.{L}.attn_norm.weight` | [4096] | BF16 | RMSNorm | done |
| `layers.{L}.ffn_norm.weight` | [4096] | BF16 | RMSNorm | done |
| `layers.{L}.attn.wq_a.weight` | [1024, 4096] | F8_E4M3 | FP8 × FP8 GEMV | done |
| `layers.{L}.attn.wq_a.scale` | [8, 32] | F8_E8M0 | tile scale | done |
| `layers.{L}.attn.wq_b.weight` | [32768, 1024] | F8_E4M3 | FP8 × BF16 GEMV | done |
| `layers.{L}.attn.wq_b.scale` | [256, 8] | F8_E8M0 | tile scale | done |
| `layers.{L}.attn.wkv.weight` | [512, 4096] | F8_E4M3 | FP8 × FP8 GEMV | done |
| `layers.{L}.attn.wkv.scale` | [4, 32] | F8_E8M0 | tile scale | done |
| `layers.{L}.attn.wo_a.weight` | [8192, 4096] | F8_E4M3 | FP8 × FP8 GEMV | done |
| `layers.{L}.attn.wo_a.scale` | [64, 32] | F8_E8M0 | tile scale | done |
| `layers.{L}.attn.wo_b.weight` | [4096, 8192] | F8_E4M3 | FP8 × FP8 GEMV | done |
| `layers.{L}.attn.wo_b.scale` | [32, 64] | F8_E8M0 | tile scale | done |
| `layers.{L}.attn.q_norm.weight` | [1024] | BF16 | RMSNorm (q after WqA) | done |
| `layers.{L}.attn.kv_norm.weight` | [512] | BF16 | RMSNorm (kv after Wkv) | done |
| `layers.{L}.attn.attn_sink` | [64] | F32 | additive bias | done |
| `layers.{L}.hc_ffn_base` | [1024] | F32 | additive residual | done |
| `layers.{L}.hc_ffn_fn` | [4096, 2048] | F8_E4M3 | FP8 × FP8 compress | done |
| `layers.{L}.hc_ffn_scale` | [32, 32] | F8_E8M0 | tile scale | done |
| `norm.weight` | [4096] | BF16 | RMSNorm (final) | done |
| `head.weight` | [129280, 4096] | BF16 | matmul (hidden→logits) | done |
| `hc_head_base` | [1024] | F32 | additive residual | done |
| `hc_head_fn` | [4096, 2048] | F8_E4M3 | FP8 × FP8 compress | done |
| `hc_head_scale` | [32, 32] | F8_E8M0 | tile scale | done |

### Unsupported (MTP only, excluded from greedy)

- `mtp.0.e_proj.weight/scale` — next-token-prediction layers, not needed
- `mtp.0.h_proj.weight/scale` — same

### Notes

- `head.weight` at BF16 (not FP8) — direct dot product with hidden state
- `embed.weight` at BF16 — simple gather by token id
- RMSNorm weights at BF16 — multiplication with normalized input
- HC FFN compress: hidden [4096] → compressed [2048] via FP8 × FP8
- HC decompress is implicit: residual is stored at coarse granularity
