# DeepSeek V4 Flash — One-Token Forward Specification

## Architecture Overview

- hidden_size = 4096
- hc_mult = 4
- Hidden state for one token: shape [hc_mult, hidden_size] = [4, 4096]
- 43 decoder layers
- 256 MoE routed experts (top_k=6), 1 shared expert
- num_attention_heads = 64, num_key_value_heads = 1
- q_lora_rank = 1024, kv_lora_rank = 512
- index_head_dim = 128, head_dim = 512
- o_lora_rank = 1024

## Hyper-Connection (HC) Semantics

### HC State
Hidden state `h` for one token is [hc_mult=4, dim=4096] = [4, 4096].
Initialized by repeating the embedding vector 4 times.

### HC Pre-FFN
Input: `h` [4, 4096]
1. Flatten: `x_flat = h.reshape(16384)`
2. `mixes_raw = hc_fn @ x_flat` where `hc_fn` has shape [output_dim, 16384]
   Note: actual `hc_ffn_fn.weight` shape is [4096, 2048], indicating that:
   - The HC pre uses a non-trivial reshape/view rather than a flat linear
   - The `hc_ffn_fn` weight is NOT the full pre-mixing matrix
   - Alternative: each HC copy independently processes through a shared weight
3. Sinkhorn balancing on 4×4 grid
4. Weighted sum: `x_single[4096] = sum(pre_i * h[i])`

### HC Post-FFN
`output_state = post * moe_output + comb @ previous_state`

### HC Head (before final norm + lm_head)
`pre = sigmoid(mixes * hc_head_scale + hc_head_base) + eps`
`output = sum(pre_i * state_i)`

## MLA Attention (seq_len=1, position=0)

- Q: WqA [1024,4096] × x [4096] → q_latent [1024]
- q_norm: RMSNorm on [1024]
- WqB [32768,1024] × q_normed [1024] → q [32768] → reshape [64, 512]
- KV: Wkv [512,4096] × x [4096] → kv_latent [512]
- kv_norm: RMSNorm on [512]
- For each head h:
  - score_h = dot(q[h,:], kv_normed) / sqrt(head_dim)
  - alpha_h = softmax over [score_h, attn_sink[h] + existing_logit]
  - o_h = alpha_h * kv_normed
- Output: reshape [64, 128] (index_head_dim from kv_lora/num_heads)

## Output Projection

### WoA (Grouped, 8 groups)
- Input: attention output [64, 512]
- Reshape to [8, 4096] (8 groups × 4096)
- wo_a.weight: [8192, 4096] physical
- View as [8, 1024, 4096]
- For group g: y[g, :] = W_wo_a[g, :, :] @ input[g, :]
- Output: [8, 1024] → flatten [8192]

### WoB (Standard)
- wo_b.weight: [4096, 8192]
- Output: [4096] = W_wo_b @ y[8192]
- Added as residual to hidden state

### Key Insight for WoA storage
wo_a.weight.shape = [8192, 4096] stored as FP8 (F8_E4M3 + F8_E8M0 tile scales).
The weight is stored in standard [output_dim, input_dim] layout.
For grouped: output_dim=8192, input_dim=4096.
Break into 8 groups: [1024, 4096] per group.

## MoE.forward Hook

After hc_pre_ffn + ffn_norm:
Call official CUDA MoE.forward with routed FP4 + shared FP8.

Input: ffn_normed [4096]
Output: moe_output [4096]

This is already validated in the existing benchmark binary.

## Accept Blockers

- MTP layers explicitly excluded
- KV compression (layers with compress_ratio > 0) not exercised at seq=1,pos=0
- RoPE is identity at position=0
- Attention sink bias is applied per-head
