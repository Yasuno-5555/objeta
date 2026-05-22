# E2E One-Token Canary Results (2026-05-22)

## Sealed Baseline

| Field | Value |
|---|---|
| Input | token 42 |
| Position | 0, seq_len=1 |
| Output token | **5** |
| Top-5 tokens | 5, 3398, 7519, 110704, 372 |
| Top-5 logits | 15.9, 13.2, 13.1, 12.9, 12.8 |
| All finite | yes |
| MoE placeholder | false |
| Deterministic | 3 runs identical |

## Global Ablations

All 43 layers modified simultaneously.

| Variant | Output Token | vs Baseline Cosine |
|---|---|---|
| official_full | **5** | 1.000 |
| no_moe_global | 124365 | 0.002 |
| routed_only_global | 83406 | 0.165 |
| shared_only_global | 201 | 0.565 |

**Finding**: Shared expert alone is closest to full MoE (cos=0.565), suggesting shared=persistent field.

## Single-Layer Interventions

One layer modified, all others official.

| Layer | Type | remove_routed | remove_shared | remove_moe |
|---|---|---|---|---|
| 0 | hash | 5 | 5 | 5 |
| **1** | **hash** | 5 | **680** | **680** |
| 2 | hash | 5 | 5 | 5 |
| 10 | — | 5 | 5 | 5 |
| 21 | — | 5 | 5 | 5 |
| 27 | — | 5 | 5 | 5 |
| 35 | — | 5 | 5 | 5 |
| 42 | — | 5 | 5 | 5 |

**Finding**: Layer 1 is the only causal critical layer. Removing shared MoE (or all MoE) at layer 1 changes the output token. All other layers are robust to single-layer MoE removal.

## Performance

| Binary | Attn Backend | Total Time |
|---|---|---|
| `deepseek_e2e` | CPU FP8 decode/GEMV | ~25s |
| `deepseek_e2e_fast` | CUDA fp8_act×fp8_wt GEMV | ~9s |
