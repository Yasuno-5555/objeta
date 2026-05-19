# Executor Bottlenecks — 2026-05-19

## Purpose

This document is the current detailed record of:

- what is correct
- what is still slow
- what is still unstable
- what the current implementation problems are
- what should be measured next

It is intentionally more operational than `CURRENT_STATUS_2026_05_18.md`.

## Executive Summary

The executor is now in a useful but awkward state:

- correctness is good enough to trust for optimization experiments
- aggressive trajectory skipping is not safe
- adaptive routed-MoE pruning looks promising for quality
- wall-clock speedup is weaker than the MoE-local savings suggest

The important current conclusion is:

> We can reduce routed expert execution without immediate text collapse, but the current runtime does not yet convert those savings cleanly into end-to-end latency reduction.

So the bottleneck has shifted from "model is broken" to:

> "why are reduced expert count and reduced bytes not showing up as proportional wall-clock wins?"

## Stable Baseline

Reference checkpoint:

- tag: `qwen36-slow-correct-shared-v0`
- commit: `f6ab7da2d2df1d90b06889d9e393561ca9ae89f8`
- artifact index: `artifacts/baselines/qwen36-slow-correct-shared-v0/README.md`

This remains the ground truth for "slow but known-good."

## Correctness State

### Resolved

1. Untied `lm_head`
   - The checkpoint uses `tie_word_embeddings: false`
   - Rust now loads `lm_head.bin` separately
   - This removed the earlier broken next-token behavior

2. GQA parity fixes
   - `rope_theta = 10_000_000`
   - partial rotary over 64 dims of each 256-dim head
   - Q/K RMSNorm uses `(1.0 + weight)`
   - per-head `[query, gate]` split for `q_proj`

3. Shared-only stateful baseline
   - pos=0..4 parity exists
   - generation is fluent again

4. Router parity
   - `router_logits`, top-k ids, top-k weights, entropy all look strong

5. Same-input MoE implementation parity
   - same input + same fp weights gives `cos = 1.0`
   - this strongly argues the routed MoE math is basically correct

### Still not fully solved

1. Metal GQA parity
   - still disabled intentionally
   - CPU fallback is the current correctness oracle

2. Full optimization path parity
   - fast paths can still preserve fluent text or destroy it depending on configuration
   - `fusion=0.80` is currently the only tested trajectory-skip setting that remains safe

## Trajectory-Skip Findings

### Fusion sweep

With `moe_on_deltanet=1` and 25-token greedy smoke:

- `fusion=1.00`: fluent, very slow
- `fusion=0.80`: fluent, safe
- `fusion=0.66`: collapse
- `fusion=0.50`: collapse
- `fusion=0.33`: collapse

Interpretation:

- Qwen3.6 tolerates some trajectory skipping
- but the safe region is narrow
- the collapse boundary is somewhere between `0.66` and `0.80`

This means aggressive fusion skip is not the next best optimization target.

## Adaptive Expert Execution Findings

All results below use:

- `fusion=0.80`
- `moe_on_deltanet=1`
- prompt: `"The capital of France is"`
- 25-token greedy smoke

### Top-p pruning

Observed:

- `top_p=1.00`
  - fluent
  - `avg_experts/layer = 8.000`
  - `avg_bytes_read = 15,728,640`
  - `forward_wall_ms/token = 4477.59`
  - `moe_wall_ms/token = 2085.02`

- `top_p=0.95`
  - fluent
  - `avg_experts/layer = 7.796`
  - `avg_mass = 0.993`
  - `avg_drop = 0.007`
  - `avg_bytes_read = 15,326,822.4`

- `top_p=0.90`
  - fluent
  - `avg_experts/layer = 6.867`
  - `avg_bytes_read = 13,502,054.4`

- `top_p=0.85`
  - fluent
  - `avg_experts/layer = 6.406`
  - `avg_mass = 0.893`
  - `avg_drop = 0.107`
  - `avg_bytes_read = 12,594,708.5`
  - `forward_wall_ms/token = 5013.57`
  - `moe_wall_ms/token = 2242.65`

- `top_p=0.80`
  - still mostly readable, but repetition appears
  - `avg_experts/layer = 5.681`
  - `avg_mass = 0.840`
  - `avg_drop = 0.160`
  - `avg_bytes_read = 11,168,563.2`

Interpretation:

- top-p pruning definitely reduces executed experts
- top-p pruning definitely reduces bytes read
- quality remains good through at least `top_p=0.85`
- `top_p=0.80` starts looking risky

### Contribution-prior pruning

Current implementation:

- `score = gate_weight * ema_output_norm[layer][expert]`
- EMA lives inside the Rust runner
- threshold currently tested at `0.90`

Observed:

- `contrib_threshold=0.90`
  - fluent
  - `avg_experts/layer = 6.649`
  - `avg_mass = 0.905`
  - `avg_drop = 0.095`
  - `avg_bytes_read = 13,072,465.9`
  - `forward_wall_ms/token = 4983.62`
  - `moe_wall_ms/token = 2197.68`

Interpretation:

- contribution-prior also preserves quality at this threshold
- but against `top_p=0.85`, it is currently slightly less aggressive
- `top_p=0.85` gets lower expert count and lower bytes with similar output quality

## MoE Microbenchmark

Reference microbench:

- layer: `L31`
- prompt-derived hidden
- q4 selected experts
- 100 iterations

Results:

- `N=8 -> 24.616 ms`
- `N=7 -> 21.072 ms`
- `N=6 -> 20.704 ms`
- `N=5 -> 15.096 ms`
- `N=4 -> 12.902 ms`
- `N=3 -> 12.545 ms`
- `N=2 -> 7.293 ms`
- `N=1 -> 4.017 ms`

Interpretation:

- expert execution cost does scale down with fewer experts
- so the weak end-to-end speedup is not because the MoE kernel is flat-cost
- the slowdown is coming from elsewhere in the runtime stack, or from how the measurement path is structured

## Current Bottlenecks

### 1. End-to-end wall-clock is dominated by more than just MoE

Even when expert count and bytes go down, total token latency does not fall proportionally.

Observed baseline split:

- `forward_wall_ms/token ≈ 4477.59`
- `moe_wall_ms/token ≈ 2085.02`
- `non_moe_wall_ms ≈ 2392.57`

This means:

- MoE is large, but not the whole story
- even perfect MoE optimization would leave a large residual runtime

### 2. Expert cache is not helping in the current smoke path

Current runs show:

- `avg_warm_hit_count = 0.000`
- `avg_cold_hit_count > 0`

Interpretation:

- every routed expert call is effectively cold in this measurement path
- the current cache/residency path is not contributing to speedup

This is probably one of the biggest implementation-level reasons the bytes savings are not translating into wall-clock savings.

### 3. Dequant path is still expensive

Current MoE timing breakdown consistently shows large dequant cost.

Important caveat:

- `avg_dequant_ms`, `avg_gemv_ms`, etc. are sum-style timing accumulators across parallel expert workers
- these are not equal to wall-clock directly

But they still tell us the expensive work categories:

- dequantization is large
- expert execution wall is large
- router/select/accumulate are comparatively small

### 4. Non-MoE forward is still large

The current runtime still spends a lot outside routed MoE:

- GQA
- DeltaNet
- shared expert
- final norm / lm_head
- general per-token orchestration

This matters because:

- even good MoE pruning will hit a speedup ceiling unless these are also improved

### 5. Measurement variance is still nontrivial

Some runs show spikes that likely reflect:

- OS scheduling
- cold page behavior
- allocator behavior
- first-use effects

This means:

- single-run numbers are directionally useful
- but not yet publication-grade timing evidence

## Current Implementation Problems

### A. Sum timing vs wall timing confusion

There are now two kinds of timing:

1. summed work timing
   - router_ms
   - dequant_ms
   - gemv_ms
   - etc.

2. wall-span timing
   - forward wall
   - MoE wall
   - exec wall

This is necessary, because parallel worker sums can easily exceed wall-clock.

Current risk:

- people may still compare `dequant_ms` directly to total token latency
- that is wrong unless explicitly labeled as summed worker time

### B. Contribution-prior EMA is young and local

Current contribution-prior pruning uses:

- `score = gate_weight * ema_output_norm[layer][expert]`

But the EMA:

- is initialized to `1.0`
- is updated online inside the same runner
- is not yet persisted or calibrated

So right now:

- contribution-prior is a useful experiment
- but not yet a stable policy artifact

### C. Cache/residency path is not validated as effective

We now know the expert cache exists, but in smoke runs:

- warm hits are zero

So either:

- the reuse pattern is too weak in this workload
- the cache capacity / keying / lifecycle is wrong
- or the current benchmark path resets state in a way that prevents reuse from showing up

This needs direct investigation.

### D. Metal GQA still off

This is not the main optimization blocker right this second, but it is still a structural issue:

- CPU fallback remains the active path
- any future speed claims should be understood in that context

### E. first5 vs last20 decode accounting was initially wrong

The first implementation subtracted averages instead of reconstructing totals.

This has now been corrected in the Python reporting layer, but it is worth documenting because:

- phase-split statistics are easy to misread
- we should treat new reporting paths carefully until they are rechecked

## Most Likely Root Cause Of Weak Wall-Clock Gains

Current best hypothesis:

1. expert count and bytes are being reduced correctly
2. MoE-local execution does benefit from fewer experts
3. but the current decode path is still dominated by:
   - cold expert execution
   - dequant overhead
   - non-MoE forward cost
   - run-level orchestration overhead

The strongest single clue is:

- microbench scales with expert count
- full decode barely does

That strongly suggests the issue is not "pruning does nothing," but rather:

> pruning works locally, while the rest of the stack is still too expensive or too cold to expose the win.

## Recommended Next Steps

### 1. Finish the comparable pruning sweep

Still needed:

- `top_p = 0.95 / 0.90 / 0.85 / 0.80` under the new wall metrics
- `contrib = 0.90 / 0.85 / 0.80`

This will establish the best quality-preserving pruning frontier.

### 2. Investigate zero warm hits

This is the most important implementation issue right now.

Questions:

- is the cache being reused within a single generation at all?
- is the LRU too small?
- are experts rotating too much to benefit?
- is the cache being bypassed by the q4/dequant path structure?

### 3. Separate true wall bottlenecks from summed work

The next comparison should prioritize:

- `avg_forward_wall_ms`
- `avg_moe_wall_ms_per_token`
- `avg_exec_wall_ms`
- `avg_non_moe_wall_ms`

over worker-sum metrics.

### 4. Compare pruning policies by equal-quality frontier

The real question is not just "which prunes more," but:

> which keeps fluent generation while minimizing experts, bytes, and wall time?

At the moment, `top_p=0.85` looks slightly better than `contrib=0.90`, but the sweep is not complete.

### 5. Do not return to aggressive fusion skip yet

The fusion experiments already established:

- trajectory skipping is much less forgiving than expert pruning

That path should stay secondary until the routed-MoE execution path is better understood.
