# Current Status — 2026-05-19

## Summary

The executor is **partially recovered, not finished**.

The biggest new fix was identifying that this checkpoint uses an **untied `lm_head`**, while the Rust executor had been using `embed_tokens.bin` as the output projection. After loading a real `lm_head.bin`, next-token behavior improved dramatically and the earlier obviously broken outputs disappeared.

What is true right now:

- prefill logits are much closer to HF
- end-to-end output is no longer collapsing into unrelated Chinese tokens
- correctness is improved, but still not exact

## Biggest Fix From This Round

### Untied output head

HF config says:

- `tie_word_embeddings: false`

Direct comparison showed:

- `embed_tokens.weight` vs `lm_head.weight`
- same shape `(248320, 2048)`
- cosine only about `0.085`

So using embeddings as lm-head was fundamentally wrong for generation.

Rust now loads `lm_head.bin` when present.

## Current Verification Snapshot

### Prefill logits

For prompt:

```text
The capital of France is
```

current comparison gave:

- Rust top-1: `'Here'`
- HF top-1: `'Here'`
- top-10 overlap: `8/10`
- `hn` cosine: `0.861411`

### Smoke generation

These now both produce `Here` instead of the older broken outputs:

```bash
python3 -u experiments/qwen36_full_rust.py 1.0 1 --warmup-tokens 0 --max-tokens 1 --prompt 'The capital of France is'
python3 -u experiments/qwen36_full_rust.py 1.0 1 --warmup-tokens 0 --max-tokens 1 --prompt '2 + 2 ='
```

This is progress, but not yet proof of full HF parity.

## What Seems Mostly Correct

- GQA formula after the recent fixes
- router top-8 selection
- q4 expert dequantization quality
- shared expert path

## Still Suspect

- late-layer residual / hidden flow under the full runner path
- final hidden mismatch before/after final norm
- strategy-time requantization assumptions
- any doc claiming “all bugs fully resolved”

## Important Runtime Note

Metal GQA is currently disabled on purpose:

```text
[objeta] Metal GQA: disabled pending kernel parity, using CPU fallback
```

## Next Suggested Checks

1. Compare `hn` and top-k logits on several prompts after the lm-head fix.
2. Re-check late-layer traces with `lko_runner_trace_layer_components(...)`.
3. Audit `strategy.json` handling before trusting precision experiments.
