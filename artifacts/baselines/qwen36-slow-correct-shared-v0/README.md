# qwen36-slow-correct-shared-v0

Frozen slow-correct shared-expert baseline for Qwen3.6 Rust executor validation.

## Scope

- Rust executor in shared-only parity mode (`moe_enabled: 0` in the comparison path)
- Layerwise/stateful parity oracle for token positions 0-4
- Greedy generation smoke test for `The capital of France is`
- Tokenizer/config snapshot copied from the exact Hugging Face snapshot

## Reproducibility

- Git tag: `qwen36-slow-correct-shared-v0`
- Git HEAD: `f6ab7da2d2df1d90b06889d9e393561ca9ae89f8`
- Worktree status: see [git_status.txt](./git_status.txt)
- Uncommitted patch: see [git_diff.patch](./git_diff.patch)
- Local environment: see [environment.txt](./environment.txt)
- Commands: see [commands.sh](./commands.sh)

## Artifacts

- [parity_pos0_4.log](./parity_pos0_4.log): HF vs Rust 40-layer trace for token positions 0-4
- [greedy_generation.log](./greedy_generation.log): end-to-end greedy generation smoke output
- [snapshot/](./snapshot): `config.json`, `tokenizer.json`, `tokenizer_config.json`

## Observed Results

- Token 0 final cosine: `0.992732`
- Token 1 final cosine: `0.999572`
- Token 2 final cosine: `0.999858`
- Token 3 final cosine: `0.999818`
- Token 4 final cosine: `0.999875`
- 25-token greedy output starts with:
  `Here's a thinking process:`
  `1.  **Analyze User Input:** The user asks "The capital of France is`
