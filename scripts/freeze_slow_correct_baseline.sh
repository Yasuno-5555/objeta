#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-qwen36-slow-correct-shared-v0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/artifacts/baselines/${LABEL}"
SNAPSHOT="/Users/yasuno/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

mkdir -p "${OUT_DIR}/snapshot"

cd "${ROOT}"

cat > "${OUT_DIR}/commands.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

python3 -u experiments/a1_full_compare.py
python3 -u experiments/qwen36_full_rust.py 1.0 1 \
  --warmup-tokens 0 \
  --max-tokens 25 \
  --temperature 0.0 \
  --prompt 'The capital of France is'
EOF
chmod +x "${OUT_DIR}/commands.sh"

git rev-parse HEAD > "${OUT_DIR}/git_head.txt"
git status --short > "${OUT_DIR}/git_status.txt"
git diff --binary > "${OUT_DIR}/git_diff.patch"
git ls-files -m -o --exclude-standard > "${OUT_DIR}/worktree_files.txt"

{
  echo "date_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "date_local=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "pwd=${ROOT}"
  echo "python=$(python3 --version 2>&1)"
  echo "cargo=$(cargo --version 2>&1)"
  echo "rustc=$(rustc --version 2>&1)"
  echo "uname=$(uname -a)"
} > "${OUT_DIR}/environment.txt"

for f in config.json tokenizer.json tokenizer_config.json; do
  cp "${SNAPSHOT}/${f}" "${OUT_DIR}/snapshot/${f}"
done
(cd "${OUT_DIR}/snapshot" && shasum -a 256 config.json tokenizer.json tokenizer_config.json > SHA256SUMS.txt)

python3 -u experiments/a1_full_compare.py > "${OUT_DIR}/parity_pos0_4.log" 2>&1
python3 -u experiments/qwen36_full_rust.py 1.0 1 \
  --warmup-tokens 0 \
  --max-tokens 25 \
  --temperature 0.0 \
  --prompt 'The capital of France is' \
  > "${OUT_DIR}/greedy_generation.log" 2>&1

mapfile -t COS_VALUES < <(rg 'cos\(hf, ours\) after 40L' "${OUT_DIR}/parity_pos0_4.log" | awk '{print $NF}')
GEN_OUTPUT="$(sed -n 's/^  Output: //p' "${OUT_DIR}/greedy_generation.log" | tail -n 1)"

cat > "${OUT_DIR}/README.md" <<EOF
# ${LABEL}

Frozen slow-correct shared-expert baseline for Qwen3.6 Rust executor validation.

## Scope

- Rust executor in shared-only parity mode (\`moe_enabled: 0\` in the comparison path)
- Layerwise/stateful parity oracle for token positions 0-4
- Greedy generation smoke test for \`The capital of France is\`
- Tokenizer/config snapshot copied from the exact Hugging Face snapshot

## Reproducibility

- Git HEAD: \`$(cat "${OUT_DIR}/git_head.txt")\`
- Worktree status: see [git_status.txt](./git_status.txt)
- Uncommitted patch: see [git_diff.patch](./git_diff.patch)
- Local environment: see [environment.txt](./environment.txt)
- Commands: see [commands.sh](./commands.sh)

## Artifacts

- [parity_pos0_4.log](./parity_pos0_4.log): HF vs Rust 40-layer trace for token positions 0-4
- [greedy_generation.log](./greedy_generation.log): end-to-end greedy generation smoke output
- [snapshot/](./snapshot): \`config.json\`, \`tokenizer.json\`, \`tokenizer_config.json\`

## Observed Results

- Token 0 final cosine: \`${COS_VALUES[0]:-missing}\`
- Token 1 final cosine: \`${COS_VALUES[1]:-missing}\`
- Token 2 final cosine: \`${COS_VALUES[2]:-missing}\`
- Token 3 final cosine: \`${COS_VALUES[3]:-missing}\`
- Token 4 final cosine: \`${COS_VALUES[4]:-missing}\`
- 25-token greedy output:
  \`${GEN_OUTPUT:-missing}\`
EOF

echo "Baseline frozen at ${OUT_DIR}"
