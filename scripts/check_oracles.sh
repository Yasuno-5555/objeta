#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 experiments/qwen36_oracle_check.py \
  --strategy configs/safe_exact.json \
  --chat-template \
  --warmup-tokens 0 \
  --compare-golden \
  --golden-name safe_exact_chat_prefill

python3 experiments/qwen36_one_token_oracle.py \
  --strategy configs/safe_exact.json \
  --chat-template \
  --warmup-tokens 0 \
  --compare-golden \
  --golden-name safe_exact_chat_1token

python3 experiments/qwen36_layer_oracle.py \
  --strategy configs/safe_exact.json \
  --chat-template \
  --warmup-tokens 0 \
  --compare-golden \
  --golden-name safe_exact_chat_layer_trace
