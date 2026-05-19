#!/usr/bin/env bash
set -euo pipefail

python3 -u experiments/a1_full_compare.py
python3 -u experiments/qwen36_full_rust.py 1.0 1 \
  --warmup-tokens 0 \
  --max-tokens 25 \
  --temperature 0.0 \
  --prompt 'The capital of France is'
