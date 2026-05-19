#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m py_compile \
  experiments/oracle_registry.py \
  experiments/qwen36_oracle_check.py \
  experiments/qwen36_one_token_oracle.py \
  experiments/qwen36_layer_oracle.py >/dev/null

bash scripts/check_oracles.sh
