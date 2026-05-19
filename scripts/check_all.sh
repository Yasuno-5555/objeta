#!/bin/bash
# Local CI verification script for objeta qwen36 executor
set -e

echo "=== 1. Checking format (cargo fmt) ==="
cargo fmt --manifest-path crates/objeta-qwen36-executor/Cargo.toml --check

echo "=== 2. Running clippy (cargo clippy) ==="
cargo clippy --manifest-path crates/objeta-qwen36-executor/Cargo.toml --all-targets -- -D warnings

echo "=== 3. Running unit and integration tests (cargo test) ==="
cargo test --manifest-path crates/objeta-qwen36-executor/Cargo.toml

echo "=== 4. Running Python e2e smoke matrix test ==="
python3 experiments/smoke_matrix.py

echo "=========================================="
echo " SUCCESS: All checks and regression tests passed!"
echo "=========================================="
