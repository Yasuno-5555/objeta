#!/bin/bash
# Fusion × Cache 2x2 smoke diagnostic
# All runs: exact policy, moe_on_deltanet=1, max_tokens=25, temp=0, prompt="The capital of France is"
set -euo pipefail

SCRIPT="experiments/qwen36_full_rust.py"
PROMPT="The capital of France is"
MAX_TOKENS=25
MOE_DN=1
STRATEGY="configs/safe_exact.json"
DIAG_DIR="runs/fusion_diag_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$DIAG_DIR"

echo "=== Fusion × Cache 2x2 Smoke Diagnostic ==="
echo "Output dir: $DIAG_DIR"
echo ""

run_one() {
    local label="$1"
    local fusion="$2"
    local cache_mb="$3"
    local out="$DIAG_DIR/${label}.txt"

    echo "──────────────────────────────────────────"
    echo "RUN $label: fusion=$fusion exact cache_mb=$cache_mb"
    echo "──────────────────────────────────────────"

    python3 "$SCRIPT" \
        "$fusion" "$MOE_DN" \
        --strategy "$STRATEGY" \
        --expert-cache-mb "$cache_mb" \
        --warmup-tokens 0 \
        --max-tokens "$MAX_TOKENS" \
        --prompt "$PROMPT" \
        --temperature 0 \
        2>&1 | tee "$out"

    echo ""
    echo "  → Results saved to $out"
    echo ""
}

# A: fusion=1.0, cache=0
run_one "A_fusion1.0_cache0" 1.0 0

# B: fusion=1.0, cache=512 (on)
run_one "B_fusion1.0_cache512" 1.0 512

# C: fusion=0.80, cache=0
run_one "C_fusion0.80_cache0" 0.80 0

# D: fusion=0.80, cache=512 (on)
run_one "D_fusion0.80_cache512" 0.80 512

echo ""
echo "=== All 4 runs complete ==="
echo "Results in: $DIAG_DIR"
echo ""

# Summarize key metrics
echo "=== QUICK SUMMARY ==="
for f in "$DIAG_DIR"/A_*.txt "$DIAG_DIR"/B_*.txt "$DIAG_DIR"/C_*.txt "$DIAG_DIR"/D_*.txt; do
    echo "── $(basename "$f") ──"
    grep -E '(Output:|First garbage|First repetition|Selected policy|executed_layers|skipped_layers|avg_experts|tok/s|entropy)' "$f" | head -20 || true
    echo ""
done
