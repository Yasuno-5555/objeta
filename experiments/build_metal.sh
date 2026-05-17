#!/bin/bash
# Build objeta Metal library from kernel sources.
# Requires Xcode Command Line Tools.
set -e

KERNEL_DIR="$(dirname "$0")/../kernels/metal"
TARGET_DIR="$(dirname "$0")/../target"
OUTPUT="${TARGET_DIR}/objeta.metallib"
AIR_DIR="${TARGET_DIR}/metal_air"

mkdir -p "${AIR_DIR}"

KERNELS=(
    "q4_expert_gemv"
    "router_forward"
    "fused_ops"
    "attention_ops"
    "fused_gqa"
)

echo "Compiling Metal kernels..."
AIR_FILES=()
for kernel in "${KERNELS[@]}"; do
    src="${KERNEL_DIR}/${kernel}.metal"
    air="${AIR_DIR}/${kernel}.air"
    echo "  ${kernel}.metal → ${kernel}.air"
    xcrun -sdk macosx metal -c "${src}" -o "${air}"
    AIR_FILES+=("${air}")
done

echo "Linking → ${OUTPUT}"
xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "${OUTPUT}"

echo "  → ${OUTPUT}"
ls -la "${OUTPUT}"
