#!/bin/bash
# Extract best checkpoints from sequential finetuning experiments
# Usage: ./scripts/extract_best_checkpoints.sh <checkpoint_dir>

set -e

CHECKPOINT_DIR="${1:-./checkpoints/sequential}"

echo "=========================================="
echo "Extracting Best Checkpoints"
echo "=========================================="
echo "Source: $CHECKPOINT_DIR"
echo ""

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Create output directory
BEST_DIR="${CHECKPOINT_DIR}/best_models"
mkdir -p "$BEST_DIR"

# Copy final checkpoints
echo "Copying final checkpoints..."

if [ -f "$CHECKPOINT_DIR/stage1_utm_final.pt" ]; then
    cp "$CHECKPOINT_DIR/stage1_utm_final.pt" "$BEST_DIR/"
    echo "✓ Copied stage1_utm_final.pt"
fi

if [ -f "$CHECKPOINT_DIR/stage2_ctw_final.pt" ]; then
    cp "$CHECKPOINT_DIR/stage2_ctw_final.pt" "$BEST_DIR/"
    echo "✓ Copied stage2_ctw_final.pt"
fi

# Copy logs and metrics
if [ -f "$CHECKPOINT_DIR/metrics.json" ]; then
    cp "$CHECKPOINT_DIR/metrics.json" "$BEST_DIR/"
    echo "✓ Copied metrics.json"
fi

if [ -f "$CHECKPOINT_DIR/train.log" ]; then
    cp "$CHECKPOINT_DIR/train.log" "$BEST_DIR/"
    echo "✓ Copied train.log"
fi

echo ""
echo "=========================================="
echo "✓ Best checkpoints extracted to:"
echo "  $BEST_DIR"
echo "=========================================="
echo ""
echo "Files:"
ls -lh "$BEST_DIR"
