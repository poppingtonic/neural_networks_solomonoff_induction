#!/bin/bash
# Quick test run with minimal steps for debugging
# Usage: ./scripts/quick_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Running quick test..."
cd "$PROJECT_ROOT"

python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
    --use_hf \
    --utm_steps 100 \
    --ctw_steps 50 \
    --batch_size 8 \
    --seq_length 64 \
    --lr 1e-4 \
    --save_every 50 \
    --log_every 10 \
    --output_dir "./checkpoints/test" \
    --seed 42

echo "✓ Quick test complete!"
