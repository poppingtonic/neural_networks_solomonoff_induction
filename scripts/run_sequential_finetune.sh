#!/bin/bash
# Sequential Finetuning Script
# Usage: ./scripts/run_sequential_finetune.sh [config_file]

set -e

# Default configuration
CONFIG_FILE="${1:-configs/sequential_finetune_default.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Sequential Finetuning Pipeline"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Project Root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Check if running with config file
if [ -f "$CONFIG_FILE" ]; then
    echo "✓ Using configuration file: $CONFIG_FILE"
    # TODO: Parse JSON config and pass arguments
    # For now, use default command line args
fi

# Run sequential finetuning
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
    --use_hf \
    --utm_steps 10000 \
    --ctw_steps 5000 \
    --batch_size 32 \
    --seq_length 256 \
    --lr 1e-5 \
    --utm_lr 1e-5 \
    --ctw_lr 5e-6 \
    --save_every 1000 \
    --log_every 100 \
    --output_dir "./checkpoints/sequential" \
    --seed 42 \
    "$@"

echo ""
echo "=========================================="
echo "✓ Sequential finetuning complete!"
echo "=========================================="
