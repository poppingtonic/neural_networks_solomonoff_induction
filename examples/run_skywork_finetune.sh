#!/bin/bash
# Sequential Fine-tuning on Skywork-Reward-V2-Qwen3-0.6B
# This script demonstrates how to fine-tune the Skywork reward model
# on UTM and CTW tasks for Solomonoff induction

set -e

# Configuration
MODEL_NAME="Skywork/Skywork-Reward-V2-Qwen3-0.6B"
OUTPUT_DIR="./checkpoints/skywork_sequential"
DEVICE="cuda"  # Change to "cpu" if no GPU available

echo "============================================================"
echo "Sequential Fine-tuning: Skywork Reward Model"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# Stage 1: UTM Fine-tuning
echo "Starting Stage 1: UTM Fine-tuning..."
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "$MODEL_NAME" \
  --use_hf \
  --device "$DEVICE" \
  --stage utm \
  --utm_steps 5000 \
  --utm_lr 1e-5 \
  --batch_size 16 \
  --seq_length 256 \
  --memory_size 10 \
  --maximum_steps 100 \
  --tokenizer ascii \
  --maximum_program_length 100 \
  --save_every 500 \
  --log_every 50 \
  --output_dir "$OUTPUT_DIR" \
  --seed 42

echo ""
echo "✓ Stage 1 complete!"
echo ""

# Stage 2: CTW Fine-tuning (starting from Stage 1 checkpoint)
echo "Starting Stage 2: CTW Fine-tuning..."
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "$OUTPUT_DIR/stage1_utm_final.pt" \
  --device "$DEVICE" \
  --stage ctw \
  --ctw_steps 2500 \
  --ctw_lr 5e-6 \
  --ctw_batch_size 16 \
  --ctw_seq_length 256 \
  --ctw_max_depth 5 \
  --save_every 250 \
  --log_every 50 \
  --output_dir "$OUTPUT_DIR" \
  --seed 42

echo ""
echo "============================================================"
echo "✓ Sequential fine-tuning complete!"
echo "============================================================"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "- stage1_utm_final.pt"
echo "- stage2_ctw_final.pt"
echo ""
