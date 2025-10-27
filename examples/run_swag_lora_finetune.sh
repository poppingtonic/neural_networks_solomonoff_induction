#!/bin/bash
# SWAG-LoRA Sequential Fine-tuning Example
# Uses LoRA for memory efficiency + SWAG for uncertainty quantification

set -e

MODEL="Skywork/Skywork-Reward-V2-Qwen3-0.6B"
OUTPUT_DIR="./checkpoints/swag_lora_sequential"
DEVICE="cuda"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║      SWAG-LoRA Sequential Fine-tuning on Skywork              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""
echo "Features:"
echo "  ✓ LoRA (Parameter-efficient fine-tuning)"
echo "  ✓ SWAG (Uncertainty quantification)"
echo "  ✓ Sequential curriculum (UTM → CTW)"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import peft" 2>/dev/null || {
    echo "❌ peft not installed. Run: pip install peft"
    exit 1
}

python3 -c "import swag_lora" 2>/dev/null || {
    echo "⚠️  swag_lora not installed (optional for SWAG)"
    echo "   Install: pip install git+https://github.com/fortuinlab/swag-lora.git"
    echo "   Or: pip install -e ~/src/swag-lora"
    echo ""
}

echo "✓ Dependencies OK"
echo ""

# LoRA configuration
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# SWAG configuration
SWAG_START_EPOCH=10
SWAG_UPDATE_FREQ=100
SWAG_LR=1e-5

# Training configuration
BATCH_SIZE=16
SEQ_LENGTH=256
UTM_STEPS=5000
CTW_STEPS=2500
LR=5e-5

echo "Configuration:"
echo "  LoRA rank: $LORA_R"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  Batch size: $BATCH_SIZE"
echo "  Sequence length: $SEQ_LENGTH"
echo "  UTM steps: $UTM_STEPS"
echo "  CTW steps: $CTW_STEPS"
echo ""

# Run training
echo "Starting SWAG-LoRA sequential fine-tuning..."
echo ""

python torch_training/train_sequential_finetune.py \
  --model_name_or_path "$MODEL" \
  --use_hf \
  --use_lora \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --use_swag \
  --swag_start_epoch $SWAG_START_EPOCH \
  --swag_update_freq $SWAG_UPDATE_FREQ \
  --swag_lr $SWAG_LR \
  --stage all \
  --utm_steps $UTM_STEPS \
  --ctw_steps $CTW_STEPS \
  --utm_lr $LR \
  --ctw_lr $(echo "$LR * 0.5" | bc) \
  --batch_size $BATCH_SIZE \
  --seq_length $SEQ_LENGTH \
  --memory_size 10 \
  --maximum_steps 100 \
  --maximum_program_length 100 \
  --ctw_max_depth 5 \
  --save_every 500 \
  --log_every 50 \
  --output_dir "$OUTPUT_DIR" \
  --device $DEVICE \
  --seed 42

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Training Complete!                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - stage1_utm_final.pt"
echo "  - stage2_ctw_final.pt"
echo "  - train.log"
echo "  - metrics.json"
echo ""
echo "Memory savings from LoRA: ~99% fewer trainable parameters"
echo "SWAG provides uncertainty estimates for predictions"
echo ""
echo "Next steps:"
echo "  1. Evaluate model: python torch_training/evaluate_sequential.py"
echo "  2. Analyze uncertainty: Load SWAG model and sample predictions"
echo "  3. Compare with full fine-tuning baseline"
echo ""
