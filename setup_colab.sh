#!/bin/bash
# One-command setup for Google Colab
# Run this in a Colab cell: !bash setup_colab.sh

set -e

echo "╔════════════════════════════════════════════════════╗"
echo "║   Colab Setup for Skywork Sequential Fine-tuning  ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Check if we're in Colab
if [ -d "/content" ]; then
    echo "✓ Google Colab environment detected"
    IN_COLAB=true
else
    echo "⚠️  Not running in Colab (continuing anyway)"
    IN_COLAB=false
fi

# Check GPU
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo "✓ GPU found: $GPU_NAME"
    echo "  Memory: $GPU_MEM"
else
    echo "❌ No GPU found - training will be very slow!"
    echo "   Enable GPU: Runtime → Change runtime type → GPU"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch transformers bitsandbytes accelerate
pip install -q jax jaxlib dm-haiku optax chex

echo "✓ Dependencies installed"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs

# Download test to verify transformers works
echo ""
echo "Testing HuggingFace access..."
python3 -c "from transformers import AutoTokenizer; print('✓ Transformers working')" 2>/dev/null || {
    echo "⚠️  Transformers test failed"
}

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║                  Setup Complete!                   ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "1. Quick test (5 minutes):"
echo "   python colab_train_minimal.py --steps 100 --batch_size 2"
echo ""
echo "2. LSTM training (recommended):"
echo "   python torch_training/train_sequential_finetune.py \\"
echo "     --architecture lstm \\"
echo "     --stage all \\"
echo "     --batch_size 16"
echo ""
echo "3. Skywork training:"
echo "   python colab_train_minimal.py \\"
echo "     --model_name 'Skywork/Skywork-Reward-V2-Qwen3-0.6B' \\"
echo "     --use_8bit \\"
echo "     --batch_size 4"
echo ""
echo "See COLAB_README.md for full documentation"
echo ""
