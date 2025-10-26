# Sequential Finetuning Recipe

## Overview
Sequential finetuning pipeline for Universal Predictors following the "Learning Universal Predictors" paper:
1. **Stage 1**: Finetune on random valid UTM data
2. **Stage 2**: Finetune on random CTW data

## Base Models & Techniques

### Base Model Options
- **Skywork-Reward-V2-Qwen3-0.6B**: `Skywork/Skywork-Reward-V2-Qwen3-0.6B` (HuggingFace)
- Custom Transformer (existing implementation)

### Optional Enhancements
- **SGM (Stochastic Gradient Matching)**: For improved optimization
- **BitNet**: 1-bit quantized networks for efficiency
- **DNI (Decoupled Neural Interfaces)**: ✅ Fully integrated for synthetic gradients (from `~/src/dni-pytorch`)
- **MultiSWAG**: ✅ Bayesian uncertainty (already integrated)

## Architecture

```
[Pretrained Base Model]
    ↓
[Stage 1: UTM Finetuning] → Checkpoint
    ↓
[Stage 2: CTW Finetuning] → Final Model
```

## Setup

### Dependencies
```bash
# Core PyTorch dependencies
pip install torch transformers accelerate

# Optional: BitNet (if using)
pip install bitnet

# Optional: DNI (if using)
# Clone and install from ~/src/dni-pytorch

# Optional: MultiSWAG
pip install push-bayes

# Existing requirements
pip install -r requirements.txt
```

### Data Sources
- **UTM Data**: Generated from `data/utm_data_generator.py`
- **CTW Data**: Generated from `data/ctw_data_generator.py`
- **Reference Logs**: `/home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log/*.log`

## Training Pipeline

### Stage 1: UTM Pretraining
```bash
python torch_training/train_sequential_finetune.py \
    --stage utm \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --batch_size 32 \
    --seq_length 256 \
    --training_steps 10000 \
    --lr 1e-5 \
    --output_dir ./checkpoints/stage1_utm \
    --save_every 1000
```

### Stage 2: CTW Finetuning
```bash
python torch_training/train_sequential_finetune.py \
    --stage ctw \
    --model_name_or_path ./checkpoints/stage1_utm/final \
    --batch_size 32 \
    --seq_length 256 \
    --training_steps 5000 \
    --lr 5e-6 \
    --output_dir ./checkpoints/stage2_ctw \
    --save_every 1000
```

### Combined Pipeline
```bash
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --utm_steps 10000 \
    --ctw_steps 5000 \
    --batch_size 32 \
    --seq_length 256 \
    --lr 1e-5 \
    --output_dir ./checkpoints/sequential
```

## Advanced Options

### With MultiSWAG
```bash
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --enable_mswag \
    --mswag_pretrain_epochs 200 \
    --mswag_swag_epochs 100
```

### With DNI (Decoupled Neural Interfaces) ✅
```bash
# DNI enables asynchronous layer updates using synthetic gradients
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_dni \
    --dni_num_points 2 \
    --dni_hidden_dim 64

# Demo DNI functionality
python examples/dni_demo.py
```

### With BitNet Quantization
```bash
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_bitnet \
    --bitnet_config configs/bitnet_config.json
```

## Hyperparameters

### Recommended Settings

#### UTM Stage
- **Learning Rate**: 1e-5 to 5e-5
- **Batch Size**: 16-32
- **Sequence Length**: 256-512
- **Steps**: 5000-20000
- **Warmup Steps**: 500

#### CTW Stage  
- **Learning Rate**: 5e-6 to 1e-5 (lower than UTM)
- **Batch Size**: 16-32
- **Sequence Length**: 256
- **Steps**: 2000-10000
- **Warmup Steps**: 200

### Optimizer
- **Adam/AdamW**: Default, stable
- **SGD with SGM**: If using stochastic gradient matching

## Monitoring

### Metrics to Track
- **Loss**: NLL loss per stage
- **Perplexity**: Token-level perplexity
- **Tree Depth**: For CTW stage (context complexity)
- **Gradient Norms**: For stability monitoring

### Tensorboard
```bash
tensorboard --logdir=./checkpoints/sequential/logs
```

## Checkpointing Strategy

1. **Intermediate Checkpoints**: Every N steps (--save_every)
2. **Best Model**: Based on validation loss
3. **Stage Boundaries**: Always save at stage transitions
4. **Final Model**: End of Stage 2

## Evaluation

### UTM Evaluation
- Prediction accuracy on held-out UTM programs
- Output sequence matching

### CTW Evaluation
- KL divergence from ground-truth CTW distribution
- Context prediction accuracy
- Tree depth recovery

### Combined Evaluation
- Universal prediction capability
- Transfer learning effectiveness

## Reference Papers
1. **Learning Universal Predictors** - Solomonoff induction via neural networks
2. **Context Tree Weighting** - Willems, Shtarkov, Tjalkens (1995)
3. **Decoupled Neural Interfaces** - Jaderberg et al. (2017)

## Logs & Debugging
- Training logs: `./checkpoints/sequential/train.log`
- Reference CTW logs: `/home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log/*.log`
- Debug mode: `--debug` flag

## Quick Start Example
```bash
# Minimal working example
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --utm_steps 1000 \
    --ctw_steps 500 \
    --batch_size 16 \
    --lr 1e-5 \
    --output_dir ./checkpoints/quickstart
```

## Troubleshooting

### Common Issues
1. **OOM Errors**: Reduce batch_size or seq_length
2. **Divergence**: Lower learning rate, add gradient clipping
3. **Slow Convergence**: Increase learning rate, check data quality
4. **Token Mismatch**: Verify tokenizer compatibility with base model

### Model-Specific Notes
- **Skywork models**: Use AutoTokenizer, check vocab size
- **Custom Transformers**: Match embedding dimensions
- **DNI integration**: ✅ Fully integrated - see `DNI_INTEGRATION.md` and `examples/dni_demo.py`
