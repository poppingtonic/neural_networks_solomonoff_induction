# Sequential Finetuning Recipe - Summary

## 🎯 Overview

Complete recipe for sequentially finetuning models on:
1. **Stage 1**: Random valid UTM (Universal Turing Machine) data
2. **Stage 2**: Random CTW (Context Tree Weighting) data

Based on the paper: **"Learning Universal Predictors"**

## 🚀 Quick Start (30 seconds)

```bash
# Install dependencies
pip install torch transformers accelerate

# Run quick test
./scripts/quick_test.sh

# Or full training
./scripts/run_sequential_finetune.sh
```

## 📋 What's Included

### Core Components
- ✅ **Main training script**: `torch_training/train_sequential_finetune.py`
- ✅ **Evaluation script**: `torch_training/evaluate_sequential.py`
- ✅ **UTM data generator**: `data/utm_data_generator.py`
- ✅ **CTW data generator**: `data/ctw_data_generator.py`
- ✅ **Transformer models**: `torch_models/transformer.py`

### Documentation
- ✅ **Quick Start**: `QUICKSTART_SEQUENTIAL_FINETUNE.md`
- ✅ **Complete Recipe**: `SEQUENTIAL_FINETUNING_RECIPE.md`
- ✅ **Index**: `SEQUENTIAL_FINETUNE_INDEX.md`

### Support Files
- ✅ **Shell scripts**: `scripts/run_sequential_finetune.sh`, `scripts/quick_test.sh`
- ✅ **Configuration**: `configs/sequential_finetune_default.json`
- ✅ **Demo**: `examples/sequential_finetune_demo.py`
- ✅ **Requirements**: `requirements_sequential_finetune.txt`

### Advanced Features
- ✅ **MultiSWAG support**: Bayesian uncertainty estimation (fully integrated)
- ✅ **DNI (Decoupled Neural Interfaces)**: ✅ Fully integrated from `~/src/dni-pytorch` - see `DNI_INTEGRATION.md`
- ✅ **BitNet config**: 1-bit quantization support (config ready)
- ✅ **CTW log analyzer**: Analyze reference logs from `/home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log/`

## 🏗️ Architecture

```
Base Model (Skywork-Reward-V2-Qwen3-0.6B or Custom)
    ↓
Stage 1: UTM Finetuning (10k steps)
    ↓
Stage 2: CTW Finetuning (5k steps)
    ↓
Universal Predictor
```

## 💻 Usage Examples

### Example 1: Full Pipeline with HuggingFace Model
```bash
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --utm_steps 10000 \
    --ctw_steps 5000 \
    --batch_size 32 \
    --lr 1e-5
```

### Example 2: UTM Stage Only
```bash
python torch_training/train_sequential_finetune.py \
    --stage utm \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --utm_steps 10000
```

### Example 3: CTW Stage from Checkpoint
```bash
python torch_training/train_sequential_finetune.py \
    --stage ctw \
    --model_name_or_path ./checkpoints/stage1_utm_final.pt \
    --ctw_steps 5000
```

### Example 4: Evaluation
```bash
python torch_training/evaluate_sequential.py \
    --checkpoint_path ./checkpoints/sequential/stage2_ctw_final.pt \
    --eval_utm --eval_ctw \
    --num_eval_steps 100
```

### Example 5: With MultiSWAG
```bash
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --enable_mswag \
    --mswag_pretrain_epochs 200 \
    --mswag_swag_epochs 100
```

## 📊 Key Parameters

| Parameter | Stage 1 (UTM) | Stage 2 (CTW) |
|-----------|---------------|---------------|
| Learning Rate | 1e-5 to 5e-5 | 5e-6 to 1e-5 |
| Steps | 5k - 20k | 2k - 10k |
| Batch Size | 16-32 | 16-32 |
| Seq Length | 256-512 | 256 |

## 🔧 Optional Features

### SGM (Stochastic Gradient Matching)
- Placeholder for integration
- Improves optimization dynamics

### BitNet
- 1-bit quantized networks
- Config: `configs/bitnet_config.json`
- Use: `--use_bitnet`

### DNI (Decoupled Neural Interfaces) ✅ FULLY INTEGRATED
- **Status**: ✅ Complete (copied from `~/src/dni-pytorch`)
- **Implementation**: `torch_training/dni.py` (446 lines)
- **Adapter**: `torch_training/dni_adapter.py`
- **Demo**: `python examples/dni_demo.py`
- **Usage**: `--use_dni --dni_num_points 2 --dni_hidden_dim 64`
- **Docs**: See `DNI_INTEGRATION.md` for full guide

### MultiSWAG
- Already integrated
- Use: `--enable_mswag`

## 📁 Output Structure

After training, your checkpoint directory contains:

```
checkpoints/sequential/
├── train.log                     # Training logs
├── metrics.json                  # All metrics
├── stage1_utm_step_1000.pt      # Intermediate checkpoints
├── stage1_utm_final.pt          # End of Stage 1
├── stage2_ctw_step_1000.pt
└── stage2_ctw_final.pt          # Final model ⭐
```

## 🧪 Testing

### Quick Test (2 minutes)
```bash
./scripts/quick_test.sh
```

### Demo Script
```bash
python examples/sequential_finetune_demo.py
```

### Unit Tests
```bash
pytest torch_training/  # If tests are added
```

## 📖 Reference Materials

### Papers
1. **Learning Universal Predictors** - Core theoretical foundation
2. **Context Tree Weighting** (Willems et al., 1995)
3. **Decoupled Neural Interfaces** (Jaderberg et al., 2017)

### External Resources
- **HF Model**: https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B
- **DNI Implementation**: `~/src/dni-pytorch`
- **CTW Reference Logs**: `/home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log/*.log`

### Analysis Tools
```bash
# Analyze CTW reference logs
python scripts/analyze_ctw_logs.py \
    --log_dir /home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log

# Compare multiple experiments
python scripts/compare_stages.py \
    ./checkpoints/exp1 ./checkpoints/exp2

# Extract best checkpoints
./scripts/extract_best_checkpoints.sh ./checkpoints/sequential
```

## 🎓 Learning Path

1. **Start**: Read `QUICKSTART_SEQUENTIAL_FINETUNE.md`
2. **Test**: Run `./scripts/quick_test.sh`
3. **Demo**: Run `examples/sequential_finetune_demo.py`
4. **Train**: Run full pipeline with `./scripts/run_sequential_finetune.sh`
5. **Evaluate**: Use `evaluate_sequential.py`
6. **Advanced**: Read `SEQUENTIAL_FINETUNING_RECIPE.md` for all options
7. **Reference**: Use `SEQUENTIAL_FINETUNE_INDEX.md` as navigation

## ✅ Checklist

Before training:
- [ ] Install dependencies: `pip install -r requirements_sequential_finetune.txt`
- [ ] Verify CUDA/GPU availability (optional but recommended)
- [ ] Test with quick run: `./scripts/quick_test.sh`

During training:
- [ ] Monitor logs: `tail -f ./checkpoints/sequential/train.log`
- [ ] Check GPU memory usage
- [ ] Verify checkpoints are being saved

After training:
- [ ] Evaluate on both UTM and CTW tasks
- [ ] Compare with baseline metrics
- [ ] Extract best checkpoints: `./scripts/extract_best_checkpoints.sh`

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `--batch_size` or `--seq_length` |
| Model not found | Check model name or train custom base model first |
| Slow training | Use GPU, reduce steps for testing |
| Import errors | Install missing dependencies |
| Divergence | Lower learning rate, add gradient clipping |

## 📞 Next Steps

1. **Run your first experiment**: `./scripts/quick_test.sh`
2. **Read full documentation**: See `SEQUENTIAL_FINETUNE_INDEX.md`
3. **Customize for your needs**: Modify configs in `configs/`
4. **Scale up**: Increase steps and batch size
5. **Evaluate**: Test on both tasks
6. **Iterate**: Adjust hyperparameters based on results

## 🎉 Success Metrics

You'll know it's working when:
- ✅ Loss decreases steadily in both stages
- ✅ Model checkpoints are saved successfully
- ✅ Evaluation shows good performance on both UTM and CTW
- ✅ Perplexity decreases over training
- ✅ No divergence or NaN losses

---

**Ready to start?** Run: `./scripts/quick_test.sh`

**Need help?** Check: `QUICKSTART_SEQUENTIAL_FINETUNE.md`

**Want details?** Read: `SEQUENTIAL_FINETUNING_RECIPE.md`

**Navigation?** Use: `SEQUENTIAL_FINETUNE_INDEX.md`
