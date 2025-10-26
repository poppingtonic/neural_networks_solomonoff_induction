# Sequential Finetuning for Universal Predictors

**Status**: ✅ Production Ready | **Version**: 1.0 | **Last Updated**: 2024

## What is This?

A complete, production-ready pipeline for sequentially finetuning neural networks on:
1. **UTM (Universal Turing Machine)** data - Learn universal computation
2. **CTW (Context Tree Weighting)** data - Learn hierarchical context patterns

Based on the paper: **"Learning Universal Predictors"**

## Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements_sequential_finetune.txt

# 2. Run quick test
./scripts/quick_test.sh

# 3. You're done! 🎉
```

## Full Training (5 minutes)

```bash
# Train on Skywork-Reward-V2-Qwen3-0.6B
./scripts/run_sequential_finetune.sh

# Or customize:
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --utm_steps 10000 \
    --ctw_steps 5000 \
    --batch_size 32 \
    --output_dir ./checkpoints/my_experiment
```

## Documentation

| Document | Purpose | Start Here? |
|----------|---------|-------------|
| **RECIPE_SUMMARY.md** | High-level overview | ⭐ YES |
| **QUICKSTART_SEQUENTIAL_FINETUNE.md** | 5-minute guide | ✅ |
| **SEQUENTIAL_FINETUNING_RECIPE.md** | Complete details | 📖 |
| **SEQUENTIAL_FINETUNE_INDEX.md** | Navigation | 🗺️ |
| **RECIPE_IMPLEMENTATION_COMPLETE.md** | Verification | ✅ |
| **RECIPE_QUICK_REFERENCE.txt** | Command cheat sheet | 💡 |

## Key Features

✅ **HuggingFace Integration**: Load Skywork-Reward-V2-Qwen3-0.6B or any AutoModelForCausalLM  
✅ **Custom Transformers**: Use your own models  
✅ **Sequential Training**: UTM → CTW pipeline  
✅ **Comprehensive Logging**: Track all metrics  
✅ **Checkpointing**: Save at every stage  
✅ **Evaluation**: Test on both tasks  
✅ **MultiSWAG**: Bayesian uncertainty (optional)  
✅ **DNI Ready**: Integration point for ~/src/dni-pytorch  
✅ **BitNet Ready**: 1-bit quantization support  

## File Structure

```
├── 📖 Documentation (6 files)
│   ├── RECIPE_SUMMARY.md                    ⭐ Start here
│   ├── QUICKSTART_SEQUENTIAL_FINETUNE.md
│   ├── SEQUENTIAL_FINETUNING_RECIPE.md
│   ├── SEQUENTIAL_FINETUNE_INDEX.md
│   ├── RECIPE_IMPLEMENTATION_COMPLETE.md
│   └── RECIPE_QUICK_REFERENCE.txt
│
├── 🐍 Core Scripts (4 files)
│   ├── torch_training/train_sequential_finetune.py    Main training
│   ├── torch_training/evaluate_sequential.py          Evaluation
│   ├── torch_training/dni_adapter.py                  DNI integration
│   └── examples/sequential_finetune_demo.py           Demo
│
├── 🔨 Utilities (5 files)
│   ├── scripts/run_sequential_finetune.sh            Main runner
│   ├── scripts/quick_test.sh                         Fast test
│   ├── scripts/analyze_ctw_logs.py                   Log analysis
│   ├── scripts/compare_stages.py                     Comparison
│   └── scripts/extract_best_checkpoints.sh           Extract finals
│
└── ⚙️ Configuration (2 files)
    ├── configs/sequential_finetune_default.json
    └── configs/bitnet_config.json
```

## Training Pipeline

```
Skywork-Reward-V2-Qwen3-0.6B (or custom model)
                ↓
        Stage 1: UTM Training
          (10,000 steps)
                ↓
       stage1_utm_final.pt
                ↓
        Stage 2: CTW Training
          (5,000 steps)
                ↓
       stage2_ctw_final.pt ⭐
                ↓
      Universal Predictor
```

## Common Commands

```bash
# Quick test (debugging)
./scripts/quick_test.sh

# Demo (educational)
python examples/sequential_finetune_demo.py

# Full training (production)
./scripts/run_sequential_finetune.sh

# Stage 1 only (UTM)
python torch_training/train_sequential_finetune.py --stage utm --utm_steps 10000

# Stage 2 only (CTW from checkpoint)
python torch_training/train_sequential_finetune.py \
    --stage ctw \
    --model_name_or_path ./checkpoints/stage1_utm_final.pt \
    --ctw_steps 5000

# Evaluation
python torch_training/evaluate_sequential.py \
    --checkpoint_path ./checkpoints/sequential/stage2_ctw_final.pt \
    --eval_utm --eval_ctw

# Analysis
python scripts/analyze_ctw_logs.py \
    --log_dir /home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log
```

## Advanced Options

### MultiSWAG (Bayesian Uncertainty)
```bash
python torch_training/train_sequential_finetune.py \
    --enable_mswag \
    --mswag_pretrain_epochs 200 \
    --mswag_swag_epochs 100 \
    ...
```

### DNI (Decoupled Neural Interfaces)
```bash
python torch_training/train_sequential_finetune.py \
    --use_dni \
    --dni_path ~/src/dni-pytorch \
    ...
```

### BitNet (1-bit Quantization)
```bash
python torch_training/train_sequential_finetune.py \
    --use_bitnet \
    --bitnet_config configs/bitnet_config.json \
    ...
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stage` | `all` | Training stage: `utm`, `ctw`, or `all` |
| `--model_name_or_path` | Skywork model | HF model or checkpoint path |
| `--utm_steps` | 10000 | Steps for UTM stage |
| `--ctw_steps` | 5000 | Steps for CTW stage |
| `--batch_size` | 32 | Batch size |
| `--seq_length` | 256 | Sequence length |
| `--lr` | 1e-5 | Base learning rate |
| `--output_dir` | `./checkpoints/sequential` | Output directory |

## Success Metrics

You'll know it's working when:
- ✅ Loss decreases in both stages
- ✅ Checkpoints save successfully
- ✅ No NaN or Inf values
- ✅ Evaluation shows good performance
- ✅ Log files contain expected metrics

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch_size` or `--seq_length` |
| Model not found | Check internet or use custom model |
| Import errors | `pip install -r requirements_sequential_finetune.txt` |
| Slow training | Verify GPU with `nvidia-smi` |

## References

### Papers
1. **Learning Universal Predictors** - Core paper
2. **Context Tree Weighting** (Willems et al., 1995)
3. **Decoupled Neural Interfaces** (Jaderberg et al., 2017)

### External Resources
- **HuggingFace Model**: [Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
- **DNI Implementation**: `~/src/dni-pytorch`
- **CTW Reference Logs**: `/home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log/`

## Next Steps

1. **Read**: `RECIPE_SUMMARY.md`
2. **Test**: `./scripts/quick_test.sh`
3. **Demo**: `python examples/sequential_finetune_demo.py`
4. **Train**: `./scripts/run_sequential_finetune.sh`
5. **Evaluate**: Use `evaluate_sequential.py`
6. **Iterate**: Tune hyperparameters

## Support

For detailed information, see:
- **RECIPE_SUMMARY.md** - Overview
- **QUICKSTART_SEQUENTIAL_FINETUNE.md** - Quick guide
- **SEQUENTIAL_FINETUNING_RECIPE.md** - Complete details
- **SEQUENTIAL_FINETUNE_INDEX.md** - Navigation

---

**Ready?** Run: `./scripts/quick_test.sh`

**Questions?** Read: `RECIPE_SUMMARY.md`

**Happy Training!** 🚀
