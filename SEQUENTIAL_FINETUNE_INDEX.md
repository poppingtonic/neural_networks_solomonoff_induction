# Sequential Finetuning Index

Complete reference for the sequential finetuning pipeline (UTM → CTW).

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART_SEQUENTIAL_FINETUNE.md](QUICKSTART_SEQUENTIAL_FINETUNE.md) | **Start here!** Quick start guide (5 minutes) |
| [SEQUENTIAL_FINETUNING_RECIPE.md](SEQUENTIAL_FINETUNING_RECIPE.md) | Complete recipe with all options and configurations |
| This file | Index of all resources |

## 🚀 Quick Commands

```bash
# Quick test (for debugging)
./scripts/quick_test.sh

# Full training pipeline
./scripts/run_sequential_finetune.sh

# Demo (minimal example)
python examples/sequential_finetune_demo.py

# Evaluation
python torch_training/evaluate_sequential.py \
    --checkpoint_path ./checkpoints/sequential/stage2_ctw_final.pt \
    --eval_utm --eval_ctw
```

## 📁 File Structure

```
neural_networks_solomonoff_induction/
│
├── 📖 Documentation
│   ├── QUICKSTART_SEQUENTIAL_FINETUNE.md      # Quick start guide
│   ├── SEQUENTIAL_FINETUNING_RECIPE.md        # Complete recipe
│   └── SEQUENTIAL_FINETUNE_INDEX.md           # This file
│
├── 🔧 Configuration Files
│   └── configs/
│       ├── sequential_finetune_default.json   # Default configuration
│       └── bitnet_config.json                 # BitNet configuration
│
├── 🐍 Core Training Scripts
│   └── torch_training/
│       ├── train_sequential_finetune.py       # Main training script ⭐
│       ├── evaluate_sequential.py             # Evaluation script
│       ├── utm_dataset.py                     # UTM dataset wrapper
│       ├── dni_adapter.py                     # DNI integration (optional)
│       ├── mswag_integration.py               # MultiSWAG support (optional)
│       └── mswag_utils.py                     # MultiSWAG utilities
│
├── 📊 Data Generators
│   └── data/
│       ├── utm_data_generator.py              # Universal Turing Machine data
│       └── ctw_data_generator.py              # Context Tree Weighting data
│
├── 🎯 Models
│   └── torch_models/
│       └── transformer.py                     # Transformer decoder model
│
├── 🔨 Shell Scripts
│   └── scripts/
│       ├── run_sequential_finetune.sh         # Main training script
│       ├── quick_test.sh                      # Quick test run
│       └── analyze_ctw_logs.py                # Analyze reference CTW logs
│
├── 🎨 Examples
│   └── examples/
│       └── sequential_finetune_demo.py        # Demo script
│
└── 📦 Dependencies
    ├── requirements.txt                       # Base requirements
    └── requirements_sequential_finetune.txt   # Sequential finetuning requirements
```

## 🎯 Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Base Model Loading                       │
│  • HuggingFace (Skywork-Reward-V2-Qwen3-0.6B)              │
│  • Custom Transformer                                        │
│  • Existing Checkpoint                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 1: UTM Finetuning                         │
│  • Universal Turing Machine data                            │
│  • Random valid programs                                     │
│  • Learn universal computation patterns                      │
│  └─► Checkpoint: stage1_utm_final.pt                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 2: CTW Finetuning                         │
│  • Context Tree Weighting data                              │
│  • Binary sequence prediction                                │
│  • Learn tree-structured context patterns                    │
│  └─► Checkpoint: stage2_ctw_final.pt                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation                                │
│  • UTM task performance                                      │
│  • CTW task performance                                      │
│  • Universal prediction capability                           │
└─────────────────────────────────────────────────────────────┘
```

## 🎓 Learning Path

### Beginner
1. Read [QUICKSTART_SEQUENTIAL_FINETUNE.md](QUICKSTART_SEQUENTIAL_FINETUNE.md)
2. Run `./scripts/quick_test.sh`
3. Run demo: `python examples/sequential_finetune_demo.py`
4. Modify hyperparameters and re-run

### Intermediate
1. Read [SEQUENTIAL_FINETUNING_RECIPE.md](SEQUENTIAL_FINETUNING_RECIPE.md)
2. Train with HuggingFace model: `./scripts/run_sequential_finetune.sh`
3. Evaluate on both tasks
4. Experiment with different model architectures

### Advanced
1. Integrate MultiSWAG for Bayesian uncertainty
2. Implement DNI (Decoupled Neural Interfaces)
3. Use BitNet quantization
4. Analyze CTW reference logs
5. Custom data generators and model architectures

## 🔑 Key Parameters

### Stage 1: UTM
- **Learning Rate**: 1e-5 to 5e-5
- **Steps**: 5,000 - 20,000
- **Batch Size**: 16-32
- **Sequence Length**: 256-512

### Stage 2: CTW
- **Learning Rate**: 5e-6 to 1e-5 (lower than Stage 1)
- **Steps**: 2,000 - 10,000
- **Batch Size**: 16-32
- **Sequence Length**: 256

## 🔬 Experiments

### Experiment 1: Baseline
```bash
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --utm_steps 10000 \
    --ctw_steps 5000 \
    --output_dir ./experiments/baseline
```

### Experiment 2: Custom Transformer
```bash
# Train base model first
python torch_training/train_sml.py \
    --training_steps 5000 \
    --save_path ./experiments/custom/base.pt

# Sequential finetune
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path ./experiments/custom/base.pt \
    --utm_steps 5000 \
    --ctw_steps 2000 \
    --output_dir ./experiments/custom/sequential
```

### Experiment 3: With MultiSWAG
```bash
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --enable_mswag \
    --mswag_pretrain_epochs 200 \
    --mswag_swag_epochs 100 \
    --output_dir ./experiments/mswag
```

## 📊 Evaluation

### Quick Evaluation
```bash
python torch_training/evaluate_sequential.py \
    --checkpoint_path ./checkpoints/sequential/stage2_ctw_final.pt \
    --eval_utm --eval_ctw \
    --num_eval_steps 100
```

### Comprehensive Evaluation
```bash
python torch_training/evaluate_sequential.py \
    --checkpoint_path ./checkpoints/sequential/stage2_ctw_final.pt \
    --eval_utm --eval_ctw \
    --num_eval_steps 1000 \
    --batch_size 64 \
    --output_file ./results/comprehensive_eval.json
```

## 🐛 Debugging

### Enable Debug Logging
```bash
python torch_training/train_sequential_finetune.py \
    --log_every 10 \
    --save_every 100 \
    ...
```

### Monitor Training
```bash
# In one terminal
python torch_training/train_sequential_finetune.py ...

# In another terminal
tail -f ./checkpoints/sequential/train.log
```

### Analyze CTW Reference Logs
```bash
python scripts/analyze_ctw_logs.py \
    --log_dir /home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log \
    --output ./analysis/ctw_logs.json
```

## 📚 References

### Papers
1. **Learning Universal Predictors** - Solomonoff induction
2. **Context Tree Weighting** (Willems et al., 1995)
3. **Decoupled Neural Interfaces** (Jaderberg et al., 2017)
4. **SWAG** - Stochastic Weight Averaging Gaussian

### External Resources
- HuggingFace Model: [Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
- DNI Implementation: `~/src/dni-pytorch`
- Reference CTW Logs: `/home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log/`

### Internal Documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Project overview
- [DEEP_ACTIVE_INFERENCE_INDEX.md](DEEP_ACTIVE_INFERENCE_INDEX.md) - Deep Active Inference
- [README.md](README.md) - General project info

## ⚙️ Configuration Examples

### Minimal Config (Fast Debug)
```json
{
  "utm_steps": 100,
  "ctw_steps": 50,
  "batch_size": 8,
  "seq_length": 64,
  "lr": 1e-4
}
```

### Production Config
```json
{
  "utm_steps": 50000,
  "ctw_steps": 20000,
  "batch_size": 64,
  "seq_length": 512,
  "lr": 5e-6,
  "save_every": 2000
}
```

## 🤝 Contributing

When adding new features to sequential finetuning:
1. Update this index
2. Add examples to `examples/`
3. Update documentation
4. Add tests if applicable

## 📞 Support

- Check existing documentation first
- Review examples and demos
- Examine log files for errors
- Verify hyperparameters match your use case

---

**Last Updated**: 2024
**Version**: 1.0
**Status**: Production Ready ✅
