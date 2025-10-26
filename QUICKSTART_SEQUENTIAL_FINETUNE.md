# Quick Start: Sequential Finetuning

Get started with sequential finetuning (UTM → CTW) in 5 minutes.

## 1. Installation

```bash
# Install dependencies
pip install -r requirements_sequential_finetune.txt

# Or minimal setup
pip install torch transformers accelerate
```

## 2. Basic Usage

### Option A: Use the Shell Script (Easiest)

```bash
# Make scripts executable
chmod +x scripts/run_sequential_finetune.sh
chmod +x scripts/quick_test.sh

# Quick test (fast, for debugging)
./scripts/quick_test.sh

# Full training
./scripts/run_sequential_finetune.sh
```

### Option B: Direct Python Command

```bash
# Full sequential finetuning
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --utm_steps 10000 \
    --ctw_steps 5000 \
    --batch_size 32 \
    --seq_length 256 \
    --output_dir ./checkpoints/sequential
```

## 3. Stage-by-Stage Training

### Stage 1 Only (UTM)

```bash
python torch_training/train_sequential_finetune.py \
    --stage utm \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --utm_steps 10000 \
    --output_dir ./checkpoints/utm_only
```

### Stage 2 Only (CTW) - From UTM Checkpoint

```bash
python torch_training/train_sequential_finetune.py \
    --stage ctw \
    --model_name_or_path ./checkpoints/utm_only/stage1_utm_final.pt \
    --ctw_steps 5000 \
    --output_dir ./checkpoints/ctw_from_utm
```

## 4. Evaluation

```bash
# Evaluate final model on both tasks
python torch_training/evaluate_sequential.py \
    --checkpoint_path ./checkpoints/sequential/stage2_ctw_final.pt \
    --eval_utm \
    --eval_ctw \
    --num_eval_steps 100 \
    --output_file ./results/eval_results.json
```

## 5. Monitor Training

```bash
# View logs
tail -f ./checkpoints/sequential/train.log

# Check metrics
cat ./checkpoints/sequential/metrics.json
```

## 6. Common Scenarios

### Scenario 1: Quick Test Run
```bash
./scripts/quick_test.sh
```

### Scenario 2: Small-Scale Experiment
```bash
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --utm_steps 2000 \
    --ctw_steps 1000 \
    --batch_size 16 \
    --seq_length 128 \
    --save_every 500
```

### Scenario 3: Large-Scale Training
```bash
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --utm_steps 50000 \
    --ctw_steps 20000 \
    --batch_size 64 \
    --seq_length 512 \
    --lr 5e-6 \
    --save_every 2000
```

### Scenario 4: Custom Transformer (No HuggingFace)
```bash
# First train a custom transformer on UTM
python torch_training/train_sml.py \
    --training_steps 5000 \
    --save_path ./checkpoints/custom_base.pt

# Then finetune sequentially
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path ./checkpoints/custom_base.pt \
    --utm_steps 5000 \
    --ctw_steps 2000
```

## 7. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stage` | `all` | Training stage: `utm`, `ctw`, or `all` |
| `--utm_steps` | 10000 | Steps for UTM stage |
| `--ctw_steps` | 5000 | Steps for CTW stage |
| `--batch_size` | 32 | Batch size |
| `--seq_length` | 256 | Sequence length |
| `--lr` | 1e-5 | Base learning rate |
| `--save_every` | 1000 | Save checkpoint every N steps |

## 8. Output Structure

After training, your checkpoint directory will contain:

```
checkpoints/sequential/
├── train.log                  # Training logs
├── metrics.json               # Training metrics
├── stage1_utm_step_1000.pt   # Intermediate checkpoints
├── stage1_utm_step_2000.pt
├── stage1_utm_final.pt       # End of Stage 1
├── stage2_ctw_step_1000.pt
├── stage2_ctw_final.pt       # Final model
```

## 9. Troubleshooting

### Issue: CUDA out of memory
```bash
# Reduce batch size and sequence length
python torch_training/train_sequential_finetune.py \
    --batch_size 8 \
    --seq_length 128 \
    ...
```

### Issue: Model not found
```bash
# Check HuggingFace model name or use custom model
# For custom model, train one first:
python torch_training/train_sml.py --training_steps 1000
```

### Issue: Slow training
```bash
# Use smaller model or fewer steps
python torch_training/train_sequential_finetune.py \
    --utm_steps 1000 \
    --ctw_steps 500 \
    --batch_size 16
```

## 10. Next Steps

- Read the full [Sequential Finetuning Recipe](SEQUENTIAL_FINETUNING_RECIPE.md)
- Analyze CTW logs: `python scripts/analyze_ctw_logs.py`
- Experiment with advanced options (MultiSWAG, DNI, BitNet)
- Run evaluation scripts
- Fine-tune hyperparameters based on your task

## Need Help?

Check the full documentation:
- `SEQUENTIAL_FINETUNING_RECIPE.md` - Complete guide
- `IMPLEMENTATION_SUMMARY.md` - Project overview
- `README.md` - General information

Happy training! 🚀
