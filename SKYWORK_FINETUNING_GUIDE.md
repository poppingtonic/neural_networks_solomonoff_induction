# Sequential Fine-tuning on Skywork Reward Model

## Quick Start

### Option 1: Run Both Stages Together (Recommended)

```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage all \
  --utm_steps 10000 \
  --ctw_steps 5000 \
  --utm_lr 1e-5 \
  --ctw_lr 5e-6 \
  --batch_size 32 \
  --seq_length 256 \
  --save_every 1000 \
  --log_every 100 \
  --output_dir ./checkpoints/skywork_sequential \
  --device cuda
```

### Option 2: Run Stages Separately

**Stage 1: UTM Fine-tuning**
```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage utm \
  --utm_steps 10000 \
  --utm_lr 1e-5 \
  --batch_size 32 \
  --seq_length 256 \
  --save_every 1000 \
  --log_every 100 \
  --output_dir ./checkpoints/skywork_sequential \
  --device cuda
```

**Stage 2: CTW Fine-tuning** (loads Stage 1 checkpoint automatically)
```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage ctw \
  --ctw_steps 5000 \
  --ctw_lr 5e-6 \
  --batch_size 32 \
  --seq_length 256 \
  --save_every 1000 \
  --log_every 100 \
  --output_dir ./checkpoints/skywork_sequential \
  --device cuda
```

### Option 3: Use the Shell Script

```bash
chmod +x examples/run_skywork_finetune.sh
./examples/run_skywork_finetune.sh
```

## Configuration Options

### Model Settings
- `--model_name_or_path`: HuggingFace model ID (e.g., "Skywork/Skywork-Reward-V2-Qwen3-0.6B")
- `--use_hf`: Force HuggingFace model loading (required for Skywork)
- `--device`: "cuda" or "cpu"

### Training Stages
- `--stage`: "utm", "ctw", or "all"
- `--utm_steps`: Number of steps for UTM stage (default: 10000)
- `--ctw_steps`: Number of steps for CTW stage (default: 5000)

### Learning Rates
- `--utm_lr`: Learning rate for UTM stage (default: 1e-5)
- `--ctw_lr`: Learning rate for CTW stage (default: 5e-6, half of utm_lr)
- `--lr`: Base learning rate (used if utm_lr/ctw_lr not specified)

### Data Configuration - UTM Stage
- `--batch_size`: Batch size (default: 32)
- `--seq_length`: Sequence length (default: 256)
- `--memory_size`: UTM memory size (default: 10)
- `--maximum_steps`: Maximum UTM steps (default: 100)
- `--tokenizer`: "ascii" or "seq_position" (default: "ascii")
- `--maximum_program_length`: Max program length (default: 100)

### Data Configuration - CTW Stage
- `--ctw_batch_size`: Override batch size for CTW (optional)
- `--ctw_seq_length`: Override sequence length for CTW (optional)
- `--ctw_max_depth`: Maximum tree depth (default: 5)

### Logging and Checkpointing
- `--output_dir`: Directory for checkpoints and logs
- `--save_every`: Save checkpoint every N steps (default: 1000)
- `--log_every`: Log metrics every N steps (default: 100)
- `--seed`: Random seed (default: 42)

### Advanced Options
- `--use_dni`: Enable Decoupled Neural Interfaces
- `--dni_num_points`: Number of DNI decoupling points (default: 2)
- `--dni_hidden_dim`: DNI synthesizer hidden dimension
- `--enable_mswag`: Enable MultiSWAG for uncertainty quantification

## Example Configurations

### Fast Test Run (Quick Validation)
```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage all \
  --utm_steps 100 \
  --ctw_steps 50 \
  --batch_size 8 \
  --seq_length 128 \
  --log_every 10 \
  --save_every 50 \
  --output_dir ./checkpoints/skywork_test \
  --device cuda
```

### Production Run (High Quality)
```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage all \
  --utm_steps 50000 \
  --ctw_steps 25000 \
  --utm_lr 5e-6 \
  --ctw_lr 2e-6 \
  --batch_size 64 \
  --seq_length 512 \
  --save_every 2000 \
  --log_every 200 \
  --output_dir ./checkpoints/skywork_production \
  --device cuda
```

### With DNI (Decoupled Neural Interfaces)
```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --use_dni \
  --dni_num_points 2 \
  --stage all \
  --utm_steps 10000 \
  --ctw_steps 5000 \
  --output_dir ./checkpoints/skywork_dni \
  --device cuda
```

### CPU-Only Run (No GPU)
```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --device cpu \
  --stage all \
  --utm_steps 1000 \
  --ctw_steps 500 \
  --batch_size 4 \
  --seq_length 128 \
  --output_dir ./checkpoints/skywork_cpu
```

## Expected Output

### Training Logs
```
Sequential Finetuning Pipeline
Model: Skywork/Skywork-Reward-V2-Qwen3-0.6B
Architecture: transformer
Device: cuda
Output: ./checkpoints/skywork_sequential

============================================================
STAGE 1: UTM Finetuning (10000 steps)
============================================================
[UTM] step=0/10000 loss=2.345678 ppl=10.438 grad_norm=1.234
[UTM] step=100/10000 loss=1.987654 ppl=7.298 grad_norm=0.987
[UTM] step=200/10000 loss=1.765432 ppl=5.845 grad_norm=0.765
...
✓ Stage 1 complete. Saved to ./checkpoints/skywork_sequential/stage1_utm_final.pt

============================================================
STAGE 2: CTW Finetuning (5000 steps)
============================================================
[CTW] step=0/5000 loss=1.654321 ppl=5.228 grad_norm=0.654
[CTW] step=100/5000 loss=1.432109 ppl=4.187 grad_norm=0.543
...
✓ Stage 2 complete. Saved to ./checkpoints/skywork_sequential/stage2_ctw_final.pt

============================================================
✓ Sequential finetuning complete!
============================================================
```

### Generated Files
```
./checkpoints/skywork_sequential/
├── train.log                      # Full training log
├── metrics.json                   # Training metrics
├── stage1_utm_step_1000.pt       # Intermediate checkpoints
├── stage1_utm_step_2000.pt
├── ...
├── stage1_utm_final.pt           # Final UTM checkpoint
├── stage2_ctw_step_1000.pt
├── ...
└── stage2_ctw_final.pt           # Final CTW checkpoint
```

## Monitoring Training

### View Logs in Real-time
```bash
tail -f ./checkpoints/skywork_sequential/train.log
```

### Check Metrics
```bash
python -c "import json; print(json.dumps(json.load(open('./checkpoints/skywork_sequential/metrics.json')), indent=2))"
```

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size and/or sequence length:
```bash
--batch_size 8 --seq_length 128
```

### Slow Training
Enable gradient accumulation (coming soon) or reduce dataset size:
```bash
--utm_steps 5000 --ctw_steps 2500
```

### Model Not Found
Ensure you have internet connection and HuggingFace access:
```bash
huggingface-cli login
```

### CUDA Not Available
Switch to CPU:
```bash
--device cpu
```

## Performance Tips

1. **Use Mixed Precision** (add to future update):
   - Can speed up training 2-3x on modern GPUs
   
2. **Tune Learning Rates**:
   - Start with 1e-5 for UTM, 5e-6 for CTW
   - Monitor loss curves and adjust
   
3. **Batch Size**:
   - Larger is generally better (32-64)
   - Limited by GPU memory
   
4. **Sequence Length**:
   - 256 is a good default
   - Increase to 512 for better quality (if memory allows)

## Next Steps

After training, evaluate the model:
```bash
python torch_training/evaluate_sequential.py \
  --checkpoint ./checkpoints/skywork_sequential/stage2_ctw_final.pt \
  --device cuda
```

Or run analysis:
```bash
python scripts/analyze_pareto_models.py \
  --checkpoint_dir ./checkpoints/skywork_sequential
```

## Comparing Skywork vs LSTM

To compare the Skywork model with the default LSTM:

**Train LSTM:**
```bash
python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --stage all \
  --output_dir ./checkpoints/lstm_sequential
```

**Train Skywork:**
```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage all \
  --output_dir ./checkpoints/skywork_sequential
```

**Compare results:**
```bash
python scripts/compare_stages.py \
  --model1 ./checkpoints/lstm_sequential \
  --model2 ./checkpoints/skywork_sequential
```
