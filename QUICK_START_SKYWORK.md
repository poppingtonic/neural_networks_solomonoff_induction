# Quick Start: Skywork Sequential Fine-tuning

## 1. Test Model Loading (Optional but Recommended)

```bash
python test_skywork_loading.py
```

Expected: ✅ Model loads successfully and forward pass works.

## 2. Run Sequential Fine-tuning

### Simple Command (Recommended)

```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage all \
  --utm_steps 10000 \
  --ctw_steps 5000 \
  --device cuda
```

### Quick Test (5 minutes)

```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage all \
  --utm_steps 500 \
  --ctw_steps 250 \
  --batch_size 8 \
  --log_every 50 \
  --device cuda
```

### Using Shell Script

```bash
./examples/run_skywork_finetune.sh
```

## 3. Monitor Progress

```bash
# In another terminal
tail -f ./checkpoints/skywork_sequential/train.log
```

## 4. Expected Results

- **Stage 1 (UTM)**: Loss decreases from ~4.0 to ~2.0
- **Stage 2 (CTW)**: Loss decreases from ~2.5 to ~1.5
- **Time**: ~2-4 hours for 15k total steps on GPU
- **Checkpoints**: Saved every 1000 steps

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch_size 8` and `--seq_length 128` |
| Model not found | Run `huggingface-cli login` first |
| No GPU | Use `--device cpu` (slower) |
| Slow download | Check internet connection |

## Compare with LSTM

```bash
# Train LSTM baseline
python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --stage all \
  --output_dir ./checkpoints/lstm_baseline

# Train Skywork
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage all \
  --output_dir ./checkpoints/skywork_baseline
```

## Full Documentation

See `SKYWORK_FINETUNING_GUIDE.md` for complete details and advanced options.
