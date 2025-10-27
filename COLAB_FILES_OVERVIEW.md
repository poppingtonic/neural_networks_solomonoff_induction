# Colab Files Overview

This repository includes several files optimized for running on Google Colab with limited GPU memory.

## 📁 Files for Colab Users

### Main Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `colab_skywork_finetune.ipynb` | Interactive notebook | **Start here** - easiest option |
| `colab_train_minimal.py` | Standalone training script | Memory-efficient training |
| `COLAB_README.md` | Complete Colab guide | Full documentation |
| `COLAB_QUICK_START.txt` | Quick reference card | Common commands |
| `requirements_colab.txt` | Colab dependencies | Minimal install list |
| `setup_colab.sh` | One-command setup | Auto-setup environment |

### Integration with Main Repo

The Colab files work with the main training pipeline:
- `torch_training/train_sequential_finetune.py` - Full pipeline (works on Colab with low settings)
- `torch_models/lstm.py` - Memory-efficient LSTM (recommended for Colab)
- `torch_models/transformer.py` - Custom transformer

## 🚀 Quick Start Paths

### Path 1: Jupyter Notebook (Recommended for Beginners)

1. Open `colab_skywork_finetune.ipynb` in Colab
2. Enable GPU runtime
3. Run all cells

**Pros:** Interactive, step-by-step, easy to modify  
**Cons:** Need to keep browser open

### Path 2: Minimal Script (Recommended for Quick Tests)

```bash
!python colab_train_minimal.py --steps 100
```

**Pros:** Fast, simple, works out-of-box  
**Cons:** Uses dummy data (need to integrate real data generators)

### Path 3: Full Pipeline (Recommended for Production)

```bash
!python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --batch_size 8 \
  --seq_length 128
```

**Pros:** Full features, real data, production-ready  
**Cons:** Higher memory usage, longer setup

## 📊 Memory Comparison

| Approach | Memory | Setup Time | Features |
|----------|--------|------------|----------|
| `colab_train_minimal.py` | ~4GB | 2 min | Basic |
| LSTM via main pipeline | ~2GB | 5 min | Full |
| Skywork 8-bit | ~6GB | 5 min | Full |
| Skywork full precision | ~12GB | 5 min | Full |

## 🎯 Recommended Workflow

### For Quick Testing
```bash
# 1. Setup
!bash setup_colab.sh

# 2. Quick test
!python colab_train_minimal.py --steps 100
```

### For Real Training (LSTM)
```bash
# 1. Setup
!bash setup_colab.sh

# 2. Train LSTM (most memory-efficient)
!python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --hidden_dim 128 \
  --num_layers 2 \
  --stage all \
  --utm_steps 5000 \
  --ctw_steps 2500 \
  --batch_size 16 \
  --seq_length 256
```

### For Skywork Baseline
```bash
# 1. Setup
!bash setup_colab.sh

# 2. Train Skywork with 8-bit quantization
!python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --stage all \
  --utm_steps 2000 \
  --ctw_steps 1000 \
  --batch_size 4 \
  --seq_length 128
```

## 📖 Documentation Files

### Quick Reference
- `COLAB_QUICK_START.txt` - Copy-paste commands
- `SKYWORK_FINETUNING_GUIDE.md` - Detailed Skywork guide
- `LSTM_ARCHITECTURE.md` - LSTM explanation

### Complete Guides
- `COLAB_README.md` - Full Colab documentation
- `README.md` - Main repository README

## 🔧 Memory Optimization Features

### Implemented in Colab Files

1. **8-bit Quantization**
   - Saves ~4x memory
   - Enabled by default in `colab_train_minimal.py`
   - Use `--use_8bit` flag

2. **Gradient Checkpointing**
   - Saves ~30% memory
   - Auto-enabled for HuggingFace models
   - Trades compute for memory

3. **Small Batch Sizes**
   - Default: 4 (vs 32 in standard training)
   - Can reduce to 1 if needed

4. **Reduced Sequence Length**
   - Default: 128 (vs 256 in standard)
   - Can reduce to 32 if needed

5. **Aggressive Memory Cleanup**
   - `torch.cuda.empty_cache()` after each step
   - `gc.collect()` periodically
   - Delete tensors immediately after use

6. **Checkpoint Management**
   - Only keep last 2 checkpoints
   - Auto-cleanup old saves
   - Compress before download

## 🎓 Learning Path

### Beginner
1. Read `COLAB_QUICK_START.txt`
2. Open `colab_skywork_finetune.ipynb`
3. Run first few cells
4. Understand the output

### Intermediate
1. Read `COLAB_README.md`
2. Run `colab_train_minimal.py` with different configs
3. Experiment with hyperparameters
4. Monitor memory usage

### Advanced
1. Read `SKYWORK_FINETUNING_GUIDE.md`
2. Modify training script for custom data
3. Implement gradient accumulation
4. Optimize for specific GPU type

## 🆘 Troubleshooting Guide

### Issue: Out of Memory

1. Check current settings:
   ```python
   import torch
   print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

2. Try solutions in order:
   ```bash
   # Level 1
   --batch_size 2 --seq_length 64
   
   # Level 2
   --batch_size 1 --seq_length 32
   
   # Level 3
   --architecture lstm --batch_size 8
   ```

### Issue: Runtime Disconnected

1. Mount Google Drive before training:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Save to Drive:
   ```bash
   --output_dir "/content/drive/MyDrive/checkpoints"
   ```

### Issue: Slow Download

Use smaller model:
```bash
--model_name_or_path "gpt2"  # 124M params instead of 600M
```

## 📦 What's Not Included

These require manual setup:

1. **HuggingFace Login**
   ```bash
   !huggingface-cli login
   ```

2. **Weights & Biases Integration**
   ```bash
   !pip install wandb
   !wandb login
   ```

3. **Custom Data**
   - Need to upload or download separately
   - Mount Drive or use `files.upload()`

## 🔄 Updates and Maintenance

The Colab files are designed to be:
- **Self-contained**: Work without external dependencies
- **Version-stable**: Pin specific package versions
- **Memory-safe**: Always fit in 15GB limit
- **User-friendly**: Clear error messages

## 💡 Tips and Tricks

1. **Save frequently**: Colab can disconnect
2. **Use Drive**: Auto-backup checkpoints
3. **Monitor memory**: Check before each run
4. **Start small**: Test with 100 steps first
5. **Use LSTM**: When possible - 10x more efficient
6. **Enable 8-bit**: For large models
7. **Close browser**: Colab continues running
8. **Set timer**: Before 12-hour limit

## 📞 Getting Help

If you encounter issues:

1. Check `COLAB_README.md` troubleshooting section
2. Verify GPU is enabled
3. Try LSTM architecture first
4. Reduce batch size to 1
5. Check Colab status page
6. Use Colab Pro if consistently hitting limits

## 🎯 Success Criteria

Your setup is working if:

✅ GPU is detected (`nvidia-smi` works)  
✅ Training starts without OOM  
✅ Loss decreases over steps  
✅ Checkpoints save successfully  
✅ Can download results  

## 🌟 Recommended Configuration

For most users on Colab Free:

```bash
!python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --hidden_dim 256 \
  --num_layers 2 \
  --embedding_dim 128 \
  --stage all \
  --utm_steps 5000 \
  --ctw_steps 2500 \
  --batch_size 16 \
  --seq_length 256 \
  --lr 1e-4 \
  --save_every 500 \
  --output_dir "/content/drive/MyDrive/checkpoints"
```

This configuration:
- Uses memory-efficient LSTM
- Fits in 15GB memory
- Completes in ~2-3 hours
- Auto-saves to Google Drive
- Produces usable model

Good luck with your training! 🚀
