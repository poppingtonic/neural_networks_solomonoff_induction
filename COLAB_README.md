# Running on Google Colab (Memory-Optimized)

This guide helps you run Skywork sequential fine-tuning on Google Colab's free tier (15GB GPU).

## Quick Start

### Option 1: Use the Colab Notebook (Easiest)

1. Open the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/neural_networks_solomonoff_induction/blob/main/colab_skywork_finetune.ipynb)

2. Enable GPU: **Runtime → Change runtime type → GPU (T4)**

3. Run all cells

### Option 2: Use the Minimal Script

```bash
# In Colab, run:
!git clone https://github.com/google-deepmind/neural_networks_solomonoff_induction.git
%cd neural_networks_solomonoff_induction

!pip install -q torch transformers bitsandbytes accelerate

!python colab_train_minimal.py \
  --model_name "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_8bit \
  --steps 1000 \
  --batch_size 4 \
  --seq_length 128 \
  --lr 5e-5
```

## Memory Optimization Strategies

### 🎯 Recommended Settings for Colab Free (15GB)

```python
CONFIG = {
    'batch_size': 16,          # With LoRA, can use larger batches!
    'seq_length': 256,         # Full length possible with LoRA
    'use_lora': True,          # ⭐ NEW - Reduces trainable params by 99%
    'lora_r': 8,              # LoRA rank
    'use_8bit': True,          # Essential - saves 4x memory
    'gradient_checkpointing': True,  # Saves ~30% memory
    'steps': 5000,             # More steps possible with LoRA efficiency
}
```

### 🚀 NEW: SWAG-LoRA (Best Option!)

```bash
!pip install peft git+https://github.com/fortuinlab/swag-lora.git

!python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --use_lora \
  --lora_r 8 \
  --use_swag \
  --stage all \
  --batch_size 16 \
  --seq_length 256
```

**Benefits:**
- ✅ 99% fewer trainable parameters
- ✅ 3-4x less memory usage
- ✅ Faster training
- ✅ Uncertainty quantification with SWAG
- ✅ Better batch sizes possible

### 🆘 If Out of Memory

**Level 1: Mild OOM**
```python
batch_size = 2
seq_length = 64
```

**Level 2: Severe OOM**
```python
batch_size = 1
seq_length = 32
use_8bit = True
```

**Level 3: Critical OOM**
Use LSTM instead of Skywork:
```bash
!python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --hidden_dim 128 \
  --num_layers 2 \
  --batch_size 16 \
  --seq_length 256
```

LSTM uses **~10x less memory** than Skywork!

## Memory Comparison

| Model | Memory (Full) | Memory (8-bit) | Memory (LoRA) | Colab Free |
|-------|---------------|----------------|---------------|------------|
| Skywork 0.6B | ~12GB | ~3GB | **~3.5GB** | ✅ Works |
| Skywork + LoRA r=8 (batch=16) | N/A | N/A | **~4GB** | ✅✅ Great |
| Skywork + LoRA r=16 (batch=16) | N/A | N/A | **~4.5GB** | ✅ Good |
| Custom LSTM | ~1GB | N/A | ~1GB | ✅✅✅ Best |

**⭐ Recommended: Skywork with LoRA r=8** - Best balance of memory, speed, and quality!

## Colab-Specific Commands

### Monitor GPU Memory
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Clear Memory
```python
import gc
import torch

# Delete model
del model
gc.collect()
torch.cuda.empty_cache()
```

### Check GPU Type
```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Colab GPUs:
- **T4**: 15GB (free tier)
- **V100**: 16GB (rarely on free)
- **A100**: 40GB (Colab Pro+)

## Training Time Estimates

### Colab Free (T4 GPU)

| Configuration | Time | Memory |
|---------------|------|--------|
| Minimal (batch=2, seq=64, 1k steps) | ~30 min | ~4GB |
| Standard (batch=4, seq=128, 2k steps) | ~2 hours | ~6GB |
| Full (batch=8, seq=256, 5k steps) | ~6 hours | ~12GB |

⚠️ **Colab free disconnects after ~12 hours**

### Tips for Long Training

1. **Save frequently**: `--save_every 250`
2. **Use Colab Pro**: 24-hour runtime, more memory
3. **Download checkpoints**: Before disconnection
4. **Resume training**: Load from checkpoint

## Download Checkpoints

```python
# Zip checkpoints
!zip -r checkpoints.zip ./checkpoints

# Download to local
from google.colab import files
files.download('checkpoints.zip')
```

## Upload to Google Drive (Auto-save)

```python
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
!cp -r ./checkpoints /content/drive/MyDrive/skywork_checkpoints
```

## Resume Training from Checkpoint

```python
# Load from Drive
!cp -r /content/drive/MyDrive/skywork_checkpoints ./checkpoints

# Resume (add checkpoint loading logic to script)
```

## Alternative: Use Smaller Model

If Skywork still OOMs, use a smaller HuggingFace model:

```bash
# GPT-2 Small (124M params)
!python colab_train_minimal.py \
  --model_name "gpt2" \
  --batch_size 8 \
  --seq_length 256

# TinyLlama (1.1B params)
!python colab_train_minimal.py \
  --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --use_8bit \
  --batch_size 4
```

## Best Practice for Colab

1. **Start small**: Test with 100 steps first
2. **Monitor memory**: Check after each epoch
3. **Use 8-bit**: Always enable for large models
4. **Save often**: Every 250-500 steps
5. **Mount Drive**: Auto-backup checkpoints
6. **Set alarms**: Before 12-hour limit
7. **Use LSTM**: If possible - much more efficient

## Troubleshooting

### "CUDA out of memory"
```python
# Emergency fix
batch_size = 1
seq_length = 32
```

### "Runtime disconnected"
- Colab free tier has usage limits
- Wait a few hours or upgrade to Pro
- Use Google Drive auto-save

### "Model download fails"
```bash
# Login to HuggingFace
!huggingface-cli login
# Then paste your token
```

### Training too slow
- Check GPU is enabled: `Runtime → Change runtime type`
- Reduce `log_every` to see less output
- Increase `batch_size` if memory allows

## Colab Pro Benefits

- **More memory**: Up to 40GB (A100)
- **Longer runtime**: 24 hours
- **Priority GPUs**: Faster T4/V100
- **Background execution**: Keep running when closed

## Cost Comparison

| Option | Cost | Memory | Runtime |
|--------|------|--------|---------|
| Colab Free | $0 | 15GB | 12h |
| Colab Pro | $10/mo | 16-40GB | 24h |
| Colab Pro+ | $50/mo | 40GB | 24h |
| Local RTX 3090 | ~$1500 | 24GB | Unlimited |

## Example Colab Workflow

```python
# 1. Setup
!git clone https://github.com/google-deepmind/neural_networks_solomonoff_induction.git
%cd neural_networks_solomonoff_induction
!pip install -q torch transformers bitsandbytes

# 2. Mount Drive for auto-save
from google.colab import drive
drive.mount('/content/drive')

# 3. Run training
!python colab_train_minimal.py \
  --model_name "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_8bit \
  --steps 2000 \
  --batch_size 4 \
  --seq_length 128 \
  --output_dir "/content/drive/MyDrive/checkpoints"

# 4. Monitor
!tail -f /content/drive/MyDrive/checkpoints/train.log
```

## Success Criteria

✅ Training runs without OOM  
✅ Loss decreases over time  
✅ Checkpoints saved successfully  
✅ Can load checkpoint and continue  
✅ Final model works on test data  

## Need Help?

1. Check memory: `nvidia-smi`
2. Reduce batch size to 1
3. Reduce seq_length to 32
4. Switch to LSTM architecture
5. Use Colab Pro for more memory
