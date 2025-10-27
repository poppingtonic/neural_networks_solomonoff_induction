# 🚀 Start Here - Skywork Sequential Fine-tuning

## For Colab Users (Recommended)

**Use the consolidated notebook**: `Skywork_Sequential_Finetuning_Complete.ipynb`

### Quick Start (3 steps)

1. **Upload notebook to Google Colab**
   - Go to https://colab.research.google.com/
   - File → Upload notebook
   - Select `Skywork_Sequential_Finetuning_Complete.ipynb`

2. **Enable GPU**
   - Runtime → Change runtime type → GPU → Save

3. **Run all cells**
   - Runtime → Run all
   - Wait 2-3 hours for training

That's it! ✅

## What You Get

- ✅ **5 pre-configured options** (LSTM, LoRA, SWAG-LoRA, Full, Quick Test)
- ✅ **Memory optimized** for Colab Free (15GB)
- ✅ **SWAG-LoRA integrated** (99% parameter reduction)
- ✅ **All documentation** included in notebook
- ✅ **One-click training** - no command line needed

## Configuration Options in Notebook

| Option | Memory | Time | Best For |
|--------|--------|------|----------|
| 1. LSTM | 2GB | 1-2h | Minimal memory |
| 2. LoRA ⭐ | 4GB | 2-3h | **Recommended** |
| 3. SWAG-LoRA | 4.5GB | 3-4h | Best quality |
| 4. Full | 12GB | 4-6h | Colab Pro |
| 5. Quick Test | 3GB | 5min | Testing |

## For Local/Server Users

Use command line:

```bash
# LSTM (most efficient)
python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --stage all

# Skywork + LoRA (recommended)
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --use_lora \
  --lora_r 8 \
  --stage all
```

## Documentation

- **NOTEBOOK_GUIDE.md** - How to use the notebook
- **SWAG_LORA_GUIDE.md** - Complete LoRA documentation
- **COLAB_README.md** - Colab-specific tips
- **SKYWORK_FINETUNING_GUIDE.md** - Skywork details

## Installation

```bash
pip install torch transformers peft bitsandbytes
pip install git+https://github.com/fortuinlab/swag-lora.git
```

## Get Help

All answers in the notebook and documentation files!
