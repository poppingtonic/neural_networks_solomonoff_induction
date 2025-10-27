# 📓 Consolidated Notebook Guide

## Main Notebook

**`Skywork_Sequential_Finetuning_Complete.ipynb`** - All-in-one training notebook

### What's Inside

This single notebook contains everything you need:

1. **Setup & Installation**
   - GPU check
   - Dependencies (torch, transformers, peft, swag-lora)
   - Repository clone

2. **5 Pre-configured Options**
   - **Option 1**: LSTM (~2GB, fastest)
   - **Option 2**: Skywork + LoRA (~4GB, recommended ⭐)
   - **Option 3**: Skywork + SWAG-LoRA (~4.5GB, best quality)
   - **Option 4**: Skywork Full (~12GB, Colab Pro)
   - **Option 5**: Quick Test (5 minutes)

3. **Training Commands**
   - One cell per option
   - Copy-paste ready
   - All parameters set

4. **Monitoring Tools**
   - GPU memory tracking
   - Log viewing
   - Progress monitoring

5. **Results Management**
   - Download checkpoints
   - Save to Google Drive
   - View metrics

6. **Troubleshooting**
   - Common issues
   - Quick fixes
   - Links to full documentation

## How to Use

### In Google Colab

1. **Upload notebook**
   ```
   File → Upload notebook → Select Skywork_Sequential_Finetuning_Complete.ipynb
   ```

2. **Enable GPU**
   ```
   Runtime → Change runtime type → GPU (T4) → Save
   ```

3. **Run cells in order**
   - Cell 1-3: Setup (5 minutes)
   - Cell 4: Select configuration (change SELECTED = 2)
   - Cell 5-9: Pick ONE training option
   - Cell 10+: Monitor and save results

### Direct Link

Upload to your Google Drive and open with:
```
https://colab.research.google.com/
```

## Configuration Selection Guide

| Option | Memory | Time | Quality | Use Case |
|--------|--------|------|---------|----------|
| 1 (LSTM) | 2GB | 1-2h | Good | Minimal memory, from scratch |
| 2 (LoRA) ⭐ | 4GB | 2-3h | Excellent | **Recommended for most** |
| 3 (SWAG-LoRA) | 4.5GB | 3-4h | Best + Uncertainty | Research, best quality |
| 4 (Full) | 12GB | 4-6h | Maximum | Colab Pro, unlimited budget |
| 5 (Test) | 3GB | 5min | N/A | Quick validation |

## Expected Results

### After Training

```
checkpoints/
├── sequential/
│   ├── stage1_utm_final.pt       # UTM trained model
│   ├── stage2_ctw_final.pt       # CTW trained model
│   ├── train.log                 # Training logs
│   └── metrics.json              # Performance metrics
```

### Memory Usage by Option

```
Option 1 (LSTM):         ████░░░░░░░░░░░ 2GB/15GB
Option 2 (LoRA):         ████████░░░░░░░ 4GB/15GB  ⭐
Option 3 (SWAG-LoRA):    █████████░░░░░░ 4.5GB/15GB
Option 4 (Full):         ████████████░░░ 12GB/15GB
Option 5 (Test):         ██████░░░░░░░░░ 3GB/15GB
```

## Features Included

### ✅ All Optimizations

- **SWAG-LoRA**: Parameter-efficient fine-tuning
- **8-bit quantization**: Automatic for HuggingFace models
- **Gradient checkpointing**: Memory savings
- **Sequential curriculum**: UTM → CTW progression
- **Automatic cleanup**: Memory management

### ✅ All Architectures

- **LSTM**: Custom, memory-efficient
- **Skywork**: Pretrained HuggingFace model
- **Transformer**: Custom implementation (via code)

### ✅ Monitoring

- Real-time GPU memory
- Training loss curves
- Log streaming
- Checkpoint saving

## Troubleshooting in Notebook

The notebook includes inline troubleshooting:

```python
# If Out of Memory
# 1. Reduce batch_size to 1
# 2. Reduce seq_length to 64
# 3. Use Option 1 (LSTM)
```

## Comparison with Other Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **Skywork_Sequential_Finetuning_Complete.ipynb** | **All-in-one notebook** | **Colab, interactive** |
| colab_skywork_finetune.ipynb | Original basic version | Learning, reference |
| colab_train_minimal.py | Standalone script | Command-line, automation |
| torch_training/train_sequential_finetune.py | Full pipeline | Local GPU, production |
| examples/run_swag_lora_finetune.sh | Shell script | Batch jobs, servers |

## Quick Start Commands

### Copy to Colab

```python
# After uploading notebook, run these cells:

# 1. Setup (run once)
!nvidia-smi
!pip install -q torch transformers peft git+https://github.com/fortuinlab/swag-lora.git
!git clone https://github.com/google-deepmind/neural_networks_solomonoff_induction.git
%cd neural_networks_solomonoff_induction

# 2. Run training (pick one)
# - Run cell for Option 1, 2, 3, 4, or 5
# - Wait for completion
# - Download results
```

## Advanced Usage

### Customize Configuration

Edit the training command in any option cell:

```bash
# Example: Increase steps
--utm_steps 10000 \
--ctw_steps 5000 \

# Example: Change LoRA rank
--lora_r 16 \
--lora_alpha 32 \

# Example: Add more logging
--log_every 25 \
--save_every 250 \
```

### Save to Google Drive

Notebook includes cell to mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r ./checkpoints /content/drive/MyDrive/skywork_checkpoints
```

## Full Documentation

For detailed information, see:

- **`SWAG_LORA_GUIDE.md`** - Complete LoRA guide
- **`SWAG_LORA_SUMMARY.md`** - Quick reference
- **`COLAB_README.md`** - Colab tips and tricks
- **`SKYWORK_FINETUNING_GUIDE.md`** - Skywork details
- **`COLAB_QUICK_START.txt`** - Command cheat sheet

## Tips for Best Results

1. **Start with Option 5** (Quick Test) to verify setup
2. **Use Option 2** (LoRA) for production training
3. **Monitor memory** in first few steps
4. **Save to Drive** to prevent data loss
5. **Download checkpoints** before session ends

## Notebook Advantages

✅ **Single file** - Everything in one place  
✅ **Interactive** - See results immediately  
✅ **Pre-configured** - No command-line needed  
✅ **Visual** - Plots and monitoring  
✅ **Documented** - Instructions included  
✅ **Shareable** - Easy to distribute  

The consolidated notebook provides the easiest way to run sequential fine-tuning on Colab! 🚀
