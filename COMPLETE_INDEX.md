# Complete Project Index

## 🎯 Quick Start

**For Colab**: Use `Skywork_Sequential_Finetuning_Complete.ipynb` (all-in-one notebook)  
**For Local**: See `START_HERE.md` for command-line instructions

---

## 📓 Notebooks (Colab)

| File | Purpose | Use When |
|------|---------|----------|
| **Skywork_Sequential_Finetuning_Complete.ipynb** | **All-in-one consolidated notebook** | **Start here for Colab** |
| colab_skywork_finetune.ipynb | Original detailed notebook | Learning, reference |

---

## 📚 Documentation

### Quick Start Guides

- **START_HERE.md** - Ultimate quick start
- **NOTEBOOK_GUIDE.md** - Notebook instructions
- **QUICK_START_SKYWORK.md** - Skywork quick reference
- **COLAB_QUICK_START.txt** - Command cheat sheet

### Complete Guides

- **SWAG_LORA_GUIDE.md** - Complete LoRA documentation (9.8KB)
- **SWAG_LORA_SUMMARY.md** - LoRA quick reference (8.1KB)
- **SKYWORK_FINETUNING_GUIDE.md** - Skywork detailed guide (8.4KB)
- **COLAB_README.md** - Colab tips and tricks (7.4KB)
- **COLAB_FILES_OVERVIEW.md** - File descriptions (7.3KB)

### Feature Guides

- **DNI_INTEGRATION.md** - Decoupled Neural Interfaces (9.7KB)
- **DNI_COMPLETE_SUMMARY.md** - DNI quick reference (7.5KB)
- **LSTM_UPDATE.md** - LSTM architecture details (3.6KB)
- **SEQUENTIAL_FINETUNE_INDEX.md** - Sequential training index (10.5KB)

### Research & Analysis

- **PARETO_REWARD_MODELS_MATRIX.md** - Model comparison (15KB)
- **PARETO_ANALYSIS_UPDATE_LOG.md** - Analysis updates (10.2KB)
- **PARETO_QUICK_REFERENCE.md** - Quick model comparison (4.2KB)
- **DEEP_ACTIVE_INFERENCE_INDEX.md** - Active inference guide (8.1KB)

---

## 🐍 Python Scripts

### Training Scripts

| Script | Purpose | Memory | Use Case |
|--------|---------|--------|----------|
| **torch_training/train_sequential_finetune.py** | Main training pipeline | Variable | Production |
| **colab_train_minimal.py** | Colab-optimized standalone | ~4GB | Quick Colab |
| **train_deep_active_inference.py** | Active inference training | ~2GB | Research |

### Adapter & Integration

- **torch_training/swag_lora_adapter.py** - SWAG-LoRA integration
- **torch_training/dni_adapter.py** - DNI integration
- **torch_training/utm_dataset.py** - UTM data handling

### Test Scripts

- **test_skywork_loading.py** - Verify Skywork loads
- **test_lstm.py** - LSTM architecture test
- **test_deep_active_inference.py** - Active inference test

---

## 🔧 Configuration Files

- **requirements.txt** - Minimal dependencies
- **requirements_colab.txt** - Colab-specific
- **requirements_swag_lora.txt** - SWAG-LoRA dependencies
- **requirements_sequential_finetune.txt** - Sequential training
- **requirements_deep_active_inference.txt** - Active inference

---

## 🚀 Example Scripts

In `examples/` directory:

- **run_skywork_finetune.sh** - Skywork sequential training
- **run_swag_lora_finetune.sh** - SWAG-LoRA training
- **run_lstm_baseline.sh** - LSTM baseline
- **run_pareto_analysis.sh** - Model comparison

---

## 🏗️ Architecture Files

In `torch_models/` directory:

- **lstm.py** - LSTM implementation (memory-efficient)
- **transformer.py** - Custom transformer
- **bitnet.py** - BitNet quantization
- **dni.py** - Decoupled Neural Interfaces

---

## 📊 Data Generators

In `torch_training/data/` directory:

- **utm_data_generator.py** - UTM sequence generation
- **ctw_data_generator.py** - CTW tree generation
- **data_generator.py** - Base data utilities
- **utms.py** - UTM implementation

---

## 📈 Analysis Scripts

In `scripts/` directory:

- **analyze_pareto_models.py** - Pareto frontier analysis
- **compare_stages.py** - Stage comparison
- **evaluate_sequential.py** - Model evaluation
- **plot_training_curves.py** - Visualization

---

## 🎨 Usage Patterns

### Pattern 1: Colab Quick Start

```python
# Upload: Skywork_Sequential_Finetuning_Complete.ipynb
# Enable GPU
# Run all cells
# Download results
```

### Pattern 2: Local Training with LoRA

```bash
python torch_training/train_sequential_finetune.py \
  --model_name_or_path "Skywork/Skywork-Reward-V2-Qwen3-0.6B" \
  --use_hf \
  --use_lora \
  --lora_r 8 \
  --stage all
```

### Pattern 3: Memory-Efficient LSTM

```bash
python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --hidden_dim 256 \
  --stage all
```

### Pattern 4: Research with SWAG-LoRA + DNI

```bash
./examples/run_swag_lora_finetune.sh
```

---

## 🔑 Key Features Matrix

| Feature | File | Status | Memory Impact |
|---------|------|--------|---------------|
| **LoRA** | swag_lora_adapter.py | ✅ Ready | -95% params |
| **SWAG** | swag_lora_adapter.py | ✅ Ready | +5% overhead |
| **DNI** | dni_adapter.py | ✅ Ready | +10% params |
| **8-bit** | load_pretrained_model() | ✅ Ready | -75% memory |
| **LSTM** | torch_models/lstm.py | ✅ Ready | 2GB total |
| **BitNet** | torch_models/bitnet.py | ✅ Ready | -50% memory |

---

## 📦 Installation Paths

### Minimal (Colab)

```bash
pip install torch transformers peft
pip install git+https://github.com/fortuinlab/swag-lora.git
```

### Complete (Local)

```bash
pip install -r requirements_sequential_finetune.txt
pip install -r requirements_swag_lora.txt
```

### Research (Full)

```bash
pip install -r requirements.txt
pip install -r requirements_deep_active_inference.txt
```

---

## 🎯 Decision Tree

```
START
  ├─ Colab Free (15GB)?
  │   ├─ Yes → Skywork_Sequential_Finetuning_Complete.ipynb (Option 2)
  │   └─ No → Continue
  ├─ Local GPU <16GB?
  │   ├─ Yes → Use LSTM or LoRA
  │   └─ No → Use Full Fine-tuning
  ├─ Need Uncertainty?
  │   ├─ Yes → Use SWAG-LoRA (Option 3)
  │   └─ No → Use LoRA (Option 2)
  └─ Research/Experimenting?
      ├─ Yes → Use DNI + SWAG-LoRA
      └─ No → Use LoRA (Option 2)
```

---

## 🆘 Troubleshooting Map

| Issue | Solution File |
|-------|---------------|
| Out of memory | COLAB_README.md → "Memory Optimization" |
| LoRA not working | SWAG_LORA_GUIDE.md → "Troubleshooting" |
| SWAG installation | SWAG_LORA_SUMMARY.md → "Installation" |
| Colab disconnects | COLAB_README.md → "Save to Google Drive" |
| Slow training | NOTEBOOK_GUIDE.md → "Configuration Options" |
| Model not loading | test_skywork_loading.py |

---

## 📊 Benchmark Summary

From `PARETO_REWARD_MODELS_MATRIX.md`:

| Model | Memory | Quality | Speed | Recommended |
|-------|--------|---------|-------|-------------|
| LSTM | 2GB | 90% | Fast | From scratch |
| LoRA r=8 | 4GB | 96% | Fast | **Best balance** |
| SWAG-LoRA | 4.5GB | 98% | Medium | Research |
| Full | 12GB | 100% | Slow | Maximum quality |

---

## 🔄 Update Log

All updates consolidated from:
- SWAG_LORA_SUMMARY.md (latest SWAG-LoRA integration)
- DNI_COMPLETE_SUMMARY.md (DNI features)
- PARETO_ANALYSIS_UPDATE_LOG.md (model comparisons)
- IMPLEMENTATION_SUMMARY.md (overall progress)

---

## 📞 Getting Started Checklist

- [ ] Read `START_HERE.md`
- [ ] Choose: Notebook (Colab) or Scripts (Local)
- [ ] If Colab: Upload `Skywork_Sequential_Finetuning_Complete.ipynb`
- [ ] If Local: Install `requirements_swag_lora.txt`
- [ ] Run Option 2 (LoRA) for best results
- [ ] Monitor memory with included tools
- [ ] Save checkpoints before completion
- [ ] Evaluate with `scripts/evaluate_sequential.py`

---

## 🎓 Learning Path

1. **Beginner**: `START_HERE.md` → Notebook Option 5 (Quick Test)
2. **Intermediate**: `NOTEBOOK_GUIDE.md` → Notebook Option 2 (LoRA)
3. **Advanced**: `SWAG_LORA_GUIDE.md` → Notebook Option 3 (SWAG-LoRA)
4. **Research**: `DNI_INTEGRATION.md` → Custom scripts with all features

---

**Everything you need is consolidated!** 🚀

For Colab users: Just use `Skywork_Sequential_Finetuning_Complete.ipynb`
