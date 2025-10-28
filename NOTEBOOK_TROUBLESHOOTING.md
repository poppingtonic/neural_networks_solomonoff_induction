# 🔧 Notebook Troubleshooting Cell

Add this cell to your notebook if you encounter errors:

```python
# ============================================================
# TROUBLESHOOTING & FIXES
# ============================================================

import subprocess
import sys

def check_and_fix():
    """Check for common issues and apply fixes."""
    
    print("🔍 Checking for common issues...")
    print()
    
    # Check 1: HuggingFace model output handling
    with open('torch_training/train_sequential_finetune.py', 'r') as f:
        content = f.read()
        if 'hasattr(outputs, \'logits\')' in content:
            print("✅ HuggingFace output handling: OK")
        else:
            print("⚠️  HuggingFace output handling: NEEDS UPDATE")
            print("   Solution: !git pull origin feat/RM-finetune")
    
    # Check 2: Dependencies
    try:
        import peft
        print("✅ peft (LoRA): Installed")
    except ImportError:
        print("❌ peft: Not installed")
        print("   Solution: !pip install peft")
    
    try:
        import swag_lora
        print("✅ swag_lora: Installed")
    except ImportError:
        print("⚠️  swag_lora: Not installed (optional)")
        print("   Solution: !pip install git+https://github.com/fortuinlab/swag-lora.git")
    
    # Check 3: GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Memory: {mem:.1f} GB")
    else:
        print("❌ GPU: Not available")
        print("   Solution: Runtime → Change runtime type → GPU")
    
    print()
    print("=" * 60)

# Run checks
check_and_fix()
```

## Common Errors & Solutions

### Error 1: AttributeError - 'CausalLMOutputWithPast' has no attribute 'shape'

**Solution:**
```python
# Pull the latest fix
!git pull origin feat/RM-finetune
```

Or manually update the file (see `HOTFIX_COLAB.md`)

---

### Error 2: ModuleNotFoundError: No module named 'peft'

**Solution:**
```python
!pip install peft
```

---

### Error 3: CUDA out of memory

**Solutions:**

```python
# Option 1: Reduce batch size
--batch_size 4  # or even 2

# Option 2: Reduce sequence length
--seq_length 128  # or 64

# Option 3: Use smaller LoRA rank
--lora_r 4

# Option 4: Switch to LSTM
# Use OPTION 1 instead (most memory efficient)
```

---

### Error 4: Cannot clone repository (already exists)

**Solution:**
```python
# If directory exists, just cd into it
%cd neural_networks_solomonoff_induction

# Update to latest
!git pull origin feat/RM-finetune
```

---

### Error 5: Training is slow

**Solutions:**

1. **Use Quick Test first** (OPTION 5) to verify setup
2. **Reduce steps** for faster iteration
3. **Use LSTM** (OPTION 1) for fastest training
4. **Check GPU usage**: Should be 80-100%

---

### Error 6: Model download fails

**Solutions:**

```python
# Option 1: Login to HuggingFace
!huggingface-cli login

# Option 2: Use LSTM instead (no download needed)
# Run OPTION 1 training cell

# Option 3: Check internet connection
!ping -c 3 huggingface.co
```

---

### Error 7: JAX/CUDA compatibility issues

**Solution:**
```python
# Install JAX with CUDA 12 support
!pip install -U "jax[cuda12]" jaxlib
```

---

## Emergency: Start Fresh

If nothing works, start completely fresh:

```python
# 1. Clear everything
!rm -rf neural_networks_solomonoff_induction
%cd /content

# 2. Reinstall
!pip install -q torch transformers peft bitsandbytes accelerate numpy
!pip install -q git+https://github.com/fortuinlab/swag-lora.git
!pip install -q "jax[cuda12]" jaxlib dm-haiku optax chex

# 3. Clone fresh
!git clone https://github.com/google-deepmind/neural_networks_solomonoff_induction.git
%cd neural_networks_solomonoff_induction
!git checkout feat/RM-finetune

# 4. Run quick test
!python torch_training/train_sequential_finetune.py \
  --model_name_or_path 'Skywork/Skywork-Reward-V2-Qwen3-0.6B' \
  --use_hf \
  --use_lora \
  --lora_r 8 \
  --stage utm \
  --utm_steps 10 \
  --batch_size 4 \
  --device cuda
```

---

## Quick Diagnostics

```python
# Run all diagnostics
import torch
import subprocess
import sys

print("=" * 60)
print("SYSTEM DIAGNOSTICS")
print("=" * 60)

# GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
else:
    print("❌ No GPU detected")

# Python packages
packages = ['torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'jax', 'swag_lora']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {pkg:15s}: {version}")
    except ImportError:
        print(f"❌ {pkg:15s}: Not installed")

# File check
import os
key_files = [
    'torch_training/train_sequential_finetune.py',
    'torch_training/swag_lora_adapter.py',
    'torch_models/lstm.py',
]
print()
for f in key_files:
    exists = "✅" if os.path.exists(f) else "❌"
    print(f"{exists} {f}")

print("=" * 60)
```

---

## Performance Tips

**For Fastest Training:**

1. Use **OPTION 1** (LSTM) - 2x faster than Skywork
2. Reduce steps: `--utm_steps 2000 --ctw_steps 1000`
3. Increase batch size if memory allows: `--batch_size 32`
4. Use `--log_every 25` for less output

**For Best Quality:**

1. Use **OPTION 3** (SWAG-LoRA) with uncertainty
2. Increase steps: `--utm_steps 10000 --ctw_steps 5000`
3. Increase LoRA rank: `--lora_r 16 --lora_alpha 32`
4. Fine-tune learning rates: `--utm_lr 5e-5 --ctw_lr 2e-5`

**For Memory Savings:**

1. Reduce batch: `--batch_size 4` or `2`
2. Reduce sequence: `--seq_length 128` or `64`
3. Lower LoRA rank: `--lora_r 4`
4. Use LSTM: Saves 80% memory vs Skywork

---

## Still Having Issues?

1. Check `HOTFIX_COLAB.md` for known issues
2. Check `SWAG_LORA_GUIDE.md` for LoRA troubleshooting
3. Check `COLAB_README.md` for Colab-specific tips
4. Make sure you're on branch `feat/RM-finetune`

## Get Help

```python
# Show current git branch and commit
!git branch
!git log -1 --oneline

# Show Python version
!python --version

# Show CUDA version
!nvcc --version
```

All troubleshooting consolidated in one place! 🎯
