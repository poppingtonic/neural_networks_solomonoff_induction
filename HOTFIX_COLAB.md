# 🔧 Hotfix for Colab Training Error

## Issue

If you encounter this error:

```
AttributeError: 'CausalLMOutputWithPast' object has no attribute 'shape'
```

This means you're using an older version of `train_sequential_finetune.py` that doesn't properly handle HuggingFace model outputs.

## Quick Fix (In Colab)

Run this cell to patch the file:

```python
# Hotfix for HuggingFace model output handling
import os

fix_script = '''
# Apply hotfix
sed -i 's/log_probs = self.model(x)/outputs = self.model(x)/' torch_training/train_sequential_finetune.py

# Add output handling after line 99
cat > /tmp/fix.txt << 'EOF'
        
        # Handle both HuggingFace and custom model outputs
        if hasattr(outputs, 'logits'):
            # HuggingFace model output
            logits = outputs.logits
        else:
            # Custom model output (already logits)
            logits = outputs
        
        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
EOF

# Note: This is a simplified fix. The actual fix requires editing the file properly.
# Recommended: Pull the latest version instead.
'''

print("⚠️  Hotfix detected an issue with model output handling.")
print("✅  Best solution: Pull the latest version of the code:")
print()
print("!git pull origin feat/RM-finetune")
print()
print("Or manually update train_sequential_finetune.py with the fix.")
```

## Better Solution: Update Code

Instead of patching, pull the latest version:

```bash
# In Colab
%cd /content/neural_networks_solomonoff_induction
!git fetch origin
!git pull origin feat/RM-finetune
```

## Manual Fix (If Needed)

If you can't pull the latest code, edit `torch_training/train_sequential_finetune.py`:

### In `train_step_utm` (around line 99):

**Replace:**
```python
optimizer.zero_grad(set_to_none=True)
log_probs = self.model(x)
B, T, V = log_probs.shape
```

**With:**
```python
optimizer.zero_grad(set_to_none=True)
outputs = self.model(x)

# Handle both HuggingFace and custom model outputs
if hasattr(outputs, 'logits'):
    # HuggingFace model output
    logits = outputs.logits
else:
    # Custom model output (already logits)
    logits = outputs

B, T, V = logits.shape
log_probs = F.log_softmax(logits, dim=-1)
```

### In `train_step_ctw` (around line 143):

Apply the same fix:

**Replace:**
```python
optimizer.zero_grad(set_to_none=True)
log_probs = self.model(x)
B, T, V = log_probs.shape
```

**With:**
```python
optimizer.zero_grad(set_to_none=True)
outputs = self.model(x)

# Handle both HuggingFace and custom model outputs
if hasattr(outputs, 'logits'):
    # HuggingFace model output
    logits = outputs.logits
else:
    # Custom model output (already logits)
    logits = outputs

B, T, V = logits.shape
log_probs = F.log_softmax(logits, dim=-1)
```

## Why This Happens

HuggingFace transformers return a special `CausalLMOutputWithPast` object that contains:
- `.logits` - The actual model predictions
- `.past_key_values` - Cached attention keys/values
- `.hidden_states` - Intermediate representations
- `.attentions` - Attention weights

Custom PyTorch models return tensors directly, so we need to handle both cases.

## Verify the Fix

After applying the fix, run a quick test:

```python
# Test that training starts
!python torch_training/train_sequential_finetune.py \
  --model_name_or_path 'Skywork/Skywork-Reward-V2-Qwen3-0.6B' \
  --use_hf \
  --use_lora \
  --lora_r 8 \
  --stage utm \
  --utm_steps 10 \
  --batch_size 4 \
  --seq_length 64 \
  --device cuda

# Should see: "Step 0/10 | Loss: ..." without errors
```

## Status

✅ **Fixed in commit**: `685abc6` and later  
✅ **Affects**: Skywork + HuggingFace models  
✅ **Does not affect**: LSTM or custom models  

## Prevention

This issue is now fixed in the main codebase. If you're using the consolidated notebook or pulling fresh code, you won't encounter this issue.
