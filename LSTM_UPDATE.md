# LSTM Architecture Update

## Summary

Updated the sequential fine-tuning pipeline to use **LSTM as the default architecture** based on research showing superior length generalization compared to Transformers.

**Reference:** [Learning Universal Predictors](https://arxiv.org/html/2401.14953v1)

## Changes Made

### 1. New LSTM Implementation
- **File:** `torch_models/lstm.py`
- Implements `LSTMDecoderLM` with same interface as `TransformerDecoderLM`
- Features:
  - Weight tying between embeddings and output projection
  - Dropout regularization
  - Autoregressive generation with temperature/top-k sampling
  - Configurable hidden dimensions and layers

### 2. Updated Training Script
- **File:** `torch_training/train_sequential_finetune.py`
- Added `--architecture` flag (choices: `lstm`, `transformer`)
- LSTM is now the default architecture
- Added hyperparameter flags: `--hidden_dim`, `--num_layers`, `--embedding_dim`
- Automatic architecture detection from checkpoints
- Reference to research paper in docstring and logs

### 3. Updated Configuration
- **File:** `configs/sequential_finetune_default.json`
- Default changed from HuggingFace model to custom LSTM
- Added LSTM-specific hyperparameters

### 4. Updated Module Exports
- **File:** `torch_models/__init__.py`
- Added LSTM imports and exports

### 5. Documentation
- **File:** `docs/LSTM_ARCHITECTURE.md`
- Comprehensive guide on LSTM architecture
- Performance comparisons
- Usage examples
- Migration guide

### 6. Test Suite
- **File:** `test_lstm.py`
- Tests for forward pass, training, generation
- Length generalization verification
- Weight tying verification

## Usage Examples

### Default (LSTM)
```bash
python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --utm_steps 10000 \
  --ctw_steps 5000
```

### With Custom Hyperparameters
```bash
python torch_training/train_sequential_finetune.py \
  --architecture lstm \
  --hidden_dim 512 \
  --num_layers 3 \
  --embedding_dim 256 \
  --lr 1e-4
```

### Using Transformer (for comparison)
```bash
python torch_training/train_sequential_finetune.py \
  --architecture transformer \
  --num_layers 4
```

## Key Benefits

1. **Better Length Generalization**: LSTMs maintain lower perplexity on sequences longer than training distribution
2. **Lower Memory**: O(1) memory vs O(n²) for attention mechanisms
3. **Stable Training**: No attention collapse or vanishing gradients issues
4. **Fast Inference**: Sequential processing without attention overhead

## Backward Compatibility

- Existing transformer checkpoints remain fully compatible
- Architecture is auto-detected from checkpoint metadata
- Can still use HuggingFace models with `--use_hf` flag

## Testing

Run the test suite:
```bash
python test_lstm.py
```

Expected output: All 5 test suites should pass, verifying:
- Forward pass correctness
- Training step functionality  
- Autoregressive generation
- Length generalization capability
- Weight tying parameter reduction

## Performance Expectations

Based on research findings, expect:
- **Similar performance** on in-distribution sequence lengths
- **Significantly better** on out-of-distribution (longer) sequences
- **Faster training** due to reduced memory and computation
- **Better sample efficiency** for learning sequential patterns

## Next Steps

1. Run comparative experiments: LSTM vs Transformer on UTM/CTW tasks
2. Measure length generalization systematically
3. Explore LSTM variants (GRU, QRNN, SRU) for further optimization
4. Consider hybrid architectures combining LSTM and sparse attention
