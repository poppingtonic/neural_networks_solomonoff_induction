# FastAI LSTM Implementation Summary

## Overview

This document summarizes the **FastAI-based implementation** of the LSTM sequential finetuning pipeline. This variant uses [fastai](https://docs.fast.ai/) primitives to provide a more streamlined, feature-rich training experience with less boilerplate code.

## What Was Implemented

### 1. **FastAI DataLoaders** (`torch_training/fastai_dataloaders.py`)

FastAI-compatible data loaders that wrap the existing UTM and CTW data generators:

- **`UTMFastAIDataset`**: Iterable dataset for UTM data
- **`CTWFastAIDataset`**: Iterable dataset for CTW data  
- **`create_utm_dataloaders()`**: Creates DataLoaders for UTM training
- **`create_ctw_dataloaders()`**: Creates DataLoaders for CTW training

**Key Features:**
- Automatic batching (datasets yield batches directly)
- Pin memory support for GPU acceleration
- Separate train/validation splits

### 2. **FastAI Callbacks & Metrics** (`torch_training/fastai_callbacks.py`)

Custom callbacks and loss functions for the training pipeline:

#### Metrics
- **`PerplexityMetric`**: Tracks perplexity (exp of loss) during training

#### Callbacks
- **`GradNormCallback`**: Logs gradient norms for monitoring training stability
- **`SequentialStageCallback`**: Tracks which stage (UTM/CTW) is currently training
- **`ModelCheckpointCallback`**: Enhanced checkpointing with:
  - Periodic saves every N steps
  - Best model tracking based on validation loss
  - Stage-aware checkpoint naming

#### Loss Functions
- **`LSTMLogProbLoss`**: For LSTM models that output log probabilities
- **`TransformerLogitsLoss`**: For transformer models that output logits

### 3. **Main Training Script** (`torch_training/train_sequential_finetune_fastai.py`)

Complete sequential finetuning pipeline with FastAI:

**Features:**
- **One-Cycle Learning Rate Policy**: Automatic LR scheduling for faster convergence
- **Sequential Stages**: UTM → CTW finetuning with separate configurations
- **Flexible Architecture**: Supports both LSTM and Transformer models
- **Rich Logging**: Progress bars, metrics tracking, and checkpoint management
- **Easy Configuration**: Command-line arguments for all hyperparameters

### 4. **Documentation**

- **`docs/FASTAI_LSTM_GUIDE.md`**: Comprehensive guide covering:
  - Installation and setup
  - Component descriptions
  - Usage examples
  - Configuration options
  - Performance tips
  - Troubleshooting

- **`examples/fastai_lstm_example.py`**: Runnable example demonstrating:
  - Model creation
  - Data generator setup
  - FastAI learner configuration
  - Sequential training (UTM → CTW)
  - Checkpoint saving

### 5. **Updated Requirements**

Updated `requirements_sequential_finetune.txt` to include:
```
fastai>=2.7.0
fastcore>=1.5.0
```

## Quick Start

### Installation

```bash
pip install -r requirements_sequential_finetune.txt
```

### Basic Training

```bash
# Default LSTM training with FastAI
python torch_training/train_sequential_finetune_fastai.py \
  --architecture lstm \
  --utm_epochs 10 \
  --ctw_epochs 5 \
  --batches_per_epoch 100 \
  --lr 1e-3
```

### Run Example

```bash
# Run the quick-start example
python examples/fastai_lstm_example.py
```

## Key Benefits of FastAI Implementation

### 1. **Less Boilerplate Code**

| Task | Pure PyTorch | FastAI |
|------|--------------|--------|
| Training loop | ~100 lines | ~10 lines |
| Learning rate scheduling | Manual | Automatic (one-cycle) |
| Callbacks | Manual implementation | Built-in + custom |
| Metrics tracking | Manual aggregation | Automatic |

### 2. **One-Cycle Learning Rate Policy**

FastAI uses the [1cycle policy](https://arxiv.org/abs/1803.09820) by default:
- Faster training convergence
- Better generalization
- No manual LR scheduling needed

### 3. **Enhanced Checkpointing**

- Automatic best model tracking
- Periodic checkpoints every N steps
- Stage-aware naming (utm/ctw)

### 4. **Rich Progress Tracking**

```
epoch    train_loss    valid_loss    perplexity    time
------   -----------   -----------   -----------   ------
0        2.345         2.312         10.09         0:42
1        2.123         2.098         8.15          0:41
```

### 5. **Easy Experimentation**

- Quick to modify and test new ideas
- Built-in tools like `lr_find()` for hyperparameter tuning
- Mixed precision training with one line: `learn.to_fp16()`

## Architecture Comparison

### FastAI vs Pure PyTorch Implementation

| Feature | FastAI | Pure PyTorch |
|---------|--------|--------------|
| **Lines of Code** | ~300 | ~600 |
| **Learning Rate** | One-cycle (auto) | Manual |
| **Callbacks** | Built-in + 4 custom | All manual |
| **Metrics** | Auto aggregation | Manual tracking |
| **Checkpointing** | Auto best model | Manual |
| **Progress Bars** | Rich, informative | Basic |
| **Experimentation** | Fast, easy | More setup |

## File Structure

```
torch_training/
├── train_sequential_finetune_fastai.py    # Main training script
├── fastai_dataloaders.py                   # DataLoader wrappers
├── fastai_callbacks.py                     # Custom callbacks & metrics
├── train_sequential_finetune.py            # Original PyTorch version
└── ...

examples/
├── fastai_lstm_example.py                  # Quick-start example
└── ...

docs/
├── FASTAI_LSTM_GUIDE.md                    # Comprehensive guide
├── LSTM_ARCHITECTURE.md                    # LSTM architecture docs
└── ...
```

## Usage Examples

### 1. Default Training (LSTM)

```bash
python torch_training/train_sequential_finetune_fastai.py
```

### 2. Custom Hyperparameters

```bash
python torch_training/train_sequential_finetune_fastai.py \
  --hidden_dim 512 \
  --num_layers 3 \
  --embedding_dim 256 \
  --utm_epochs 20 \
  --ctw_epochs 10 \
  --lr 5e-4
```

### 3. Single Stage Training

```bash
# UTM only
python torch_training/train_sequential_finetune_fastai.py --stage utm

# CTW only  
python torch_training/train_sequential_finetune_fastai.py --stage ctw
```

### 4. Different LRs per Stage

```bash
python torch_training/train_sequential_finetune_fastai.py \
  --utm_lr 1e-3 \
  --ctw_lr 5e-4
```

## Configuration Options

### Model Architecture

- `--architecture`: `lstm` (default) or `transformer`
- `--hidden_dim`: Hidden dimension size (default: 256)
- `--num_layers`: Number of layers (default: 2)
- `--embedding_dim`: Embedding dimension (default: 128)

### Training

- `--stage`: `all` (default), `utm`, or `ctw`
- `--utm_epochs`: Epochs for UTM stage (default: 10)
- `--ctw_epochs`: Epochs for CTW stage (default: 5)
- `--batches_per_epoch`: Batches per epoch (default: 100)
- `--lr`: Max learning rate for one-cycle (default: 1e-3)
- `--save_every`: Checkpoint frequency in steps (default: 1000)

### Data

- `--batch_size`: Batch size (default: 32)
- `--seq_length`: Sequence length (default: 256)
- `--tokenizer`: `ascii` (default) or `seq_position`
- `--memory_size`: UTM memory size (default: 10)
- `--ctw_max_depth`: CTW tree depth (default: 5)

## Performance Tips

1. **Start with default LR (1e-3)**: Works well with one-cycle policy
2. **Use learning rate finder**: `learn.lr_find()` to find optimal LR
3. **Monitor perplexity**: Should decrease over time
4. **Check gradient norms**: High values indicate instability
5. **Enable mixed precision**: `learn.to_fp16()` for faster training

## Troubleshooting

### High Loss/Perplexity
- Reduce learning rate: `--lr 5e-4`
- Increase model capacity: `--hidden_dim 512 --num_layers 3`
- Train longer: `--utm_epochs 20`

### Slow Training
- Reduce batches per epoch: `--batches_per_epoch 50`
- Use GPU: `--device cuda`
- Enable mixed precision (see guide)

### Out of Memory
- Reduce batch size: `--batch_size 16`
- Reduce sequence length: `--seq_length 128`
- Reduce model size: `--hidden_dim 128`

## Next Steps

1. **Run the example**: `python examples/fastai_lstm_example.py`
2. **Read the guide**: See `docs/FASTAI_LSTM_GUIDE.md`
3. **Experiment**: Try different hyperparameters
4. **Compare**: Run both FastAI and PyTorch versions
5. **Extend**: Add custom callbacks for your use case

## Related Documentation

- [FASTAI_LSTM_GUIDE.md](docs/FASTAI_LSTM_GUIDE.md): Comprehensive guide
- [LSTM_ARCHITECTURE.md](docs/LSTM_ARCHITECTURE.md): LSTM architecture details
- [README_SEQUENTIAL_FINETUNE.md](README_SEQUENTIAL_FINETUNE.md): Original implementation
- [FastAI Documentation](https://docs.fast.ai/): FastAI library docs

## Citation

If you use this implementation, please cite:

```bibtex
@article{deletang2024language,
  title={Learning Universal Predictors},
  author={Deletang, Gr{\'e}goire and others},
  journal={arXiv preprint arXiv:2401.14953},
  year={2024}
}
```

## License

Same as parent project (see LICENSE file).
