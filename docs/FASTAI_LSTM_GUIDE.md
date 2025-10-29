# FastAI LSTM Sequential Finetuning Guide

This guide covers the **FastAI-based implementation** of the LSTM sequential finetuning pipeline. This variant uses [fastai](https://docs.fast.ai/) primitives for a more streamlined and feature-rich training experience.

## Overview

The FastAI implementation provides:

- **High-level API**: Less boilerplate code with fastai's `Learner` and callbacks
- **One-Cycle Training**: Automatically managed learning rate schedules
- **Rich Callbacks**: Built-in support for logging, checkpointing, and metrics
- **Better Defaults**: Optimized training practices out-of-the-box
- **Easy Experimentation**: Quickly try different architectures and hyperparameters

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  FastAI Training Pipeline                │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Stage 1: UTM Data                                       │
│  ┌─────────────────────────────────────────────┐        │
│  │ UTM Data Generator → FastAI DataLoader      │        │
│  │         ↓                                    │        │
│  │  LSTM Model (log probs)                     │        │
│  │         ↓                                    │        │
│  │  FastAI Learner (One-Cycle LR)              │        │
│  │         ↓                                    │        │
│  │  Callbacks: Checkpoint, Metrics, Logging    │        │
│  └─────────────────────────────────────────────┘        │
│                     ↓                                     │
│  Stage 2: CTW Data                                       │
│  ┌─────────────────────────────────────────────┐        │
│  │ CTW Data Generator → FastAI DataLoader      │        │
│  │         ↓                                    │        │
│  │  LSTM Model (log probs)                     │        │
│  │         ↓                                    │        │
│  │  FastAI Learner (One-Cycle LR)              │        │
│  │         ↓                                    │        │
│  │  Callbacks: Checkpoint, Metrics, Logging    │        │
│  └─────────────────────────────────────────────┘        │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install dependencies including fastai
pip install -r requirements_sequential_finetune.txt
```

## Components

### 1. FastAI DataLoaders (`fastai_dataloaders.py`)

Wraps the existing UTM and CTW data generators into fastai-compatible `DataLoaders`:

- **`UTMFastAIDataset`**: Iterable dataset for UTM data
- **`CTWFastAIDataset`**: Iterable dataset for CTW data
- **`create_utm_dataloaders()`**: Creates train/valid DataLoaders for UTM
- **`create_ctw_dataloaders()`**: Creates train/valid DataLoaders for CTW

### 2. FastAI Callbacks (`fastai_callbacks.py`)

Custom callbacks and metrics for the training pipeline:

- **`PerplexityMetric`**: Tracks perplexity during training
- **`GradNormCallback`**: Logs gradient norms
- **`SequentialStageCallback`**: Tracks current training stage (UTM/CTW)
- **`ModelCheckpointCallback`**: Enhanced checkpoint saving with best model tracking
- **`LSTMLogProbLoss`**: Loss function for LSTM models (outputs log probs)
- **`TransformerLogitsLoss`**: Loss function for transformer models (outputs logits)

### 3. Main Training Script (`train_sequential_finetune_fastai.py`)

FastAI-based sequential finetuning pipeline with:

- Automatic one-cycle learning rate scheduling
- Progressive training with callbacks
- Flexible configuration
- Support for LSTM and Transformer architectures

## Usage Examples

### Basic Usage (LSTM with default settings)

```bash
python torch_training/train_sequential_finetune_fastai.py \
  --architecture lstm \
  --utm_epochs 10 \
  --ctw_epochs 5 \
  --batches_per_epoch 100 \
  --lr 1e-3
```

### Custom Hyperparameters

```bash
python torch_training/train_sequential_finetune_fastai.py \
  --architecture lstm \
  --hidden_dim 512 \
  --num_layers 3 \
  --embedding_dim 256 \
  --utm_epochs 20 \
  --ctw_epochs 10 \
  --lr 5e-4 \
  --batches_per_epoch 200
```

### Single Stage Training

Train only UTM stage:
```bash
python torch_training/train_sequential_finetune_fastai.py \
  --stage utm \
  --utm_epochs 15 \
  --lr 1e-3
```

Train only CTW stage (requires pre-trained model):
```bash
python torch_training/train_sequential_finetune_fastai.py \
  --stage ctw \
  --ctw_epochs 10 \
  --lr 5e-4
```

### Different Learning Rates per Stage

```bash
python torch_training/train_sequential_finetune_fastai.py \
  --utm_lr 1e-3 \
  --ctw_lr 5e-4 \
  --utm_epochs 10 \
  --ctw_epochs 5
```

## Key Features

### 1. One-Cycle Learning Rate Policy

FastAI uses the [1cycle policy](https://arxiv.org/abs/1803.09820) by default:

- Starts with low learning rate
- Gradually increases to `lr_max`
- Then decreases to very low learning rate
- Provides faster training and better convergence

### 2. Automatic Metrics Tracking

The implementation automatically tracks:

- **Loss**: Cross-entropy loss per batch
- **Perplexity**: exp(loss) for interpretability
- **Gradient Norm**: Monitor gradient flow

### 3. Enhanced Checkpointing

- **Periodic checkpoints**: Save every N steps
- **Best model tracking**: Automatically saves best model based on validation loss
- **Stage-aware naming**: Checkpoints include stage information

### 4. Progress Visualization

FastAI provides rich progress bars and logging:

```
epoch    train_loss    valid_loss    perplexity    time
------   -----------   -----------   -----------   ------
0        2.345         2.312         10.09         0:42
1        2.123         2.098         8.15          0:41
...
```

## Configuration Options

### Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--architecture` | `lstm` | Model type (`lstm` or `transformer`) |
| `--hidden_dim` | `256` | Hidden dimension size |
| `--num_layers` | `2` | Number of layers |
| `--embedding_dim` | `128` | Embedding dimension |

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage` | `all` | Stage to run (`utm`, `ctw`, or `all`) |
| `--utm_epochs` | `10` | Epochs for UTM stage |
| `--ctw_epochs` | `5` | Epochs for CTW stage |
| `--batches_per_epoch` | `100` | Batches per epoch |
| `--lr` | `1e-3` | Max learning rate (one-cycle) |
| `--save_every` | `1000` | Checkpoint frequency (steps) |

### Data Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | `32` | Batch size |
| `--seq_length` | `256` | Sequence length |
| `--tokenizer` | `ascii` | Tokenizer type (`ascii` or `seq_position`) |
| `--memory_size` | `10` | UTM memory size |
| `--maximum_steps` | `100` | Max UTM steps |
| `--ctw_max_depth` | `5` | Max CTW tree depth |

## Comparison: FastAI vs Pure PyTorch

| Feature | FastAI Implementation | Pure PyTorch Implementation |
|---------|----------------------|----------------------------|
| **Code Length** | ~300 lines | ~600 lines |
| **Learning Rate** | One-cycle (automatic) | Manual scheduling |
| **Callbacks** | Built-in + custom | Manual implementation |
| **Progress Tracking** | Rich progress bars | Basic logging |
| **Checkpointing** | Automatic best model | Manual tracking |
| **Metrics** | Automatic aggregation | Manual calculation |
| **Experimentation** | Easy to modify | More boilerplate |

## Advanced Usage

### Custom Callbacks

You can add custom fastai callbacks:

```python
from fastai.callback.core import Callback

class MyCustomCallback(Callback):
    def after_batch(self):
        # Custom logic after each batch
        pass

# Add to training
learn.fit_one_cycle(n_epoch=10, cbs=[MyCustomCallback()])
```

### Learning Rate Finder

Use fastai's learning rate finder to find optimal LR:

```python
from fastai_dataloaders import create_utm_dataloaders
from fastai.learner import Learner

# Create learner
dls = create_utm_dataloaders(utm_generator, batches_per_epoch=100)
learn = Learner(dls, model, loss_func=LSTMLogProbLoss())

# Find learning rate
learn.lr_find()
```

### Mixed Precision Training

Enable mixed precision for faster training:

```python
learn.to_fp16()  # Use mixed precision
learn.fit_one_cycle(n_epoch=10, lr_max=1e-3)
```

## Output Structure

```
checkpoints/sequential_fastai/
├── train.log                    # Training log
├── metrics.json                 # Metrics history
├── stage_utm_step_1000.pth     # Periodic checkpoint
├── stage_utm_best.pth          # Best UTM model
├── stage1_utm_final.pth        # Final UTM checkpoint
├── stage_ctw_step_1000.pth     # Periodic checkpoint
├── stage_ctw_best.pth          # Best CTW model
└── stage2_ctw_final.pth        # Final CTW checkpoint
```

## Performance Tips

1. **Use one-cycle policy**: The default one-cycle LR schedule often works best
2. **Tune batches_per_epoch**: Higher values = more stable gradients, longer epochs
3. **Monitor perplexity**: Should decrease over time; if not, adjust LR
4. **Use gradient norm**: High values may indicate instability
5. **Start with default LR**: 1e-3 works well for most cases with one-cycle

## Troubleshooting

### High Loss/Perplexity

- Reduce learning rate (`--lr 5e-4`)
- Increase model capacity (`--hidden_dim 512 --num_layers 3`)
- More training (`--utm_epochs 20`)

### Slow Training

- Reduce `batches_per_epoch` for faster epochs
- Use GPU (`--device cuda`)
- Enable mixed precision (see Advanced Usage)

### OOM (Out of Memory)

- Reduce batch size (`--batch_size 16`)
- Reduce sequence length (`--seq_length 128`)
- Reduce model size (`--hidden_dim 128 --num_layers 2`)

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

## See Also

- [LSTM Architecture Documentation](LSTM_ARCHITECTURE.md)
- [Sequential Finetuning README](../README_SEQUENTIAL_FINETUNE.md)
- [FastAI Documentation](https://docs.fast.ai/)
