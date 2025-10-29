# FastAI LSTM Quick Start Guide

**TL;DR**: Train an LSTM on UTM and CTW data using FastAI in 3 commands.

## Installation

```bash
pip install -r requirements_sequential_finetune.txt
```

## Run Training (Default Settings)

```bash
python torch_training/train_sequential_finetune_fastai.py
```

This will:
- Create LSTM model (256 hidden, 2 layers)
- Train on UTM data for 10 epochs
- Train on CTW data for 5 epochs  
- Use one-cycle LR policy (max_lr=1e-3)
- Save checkpoints to `./checkpoints/sequential_fastai/`

## Run Example Script

```bash
python examples/fastai_lstm_example.py
```

## Common Commands

### Fast Training (Fewer Epochs)
```bash
python torch_training/train_sequential_finetune_fastai.py \
  --utm_epochs 5 \
  --ctw_epochs 3 \
  --batches_per_epoch 50
```

### High-Quality Training (More Epochs, Larger Model)
```bash
python torch_training/train_sequential_finetune_fastai.py \
  --hidden_dim 512 \
  --num_layers 3 \
  --utm_epochs 20 \
  --ctw_epochs 10 \
  --batches_per_epoch 200 \
  --lr 5e-4
```

### UTM Only
```bash
python torch_training/train_sequential_finetune_fastai.py \
  --stage utm \
  --utm_epochs 15
```

### CTW Only
```bash
python torch_training/train_sequential_finetune_fastai.py \
  --stage ctw \
  --ctw_epochs 10
```

## Output Files

After training, you'll find:

```
checkpoints/sequential_fastai/
├── train.log                    # Training log
├── metrics.json                 # Metrics history
├── stage_utm_best.pth          # Best UTM model
├── stage1_utm_final.pth        # Final UTM checkpoint
├── stage_ctw_best.pth          # Best CTW model
└── stage2_ctw_final.pth        # Final CTW checkpoint
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--architecture` | `lstm` | Model type (`lstm` or `transformer`) |
| `--utm_epochs` | `10` | UTM training epochs |
| `--ctw_epochs` | `5` | CTW training epochs |
| `--lr` | `1e-3` | Max learning rate (one-cycle) |
| `--hidden_dim` | `256` | Hidden dimension |
| `--num_layers` | `2` | Number of layers |
| `--batches_per_epoch` | `100` | Batches per epoch |
| `--batch_size` | `32` | Batch size |
| `--seq_length` | `256` | Sequence length |

## What is FastAI?

FastAI provides:
- **One-cycle LR scheduling**: Automatic learning rate management
- **Rich callbacks**: Progress bars, metrics, checkpointing
- **Less code**: ~50% less boilerplate than pure PyTorch
- **Better defaults**: Optimized training practices

## FastAI vs PyTorch Comparison

| Feature | FastAI | Pure PyTorch |
|---------|--------|--------------|
| **Command** | `train_sequential_finetune_fastai.py` | `train_sequential_finetune.py` |
| **Training** | Epochs-based | Steps-based |
| **LR Schedule** | One-cycle (auto) | Manual |
| **Code Length** | ~300 lines | ~600 lines |
| **Flexibility** | High-level | Low-level control |

## Next Steps

1. ✅ Run the example: `python examples/fastai_lstm_example.py`
2. 📖 Read the guide: [docs/FASTAI_LSTM_GUIDE.md](docs/FASTAI_LSTM_GUIDE.md)
3. 🧪 Experiment with hyperparameters
4. 📊 Compare with PyTorch version
5. 🚀 Deploy your trained model

## Troubleshooting

**High loss?**
```bash
--lr 5e-4 --hidden_dim 512
```

**Out of memory?**
```bash
--batch_size 16 --seq_length 128
```

**Too slow?**
```bash
--batches_per_epoch 50 --device cuda
```

## Full Documentation

See [FASTAI_IMPLEMENTATION.md](FASTAI_IMPLEMENTATION.md) for complete documentation.
