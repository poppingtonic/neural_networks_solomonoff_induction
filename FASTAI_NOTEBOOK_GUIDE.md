# FastAI LSTM Sequential Finetuning - Notebook Guide

## Overview

**`FastAI_LSTM_Sequential_Finetuning.ipynb`** is an **educational, explanatory notebook** that walks through the FastAI-based LSTM training pipeline step-by-step.

## What's Special About This Notebook?

Unlike typical training notebooks, this one is designed for **learning and understanding**:

- 📚 **Explanatory**: Each step is explained with markdown cells
- 🔬 **Interactive**: Run cells individually to understand each component
- 📊 **Visual**: Shows training curves and metrics
- 🎓 **Educational**: Perfect for learning FastAI and LSTM training
- 🚀 **Complete**: Covers both UTM and CTW training stages

## Structure

The notebook is organized into **16 clear sections**:

### Setup (Sections 1-2)
1. **Imports**: All necessary libraries
2. **Configuration**: Hyperparameters and settings

### Model Creation (Sections 3-4)
3. **Create LSTM Model**: Build the model architecture
4. **Test Forward Pass**: Verify model works

### Stage 1: UTM Training (Sections 5-10)
5. **Create UTM Data Generator**: Universal Turing Machine data
6. **Create FastAI DataLoaders**: Wrap data for FastAI
7. **Create Learner**: Combine model, data, and optimizer
8. **Train Stage 1**: Train on UTM data
9. **Analyze Training**: Review results
10. **Test Generation**: Generate sequences

### Stage 2: CTW Training (Sections 11-14)
11. **Create CTW Data Generator**: Context Tree Weighting data
12. **Create CTW DataLoaders**: Wrap CTW data
13. **Train Stage 2**: Fine-tune on CTW data
14. **Analyze Fine-tuning**: Review results

### Saving & Evaluation (Sections 15-16)
15. **Save Model**: Export trained model
16. **Final Evaluation**: Test on both datasets

## How to Use

### Option 1: Google Colab (Recommended for Learning)

1. **Upload the notebook**:
   ```
   Go to: https://colab.research.google.com/
   File → Upload notebook → Select FastAI_LSTM_Sequential_Finetuning.ipynb
   ```

2. **Enable GPU** (optional but recommended):
   ```
   Runtime → Change runtime type → GPU (T4) → Save
   ```

3. **Install dependencies**:
   ```python
   !pip install fastai torch transformers
   ```

4. **Run cells sequentially**:
   - Read the markdown explanations
   - Run each code cell
   - Observe the outputs
   - Experiment by changing parameters

### Option 2: Local Jupyter

1. **Install Jupyter**:
   ```bash
   pip install jupyter notebook
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_sequential_finetune.txt
   ```

3. **Start Jupyter**:
   ```bash
   cd /path/to/neural_networks_solomonoff_induction
   jupyter notebook
   ```

4. **Open the notebook**:
   - Navigate to `FastAI_LSTM_Sequential_Finetuning.ipynb`
   - Run cells sequentially

### Option 3: JupyterLab

```bash
pip install jupyterlab
jupyter lab
```

## Key Features

### 🎯 Focused and Concise

- **~350 lines** of code + explanations
- **16 clear sections** with specific purposes
- **No unnecessary complexity**

### 📊 Visual Learning

- Shows model architecture
- Displays training progress bars
- Plots loss curves (if matplotlib available)
- Shows generated sequences

### 🔧 Highly Configurable

Easy to modify hyperparameters:

```python
# In Section 2: Configuration
HIDDEN_DIM = 512        # Change hidden size
NUM_LAYERS = 3          # Change depth
UTM_EPOCHS = 10         # More training
MAX_LR = 5e-4           # Different learning rate
```

### ⚡ Quick Execution

With reduced settings for learning:
- **UTM Stage**: ~5-10 minutes (5 epochs, 50 batches/epoch)
- **CTW Stage**: ~3-5 minutes (3 epochs, 50 batches/epoch)
- **Total**: ~10-15 minutes on GPU

## What You'll Learn

### FastAI Concepts

1. **DataLoaders**: How to wrap custom data generators
2. **Learner**: Combining model, data, loss, optimizer
3. **One-Cycle Policy**: Automatic learning rate scheduling
4. **Callbacks**: Custom behavior during training
5. **Metrics**: Tracking perplexity and other measures

### LSTM Training

1. **Model Architecture**: Embedding → LSTM → Output
2. **Weight Tying**: Sharing weights between layers
3. **Log Probabilities**: Why LSTM outputs log probs
4. **Sequential Training**: UTM → CTW curriculum
5. **Generation**: Autoregressive sequence generation

### Sequential Finetuning

1. **Stage 1 (UTM)**: Learning universal computation
2. **Stage 2 (CTW)**: Learning sequence compression
3. **Transfer Learning**: How pre-training helps
4. **Curriculum Learning**: Progressive task difficulty

## Expected Outputs

After running the notebook, you'll have:

```
checkpoints/notebook_fastai/
└── final_model.pth          # Trained model checkpoint
```

Console output will show:
- Model architecture and parameter count
- Training progress bars with metrics
- Final loss and perplexity for each stage
- Evaluation results on both datasets

## Customization Examples

### Faster Training (for quick tests)

```python
# Section 2: Configuration
UTM_EPOCHS = 2           # Reduce epochs
CTW_EPOCHS = 1
BATCHES_PER_EPOCH = 20   # Fewer batches
```

### Better Quality (for production)

```python
# Section 2: Configuration
HIDDEN_DIM = 512         # Larger model
NUM_LAYERS = 3
UTM_EPOCHS = 20          # More training
CTW_EPOCHS = 10
BATCHES_PER_EPOCH = 200  # More data
MAX_LR = 5e-4            # Lower LR
```

### Different Architecture

```python
# Section 3: Create Model
config = LSTMConfig(
    vocab_size=VOCAB_SIZE,
    embedding_dim=256,     # Larger embeddings
    hidden_dim=512,        # Larger hidden
    num_layers=4,          # Deeper network
    dropout=0.3,           # More regularization
)
```

## Troubleshooting

### Out of Memory

Reduce memory usage:

```python
# Section 2: Configuration
BATCH_SIZE = 16          # Smaller batches
SEQ_LENGTH = 128         # Shorter sequences
HIDDEN_DIM = 128         # Smaller model
BATCHES_PER_EPOCH = 25   # Fewer batches
```

### Slow Training

Speed up training:

```python
# Use GPU if available
device = "cuda"

# Reduce epochs for testing
UTM_EPOCHS = 2
CTW_EPOCHS = 1

# Fewer batches
BATCHES_PER_EPOCH = 20
```

### Import Errors

Install missing dependencies:

```bash
pip install fastai fastcore torch transformers
```

Or use the full requirements:

```bash
pip install -r requirements_sequential_finetune.txt
```

## Comparison with Other Implementations

| Feature | This Notebook | train_sequential_finetune.py | train_sequential_finetune_fastai.py |
|---------|---------------|------------------------------|-------------------------------------|
| **Format** | Notebook | Python script | Python script |
| **Purpose** | Learning | Production | Production (FastAI) |
| **Explanations** | Extensive | Minimal | Minimal |
| **Interactive** | Yes | No | No |
| **Visualization** | Yes | No | No |
| **Training** | Epochs-based | Steps-based | Epochs-based |
| **Flexibility** | High | High | High |
| **Best For** | Education | Batch jobs | FastAI users |

## Tips for Best Learning Experience

1. **Read Before Running**: Read each markdown cell before executing code
2. **Experiment**: Change parameters and see what happens
3. **Add Cells**: Insert cells to explore further
4. **Take Notes**: Add markdown cells with your observations
5. **Compare**: Run with different configurations and compare results

## Advanced Usage

### Add Custom Metrics

```python
# In Section 7: Create Learner
from fastai.metrics import Metric

class CustomMetric(Metric):
    def reset(self): 
        self.total = 0
        self.count = 0
    
    def accumulate(self, learn):
        # Your metric logic
        pass
    
    @property
    def value(self):
        return self.total / self.count if self.count > 0 else 0

# Add to learner
learn_utm = Learner(..., metrics=[PerplexityMetric(), CustomMetric()])
```

### Use Learning Rate Finder

```python
# After Section 7 (Create Learner)
# Find optimal learning rate
learn_utm.lr_find()

# This will show a plot of loss vs learning rate
# Use the value where loss is steepest
```

### Export Model for Production

```python
# After Section 15 (Save Model)
# Export for deployment
learn_utm.export('model.pkl')

# Later, load with:
# learn = load_learner('model.pkl')
```

## Related Resources

### Documentation
- [FastAI LSTM Guide](docs/FASTAI_LSTM_GUIDE.md): Comprehensive guide
- [FastAI Implementation](FASTAI_IMPLEMENTATION.md): Implementation details
- [LSTM Architecture](docs/LSTM_ARCHITECTURE.md): LSTM model details

### Scripts
- `train_sequential_finetune_fastai.py`: Production script
- `examples/fastai_lstm_example.py`: Standalone example
- `test_fastai_implementation.py`: Tests

### Other Notebooks
- `Skywork_Sequential_Finetuning_Complete.ipynb`: Alternative approach

## Next Steps

After completing this notebook:

1. ✅ **Understand the pipeline**: You now know how FastAI training works
2. 🚀 **Run production script**: Use `train_sequential_finetune_fastai.py`
3. 🔬 **Experiment**: Try different architectures and hyperparameters
4. 📊 **Compare**: Run with Transformer and compare results
5. 📚 **Deep dive**: Read the comprehensive guides

## Why This Notebook?

This notebook bridges the gap between:
- **Theory**: Understanding how LSTM and FastAI work
- **Practice**: Actually training models effectively

It's perfect for:
- 🎓 Students learning deep learning
- 👩‍🔬 Researchers exploring sequence modeling
- 👨‍💻 Engineers new to FastAI
- 📖 Anyone wanting to understand the implementation

**Start your FastAI LSTM journey here!** 🚀
