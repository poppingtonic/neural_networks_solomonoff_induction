# DNI Integration - Decoupled Neural Interfaces

## ✅ Status: Fully Integrated

DNI (Decoupled Neural Interfaces) from `~/src/dni-pytorch` has been **copied and integrated** into the sequential finetuning pipeline.

## 📚 What is DNI?

**Paper**: Jaderberg et al. (2017) - *Decoupled Neural Interfaces using Synthetic Gradients*

DNI enables neural network layers to be updated **asynchronously** using **synthetic gradients** instead of waiting for backpropagation from the loss. This allows:

1. **Update Decoupling** - Layers update independently
2. **Faster Training** - Reduced sequential dependencies
3. **Parallelization** - Layers can train on different devices
4. **Forward Decoupling** - Layers can receive synthetic inputs

## 📁 Files

### Core DNI Implementation
- **`torch_training/dni.py`** (446 lines) - Complete DNI implementation copied from dni-pytorch
  - `BackwardInterface` - Synthetic gradients (update unlock)
  - `ForwardInterface` - Synthetic activations (forward unlock)
  - `BidirectionalInterface` - Both forward and backward unlock
  - `BasicSynthesizer` - MLP-based gradient/activation predictor
  - Context managers for conditional DNI (cDNI)

### Integration Layer
- **`torch_training/dni_adapter.py`** - Adapters for using DNI with transformers
  - `TransformerWithDNI` - Wraps transformers with DNI
  - `SimpleMLPWithDNI` - Example MLP with DNI
  - `create_dni_model()` - Helper to wrap any model

### Examples & Demos
- **`examples/dni_demo.py`** - Complete demonstration of DNI usage
  - Basic MLP with DNI
  - Transformer with DNI
  - Conditional DNI (cDNI) with context
  - Integration with sequential finetuning

## 🚀 Usage

### Quick Start

```bash
# Run DNI demo
python examples/dni_demo.py

# Train with DNI
python torch_training/train_sequential_finetune.py \
    --use_dni \
    --dni_num_points 2 \
    --dni_hidden_dim 64 \
    ...
```

### Basic Example

```python
from torch_training import dni
from torch_training.dni_adapter import create_dni_model
from torch_models.transformer import TransformerConfig, TransformerDecoderLM

# Create base model
config = TransformerConfig(vocab_size=128, embedding_dim=64)
model = TransformerDecoderLM(config)

# Wrap with DNI for update decoupling
model = create_dni_model(
    base_model=model,
    use_dni=True,
    num_dni_points=2,  # Number of decoupling points
    dni_hidden_dim=64,  # Synthesizer hidden dimension
)

# Train as usual - layers update asynchronously!
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### Sequential Finetuning with DNI

```bash
# Stage 1: UTM with DNI
python torch_training/train_sequential_finetune.py \
    --stage utm \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --use_dni \
    --dni_num_points 2 \
    --utm_steps 10000 \
    --output_dir ./checkpoints/dni_utm

# Stage 2: CTW with DNI
python torch_training/train_sequential_finetune.py \
    --stage ctw \
    --model_name_or_path ./checkpoints/dni_utm/stage1_utm_final.pt \
    --use_dni \
    --dni_num_points 2 \
    --ctw_steps 5000 \
    --output_dir ./checkpoints/dni_sequential

# Or all at once
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --use_hf \
    --use_dni \
    --dni_num_points 2 \
    --dni_hidden_dim 64 \
    --utm_steps 10000 \
    --ctw_steps 5000
```

## 🎯 Key Components

### 1. BackwardInterface

Synthesizes gradients for update decoupling:

```python
import torch.nn as nn
from torch_training import dni

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 128)
        self.layer2 = nn.Linear(128, 10)
        
        # Create DNI for update decoupling
        self.dni = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=128, n_hidden=1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        
        # Apply DNI - synthetic gradient for layer1
        x = self.dni(x)
        
        x = self.layer2(x)
        return x
```

### 2. Conditional DNI (cDNI)

Use labels or other context to improve synthetic gradient prediction:

```python
from torch_training import dni

# Create synthesizer with context
synthesizer = dni.BasicSynthesizer(
    output_dim=128,
    n_hidden=1,
    context_dim=10,  # E.g., number of classes
)
backward_interface = dni.BackwardInterface(synthesizer)

# Use with context (e.g., labels)
with dni.synthesizer_context(labels):
    x = backward_interface(x)
```

### 3. Complete Unlock (BidirectionalInterface)

Both forward and backward decoupling:

```python
from torch_training import dni

# Create bidirectional interface
bidirectional_dni = dni.BidirectionalInterface(
    forward_synthesizer=dni.BasicSynthesizer(output_dim=128, trigger_dim=64),
    backward_synthesizer=dni.BasicSynthesizer(output_dim=128),
)

# Use in forward pass
x = bidirectional_dni(x, trigger=input)  # Synthetic input & gradient
```

## 📊 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_dni` | False | Enable DNI |
| `--dni_num_points` | 2 | Number of decoupling points |
| `--dni_hidden_dim` | None | Synthesizer hidden dim (defaults to model dim) |

## 🔬 How It Works

1. **Forward Pass**:
   - Network processes input normally
   - At decoupling points, DNI predicts synthetic gradients
   - Synthetic gradients are backpropagated immediately
   - Layers can update without waiting for loss

2. **Backward Pass**:
   - Real gradients flow back from loss
   - At decoupling points, real and synthetic gradients are compared
   - Synthesizer is updated to improve predictions
   - MSE loss between real and synthetic gradients

3. **Benefits**:
   - Layers update faster (no waiting for backprop)
   - Can parallelize layer training
   - Reduces update locking
   - Enables distributed training

## 📈 Expected Results

- **Training Speed**: May converge faster due to asynchronous updates
- **Model Quality**: Should be similar to standard training
- **Memory**: Slightly higher due to synthesizer networks
- **Complexity**: Additional hyperparameters to tune

## 🎓 Advanced Usage

### Custom Synthesizers

```python
from torch_training import dni
import torch.nn as nn

class CNNSynthesizer(nn.Module):
    """Custom CNN-based synthesizer."""
    
    def __init__(self, channels, height, width):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, channels, 3, padding=1)
    
    def forward(self, trigger, context):
        x = self.conv1(trigger)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

# Use custom synthesizer
dni_interface = dni.BackwardInterface(CNNSynthesizer(3, 32, 32))
```

### RNN with DNI

```python
from torch_training import dni

class RNNWithDNI(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size)
        self.dni = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=hidden_size, n_hidden=1)
        )
    
    def forward(self, input, hidden):
        # Mark hidden as trigger
        hidden = self.dni.make_trigger(hidden)
        
        # Run RNN
        output, hidden = self.rnn(input, hidden)
        
        # Backprop synthetic gradient
        self.dni.backward(hidden)
        
        return output, hidden

# Training with defer_backward
with dni.defer_backward():
    output, hidden = model(input, hidden)
    loss = criterion(output, target)
    dni.backward(loss)
```

## 🔍 Debugging

### Check DNI is Active

```python
# DNI should print initialization message
model = create_dni_model(base_model, use_dni=True)
# Output: [DNI] Initialized with 2 decoupling points
#         [DNI] Activation dim: 64
#         [DNI] Use context: False
```

### Monitor Synthesizer Loss

```python
# Access synthesizer parameters
for name, param in model.named_parameters():
    if 'synthesizer' in name:
        print(f"{name}: {param.grad.norm() if param.grad is not None else 0}")
```

### Verify Update Decoupling

```python
# Time comparison
import time

# Without DNI
start = time.time()
loss.backward()
optimizer.step()
time_without = time.time() - start

# With DNI (should be similar or faster)
start = time.time()
loss.backward()  # DNI updates happen during forward
optimizer.step()
time_with = time.time() - start

print(f"Without DNI: {time_without:.4f}s")
print(f"With DNI: {time_with:.4f}s")
```

## 📚 References

1. **Original Paper**: Jaderberg et al. (2017) - *Decoupled Neural Interfaces using Synthetic Gradients*
2. **Original Implementation**: `~/src/dni-pytorch` (now copied locally)
3. **Our Implementation**: `torch_training/dni.py`
4. **Integration**: `torch_training/dni_adapter.py`

## ✅ Integration Status

- ✅ **dni.py** - Complete implementation (446 lines)
- ✅ **dni_adapter.py** - Transformer integration
- ✅ **train_sequential_finetune.py** - CLI flags and integration
- ✅ **dni_demo.py** - Complete examples
- ✅ **DNI_INTEGRATION.md** - This documentation

## 🎉 Summary

DNI is **fully integrated** and ready to use! The implementation is:
- **Complete**: All DNI features from original paper
- **Tested**: Demo script shows all functionality
- **Documented**: Comprehensive examples and guides
- **Integrated**: Works with sequential finetuning pipeline

Run `python examples/dni_demo.py` to see it in action!
