# Deep Active Inference Implementation (PyTorch)

This is a PyTorch implementation of the Deep Active Inference agent from Kai Ueltzhoeffer (2017), originally implemented in Theano.

## Overview

The agent learns to control a mountain car-like environment through variational free energy minimization. The key components are:

1. **Variational Inference**: The agent maintains beliefs about hidden states using a recognition network
2. **Generative Model**: Predicts observations from hidden states
3. **Action Selection**: Actions minimize expected free energy
4. **Parameter Perturbation Optimization**: Uses evolution strategies-style optimization with antithetic sampling

## Architecture

### Model Components

- **Recognition Network** `Q(s_t | s_t-1, o_t, oh_t, oa_t)`: Infers hidden states from observations
  - 2-layer network with ReLU activations
  - Outputs mean and std for variational posterior

- **Generative Model** `p(o_t | s_t)`: Predicts observations from states
  - 3-layer network for observation generation
  - Separate likelihood models for position, nonlinear channel, and proprioception

- **Transition Model** `p(s_t | s_t-1)`: Prior over state dynamics
  - Linear dynamics with learned transition matrix

- **Action Network**: Generates actions from hidden states
  - Policy outputs mean and std for action distribution

### Environment

Mountain car-like task with:
- Position dynamics with gravity-like force
- Nonlinear observation channel (Gaussian bump at position 1.0)
- Action-dependent force
- Goal: Navigate to position near 1.0

## Usage

### Training

```bash
# Basic training (1000 steps, 1000 perturbations)
python train_deep_active_inference.py --mode train --n_steps 1000 --n_perturbations 1000

# Extended training with more perturbations
python train_deep_active_inference.py --mode train \
    --n_steps 10000 \
    --n_perturbations 5000 \
    --n_run_steps 30 \
    --learning_rate 1e-3 \
    --saving_steps 50 \
    --base_name deepAI_exp1

# Resume from checkpoint
python train_deep_active_inference.py --mode train \
    --checkpoint deepAI_exp1_best.pt \
    --n_steps 20000
```

### Evaluation

```bash
# Evaluate best checkpoint
python train_deep_active_inference.py --mode eval --checkpoint deepAI_pytorch_best.pt

# Evaluate with longer rollout
python train_deep_active_inference.py --mode eval \
    --checkpoint deepAI_pytorch_best.pt \
    --n_run_steps 100

# Evaluate without plotting
python train_deep_active_inference.py --mode eval \
    --checkpoint deepAI_pytorch_best.pt \
    --no_plot
```

### Programmatic Usage

```python
import torch
from torch_models.deep_active_inference import DeepActiveInferenceAgent
from torch_models.perturbation_optimizer import PerturbationOptimizer, PerturbedModel

# Create agent
agent = DeepActiveInferenceAgent(
    n_s=10,  # Hidden states
    n_o=1,   # Position observations
    n_oh=1,  # Nonlinear observations
    n_oa=1,  # Action/proprioception
    device='cuda'
)

# Run simulation
trajectories = agent(n_run_steps=50, n_proc=1)

# Access results
positions = trajectories['positions']
actions = trajectories['actions']
free_energy = trajectories['free_energy']

print(f"Mean free energy: {free_energy.mean():.4f}")

# Train with perturbation optimizer
optimizer = PerturbationOptimizer(
    parameters=list(agent.parameters()),
    n_perturbations=1000,
    learning_rate=1e-3
)

# Generate perturbations
r_params, r_epsilons = optimizer.generate_perturbations()

# Evaluate perturbations
perturbed_model = PerturbedModel(agent, r_params)
free_energies = perturbed_model.evaluate(n_run_steps=30, n_proc=1)

# Update parameters
stats = optimizer.step(free_energies, r_epsilons)
```

## Files

- `torch_models/deep_active_inference.py`: Main agent implementation
- `torch_models/perturbation_optimizer.py`: Parameter perturbation optimization
- `train_deep_active_inference.py`: Training and evaluation script
- `deep_active_inference_README.md`: This file

## Training Details

### Optimization Method

The agent uses **parameter perturbation** (evolution strategies) rather than backpropagation:

1. Sample N perturbations of all parameters using antithetic sampling
2. Evaluate free energy for each perturbation
3. Estimate gradients: `∇θ ≈ E[FE * ε / σ]`
4. Update parameters and their perturbation variances with ADAM

This approach:
- Doesn't require computing gradients through the recurrent dynamics
- Is robust to non-differentiable operations
- Naturally explores parameter space

### Hyperparameters

Default settings (matching original paper):
- `n_s = 10`: Hidden state dimensions
- `n_perturbations = 10000`: Parameter samples per iteration (original paper)
- `n_perturbations = 1000`: Reduced for faster experimentation
- `n_run_steps = 30`: Simulation timesteps per evaluation
- `learning_rate = 1e-3`: ADAM learning rate
- Antithetic sampling for variance reduction

### Free Energy Components

The variational free energy decomposes as:
```
FE = KL[Q(s_t|o_t) || P(s_t|s_{t-1})] + E_Q[-log P(o_t|s_t)]
   = KL divergence + Reconstruction error
```

Where:
- KL term: Difference between posterior and prior beliefs
- Reconstruction: Negative log-likelihood of observations

## Performance Notes

- **GPU highly recommended**: With 10000 perturbations, each iteration is computationally intensive
- **Memory usage**: Scales with `n_perturbations` × model_size
- **Training time**: ~1-2 seconds per iteration on GPU (1000 perturbations, 30 timesteps)
- **Convergence**: Typically requires 1000-10000 iterations to see good behavior

## Differences from Original

This PyTorch implementation differs slightly from the original Theano code:

1. **Batching**: Uses explicit batching rather than scan operations
2. **Parameter initialization**: Uses PyTorch conventions (QR decomposition for orthogonal)
3. **Optimization**: Simplified perturbation evaluation (sequential rather than fully vectorized)
4. **Logging**: Added structured logging and visualization

Core algorithm and architecture remain faithful to the original.

## Troubleshooting

### Out of Memory

Reduce `n_perturbations`:
```bash
python train_deep_active_inference.py --n_perturbations 500
```

### Slow Training

- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce `n_run_steps` during initial training
- Use fewer perturbations for hyperparameter search

### Poor Performance

- Train longer (10000+ steps)
- Increase perturbations for better gradient estimates
- Check learning rate (try 1e-4 or 1e-2)
- Monitor gradient norms in log file

## Citation

If you use this implementation, please cite the original work:

```bibtex
@article{ueltzhoeffer2017deep,
  title={Deep Active Inference},
  author={Ueltzhoeffer, Kai},
  year={2017}
}
```

## References

- Original Theano implementation concepts
- [Active Inference](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf)
- [Evolution Strategies](https://arxiv.org/abs/1703.03864)
