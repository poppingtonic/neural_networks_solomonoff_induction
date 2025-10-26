# Deep Active Inference - Quick Start Guide

## 30-Second Start

```bash
# Install dependencies
pip install torch numpy matplotlib scipy

# Run demo (2-3 minutes)
python demo_deep_active_inference.py

# Run tests
python test_deep_active_inference.py

# Train briefly (10 minutes)
python train_deep_active_inference.py --mode train \
    --n_steps 100 --n_perturbations 500

# Evaluate
python train_deep_active_inference.py --mode eval
```

## What is This?

An agent that learns to control a mountain car-like environment by minimizing **variational free energy**. The agent:

1. **Observes**: Position (noisy), nonlinear channel (Gaussian bump), action feedback
2. **Infers**: Hidden state representation using variational inference
3. **Predicts**: What observations should occur given beliefs
4. **Acts**: To minimize surprise (free energy)

## Key Concepts

### Free Energy
```
FE = KL[Beliefs || Prior] + Surprise
   = How different are beliefs from expectations + How unexpected are observations
```

The agent minimizes this by:
- Better state inference (reduce KL)
- Better predictions (reduce surprise)
- Better actions (move to predictable states)

### Parameter Perturbation
Instead of backpropagation, uses evolution strategies:
1. Add random noise to all parameters (N times)
2. Evaluate each noisy version
3. Update toward noise that reduced free energy

This works without computing gradients through time!

## File Guide

| File | Purpose | Lines |
|------|---------|-------|
| `torch_models/deep_active_inference.py` | Agent model | 430 |
| `torch_models/perturbation_optimizer.py` | Optimizer | 270 |
| `train_deep_active_inference.py` | Train/eval script | 350 |
| `demo_deep_active_inference.py` | Quick demo | 200 |
| `test_deep_active_inference.py` | Unit tests | 330 |
| `deep_active_inference_README.md` | Full docs | 320 |

## Command Reference

### Training

```bash
# Quick test (100 steps, ~10 min)
python train_deep_active_inference.py --mode train \
    --n_steps 100 --n_perturbations 500

# Standard (5000 steps, ~1 hour)
python train_deep_active_inference.py --mode train \
    --n_steps 5000 --n_perturbations 1000

# Full (50000 steps, ~10 hours)
python train_deep_active_inference.py --mode train \
    --n_steps 50000 --n_perturbations 10000

# Resume training
python train_deep_active_inference.py --mode train \
    --checkpoint deepAI_pytorch_best.pt \
    --n_steps 10000

# Custom name
python train_deep_active_inference.py --mode train \
    --base_name my_experiment \
    --n_steps 1000
```

### Evaluation

```bash
# Evaluate best model
python train_deep_active_inference.py --mode eval

# Specific checkpoint
python train_deep_active_inference.py --mode eval \
    --checkpoint deepAI_pytorch_500.pt

# Longer rollout
python train_deep_active_inference.py --mode eval \
    --n_run_steps 100

# No visualization
python train_deep_active_inference.py --mode eval --no_plot
```

### Programmatic Use

```python
import torch
from torch_models.deep_active_inference import DeepActiveInferenceAgent

# Create and run agent
agent = DeepActiveInferenceAgent()
with torch.no_grad():
    results = agent(n_run_steps=50)

# Check results
print(f"Free energy: {results['free_energy'].mean():.2f}")
print(f"Final position: {results['positions'][-1, 0, 0, 0]:.2f}")

# Train
from torch_models.perturbation_optimizer import PerturbationOptimizer, PerturbedModel

optimizer = PerturbationOptimizer(
    parameters=list(agent.parameters()),
    n_perturbations=100
)

for step in range(100):
    # Generate perturbations
    r_params, r_epsilons = optimizer.generate_perturbations()
    
    # Evaluate
    perturbed = PerturbedModel(agent, r_params)
    free_energies = perturbed.evaluate(n_run_steps=30)
    
    # Update
    optimizer.step(free_energies, r_epsilons)
    
    if step % 10 == 0:
        print(f"Step {step}, FE: {free_energies.mean():.2f}")
```

## Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `n_perturbations` | 1000 | 100-10000 | More = better gradients, slower |
| `n_run_steps` | 30 | 10-100 | Episode length |
| `learning_rate` | 1e-3 | 1e-4 - 1e-2 | Update step size |
| `n_s` | 10 | 5-50 | Hidden state capacity |

## Expected Results

### Untrained Agent
- Free energy: ~150-300
- Random movements
- Position: oscillates around -0.5

### After 100 steps
- Free energy: ~80-120
- Some directed movement
- Position: reaches -0.2 to 0.5

### After 1000 steps
- Free energy: ~40-70
- Clear goal-directed behavior
- Position: often reaches 0.8-1.2

### After 10000 steps
- Free energy: ~20-40
- Consistent reaching of goal
- Position: reliably reaches ~1.0

## Outputs

### During Training
- `log_deepAI_pytorch.txt`: Training log
  - Format: `step fe_mean fe_std fe_min grad_norm sigma_grad_norm`
- `deepAI_pytorch_current.pt`: Latest checkpoint
- `deepAI_pytorch_best.pt`: Best checkpoint so far
- `deepAI_pytorch_{step}.pt`: Periodic checkpoints

### During Evaluation
- Terminal output with statistics
- `{checkpoint_name}_evaluation.png`: Trajectory plots
- Returns trajectory dictionary

## Monitoring Training

```bash
# Watch log file
tail -f log_deepAI_pytorch.txt

# Plot learning curve
python -c "
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('log_deepAI_pytorch.txt')
plt.plot(data[:, 0], data[:, 1])  # step vs FE
plt.xlabel('Step')
plt.ylabel('Free Energy')
plt.savefig('learning_curve.png')
"
```

## FAQ

**Q: How long does training take?**
A: 100 steps with 500 perturbations: ~10 minutes on GPU, ~30 minutes on CPU

**Q: Do I need a GPU?**
A: Recommended for n_perturbations > 1000. Works fine on CPU for smaller experiments.

**Q: Why is it so slow?**
A: Each iteration evaluates the model n_perturbations times. This is the cost of gradient-free optimization.

**Q: Can I use standard gradient descent?**
A: Yes! The model supports `.backward()`. But perturbation method was used in the original paper.

**Q: How do I know if it's working?**
A: Free energy should decrease over time. Final position should approach 1.0.

**Q: What if I get NaN?**
A: Lower learning rate to 1e-4 or reduce n_perturbations.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `n_perturbations` to 100-500 |
| Too slow | Use GPU, reduce `n_perturbations` or `n_run_steps` |
| No improvement | Train longer, increase `n_perturbations` |
| Unstable | Lower `learning_rate` to 1e-4 |
| Import error | Run from project root, install dependencies |

## Next Steps

1. Read `deep_active_inference_README.md` for details
2. Check `IMPLEMENTATION_SUMMARY.md` for architecture
3. Modify `step()` method for custom environments
4. Experiment with different architectures

## Resources

- Original paper: Ueltzhoeffer (2017) "Deep Active Inference"
- Active Inference: Friston (2010) "The free-energy principle"
- Evolution Strategies: Salimans et al. (2017)
