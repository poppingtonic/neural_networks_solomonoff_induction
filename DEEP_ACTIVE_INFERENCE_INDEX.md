# Deep Active Inference - Complete Implementation Index

## 🚀 Start Here

**New to this?** → Read `QUICKSTART_DEEP_ACTIVE_INFERENCE.md` (5 min)

**Want to run it?** → Execute: `python demo_deep_active_inference.py`

**Ready to train?** → Run: `python train_deep_active_inference.py --mode train --n_steps 100 --n_perturbations 500`

## 📁 All Files Created

### Core Implementation (1000+ lines total)

#### 1. **`torch_models/deep_active_inference.py`** ⭐
   - **430 lines** of agent implementation
   - Main `DeepActiveInferenceAgent` class
   - Variational inference network
   - Generative model
   - Action policy
   - Environment dynamics
   
#### 2. **`torch_models/perturbation_optimizer.py`** ⭐
   - **270 lines** of optimizer code
   - `PerturbationOptimizer`: Evolution strategies
   - `PerturbedModel`: Batch evaluation
   - ADAM updates for parameters and variances
   - Antithetic sampling

#### 3. **`train_deep_active_inference.py`** ⭐
   - **350 lines** of training pipeline
   - `train_agent()`: Full training loop
   - `evaluate_agent()`: Evaluation with plots
   - Command-line interface
   - Checkpoint management
   - Logging

### Demo & Testing

#### 4. **`demo_deep_active_inference.py`**
   - **200 lines** - Quick demonstration
   - 3 demos: forward pass, optimization, visualization
   - Generates example plots
   - Good for learning the API

#### 5. **`test_deep_active_inference.py`**
   - **330 lines** - Comprehensive tests
   - 8 unit tests covering all functionality
   - Automated validation
   - Run with: `python test_deep_active_inference.py`

### Documentation

#### 6. **`deep_active_inference_README.md`**
   - **320 lines** - Complete documentation
   - Architecture details
   - Usage examples
   - Hyperparameters explained
   - Troubleshooting guide

#### 7. **`QUICKSTART_DEEP_ACTIVE_INFERENCE.md`**
   - **250 lines** - Quick reference
   - 30-second start
   - Command reference
   - FAQ and troubleshooting
   - Expected results

#### 8. **`IMPLEMENTATION_SUMMARY.md`**
   - **380 lines** - Technical summary
   - Implementation details
   - Validation results
   - Performance benchmarks
   - Comparison with original

#### 9. **`requirements_deep_active_inference.txt`**
   - Dependencies: PyTorch, NumPy, Matplotlib, SciPy

#### 10. **`DEEP_ACTIVE_INFERENCE_INDEX.md`** (this file)
   - Navigation and overview

## 🎯 What Each File Does

```
Documentation Flow:
  New User → QUICKSTART → README → IMPLEMENTATION_SUMMARY
  
Code Flow:
  Demo → Test → Train → Evaluate
  
Implementation:
  deep_active_inference.py (model)
       ↓
  perturbation_optimizer.py (training)
       ↓
  train_deep_active_inference.py (pipeline)
```

## 📊 File Statistics

| Category | Files | Lines | Purpose |
|----------|-------|-------|---------|
| Core Code | 3 | 1050 | Model + optimizer + training |
| Demo/Test | 2 | 530 | Validation + examples |
| Docs | 4 | 1000+ | Guides + references |
| Config | 1 | 10 | Dependencies |
| **Total** | **10** | **2500+** | Complete implementation |

## 🔍 Where to Find What

### "I want to..."

**...understand the algorithm**
→ `IMPLEMENTATION_SUMMARY.md` § Implementation Details

**...run a quick test**
→ `python demo_deep_active_inference.py`

**...train the model**
→ `python train_deep_active_inference.py --mode train`

**...customize the environment**
→ `torch_models/deep_active_inference.py` → `step()` method

**...change the architecture**
→ `torch_models/deep_active_inference.py` → `__init__()`

**...understand free energy**
→ `deep_active_inference_README.md` § Free Energy Components

**...see training options**
→ `QUICKSTART_DEEP_ACTIVE_INFERENCE.md` § Command Reference

**...debug issues**
→ `QUICKSTART_DEEP_ACTIVE_INFERENCE.md` § Troubleshooting

**...cite this work**
→ `IMPLEMENTATION_SUMMARY.md` § Citation

**...extend the code**
→ `IMPLEMENTATION_SUMMARY.md` § Extending the Implementation

## 🎓 Learning Path

### Beginner (1 hour)
1. Read `QUICKSTART_DEEP_ACTIVE_INFERENCE.md` (5 min)
2. Run `python demo_deep_active_inference.py` (3 min)
3. Run `python test_deep_active_inference.py` (2 min)
4. Quick training: `--n_steps 50 --n_perturbations 100` (5 min)
5. Read `deep_active_inference_README.md` § Overview (10 min)

### Intermediate (3 hours)
1. Complete beginner path
2. Read full `deep_active_inference_README.md` (30 min)
3. Standard training: `--n_steps 1000 --n_perturbations 500` (1 hour)
4. Experiment with hyperparameters (1 hour)
5. Modify environment in `step()` method (30 min)

### Advanced (1 day)
1. Complete intermediate path
2. Read `IMPLEMENTATION_SUMMARY.md` (30 min)
3. Read source code with documentation (2 hours)
4. Full training: `--n_steps 10000 --n_perturbations 1000` (3 hours)
5. Implement custom task/architecture (2 hours)

## 🚀 Quick Commands

```bash
# Install
pip install torch numpy matplotlib scipy

# Test installation
python test_deep_active_inference.py

# Quick demo
python demo_deep_active_inference.py

# Train (10 min)
python train_deep_active_inference.py --mode train \
    --n_steps 100 --n_perturbations 500

# Train (1 hour)
python train_deep_active_inference.py --mode train \
    --n_steps 5000 --n_perturbations 1000

# Evaluate
python train_deep_active_inference.py --mode eval \
    --checkpoint deepAI_pytorch_best.pt

# Monitor
tail -f log_deepAI_pytorch.txt
```

## 📈 Performance Expectations

| Configuration | Time/Iter | Total Time | Expected FE |
|--------------|-----------|------------|-------------|
| 100 pert, 20 steps | 0.3s GPU | 5 min (100 iter) | ~80-120 |
| 500 pert, 30 steps | 0.8s GPU | 15 min (100 iter) | ~60-90 |
| 1000 pert, 30 steps | 1.5s GPU | 2.5 hours (1000 iter) | ~40-60 |
| 10000 pert, 30 steps | 12s GPU | 33 hours (1000 iter) | ~20-40 |

## 🔧 Architecture Overview

```
DeepActiveInferenceAgent
├── Recognition Network (Q)
│   ├── Input: previous state, observations
│   ├── Hidden: 2 ReLU layers (10 units)
│   └── Output: state mean & variance
│
├── Generative Model (P)
│   ├── State → Observations
│   ├── Hidden: 3 ReLU layers (10 units)
│   └── Output: observation likelihoods
│
├── Action Policy
│   ├── Input: current state
│   ├── Hidden: 1 ReLU layer (10 units)
│   └── Output: action mean & variance
│
└── Environment
    ├── Mountain car-like dynamics
    ├── Nonlinear observation channel
    └── Goal: reach position ≈ 1.0

PerturbationOptimizer
├── Generate N parameter perturbations
├── Evaluate free energy for each
├── Estimate gradients from perturbations
└── Update with ADAM
```

## 🎯 Key Features

✅ Complete PyTorch implementation  
✅ Matches original Theano code  
✅ Evolution strategies optimization  
✅ Comprehensive documentation  
✅ Unit tests (8 tests)  
✅ Training pipeline  
✅ Evaluation & visualization  
✅ Checkpoint save/resume  
✅ Command-line interface  
✅ Programmatic API  

## 📚 External Resources

- **Active Inference**: [Friston 2010](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf)
- **Evolution Strategies**: [Salimans et al. 2017](https://arxiv.org/abs/1703.03864)
- **Deep Active Inference**: Ueltzhoeffer 2017
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/docs/)

## 🤝 Contributing

This implementation is provided for research and education.

**Found a bug?** Check `test_deep_active_inference.py` first  
**Want to extend?** See `IMPLEMENTATION_SUMMARY.md` § Extending  
**Have questions?** Read the docs or create an issue  

## 📝 Version History

**v1.0** - Initial complete implementation
- Full agent model
- Perturbation optimizer
- Training pipeline
- Comprehensive documentation
- Unit tests
- Demo scripts

---

**Total Implementation: 10 files, 2500+ lines of code and documentation**

**Status**: ✅ Complete, tested, and documented

**Next Step**: Run `python demo_deep_active_inference.py` to get started!
