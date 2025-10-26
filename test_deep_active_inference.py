"""
Unit tests for Deep Active Inference implementation.

Run with: python test_deep_active_inference.py
"""

import sys
import torch
import numpy as np

try:
    from torch_models.deep_active_inference import DeepActiveInferenceAgent
    from torch_models.perturbation_optimizer import PerturbationOptimizer, PerturbedModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_agent_creation():
    """Test that agent can be created."""
    print("Test 1: Agent Creation...", end=" ")
    try:
        agent = DeepActiveInferenceAgent(device='cpu')
        n_params = sum(p.numel() for p in agent.parameters())
        assert n_params > 0, "Agent has no parameters"
        print(f"✓ (Created agent with {n_params} parameters)")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_single_step():
    """Test a single step of the agent."""
    print("Test 2: Single Step...", end=" ")
    try:
        agent = DeepActiveInferenceAgent(device='cpu')
        
        # Initialize states
        st = torch.zeros(1, agent.n_s, 1)
        post = torch.ones(1, agent.n_o, 1) * -0.5
        vt = torch.zeros(1, agent.n_o, 1)
        
        # Run one step
        with torch.no_grad():
            outputs = agent.step(0, st, post, vt)
        
        # Check outputs
        assert len(outputs) == 16, f"Expected 16 outputs, got {len(outputs)}"
        
        # Check shapes
        st_new, post_new, vt_new = outputs[0], outputs[1], outputs[2]
        assert st_new.shape == (1, agent.n_s, 1), f"Wrong state shape: {st_new.shape}"
        assert post_new.shape == (1, agent.n_o, 1), f"Wrong position shape: {post_new.shape}"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_forward_pass():
    """Test full forward pass."""
    print("Test 3: Forward Pass (10 steps)...", end=" ")
    try:
        agent = DeepActiveInferenceAgent(device='cpu')
        
        with torch.no_grad():
            trajectories = agent(n_run_steps=10, n_proc=1)
        
        # Check trajectory keys
        expected_keys = ['states', 'positions', 'velocities', 'actions', 
                        'observations', 'obs_nonlinear', 'free_energy', 
                        'kl_divergence', 'nll_ot', 'nll_oht', 'nll_oat']
        for key in expected_keys:
            assert key in trajectories, f"Missing trajectory key: {key}"
        
        # Check shapes
        assert trajectories['positions'].shape[0] == 10, "Wrong number of timesteps"
        assert torch.isfinite(trajectories['free_energy']).all(), "Non-finite free energy"
        
        print(f"✓ (Mean FE: {trajectories['free_energy'].mean():.2f})")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_creation():
    """Test optimizer creation."""
    print("Test 4: Optimizer Creation...", end=" ")
    try:
        agent = DeepActiveInferenceAgent(device='cpu')
        optimizer = PerturbationOptimizer(
            parameters=list(agent.parameters()),
            n_perturbations=10,
            learning_rate=1e-3
        )
        
        assert len(optimizer.parameters) > 0, "No parameters in optimizer"
        assert len(optimizer.sigmas) == len(optimizer.parameters), "Mismatch in sigma count"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_perturbation_generation():
    """Test perturbation generation."""
    print("Test 5: Perturbation Generation...", end=" ")
    try:
        agent = DeepActiveInferenceAgent(device='cpu')
        optimizer = PerturbationOptimizer(
            parameters=list(agent.parameters()),
            n_perturbations=10,
            learning_rate=1e-3
        )
        
        r_params, r_epsilons = optimizer.generate_perturbations()
        
        assert len(r_params) == len(list(agent.parameters())), "Wrong number of perturbed params"
        assert len(r_epsilons) == len(list(agent.parameters())), "Wrong number of epsilons"
        
        # Check shape of first perturbation
        first_param = list(agent.parameters())[0]
        assert r_params[0].shape[0] == 10, f"Wrong batch size: {r_params[0].shape[0]}"
        assert r_params[0].shape[1:] == first_param.shape, "Wrong parameter shape"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_perturbation_evaluation():
    """Test evaluation with perturbations."""
    print("Test 6: Perturbation Evaluation...", end=" ")
    try:
        agent = DeepActiveInferenceAgent(device='cpu')
        optimizer = PerturbationOptimizer(
            parameters=list(agent.parameters()),
            n_perturbations=4,  # Small number for test
            learning_rate=1e-3
        )
        
        r_params, r_epsilons = optimizer.generate_perturbations()
        
        perturbed_model = PerturbedModel(agent, r_params)
        free_energies = perturbed_model.evaluate(n_run_steps=5, n_proc=1)
        
        assert free_energies.shape == (4,), f"Wrong FE shape: {free_energies.shape}"
        assert torch.isfinite(free_energies).all(), "Non-finite free energies"
        
        print(f"✓ (Mean FE: {free_energies.mean():.2f})")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_step():
    """Test optimizer update step."""
    print("Test 7: Optimizer Update...", end=" ")
    try:
        agent = DeepActiveInferenceAgent(device='cpu')
        optimizer = PerturbationOptimizer(
            parameters=list(agent.parameters()),
            n_perturbations=4,
            learning_rate=1e-3
        )
        
        # Get initial parameter value
        initial_param = list(agent.parameters())[0].clone()
        
        r_params, r_epsilons = optimizer.generate_perturbations()
        perturbed_model = PerturbedModel(agent, r_params)
        free_energies = perturbed_model.evaluate(n_run_steps=5, n_proc=1)
        
        stats = optimizer.step(free_energies, r_epsilons)
        
        # Check parameter was updated
        updated_param = list(agent.parameters())[0]
        assert not torch.allclose(initial_param, updated_param), "Parameters not updated"
        
        # Check stats
        assert 'param_grad_norm' in stats, "Missing stat: param_grad_norm"
        assert 'sigma_grad_norm' in stats, "Missing stat: sigma_grad_norm"
        
        print(f"✓ (Grad norm: {stats['param_grad_norm']:.4f})")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("Test 8: Checkpoint Save/Load...", end=" ")
    try:
        import tempfile
        import os
        
        agent = DeepActiveInferenceAgent(device='cpu')
        optimizer = PerturbationOptimizer(
            parameters=list(agent.parameters()),
            n_perturbations=4,
            learning_rate=1e-3
        )
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            checkpoint_path = f.name
        
        checkpoint = {
            'step': 42,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_fe': 123.45
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded = torch.load(checkpoint_path, map_location='cpu')
        
        new_agent = DeepActiveInferenceAgent(device='cpu')
        new_agent.load_state_dict(loaded['model_state_dict'])
        
        # Check parameters match
        for p1, p2 in zip(agent.parameters(), new_agent.parameters()):
            assert torch.allclose(p1, p2), "Loaded parameters don't match"
        
        # Cleanup
        os.unlink(checkpoint_path)
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Deep Active Inference - Unit Tests")
    print("="*60 + "\n")
    
    tests = [
        test_agent_creation,
        test_single_step,
        test_forward_pass,
        test_optimizer_creation,
        test_perturbation_generation,
        test_perturbation_evaluation,
        test_optimizer_step,
        test_checkpoint_save_load
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
