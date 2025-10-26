"""
Quick demo of the Deep Active Inference agent.

This script runs a short evaluation to verify the implementation works.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_models.deep_active_inference import DeepActiveInferenceAgent
from torch_models.perturbation_optimizer import PerturbationOptimizer, PerturbedModel


def demo_forward_pass():
    """Test a single forward pass through the agent."""
    print("="*60)
    print("Demo 1: Single Forward Pass")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create agent
    agent = DeepActiveInferenceAgent(
        n_s=10,
        n_o=1,
        n_oh=1,
        n_oa=1,
        device=device
    )
    
    print(f"Model created with {sum(p.numel() for p in agent.parameters())} parameters")
    
    # Run simulation
    print("\nRunning 30-step simulation...")
    with torch.no_grad():
        trajectories = agent(n_run_steps=30, n_proc=1)
    
    # Print results
    print("\nResults:")
    print(f"  Mean Free Energy: {trajectories['free_energy'].mean().item():.4f}")
    print(f"  Final position: {trajectories['positions'][-1, 0, 0, 0].item():.4f}")
    print(f"  Position range: [{trajectories['positions'][:, 0, 0, 0].min().item():.3f}, "
          f"{trajectories['positions'][:, 0, 0, 0].max().item():.3f}]")
    
    return agent, trajectories


def demo_perturbation_optimization():
    """Test parameter perturbation optimization."""
    print("\n" + "="*60)
    print("Demo 2: Perturbation Optimization (3 iterations)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create agent
    agent = DeepActiveInferenceAgent(device=device)
    
    # Create optimizer with fewer perturbations for demo
    optimizer = PerturbationOptimizer(
        parameters=list(agent.parameters()),
        n_perturbations=100,  # Small for demo
        learning_rate=1e-3
    )
    
    print(f"Optimizer created with {optimizer.n_perturbations} perturbations")
    
    # Run a few optimization steps
    for step in range(3):
        print(f"\nIteration {step + 1}/3")
        
        # Generate perturbations
        r_params, r_epsilons = optimizer.generate_perturbations()
        print(f"  Generated {len(r_params)} perturbed parameter sets")
        
        # Evaluate
        perturbed_model = PerturbedModel(agent, r_params)
        free_energies = perturbed_model.evaluate(n_run_steps=20, n_proc=1)
        
        fe_mean = free_energies.mean().item()
        fe_std = free_energies.std().item()
        print(f"  Free Energy: {fe_mean:.4f} ± {fe_std:.4f}")
        
        # Update
        stats = optimizer.step(free_energies, r_epsilons)
        print(f"  Gradient norms - Params: {stats['param_grad_norm']:.6f}, "
              f"Sigmas: {stats['sigma_grad_norm']:.6f}")
    
    # Evaluate final model
    print("\nFinal evaluation:")
    with torch.no_grad():
        trajectories = agent(n_run_steps=30, n_proc=1)
    
    print(f"  Mean Free Energy: {trajectories['free_energy'].mean().item():.4f}")
    print(f"  Final position: {trajectories['positions'][-1, 0, 0, 0].item():.4f}")
    
    return agent, optimizer


def demo_visualization(agent, trajectories):
    """Visualize agent trajectories."""
    print("\n" + "="*60)
    print("Demo 3: Visualization")
    print("="*60)
    
    # Extract data
    positions = trajectories['positions'][:, 0, 0, 0].cpu().numpy()
    velocities = trajectories['velocities'][:, 0, 0, 0].cpu().numpy()
    actions = trajectories['actions'][:, 0, 0, 0].cpu().numpy()
    obs_nonlinear = trajectories['obs_nonlinear'][:, 0, 0, 0].cpu().numpy()
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Position
    axes[0, 0].plot(positions, linewidth=2)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Goal')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].set_title('Position Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Action
    axes[0, 1].plot(actions, linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Action')
    axes[0, 1].set_title('Action Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Phase plot
    axes[1, 0].plot(positions, velocities, linewidth=2, color='green')
    axes[1, 0].scatter(positions[0], velocities[0], c='blue', s=100, 
                       label='Start', zorder=5)
    axes[1, 0].scatter(positions[-1], velocities[-1], c='red', s=100, 
                       label='End', zorder=5)
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Velocity')
    axes[1, 0].set_title('Phase Space')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Nonlinear observation
    axes[1, 1].plot(obs_nonlinear, linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Nonlinear Obs')
    axes[1, 1].set_title('Nonlinear Observation (Goal Detector)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_deep_active_inference.png', dpi=150, bbox_inches='tight')
    print("Saved plot to demo_deep_active_inference.png")
    plt.show()


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Deep Active Inference - PyTorch Implementation Demo")
    print("="*60)
    
    # Demo 1: Forward pass
    agent, trajectories = demo_forward_pass()
    
    # Demo 2: Optimization
    agent, optimizer = demo_perturbation_optimization()
    
    # Demo 3: Visualization
    with torch.no_grad():
        trajectories = agent(n_run_steps=50, n_proc=1)
    demo_visualization(agent, trajectories)
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train the agent: python train_deep_active_inference.py --mode train")
    print("2. Evaluate trained agent: python train_deep_active_inference.py --mode eval")
    print("3. See deep_active_inference_README.md for more details")


if __name__ == '__main__':
    main()
