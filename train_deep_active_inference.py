"""
Training script for Deep Active Inference Agent

This script trains the agent using parameter perturbation optimization
to minimize variational free energy on a mountain car-like task.
"""

import os
import argparse
import time
import pickle
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_models.deep_active_inference import DeepActiveInferenceAgent
from torch_models.perturbation_optimizer import PerturbationOptimizer, PerturbedModel


def train_agent(
    n_steps: int = 1000000,
    n_perturbations: int = 10000,
    n_run_steps: int = 30,
    n_proc: int = 1,
    learning_rate: float = 1e-3,
    saving_steps: int = 10,
    save_best_trajectory: bool = True,
    base_name: str = 'deepAI_pytorch',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    resume_from: Optional[str] = None
):
    """
    Train the Deep Active Inference agent.
    
    Args:
        n_steps: Maximum number of optimization steps
        n_perturbations: Number of parameter perturbations per iteration
        n_run_steps: Number of timesteps to simulate
        n_proc: Number of parallel processes
        learning_rate: Learning rate for optimization
        saving_steps: Save progress every nth step
        save_best_trajectory: Save best parameters at each improvement
        base_name: Name for saves and logfile
        device: Device to run on ('cuda' or 'cpu')
        resume_from: Path to checkpoint to resume from
    """
    print(f"Training on device: {device}")
    print(f"Parameters: n_perturbations={n_perturbations}, n_run_steps={n_run_steps}")
    
    # Create model
    agent = DeepActiveInferenceAgent(device=device)
    print(f"Model has {sum(p.numel() for p in agent.parameters())} parameters")
    
    # Create optimizer
    optimizer = PerturbationOptimizer(
        parameters=list(agent.parameters()),
        n_perturbations=n_perturbations,
        learning_rate=learning_rate
    )
    
    # Initialize tracking
    start_step = 0
    best_fe = float('inf')
    log_file = f'log_{base_name}.txt'
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        best_fe = checkpoint['best_fe']
        print(f"Resumed from step {start_step}, best FE: {best_fe:.4f}")
    
    # Initial evaluation
    print("\nEvaluating initial model...")
    with torch.no_grad():
        trajectories = agent(n_run_steps=n_run_steps, n_proc=n_proc)
        initial_fe = trajectories['free_energy'].mean().item()
        print(f"Initial Free Energy: {initial_fe:.4f}")
        print(f"  KL divergence: {trajectories['kl_divergence'].mean().item():.4f}")
        print(f"  NLL position: {trajectories['nll_ot'].mean().item():.4f}")
        print(f"  NLL nonlinear: {trajectories['nll_oht'].mean().item():.4f}")
        print(f"  NLL action: {trajectories['nll_oat'].mean().item():.4f}")
    
    best_fe = min(best_fe, initial_fe)
    
    # Training loop
    print(f"\nStarting training for {n_steps} steps...")
    for step in range(start_step, n_steps):
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Iteration: {step}")
        
        # Generate perturbed parameters
        r_params, r_epsilons = optimizer.generate_perturbations()
        
        # Evaluate all perturbations
        print("Evaluating perturbations...")
        perturbed_model = PerturbedModel(agent, r_params)
        free_energies = perturbed_model.evaluate(n_run_steps=n_run_steps, n_proc=n_proc)
        
        # Compute statistics
        fe_mean = free_energies.mean().item()
        fe_std = free_energies.std().item()
        fe_min = free_energies.min().item()
        fe_max = free_energies.max().item()
        
        print(f"Free Energy - Mean: {fe_mean:.4f}, Std: {fe_std:.4f}, "
              f"Min: {fe_min:.4f}, Max: {fe_max:.4f}")
        
        # Update parameters
        print("Updating parameters...")
        stats = optimizer.step(free_energies, r_epsilons)
        
        print(f"Gradient norms - Params: {stats['param_grad_norm']:.6f}, "
              f"Sigmas: {stats['sigma_grad_norm']:.6f}")
        
        # Log results
        with open(log_file, 'a' if step > 0 else 'w') as f:
            f.write(f"{step} {fe_mean:.6f} {fe_std:.6f} {fe_min:.6f} "
                   f"{stats['param_grad_norm']:.6f} {stats['sigma_grad_norm']:.6f}\n")
        
        # Save checkpoints
        if step % saving_steps == 0:
            checkpoint_path = f"{base_name}_{step}.pt"
            save_checkpoint(agent, optimizer, step, best_fe, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Also save as current
            save_checkpoint(agent, optimizer, step, best_fe, f"{base_name}_current.pt")
        
        # Save best model
        if fe_mean < best_fe:
            print(f"New best FE: {fe_mean:.4f} (previous: {best_fe:.4f})")
            best_fe = fe_mean
            save_checkpoint(agent, optimizer, step, best_fe, f"{base_name}_best.pt")
            
            if save_best_trajectory:
                save_checkpoint(agent, optimizer, step, best_fe, 
                              f"{base_name}_best_{step}.pt")
        
        # Time tracking
        elapsed = time.time() - start_time
        print(f"Time for iteration: {elapsed:.2f}s")
        
        # Evaluate current best parameters periodically
        if step % (saving_steps * 5) == 0:
            print("\nEvaluating current model...")
            with torch.no_grad():
                trajectories = agent(n_run_steps=n_run_steps, n_proc=n_proc)
                current_fe = trajectories['free_energy'].mean().item()
                print(f"Current model FE: {current_fe:.4f}")
    
    # Save final model
    save_checkpoint(agent, optimizer, n_steps - 1, best_fe, f"{base_name}_final.pt")
    print(f"\nTraining complete! Best FE: {best_fe:.4f}")
    
    return agent, optimizer


def save_checkpoint(agent, optimizer, step, best_fe, path):
    """Save model and optimizer state."""
    torch.save({
        'step': step,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_fe': best_fe,
    }, path)


def evaluate_agent(
    checkpoint_path: str,
    n_run_steps: int = 100,
    n_proc: int = 1,
    plot: bool = True,
    save_plots: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Evaluate a trained agent and optionally visualize results.
    
    Args:
        checkpoint_path: Path to model checkpoint
        n_run_steps: Number of timesteps to simulate
        n_proc: Number of parallel processes
        plot: Whether to create plots
        save_plots: Whether to save plots to files
        device: Device to run on
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load model
    agent = DeepActiveInferenceAgent(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    print(f"Loaded model from step {checkpoint['step']}, best FE: {checkpoint['best_fe']:.4f}")
    
    # Run simulation
    print(f"Running simulation for {n_run_steps} steps...")
    with torch.no_grad():
        trajectories = agent(n_run_steps=n_run_steps, n_proc=n_proc)
    
    # Print statistics
    print("\nResults:")
    print(f"  Mean Free Energy: {trajectories['free_energy'].mean().item():.4f}")
    print(f"  Mean KL divergence: {trajectories['kl_divergence'].mean().item():.4f}")
    print(f"  Mean NLL position: {trajectories['nll_ot'].mean().item():.4f}")
    print(f"  Mean NLL nonlinear: {trajectories['nll_oht'].mean().item():.4f}")
    print(f"  Mean NLL action: {trajectories['nll_oat'].mean().item():.4f}")
    
    # Convert to numpy for plotting
    positions = trajectories['positions'][:, 0, 0, 0].cpu().numpy()
    velocities = trajectories['velocities'][:, 0, 0, 0].cpu().numpy()
    actions = trajectories['actions'][:, 0, 0, 0].cpu().numpy()
    obs_nonlinear = trajectories['obs_nonlinear'][:, 0, 0, 0].cpu().numpy()
    free_energies = trajectories['free_energy'].cpu().numpy()
    
    print(f"\nPosition range: [{positions.min():.3f}, {positions.max():.3f}]")
    print(f"Final position: {positions[-1]:.3f}")
    
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Position
        axes[0, 0].plot(positions)
        axes[0, 0].set_ylabel('Position')
        axes[0, 0].set_title('Position over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Goal')
        axes[0, 0].legend()
        
        # Velocity
        axes[0, 1].plot(velocities)
        axes[0, 1].set_ylabel('Velocity')
        axes[0, 1].set_title('Velocity over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Action
        axes[1, 0].plot(actions)
        axes[1, 0].set_ylabel('Action')
        axes[1, 0].set_title('Action over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Nonlinear observation
        axes[1, 1].plot(obs_nonlinear)
        axes[1, 1].set_ylabel('Nonlinear Obs')
        axes[1, 1].set_title('Nonlinear Observation over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Free energy components
        axes[2, 0].plot(free_energies.mean(axis=1), label='Total FE')
        axes[2, 0].plot(trajectories['kl_divergence'].cpu().numpy().mean(axis=1), 
                       label='KL', alpha=0.7)
        axes[2, 0].set_ylabel('Free Energy')
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_title('Free Energy Components')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Phase plot (position vs velocity)
        axes[2, 1].plot(positions, velocities)
        axes[2, 1].scatter(positions[0], velocities[0], c='green', s=100, 
                          label='Start', zorder=5)
        axes[2, 1].scatter(positions[-1], velocities[-1], c='red', s=100, 
                          label='End', zorder=5)
        axes[2, 1].set_xlabel('Position')
        axes[2, 1].set_ylabel('Velocity')
        axes[2, 1].set_title('Phase Plot')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = checkpoint_path.replace('.pt', '_evaluation.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {plot_path}")
        
        plt.show()
    
    return trajectories


def main():
    parser = argparse.ArgumentParser(description='Train or evaluate Deep Active Inference agent')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                       help='Mode: train or eval')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path (for eval or resume training)')
    parser.add_argument('--n_steps', type=int, default=1000,
                       help='Number of training steps')
    parser.add_argument('--n_perturbations', type=int, default=1000,
                       help='Number of parameter perturbations')
    parser.add_argument('--n_run_steps', type=int, default=30,
                       help='Number of simulation timesteps')
    parser.add_argument('--n_proc', type=int, default=1,
                       help='Number of parallel processes')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--saving_steps', type=int, default=10,
                       help='Save every n steps')
    parser.add_argument('--base_name', type=str, default='deepAI_pytorch',
                       help='Base name for saves')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Disable plotting in eval mode')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if args.mode == 'train':
        train_agent(
            n_steps=args.n_steps,
            n_perturbations=args.n_perturbations,
            n_run_steps=args.n_run_steps,
            n_proc=args.n_proc,
            learning_rate=args.learning_rate,
            saving_steps=args.saving_steps,
            base_name=args.base_name,
            device=device,
            resume_from=args.checkpoint
        )
    else:  # eval
        if args.checkpoint is None:
            args.checkpoint = f"{args.base_name}_best.pt"
        
        evaluate_agent(
            checkpoint_path=args.checkpoint,
            n_run_steps=args.n_run_steps,
            n_proc=args.n_proc,
            plot=not args.no_plot,
            save_plots=True,
            device=device
        )


if __name__ == '__main__':
    main()
