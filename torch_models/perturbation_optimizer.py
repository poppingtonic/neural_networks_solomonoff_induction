"""
Parameter Perturbation Optimizer for Deep Active Inference

Implements evolution strategies-style optimization using parameter perturbations
with ADAM updates for both parameters and their perturbation variances.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class PerturbationOptimizer:
    """
    Optimizer that uses parameter perturbations (evolution strategies).
    
    This implements the optimization method from the original paper where:
    1. Parameters are perturbed with Gaussian noise
    2. Free energy is evaluated for each perturbation
    3. Gradients are estimated using the perturbations
    4. Parameters and their std devs are updated with ADAM
    """
    
    def __init__(
        self,
        parameters: List[nn.Parameter],
        n_perturbations: int = 10000,
        learning_rate: float = 1e-3,
        sig_min_perturbations: float = 1e-6,
        init_sig_perturbations: float = -3.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-6
    ):
        """
        Args:
            parameters: List of model parameters to optimize
            n_perturbations: Number of parameter perturbations per iteration
            learning_rate: Learning rate for ADAM
            sig_min_perturbations: Minimum std dev for perturbations
            init_sig_perturbations: Initial log std dev for perturbations
            beta1: ADAM beta1 parameter
            beta2: ADAM beta2 parameter
            epsilon: ADAM epsilon parameter
        """
        self.parameters = list(parameters)
        self.n_perturbations = n_perturbations
        self.learning_rate = learning_rate
        self.sig_min_perturbations = sig_min_perturbations
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize perturbation variances (in log space)
        self.sigmas = []
        for param in self.parameters:
            sigma = torch.ones_like(param) * init_sig_perturbations
            self.sigmas.append(sigma)
        
        # ADAM state for parameters
        self.param_m = [torch.zeros_like(p) for p in self.parameters]
        self.param_v = [torch.zeros_like(p) for p in self.parameters]
        
        # ADAM state for sigmas
        self.sigma_m = [torch.zeros_like(s) for s in self.sigmas]
        self.sigma_v = [torch.zeros_like(s) for s in self.sigmas]
        
        self.t = 0  # Time step
    
    def generate_perturbations(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate perturbed parameters using antithetic sampling.
        
        Returns:
            r_params: List of perturbed parameter tensors
            r_epsilons: List of perturbation noise tensors
        """
        r_params = []
        r_epsilons = []
        
        for i, (param, sigma) in enumerate(zip(self.parameters, self.sigmas)):
            # Generate half the perturbations
            epsilon_half = torch.randn(
                self.n_perturbations // 2, *param.shape,
                device=param.device
            )
            
            # Antithetic sampling: use both +epsilon and -epsilon
            epsilon = torch.cat([epsilon_half, -epsilon_half], dim=0)
            
            # Compute actual std dev from log-space sigma
            std = torch.nn.functional.softplus(sigma) + self.sig_min_perturbations
            
            # Perturb parameters
            r_param = param.unsqueeze(0) + epsilon * std.unsqueeze(0)
            
            r_params.append(r_param)
            r_epsilons.append(epsilon)
        
        return r_params, r_epsilons
    
    def step(self, free_energies: torch.Tensor, r_epsilons: List[torch.Tensor]) -> dict:
        """
        Perform one optimization step.
        
        Args:
            free_energies: Free energy values for each perturbation [n_perturbations]
            r_epsilons: List of perturbation noise tensors from generate_perturbations
            
        Returns:
            Dictionary with gradient statistics
        """
        self.t += 1
        
        # Compute bias correction
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        stats = {
            'param_grad_norm': 0.0,
            'sigma_grad_norm': 0.0,
            'param_updates': [],
            'sigma_updates': []
        }
        
        # Update parameters
        for i, (param, sigma, epsilon) in enumerate(zip(self.parameters, 
                                                         self.sigmas, 
                                                         r_epsilons)):
            # Compute gradient estimate for parameters
            std = torch.nn.functional.softplus(sigma) + self.sig_min_perturbations
            
            # Average gradient: E[FE * epsilon / std]
            grad_param = torch.mean(
                free_energies.view(-1, *([1] * len(param.shape))) * epsilon,
                dim=0
            ) / std
            
            # ADAM update for parameters
            self.param_m[i] = self.beta1 * self.param_m[i] + (1 - self.beta1) * grad_param
            self.param_v[i] = self.beta2 * self.param_v[i] + (1 - self.beta2) * grad_param ** 2
            
            m_hat = self.param_m[i] / bias_correction1
            v_hat = self.param_v[i] / bias_correction2
            
            param_update = -self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
            
            with torch.no_grad():
                param.add_(param_update)
            
            stats['param_grad_norm'] += torch.norm(grad_param).item() ** 2
            stats['param_updates'].append(torch.norm(param_update).item())
            
            # Compute gradient estimate for sigma
            outer_der = (epsilon ** 2 - 1.0) / std
            inner_der = torch.exp(sigma) / (1.0 + torch.exp(sigma))
            
            grad_sigma = torch.mean(
                free_energies.view(-1, *([1] * len(sigma.shape))) * outer_der * inner_der,
                dim=0
            )
            
            # ADAM update for sigma
            self.sigma_m[i] = self.beta1 * self.sigma_m[i] + (1 - self.beta1) * grad_sigma
            self.sigma_v[i] = self.beta2 * self.sigma_v[i] + (1 - self.beta2) * grad_sigma ** 2
            
            m_hat_sigma = self.sigma_m[i] / bias_correction1
            v_hat_sigma = self.sigma_v[i] / bias_correction2
            
            sigma_update = -self.learning_rate * m_hat_sigma / (torch.sqrt(v_hat_sigma) + self.epsilon)
            
            with torch.no_grad():
                sigma.add_(sigma_update)
            
            stats['sigma_grad_norm'] += torch.norm(grad_sigma).item() ** 2
            stats['sigma_updates'].append(torch.norm(sigma_update).item())
        
        stats['param_grad_norm'] = np.sqrt(stats['param_grad_norm'])
        stats['sigma_grad_norm'] = np.sqrt(stats['sigma_grad_norm'])
        
        return stats
    
    def get_sigmas(self) -> List[torch.Tensor]:
        """Get current perturbation std devs."""
        return [torch.nn.functional.softplus(s) + self.sig_min_perturbations 
                for s in self.sigmas]
    
    def state_dict(self) -> dict:
        """Get optimizer state for checkpointing."""
        return {
            'sigmas': self.sigmas,
            'param_m': self.param_m,
            'param_v': self.param_v,
            'sigma_m': self.sigma_m,
            'sigma_v': self.sigma_v,
            't': self.t
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load optimizer state from checkpoint."""
        self.sigmas = state_dict['sigmas']
        self.param_m = state_dict['param_m']
        self.param_v = state_dict['param_v']
        self.sigma_m = state_dict['sigma_m']
        self.sigma_v = state_dict['sigma_v']
        self.t = state_dict['t']


class PerturbedModel:
    """
    Wrapper that evaluates a model with perturbed parameters.
    """
    
    def __init__(self, model: nn.Module, r_params: List[torch.Tensor]):
        """
        Args:
            model: Base model
            r_params: List of perturbed parameters [n_perturbations, *param_shape]
        """
        self.model = model
        self.r_params = r_params
        self.n_perturbations = r_params[0].shape[0]
    
    def evaluate(self, n_run_steps: int = 30, n_proc: int = 1) -> torch.Tensor:
        """
        Evaluate model with all perturbations in parallel.
        
        Args:
            n_run_steps: Number of timesteps to simulate
            n_proc: Number of parallel processes per perturbation
            
        Returns:
            free_energies: Mean free energy for each perturbation [n_perturbations]
        """
        device = self.r_params[0].device
        
        # Initialize states for all perturbations
        st = torch.zeros(self.n_perturbations, self.model.n_s, n_proc, device=device)
        post = torch.ones(self.n_perturbations, self.model.n_o, n_proc, device=device) * -0.5
        vt = torch.zeros(self.n_perturbations, self.model.n_o, n_proc, device=device)
        
        total_free_energy = torch.zeros(self.n_perturbations, device=device)
        
        # Temporarily replace model parameters with perturbed versions
        original_params = []
        param_list = list(self.model.parameters())
        
        for param, r_param in zip(param_list, self.r_params):
            original_params.append(param.data.clone())
        
        # Run simulation for all perturbations
        for pert_idx in range(self.n_perturbations):
            # Set perturbed parameters
            for param_idx, param in enumerate(param_list):
                param.data = self.r_params[param_idx][pert_idx]
            
            # Run one trajectory
            st_i = st[pert_idx:pert_idx+1]
            post_i = post[pert_idx:pert_idx+1]
            vt_i = vt[pert_idx:pert_idx+1]
            
            for t in range(n_run_steps):
                (st_i, post_i, vt_i, oat, ot, oht, FEt, KL_st, hst, hst2, 
                 stmu, stsig, force, p_ot, p_oht, p_oat) = self.model.step(t, st_i, post_i, vt_i)
                
                total_free_energy[pert_idx] += FEt.mean()
            
            st[pert_idx:pert_idx+1] = st_i
            post[pert_idx:pert_idx+1] = post_i
            vt[pert_idx:pert_idx+1] = vt_i
        
        # Restore original parameters
        for param, original in zip(param_list, original_params):
            param.data = original
        
        # Return mean free energy per timestep
        return total_free_energy / n_run_steps
