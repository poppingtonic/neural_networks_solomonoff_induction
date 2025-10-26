"""
Implementation of Deep Active Inference for General Artificial Intelligence
PyTorch Implementation

Based on: Kai Ueltzhoeffer, 2017

This module implements a deep active inference agent that learns through
variational free energy minimization with parameter perturbation optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class DeepActiveInferenceAgent(nn.Module):
    """
    Deep Active Inference Agent with variational free energy minimization.
    
    The agent learns to:
    1. Infer hidden states from observations using a variational posterior
    2. Predict observations from hidden states using a generative model
    3. Select actions to minimize expected free energy
    """
    
    def __init__(
        self,
        n_s: int = 10,  # Number of hidden states
        n_o: int = 1,   # Sensory input encoding position
        n_oh: int = 1,  # Nonlinearly transformed channel
        n_oa: int = 1,  # Proprioception/action
        sig_min_obs: float = 1e-6,
        sig_min_states: float = 1e-6,
        sig_min_action: float = 1e-6,
        init_sig_obs: float = 0.0,
        init_sig_states_likelihood: float = 0.0,
        init_sig_states: float = -3.0,
        init_sig_action: float = -3.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.n_s = n_s
        self.n_o = n_o
        self.n_oh = n_oh
        self.n_oa = n_oa
        self.sig_min_obs = sig_min_obs
        self.sig_min_states = sig_min_states
        self.sig_min_action = sig_min_action
        self.device = device
        
        # ========================================
        # Approximate Posterior Q(s_t | s_t-1, o_t, oh_t, oa_t)
        # ========================================
        
        self.Wq_hst_ot = nn.Parameter(self._init_weight(n_s, n_o, -0.5, 0.5))
        self.Wq_hst_oht = nn.Parameter(self._init_weight(n_s, n_oh, -0.5, 0.5))
        self.Wq_hst_oat = nn.Parameter(self._init_weight(n_s, n_oa, -0.5, 0.5))
        self.Wq_hst_stm1 = nn.Parameter(self._init_ortho(n_s, n_s))
        self.bq_hst = nn.Parameter(self._init_const(n_s, 1))
        
        self.Wq_hst2_hst = nn.Parameter(self._init_ortho(n_s, n_s))
        self.bq_hst2 = nn.Parameter(self._init_const(n_s, 1))
        
        self.Wq_stmu_hst2 = nn.Parameter(self._init_ortho(n_s, n_s))
        self.bq_stmu = nn.Parameter(self._init_const(n_s, 1))
        
        self.Wq_stsig_hst2 = nn.Parameter(self._init_weight(n_s, n_s))
        self.bq_stsig = nn.Parameter(self._init_const(n_s, 1, init_sig_states))
        
        # ========================================
        # Prior/Likelihood p(s_t | s_t-1)
        # ========================================
        
        self.Wl_stmu_stm1 = nn.Parameter(self._init_ortho(n_s, n_s))
        self.bl_stmu = nn.Parameter(self._init_const(n_s, 1))
        
        self.Wl_stsig_stm1 = nn.Parameter(self._init_weight(n_s, n_s))
        self.bl_stsig = nn.Parameter(self._init_const(n_s, 1, init_sig_states_likelihood))
        
        # Hidden state to observation features
        self.Wl_ost_st = nn.Parameter(self._init_ortho(n_s, n_s))
        self.bl_ost = nn.Parameter(self._init_const(n_s, 1))
        
        self.Wl_ost2_ost = nn.Parameter(self._init_ortho(n_s, n_s))
        self.bl_ost2 = nn.Parameter(self._init_const(n_s, 1))
        
        self.Wl_ost3_ost2 = nn.Parameter(self._init_ortho(n_s, n_s))
        self.bl_ost3 = nn.Parameter(self._init_const(n_s, 1))
        
        # ========================================
        # Observation Likelihoods p(o_t | s_t)
        # ========================================
        
        self.Wl_otmu_st = nn.Parameter(self._init_weight(n_o, n_s))
        self.bl_otmu = nn.Parameter(self._init_const(n_o, 1))
        self.Wl_otsig_st = nn.Parameter(self._init_weight(n_o, n_s))
        self.bl_otsig = nn.Parameter(self._init_const(n_o, 1, init_sig_obs))
        
        self.Wl_ohtmu_st = nn.Parameter(self._init_weight(n_oh, n_s))
        self.bl_ohtmu = nn.Parameter(self._init_const(n_oh, 1))
        self.Wl_ohtsig_st = nn.Parameter(self._init_weight(n_oh, n_s))
        self.bl_ohtsig = nn.Parameter(self._init_const(n_oh, 1, init_sig_obs))
        
        self.Wl_oatmu_st = nn.Parameter(self._init_weight(n_oa, n_s))
        self.bl_oatmu = nn.Parameter(self._init_const(n_oa, 1))
        self.Wl_oatsig_st = nn.Parameter(self._init_weight(n_oa, n_s))
        self.bl_oatsig = nn.Parameter(self._init_const(n_oa, 1, init_sig_obs))
        
        # ========================================
        # Action Function
        # ========================================
        
        self.Wa_aht_st = nn.Parameter(self._init_ortho(n_s, n_s))
        self.ba_aht = nn.Parameter(self._init_const(n_s, 1))
        
        self.Wa_atmu_aht = nn.Parameter(self._init_weight(n_oa, n_s, -1.0, 1.0))
        self.ba_atmu = nn.Parameter(self._init_const(n_oa, 1))
        
        self.Wa_atsig_aht = nn.Parameter(self._init_weight(n_oa, n_s))
        self.ba_atsig = nn.Parameter(self._init_const(n_oa, 1, init_sig_action))
        
        self.to(device)
    
    def _init_weight(self, shape1: int, shape2: int, minval: float = -0.05, 
                     maxval: float = 0.05) -> torch.Tensor:
        """Uniform weight initialization."""
        val = torch.rand(shape1, shape2)
        val = minval + (maxval - minval) * val
        return val
    
    def _init_xavier(self, shape1: int, shape2: int) -> torch.Tensor:
        """Xavier initialization."""
        val = torch.randn(shape1, shape2) / np.sqrt(shape2)
        return val
    
    def _init_ortho(self, shape1: int, shape2: int) -> torch.Tensor:
        """Orthogonal initialization."""
        x = torch.randn(shape1, shape2) * 0.1
        if shape1 >= shape2:
            xo, _ = torch.linalg.qr(x)
        else:
            xo, _ = torch.linalg.qr(x.T)
            xo = xo.T
        return xo
    
    def _init_const(self, shape1: int, shape2: int, val: float = 0.0) -> torch.Tensor:
        """Constant initialization."""
        return torch.ones(shape1, shape2) * val
    
    def _gaussian_nll(self, y: torch.Tensor, mu: torch.Tensor, 
                      sig: torch.Tensor) -> torch.Tensor:
        """Gaussian negative log-likelihood."""
        nll = 0.5 * torch.sum(
            torch.square(y - mu) / (sig ** 2) + 2 * torch.log(sig) + 
            np.log(2 * np.pi), 
            dim=1
        )
        return nll
    
    def _kl_gaussian_gaussian(self, mu1: torch.Tensor, sig1: torch.Tensor,
                              mu2: torch.Tensor, sig2: torch.Tensor) -> torch.Tensor:
        """KL divergence between two diagonal Gaussians."""
        kl = torch.sum(
            0.5 * (2 * torch.log(sig2) - 2 * torch.log(sig1) +
                   (sig1 ** 2 + (mu1 - mu2) ** 2) / (sig2 ** 2) - 1),
            dim=1
        )
        return kl
    
    def step(self, t: int, stm1: torch.Tensor, postm1: torch.Tensor, 
             vtm1: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Single timestep of the agent.
        
        Args:
            t: Current timestep
            stm1: Previous hidden state [batch, n_s, n_proc]
            postm1: Previous position [batch, n_o, n_proc]
            vtm1: Previous velocity [batch, n_o, n_proc]
            
        Returns:
            Tuple of (st, post, vt, oat, ot, oht, FEt, KL_st, hst, hst2, 
                     stmu, stsig, force, p_ot, p_oht, p_oat)
        """
        batch_size, _, n_proc = stm1.shape
        
        # ========================================
        # Generate Action
        # ========================================
        
        # Action hidden layer
        aht = torch.bmm(self.Wa_aht_st.expand(batch_size, -1, -1), 
                        stm1.reshape(batch_size, self.n_s, n_proc)) + self.ba_aht
        
        # Action mean and std
        at_mu = torch.bmm(self.Wa_atmu_aht.expand(batch_size, -1, -1), aht) + self.ba_atmu
        at_sig = F.softplus(torch.bmm(self.Wa_atsig_aht.expand(batch_size, -1, -1), 
                                      aht) + self.ba_atsig) + self.sig_min_action
        
        # Sample action
        at = at_mu + torch.randn_like(at_mu) * at_sig
        
        # ========================================
        # Update Environment (Mountain Car-like)
        # ========================================
        
        action_force = torch.tanh(at)
        
        # Complex force function
        force = torch.where(
            postm1 < 0.0,
            -2 * postm1 - 1,
            -(1 + 5 * postm1**2) ** (-0.5) - 
            postm1**2 * (1 + 5 * postm1**2) ** (-1.5) - 
            postm1**4 / 16.0
        ) - 0.25 * vtm1
        
        vt = vtm1 + 0.05 * force + 0.03 * action_force
        post = postm1 + vt
        
        # ========================================
        # Generate Observations
        # ========================================
        
        # 1. Action observation (proprioception)
        oat = at
        
        # 2. Noisy position observation
        ot = post + torch.randn_like(post) * 0.01
        
        # 3. Nonlinear transformed sensory channel
        oht = torch.exp(-torch.square(post - 1.0) / 2.0 / 0.3 / 0.3)
        
        # ========================================
        # Infer Hidden State (Variational Posterior)
        # ========================================
        
        # First hidden layer
        hst = F.relu(
            torch.bmm(self.Wq_hst_stm1.expand(batch_size, -1, -1), stm1) +
            torch.bmm(self.Wq_hst_ot.expand(batch_size, -1, -1), ot) +
            torch.bmm(self.Wq_hst_oht.expand(batch_size, -1, -1), oht) +
            torch.bmm(self.Wq_hst_oat.expand(batch_size, -1, -1), oat) +
            self.bq_hst
        )
        
        # Second hidden layer
        hst2 = F.relu(
            torch.bmm(self.Wq_hst2_hst.expand(batch_size, -1, -1), hst) + 
            self.bq_hst2
        )
        
        # State mean and std
        stmu = torch.tanh(
            torch.bmm(self.Wq_stmu_hst2.expand(batch_size, -1, -1), hst2) + 
            self.bq_stmu
        )
        stsig = F.softplus(
            torch.bmm(self.Wq_stsig_hst2.expand(batch_size, -1, -1), hst2) + 
            self.bq_stsig
        ) + self.sig_min_states
        
        # Explicitly encode position as homeostatic state variable
        stmu[:, 0:1, :] = 0.1 * ot[:, 0:1, :]
        stsig[:, 0:1, :] = 0.005
        
        # Sample from variational density
        st = stmu + torch.randn_like(stmu) * stsig
        
        # ========================================
        # Compute Likelihood (Generative Model)
        # ========================================
        
        # Hidden layers for observation generation
        ost = F.relu(torch.bmm(self.Wl_ost_st.expand(batch_size, -1, -1), st) + self.bl_ost)
        ost2 = F.relu(torch.bmm(self.Wl_ost2_ost.expand(batch_size, -1, -1), ost) + self.bl_ost2)
        ost3 = F.relu(torch.bmm(self.Wl_ost3_ost2.expand(batch_size, -1, -1), ost2) + self.bl_ost3)
        
        # Position observation likelihood
        otmu = torch.bmm(self.Wl_otmu_st.expand(batch_size, -1, -1), ost3) + self.bl_otmu
        otsig = F.softplus(torch.bmm(self.Wl_otsig_st.expand(batch_size, -1, -1), 
                                     ost3) + self.bl_otsig) + self.sig_min_obs
        
        # Nonlinear channel likelihood
        ohtmu = torch.bmm(self.Wl_ohtmu_st.expand(batch_size, -1, -1), ost3) + self.bl_ohtmu
        ohtsig = F.softplus(torch.bmm(self.Wl_ohtsig_st.expand(batch_size, -1, -1), 
                                      ost3) + self.bl_ohtsig) + self.sig_min_obs
        
        # Action observation likelihood
        oatmu = torch.bmm(self.Wl_oatmu_st.expand(batch_size, -1, -1), ost3) + self.bl_oatmu
        oatsig = F.softplus(torch.bmm(self.Wl_oatsig_st.expand(batch_size, -1, -1), 
                                      ost3) + self.bl_oatsig) + self.sig_min_obs
        
        # Compute negative log-likelihoods
        p_ot = self._gaussian_nll(ot.permute(0, 2, 1).reshape(-1, self.n_o),
                                   otmu.permute(0, 2, 1).reshape(-1, self.n_o),
                                   otsig.permute(0, 2, 1).reshape(-1, self.n_o))
        p_oht = self._gaussian_nll(oht.permute(0, 2, 1).reshape(-1, self.n_oh),
                                    ohtmu.permute(0, 2, 1).reshape(-1, self.n_oh),
                                    ohtsig.permute(0, 2, 1).reshape(-1, self.n_oh))
        p_oat = self._gaussian_nll(oat.permute(0, 2, 1).reshape(-1, self.n_oa),
                                    oatmu.permute(0, 2, 1).reshape(-1, self.n_oa),
                                    oatsig.permute(0, 2, 1).reshape(-1, self.n_oa))
        
        # ========================================
        # Compute Prior
        # ========================================
        
        prior_stmu = torch.tanh(
            torch.bmm(self.Wl_stmu_stm1.expand(batch_size, -1, -1), stm1) + 
            self.bl_stmu
        )
        prior_stsig = F.softplus(
            torch.bmm(self.Wl_stsig_stm1.expand(batch_size, -1, -1), stm1) + 
            self.bl_stsig
        ) + self.sig_min_states
        
        # Explicitly encode expectations on homeostatic state variable
        if t >= 20:
            prior_stmu[:, 0:1, :] = 0.1
            prior_stsig[:, 0:1, :] = 0.005
        
        # ========================================
        # Compute Free Energy
        # ========================================
        
        # KL divergence between posterior and prior
        KL_st = self._kl_gaussian_gaussian(
            stmu.permute(0, 2, 1).reshape(-1, self.n_s),
            stsig.permute(0, 2, 1).reshape(-1, self.n_s),
            prior_stmu.permute(0, 2, 1).reshape(-1, self.n_s),
            prior_stsig.permute(0, 2, 1).reshape(-1, self.n_s)
        )
        
        # Total free energy
        FEt = KL_st + p_ot + p_oht + p_oat
        
        return (st, post, vt, oat, ot, oht, FEt, KL_st, hst, hst2, 
                stmu, stsig, force, p_ot, p_oht, p_oat)
    
    def forward(self, n_run_steps: int = 30, n_proc: int = 1) -> dict:
        """
        Run the agent for multiple timesteps.
        
        Args:
            n_run_steps: Number of timesteps to simulate
            n_proc: Number of parallel processes
            
        Returns:
            Dictionary containing trajectories and free energy components
        """
        batch_size = 1  # For forward pass, batch_size is 1
        
        # Initialize
        st = torch.zeros(batch_size, self.n_s, n_proc, device=self.device)
        post = torch.ones(batch_size, self.n_o, n_proc, device=self.device) * -0.5
        vt = torch.zeros(batch_size, self.n_o, n_proc, device=self.device)
        
        # Storage for trajectories
        trajectories = {
            'states': [],
            'positions': [],
            'velocities': [],
            'actions': [],
            'observations': [],
            'obs_nonlinear': [],
            'free_energy': [],
            'kl_divergence': [],
            'nll_ot': [],
            'nll_oht': [],
            'nll_oat': []
        }
        
        # Run simulation
        for t in range(n_run_steps):
            (st, post, vt, oat, ot, oht, FEt, KL_st, hst, hst2, 
             stmu, stsig, force, p_ot, p_oht, p_oat) = self.step(t, st, post, vt)
            
            trajectories['states'].append(st.detach())
            trajectories['positions'].append(post.detach())
            trajectories['velocities'].append(vt.detach())
            trajectories['actions'].append(oat.detach())
            trajectories['observations'].append(ot.detach())
            trajectories['obs_nonlinear'].append(oht.detach())
            trajectories['free_energy'].append(FEt.detach())
            trajectories['kl_divergence'].append(KL_st.detach())
            trajectories['nll_ot'].append(p_ot.detach())
            trajectories['nll_oht'].append(p_oht.detach())
            trajectories['nll_oat'].append(p_oat.detach())
        
        # Stack trajectories
        for key in trajectories:
            if len(trajectories[key]) > 0 and trajectories[key][0] is not None:
                trajectories[key] = torch.stack(trajectories[key])
        
        return trajectories
