"""
Active Inference for Universal Prediction

Adapts the Deep Active Inference framework for sequence prediction tasks,
compatible with the UTM data generators in this repository.

The key insight: Prediction can be viewed as "epistemic" action - where
the agent acts to reduce uncertainty about the environment (future tokens).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict


class ActiveInferencePredictor(nn.Module):
    """
    Active Inference agent adapted for sequence prediction.
    
    Instead of controlling an environment, this agent:
    1. Observes token sequences
    2. Infers latent structure (hidden states)
    3. Predicts next tokens
    4. Minimizes free energy = prediction error + KL divergence
    
    This is compatible with UTM data generators and can be used for
    meta-learning universal prediction strategies.
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_s: int = 64,  # Hidden state dimension (larger for sequences)
        embedding_dim: int = 32,
        n_layers: int = 2,
        sig_min_states: float = 1e-6,
        sig_min_obs: float = 1e-6,
        init_sig_states: float = -3.0,
        init_sig_obs: float = -2.0,
        use_actions: bool = False,  # Optional: epistemic actions
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            vocab_size: Size of token vocabulary
            n_s: Hidden state dimension
            embedding_dim: Token embedding dimension
            n_layers: Number of layers in networks
            sig_min_states: Minimum std for state distribution
            sig_min_obs: Minimum std for observation distribution
            init_sig_states: Initial log-std for states
            init_sig_obs: Initial log-std for observations
            use_actions: If True, agent can take "epistemic actions" (queries)
            device: Computation device
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_s = n_s
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.sig_min_states = sig_min_states
        self.sig_min_obs = sig_min_obs
        self.use_actions = use_actions
        self.device = device
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # ========================================
        # Recognition Network: Q(s_t | s_t-1, o_t)
        # ========================================
        
        # Encodes observation and previous state
        recognition_layers = []
        input_dim = embedding_dim + n_s
        
        for i in range(n_layers):
            output_dim = n_s if i < n_layers - 1 else n_s
            recognition_layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.ReLU() if i < n_layers - 1 else nn.Identity()
            ])
            input_dim = output_dim
        
        self.recognition_net = nn.Sequential(*recognition_layers)
        
        # Output: mean and log-std of state posterior
        self.q_mean = nn.Linear(n_s, n_s)
        self.q_logstd = nn.Linear(n_s, n_s)
        nn.init.constant_(self.q_logstd.bias, init_sig_states)
        
        # ========================================
        # Generative Model: P(o_t | s_t)
        # ========================================
        
        # Predicts next token from state
        generative_layers = []
        input_dim = n_s
        
        for i in range(n_layers):
            output_dim = n_s if i < n_layers - 1 else n_s
            generative_layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.ReLU()
            ])
            input_dim = output_dim
        
        self.generative_net = nn.Sequential(*generative_layers)
        
        # Output: logits over vocabulary (categorical distribution)
        self.pred_logits = nn.Linear(n_s, vocab_size)
        
        # ========================================
        # Transition Model: P(s_t | s_t-1)
        # ========================================
        
        transition_layers = []
        input_dim = n_s
        
        for i in range(max(1, n_layers - 1)):
            output_dim = n_s
            transition_layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.Tanh() if i == max(1, n_layers - 1) - 1 else nn.ReLU()
            ])
            input_dim = output_dim
        
        self.transition_net = nn.Sequential(*transition_layers)
        
        self.p_mean = nn.Linear(n_s, n_s)
        self.p_logstd = nn.Linear(n_s, n_s)
        nn.init.constant_(self.p_logstd.bias, init_sig_states)
        
        # ========================================
        # Optional: Epistemic Action Network
        # ========================================
        
        if use_actions:
            # Agent can "query" the environment (e.g., which position to look at)
            self.action_net = nn.Sequential(
                nn.Linear(n_s, n_s),
                nn.ReLU(),
                nn.Linear(n_s, embedding_dim)  # Action embedding
            )
        
        self.to(device)
    
    def _kl_gaussian_gaussian(self, mu1: torch.Tensor, logstd1: torch.Tensor,
                              mu2: torch.Tensor, logstd2: torch.Tensor) -> torch.Tensor:
        """KL divergence between two diagonal Gaussians."""
        var1 = torch.exp(2 * logstd1)
        var2 = torch.exp(2 * logstd2)
        
        kl = 0.5 * torch.sum(
            2 * (logstd2 - logstd1) + 
            (var1 + (mu1 - mu2) ** 2) / var2 - 1,
            dim=-1
        )
        return kl
    
    def step(self, token: torch.Tensor, state_prev: torch.Tensor,
             return_components: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Process one token and predict the next.
        
        Args:
            token: Current token [batch_size]
            state_prev: Previous hidden state [batch_size, n_s]
            return_components: If True, return detailed components
            
        Returns:
            next_token_logits: Logits for next token [batch_size, vocab_size]
            state_current: Current state [batch_size, n_s]
            free_energy: Variational free energy [batch_size]
            (optional) components: Dict with KL, NLL, etc.
        """
        batch_size = token.shape[0]
        
        # Embed current token
        token_emb = self.token_embedding(token)  # [batch, embedding_dim]
        
        # ========================================
        # Recognition: Infer current state Q(s_t | s_t-1, o_t)
        # ========================================
        
        # Combine previous state and current observation
        recognition_input = torch.cat([state_prev, token_emb], dim=-1)
        recognition_hidden = self.recognition_net(recognition_input)
        
        # Posterior parameters
        q_mean = self.q_mean(recognition_hidden)
        q_logstd = self.q_logstd(recognition_hidden)
        q_logstd = torch.clamp(q_logstd, -10, 2)  # Numerical stability
        
        # Sample state from posterior
        q_std = F.softplus(q_logstd) + self.sig_min_states
        epsilon = torch.randn_like(q_mean)
        state_current = q_mean + epsilon * q_std
        
        # ========================================
        # Prior: P(s_t | s_t-1)
        # ========================================
        
        transition_hidden = self.transition_net(state_prev)
        p_mean = self.p_mean(transition_hidden)
        p_logstd = self.p_logstd(transition_hidden)
        p_logstd = torch.clamp(p_logstd, -10, 2)
        
        # ========================================
        # Generative Model: P(o_{t+1} | s_t)
        # ========================================
        
        generative_hidden = self.generative_net(state_current)
        next_token_logits = self.pred_logits(generative_hidden)
        
        # ========================================
        # Free Energy Computation
        # ========================================
        
        # KL divergence: KL[Q(s_t | s_t-1, o_t) || P(s_t | s_t-1)]
        kl_div = self._kl_gaussian_gaussian(q_mean, q_logstd, p_mean, p_logstd)
        
        # Note: We don't compute NLL here because we don't have the next token yet
        # This will be computed externally when we have the target
        # Free energy = KL + NLL (to be added by caller)
        
        if return_components:
            components = {
                'kl_divergence': kl_div,
                'q_mean': q_mean,
                'q_std': q_std,
                'p_mean': p_mean,
                'p_std': F.softplus(p_logstd) + self.sig_min_states,
                'state': state_current
            }
            return next_token_logits, state_current, kl_div, components
        
        return next_token_logits, state_current, kl_div
    
    def forward(self, sequence: torch.Tensor, 
                return_all_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process a full sequence.
        
        Args:
            sequence: Token sequence [batch_size, seq_len]
            return_all_states: If True, return states at each timestep
            
        Returns:
            Dictionary containing:
                - logits: Predictions for next tokens [batch, seq_len-1, vocab_size]
                - free_energy: Free energy at each step [batch, seq_len-1]
                - kl_divergence: KL divergence at each step [batch, seq_len-1]
                - nll: Negative log-likelihood at each step [batch, seq_len-1]
                - states: (optional) Hidden states [batch, seq_len, n_s]
        """
        batch_size, seq_len = sequence.shape
        
        # Initialize state
        state = torch.zeros(batch_size, self.n_s, device=self.device)
        
        # Storage
        all_logits = []
        all_kl = []
        all_nll = []
        all_states = [] if return_all_states else None
        
        # Process sequence
        for t in range(seq_len - 1):
            # Current token
            token = sequence[:, t]
            
            # Step
            logits, state, kl_div = self.step(token, state)
            
            # Compute NLL for next token
            target = sequence[:, t + 1]
            nll = F.cross_entropy(logits, target, reduction='none')
            
            # Store
            all_logits.append(logits)
            all_kl.append(kl_div)
            all_nll.append(nll)
            
            if return_all_states:
                all_states.append(state)
        
        # Stack results
        logits = torch.stack(all_logits, dim=1)  # [batch, seq_len-1, vocab]
        kl_divergence = torch.stack(all_kl, dim=1)  # [batch, seq_len-1]
        nll = torch.stack(all_nll, dim=1)  # [batch, seq_len-1]
        
        # Free energy = KL + NLL
        free_energy = kl_divergence + nll
        
        results = {
            'logits': logits,
            'free_energy': free_energy,
            'kl_divergence': kl_divergence,
            'nll': nll,
        }
        
        if return_all_states:
            results['states'] = torch.stack(all_states, dim=1)
        
        return results
    
    def predict_next(self, sequence: torch.Tensor, 
                     temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the next token given a sequence.
        
        Args:
            sequence: Input sequence [batch_size, seq_len]
            temperature: Sampling temperature (1.0 = normal, <1 = sharper)
            
        Returns:
            predictions: Predicted next tokens [batch_size]
            logits: Logits for all tokens [batch_size, vocab_size]
        """
        with torch.no_grad():
            results = self.forward(sequence)
            logits = results['logits'][:, -1, :]  # Last prediction
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            predictions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            return predictions, logits
    
    def compute_loss(self, sequence: torch.Tensor, 
                     beta_kl: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Compute variational free energy loss.
        
        Args:
            sequence: Input sequence [batch_size, seq_len]
            beta_kl: Weight for KL term (beta-VAE style)
            
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        results = self.forward(sequence)
        
        # Free energy = weighted KL + NLL
        kl_term = results['kl_divergence'].mean()
        nll_term = results['nll'].mean()
        
        loss = beta_kl * kl_term + nll_term
        
        metrics = {
            'loss': loss.item(),
            'free_energy': results['free_energy'].mean().item(),
            'kl_divergence': kl_term.item(),
            'nll': nll_term.item(),
            'perplexity': torch.exp(nll_term).item()
        }
        
        return loss, metrics


class ActiveInferencePredictorWithMemory(ActiveInferencePredictor):
    """
    Extended version with explicit memory mechanism.
    
    This adds an attention-based memory that allows the model to
    maintain and query past states, similar to Transformers but
    within the active inference framework.
    """
    
    def __init__(self, *args, memory_size: int = 256, n_heads: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.memory_size = memory_size
        self.n_heads = n_heads
        
        # Memory attention
        self.memory_query = nn.Linear(self.n_s, self.n_s)
        self.memory_key = nn.Linear(self.n_s, self.n_s)
        self.memory_value = nn.Linear(self.n_s, self.n_s)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.n_s,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Combine attended memory with current state
        self.memory_combine = nn.Linear(self.n_s * 2, self.n_s)
    
    def forward_with_memory(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process sequence with memory-augmented inference."""
        batch_size, seq_len = sequence.shape
        
        # Initialize
        state = torch.zeros(batch_size, self.n_s, device=self.device)
        memory = []
        
        all_logits = []
        all_kl = []
        all_nll = []
        
        for t in range(seq_len - 1):
            token = sequence[:, t]
            
            # Standard step
            logits, state_new, kl_div, components = self.step(
                token, state, return_components=True
            )
            
            # Update memory
            memory.append(state_new.unsqueeze(1))
            if len(memory) > self.memory_size:
                memory.pop(0)
            
            # Attend to memory if we have enough history
            if len(memory) > 1:
                memory_tensor = torch.cat(memory, dim=1)  # [batch, mem_len, n_s]
                
                # Attention
                query = state_new.unsqueeze(1)  # [batch, 1, n_s]
                attended, _ = self.attention(query, memory_tensor, memory_tensor)
                attended = attended.squeeze(1)  # [batch, n_s]
                
                # Combine
                state_combined = self.memory_combine(
                    torch.cat([state_new, attended], dim=-1)
                )
                state = state_combined
            else:
                state = state_new
            
            # Compute NLL
            target = sequence[:, t + 1]
            nll = F.cross_entropy(logits, target, reduction='none')
            
            all_logits.append(logits)
            all_kl.append(kl_div)
            all_nll.append(nll)
        
        logits = torch.stack(all_logits, dim=1)
        kl_divergence = torch.stack(all_kl, dim=1)
        nll = torch.stack(all_nll, dim=1)
        free_energy = kl_divergence + nll
        
        return {
            'logits': logits,
            'free_energy': free_energy,
            'kl_divergence': kl_divergence,
            'nll': nll,
        }
