# Copyright 2025
# PyTorch LSTM implementation for sequence modeling
# LSTMs have been shown to generalize better to longer sequences
# Reference: https://arxiv.org/html/2401.14953v1

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(kw_only=True)
class LSTMConfig:
    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    emb_init_scale: float = 0.02
    tie_weights: bool = True  # Tie input embedding and output projection weights


class LSTMDecoderLM(nn.Module):
    """LSTM-based language model producing log-probabilities over the vocabulary.
    
    LSTMs have been shown to generalize better to longer sequences than transformers,
    making them particularly suitable for sequence prediction tasks.
    
    Forward signature:
      inputs: integer tokens of shape (B, T)
      returns: log-probs of shape (B, T, V)
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        nn.init.trunc_normal_(self.embed.weight, std=config.emb_init_scale)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        
        # Output projection
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.hidden_dim, config.vocab_size)
        
        # Weight tying (share embedding and output weights)
        if config.tie_weights:
            if config.embedding_dim != config.hidden_dim:
                self.projection = nn.Linear(config.hidden_dim, config.embedding_dim, bias=False)
            else:
                self.projection = None
            self.out.weight = self.embed.weight
        else:
            self.projection = None
    
    @staticmethod
    def _shift_right(targets: torch.Tensor) -> torch.Tensor:
        """Right-shift by prepending BOS token id 0."""
        B, T = targets.shape
        bos = torch.zeros((B, 1), dtype=targets.dtype, device=targets.device)
        return torch.cat([bos, targets], dim=1)[:, :T]
    
    def forward(
        self,
        targets: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            targets: integer tokens of shape (B, T)
            hidden: optional hidden state tuple (h, c) from previous forward pass
        
        Returns:
            log-probs of shape (B, T, V)
        """
        # Shift targets to create inputs (teacher forcing)
        inputs = self._shift_right(targets)  # (B, T)
        
        # Embed inputs
        x = self.embed(inputs)  # (B, T, embedding_dim)
        
        # Apply LSTM
        x, _ = self.lstm(x, hidden)  # (B, T, hidden_dim)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Project to embedding dimension if needed (for weight tying)
        if self.projection is not None:
            x = self.projection(x)
        
        # Project to vocabulary
        logits = self.out(x)  # (B, T, vocab_size)
        
        return F.log_softmax(logits, dim=-1)
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate sequences autoregressively.
        
        Args:
            prompt: initial tokens of shape (B, L)
            max_length: maximum sequence length to generate
            temperature: sampling temperature
            top_k: if set, only sample from top-k tokens
        
        Returns:
            generated sequence of shape (B, max_length)
        """
        B, L = prompt.shape
        generated = prompt.clone()
        device = prompt.device
        
        # Initialize hidden state
        hidden = None
        
        for _ in range(max_length - L):
            # Get logits for last token
            log_probs = self.forward(generated, hidden)
            logits = log_probs[:, -1, :] / temperature  # (B, V)
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
