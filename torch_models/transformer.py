# Copyright 2025
# PyTorch implementation of a decoder-only Transformer mirroring models/transformer.py (Haiku/JAX)

from __future__ import annotations

import dataclasses
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
    vocab_size: int
    embedding_dim: int = 64
    num_layers: int = 4
    num_heads: int = 8
    emb_init_scale: float = 0.02
    widening_factor: int = 4
    # Randomized positional encoding params (for length generalization)
    # See: https://arxiv.org/abs/2305.16843
    use_randomized_pe: bool = False
    pe_max_length: int = 2048  # Max length to sample positions from during training


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class RandomizedPositionalEncoding(nn.Module):
    """Randomized positional encoding for improved length generalization.
    
    During training, randomly samples and sorts positions from [0, max_length)
    instead of using sequential positions [0, 1, 2, ..., seq_len-1].
    This simulates seeing longer sequences and makes the model robust to
    out-of-distribution positions at test time.
    
    Reference: https://arxiv.org/abs/2305.16843
    "Randomized Positional Encodings Boost Length Generalization of Transformers"
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.max_len = max_len
        # Precompute full positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        
        if self.training and T < self.max_len:
            # During training: randomly sample T positions from [0, max_len) and sort
            # Each batch element gets the same random positions for consistency
            indices = torch.randperm(self.max_len, device=x.device)[:T]
            indices = torch.sort(indices).values
            pos_enc = self.pe[indices].unsqueeze(0)  # (1, T, D)
        else:
            # During eval or if seq_len >= max_len: use standard sequential positions
            pos_enc = self.pe[:T].unsqueeze(0)
        
        return x + pos_enc


class TransformerDecoderLM(nn.Module):
    """Decoder-only transformer producing log-probabilities over the vocabulary.

    Forward signature:
      inputs: integer tokens of shape (B, T)
      returns: log-probs of shape (B, T, V)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        nn.init.trunc_normal_(self.embed.weight, std=config.emb_init_scale)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * config.widening_factor,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Use randomized PE for length generalization if configured
        if config.use_randomized_pe:
            self.pos_enc = RandomizedPositionalEncoding(
                config.embedding_dim, max_len=config.pe_max_length
            )
        else:
            self.pos_enc = PositionalEncoding(config.embedding_dim)
        
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.out = nn.Linear(config.embedding_dim, config.vocab_size)

    @staticmethod
    def _shift_right(targets: torch.Tensor) -> torch.Tensor:
        # Right-shift by prepending BOS token id 0 (matching JAX shift behavior with zeros)
        B, T = targets.shape
        bos = torch.zeros((B, 1), dtype=targets.dtype, device=targets.device)
        return torch.cat([bos, targets], dim=1)[:, :T]

    @staticmethod
    def _causal_mask(T: int, device: Optional[torch.device] = None) -> torch.Tensor:
        # Bool mask with True in positions that should be masked (no future attention)
        # nn.Transformer uses True to indicate masked positions in attn_mask when bool.
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        # Shift targets to create inputs
        inputs = self._shift_right(targets)  # (B, T)
        x = self.embed(inputs) * math.sqrt(self.config.embedding_dim)  # (B, T, D)
        x = self.pos_enc(x)
        T = x.size(1)
        attn_mask = self._causal_mask(T, x.device)
        x = self.encoder(x, mask=attn_mask)
        x = self.norm(x)
        logits = self.out(x)
        return F.log_softmax(logits, dim=-1)
