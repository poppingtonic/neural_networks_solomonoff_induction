# Copyright 2025
# Interpretable Transformer with Future Lens, Linear Probing, and Activation Engineering
# Extends basic transformer.py with hooks for interpretability research

from __future__ import annotations

import dataclasses
from typing import Optional, Callable, Dict, List, Tuple, Any
from collections import defaultdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclasses.dataclass(kw_only=True)
class InterpretableTransformerConfig:
    """Configuration for interpretable transformer."""
    vocab_size: int
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    emb_init_scale: float = 0.02
    widening_factor: int = 4
    dropout: float = 0.1
    max_seq_len: int = 2048
    # Interpretability options
    store_activations: bool = True
    store_attention: bool = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        return self.dropout(x)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with hooks for activation storage and intervention.

    Supports:
    - Attention pattern extraction
    - Key/Query/Value vector extraction
    - Activation patching
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Storage for interpretability
        self.stored_attention: Optional[torch.Tensor] = None
        self.stored_keys: Optional[torch.Tensor] = None
        self.stored_queries: Optional[torch.Tensor] = None
        self.stored_values: Optional[torch.Tensor] = None

        # Intervention hooks
        self.attention_intervention: Optional[Callable] = None
        self.key_intervention: Optional[Callable] = None
        self.query_intervention: Optional[Callable] = None
        self.value_intervention: Optional[Callable] = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        store: bool = True,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Compute Q, K, V
        Q = self.q_proj(x)  # (B, T, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Apply interventions if set
        if self.query_intervention is not None:
            Q = self.query_intervention(Q, self.layer_idx)
        if self.key_intervention is not None:
            K = self.key_intervention(K, self.layer_idx)
        if self.value_intervention is not None:
            V = self.value_intervention(V, self.layer_idx)

        # Store for interpretability
        if store:
            self.stored_queries = Q.detach()
            self.stored_keys = K.detach()
            self.stored_values = V.detach()

        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Apply causal mask
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention intervention if set
        if self.attention_intervention is not None:
            attn_probs = self.attention_intervention(attn_probs, self.layer_idx)

        # Store attention patterns
        if store:
            self.stored_attention = attn_probs.detach()

        # Apply attention to values
        out = torch.matmul(attn_probs, V)  # (B, H, T, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out


class InterpretableTransformerBlock(nn.Module):
    """Transformer block with hooks for residual stream analysis."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        widening_factor: int = 4,
        dropout: float = 0.1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = InterpretableMultiHeadAttention(
            embed_dim, num_heads, dropout, layer_idx
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * widening_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * widening_factor, embed_dim),
            nn.Dropout(dropout),
        )

        # Storage for residual stream
        self.stored_pre_attn: Optional[torch.Tensor] = None
        self.stored_post_attn: Optional[torch.Tensor] = None
        self.stored_pre_mlp: Optional[torch.Tensor] = None
        self.stored_post_mlp: Optional[torch.Tensor] = None

        # Intervention hooks
        self.residual_intervention: Optional[Callable] = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        store: bool = True,
    ) -> torch.Tensor:
        # Pre-attention
        if store:
            self.stored_pre_attn = x.detach()

        # Attention with residual
        attn_out = self.attn(self.ln1(x), attn_mask, store=store)
        x = x + attn_out

        if store:
            self.stored_post_attn = x.detach()

        # Apply residual intervention if set
        if self.residual_intervention is not None:
            x = self.residual_intervention(x, self.layer_idx, "post_attn")

        # Pre-MLP
        if store:
            self.stored_pre_mlp = x.detach()

        # MLP with residual
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out

        if store:
            self.stored_post_mlp = x.detach()

        # Apply residual intervention if set
        if self.residual_intervention is not None:
            x = self.residual_intervention(x, self.layer_idx, "post_mlp")

        return x


class InterpretableTransformerLM(nn.Module):
    """
    Decoder-only transformer with comprehensive interpretability support.

    Features:
    - Future Lens: Predict future tokens from intermediate states
    - Linear Probing: Train linear classifiers on hidden states
    - Activation Engineering: Patch/steer activations at any layer
    - Attention Analysis: Extract and modify attention patterns
    """

    def __init__(self, config: InterpretableTransformerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        nn.init.trunc_normal_(self.embed.weight, std=config.emb_init_scale)

        self.pos_enc = PositionalEncoding(
            config.embedding_dim,
            config.max_seq_len,
            config.dropout
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            InterpretableTransformerBlock(
                config.embedding_dim,
                config.num_heads,
                config.widening_factor,
                config.dropout,
                layer_idx=i,
            )
            for i in range(config.num_layers)
        ])

        self.ln_final = nn.LayerNorm(config.embedding_dim)
        self.out = nn.Linear(config.embedding_dim, config.vocab_size)

        # For Future Lens linear probes
        self.future_probes: Dict[int, nn.Linear] = {}

    @staticmethod
    def _shift_right(targets: torch.Tensor) -> torch.Tensor:
        B, T = targets.shape
        bos = torch.zeros((B, 1), dtype=targets.dtype, device=targets.device)
        return torch.cat([bos, targets], dim=1)[:, :T]

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        targets: torch.Tensor,
        store_activations: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with optional activation storage.

        Args:
            targets: Token IDs of shape (B, T)
            store_activations: Whether to store intermediate activations

        Returns:
            Log probabilities of shape (B, T, V)
        """
        inputs = self._shift_right(targets)
        B, T = inputs.shape

        x = self.embed(inputs) * math.sqrt(self.config.embedding_dim)
        x = self.pos_enc(x)

        attn_mask = self._causal_mask(T, x.device)

        for block in self.blocks:
            x = block(x, attn_mask, store=store_activations)

        x = self.ln_final(x)
        logits = self.out(x)

        return F.log_softmax(logits, dim=-1)

    # ==================== FUTURE LENS ====================

    def get_hidden_states(self, layer_idx: int, position: str = "post_mlp") -> torch.Tensor:
        """
        Get stored hidden states from a specific layer.

        Args:
            layer_idx: Layer index (0 to num_layers-1)
            position: One of 'pre_attn', 'post_attn', 'pre_mlp', 'post_mlp'

        Returns:
            Tensor of shape (B, T, D)
        """
        block = self.blocks[layer_idx]
        if position == "pre_attn":
            return block.stored_pre_attn
        elif position == "post_attn":
            return block.stored_post_attn
        elif position == "pre_mlp":
            return block.stored_pre_mlp
        elif position == "post_mlp":
            return block.stored_post_mlp
        else:
            raise ValueError(f"Unknown position: {position}")

    def train_future_probe(
        self,
        layer_idx: int,
        train_data: torch.Tensor,
        num_future: int = 1,
        num_epochs: int = 100,
        lr: float = 1e-3,
    ) -> nn.Linear:
        """
        Train a linear probe to predict future tokens from hidden states.

        This implements the Future Lens linear approximation method.

        Args:
            layer_idx: Which layer's hidden states to use
            train_data: Training sequences of shape (N, T)
            num_future: How many tokens ahead to predict
            num_epochs: Training epochs
            lr: Learning rate

        Returns:
            Trained linear probe
        """
        device = next(self.parameters()).device

        # Create probe: hidden_dim -> vocab_size
        probe = nn.Linear(self.config.embedding_dim, self.config.vocab_size).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.eval()
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_start in range(0, len(train_data), 32):
                batch = train_data[batch_start:batch_start + 32].to(device)

                with torch.no_grad():
                    _ = self.forward(batch, store_activations=True)
                    hidden = self.get_hidden_states(layer_idx, "post_mlp")

                # Get positions and targets for future prediction
                B, T, D = hidden.shape
                if T <= num_future:
                    continue

                # Hidden states at positions 0 to T-num_future-1
                h = hidden[:, :-num_future, :].reshape(-1, D)
                # Target tokens at positions num_future to T-1
                targets = batch[:, num_future:].reshape(-1)

                # Train probe
                optimizer.zero_grad()
                logits = probe(h)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if epoch % 20 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                print(f"  Future probe epoch {epoch}: loss = {avg_loss:.4f}")

        self.future_probes[(layer_idx, num_future)] = probe
        return probe

    def future_lens_predict(
        self,
        tokens: torch.Tensor,
        layer_idx: int,
        position_idx: int,
        num_future: int = 1,
    ) -> torch.Tensor:
        """
        Predict future tokens using trained Future Lens probe.

        Args:
            tokens: Input sequence (B, T)
            layer_idx: Layer to extract hidden states from
            position_idx: Which position's hidden state to use
            num_future: How far ahead to predict

        Returns:
            Predicted token probabilities (B, V)
        """
        key = (layer_idx, num_future)
        if key not in self.future_probes:
            raise ValueError(f"No probe trained for layer {layer_idx}, future {num_future}")

        probe = self.future_probes[key]

        with torch.no_grad():
            _ = self.forward(tokens, store_activations=True)
            hidden = self.get_hidden_states(layer_idx, "post_mlp")
            h = hidden[:, position_idx, :]  # (B, D)
            logits = probe(h)

        return F.softmax(logits, dim=-1)

    # ==================== LINEAR PROBING ====================

    def extract_features(
        self,
        tokens: torch.Tensor,
        layer_idx: int,
        position: str = "post_mlp",
        pool: str = "last",
    ) -> torch.Tensor:
        """
        Extract features from hidden states for linear probing.

        Args:
            tokens: Input tokens (B, T)
            layer_idx: Which layer to extract from
            position: Where in the block ('pre_attn', 'post_attn', 'pre_mlp', 'post_mlp')
            pool: Pooling strategy ('last', 'mean', 'max', 'first')

        Returns:
            Feature vectors (B, D)
        """
        with torch.no_grad():
            _ = self.forward(tokens, store_activations=True)
            hidden = self.get_hidden_states(layer_idx, position)  # (B, T, D)

        if pool == "last":
            return hidden[:, -1, :]
        elif pool == "first":
            return hidden[:, 0, :]
        elif pool == "mean":
            return hidden.mean(dim=1)
        elif pool == "max":
            return hidden.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {pool}")

    def train_linear_probe(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        num_classes: int,
        num_epochs: int = 100,
        lr: float = 1e-2,
    ) -> nn.Linear:
        """
        Train a linear classifier on extracted features.

        Args:
            train_features: Features (N, D)
            train_labels: Labels (N,)
            num_classes: Number of output classes
            num_epochs: Training epochs
            lr: Learning rate

        Returns:
            Trained linear classifier
        """
        device = train_features.device
        probe = nn.Linear(self.config.embedding_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            logits = probe(train_features)
            loss = criterion(logits, train_labels)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                acc = (logits.argmax(-1) == train_labels).float().mean()
                print(f"  Linear probe epoch {epoch}: loss={loss:.4f}, acc={acc:.3f}")

        return probe

    # ==================== ACTIVATION ENGINEERING ====================

    def set_residual_intervention(
        self,
        intervention_fn: Callable[[torch.Tensor, int, str], torch.Tensor],
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Set an intervention function for residual stream manipulation.

        The intervention function receives:
        - x: The current residual stream tensor (B, T, D)
        - layer_idx: The current layer index
        - position: Either 'post_attn' or 'post_mlp'

        And should return the modified tensor.

        Args:
            intervention_fn: The intervention function
            layer_indices: Which layers to apply to (None = all)
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.blocks)))

        for i in layer_indices:
            self.blocks[i].residual_intervention = intervention_fn

    def clear_interventions(self):
        """Clear all intervention hooks."""
        for block in self.blocks:
            block.residual_intervention = None
            block.attn.attention_intervention = None
            block.attn.key_intervention = None
            block.attn.query_intervention = None
            block.attn.value_intervention = None

    def steer_with_direction(
        self,
        direction: torch.Tensor,
        scale: float = 1.0,
        layer_indices: Optional[List[int]] = None,
        position: str = "post_mlp",
    ):
        """
        Add a steering vector to the residual stream.

        Implements activation addition for behavioral steering.

        Args:
            direction: Steering direction (D,) or (1, 1, D)
            scale: Scaling factor for the direction
            layer_indices: Which layers to steer
            position: Where to apply steering
        """
        if direction.dim() == 1:
            direction = direction.unsqueeze(0).unsqueeze(0)

        def intervention(x, layer_idx, pos):
            if pos == position:
                return x + scale * direction.to(x.device)
            return x

        self.set_residual_intervention(intervention, layer_indices)

    def ablate_component(
        self,
        layer_idx: int,
        head_idx: Optional[int] = None,
        position: str = "post_attn",
    ):
        """
        Ablate (zero out) a specific component.

        Args:
            layer_idx: Layer to ablate
            head_idx: If set, ablate specific attention head
            position: Where to ablate
        """
        if head_idx is not None:
            # Ablate specific attention head
            def intervention(attn_probs, layer):
                if layer == layer_idx:
                    attn_probs[:, head_idx, :, :] = 0
                return attn_probs

            self.blocks[layer_idx].attn.attention_intervention = intervention
        else:
            # Ablate entire layer output
            def intervention(x, layer, pos):
                if layer == layer_idx and pos == position:
                    return torch.zeros_like(x)
                return x

            self.set_residual_intervention(intervention, [layer_idx])

    # ==================== ATTENTION ANALYSIS ====================

    def get_attention_patterns(self, layer_idx: int) -> torch.Tensor:
        """
        Get attention patterns from a specific layer.

        Returns:
            Attention patterns of shape (B, H, T, T)
        """
        return self.blocks[layer_idx].attn.stored_attention

    def get_attention_heads(
        self,
        tokens: torch.Tensor,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Get all attention-related tensors from a layer.

        Returns dict with keys: 'attention', 'queries', 'keys', 'values'
        """
        with torch.no_grad():
            _ = self.forward(tokens, store_activations=True)

        attn = self.blocks[layer_idx].attn
        return {
            "attention": attn.stored_attention,
            "queries": attn.stored_queries,
            "keys": attn.stored_keys,
            "values": attn.stored_values,
        }

    # ==================== LENGTH GENERALIZATION ====================

    def test_length_generalization(
        self,
        test_data_generator: Callable[[int], torch.Tensor],
        lengths: List[int],
    ) -> Dict[int, float]:
        """
        Test model performance on various sequence lengths.

        Args:
            test_data_generator: Function that generates test data of given length
            lengths: List of sequence lengths to test

        Returns:
            Dict mapping length to loss
        """
        results = {}
        device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
            for length in lengths:
                test_data = test_data_generator(length).to(device)
                log_probs = self.forward(test_data, store_activations=False)

                # Compute loss (ignoring first position)
                targets = test_data[:, 1:]
                preds = log_probs[:, :-1, :]
                loss = F.nll_loss(
                    preds.reshape(-1, self.config.vocab_size),
                    targets.reshape(-1),
                )
                results[length] = loss.item()

        return results

    # ==================== GENERATION ====================

    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            prompt: Initial tokens (B, L)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            Generated sequence (B, max_length)
        """
        generated = prompt.clone()
        device = prompt.device

        self.eval()
        with torch.no_grad():
            for _ in range(max_length - prompt.size(1)):
                log_probs = self.forward(generated, store_activations=False)
                logits = log_probs[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated


def create_steering_direction_from_examples(
    model: InterpretableTransformerLM,
    positive_examples: torch.Tensor,
    negative_examples: torch.Tensor,
    layer_idx: int,
    position: str = "post_mlp",
) -> torch.Tensor:
    """
    Create a steering direction using contrastive activation analysis.

    Args:
        model: The transformer model
        positive_examples: Sequences exhibiting desired behavior
        negative_examples: Sequences exhibiting undesired behavior
        layer_idx: Layer to extract activations from
        position: Position in block

    Returns:
        Steering direction vector (D,)
    """
    # Get activations for positive examples
    with torch.no_grad():
        _ = model.forward(positive_examples, store_activations=True)
        pos_acts = model.get_hidden_states(layer_idx, position)
        pos_mean = pos_acts.mean(dim=(0, 1))  # Average over batch and sequence

        _ = model.forward(negative_examples, store_activations=True)
        neg_acts = model.get_hidden_states(layer_idx, position)
        neg_mean = neg_acts.mean(dim=(0, 1))

    # Steering direction is the difference
    direction = pos_mean - neg_mean
    direction = direction / direction.norm()  # Normalize

    return direction
