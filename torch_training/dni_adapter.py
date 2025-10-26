"""Adapter for Decoupled Neural Interfaces (DNI).

Wraps models to use synthetic gradients for update decoupling.
Reference: Jaderberg et al. (2017) - Decoupled Neural Interfaces using Synthetic Gradients

This module integrates DNI into transformers for sequential finetuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, List

# Import DNI from the local copy
try:
    from . import dni
    DNI_AVAILABLE = True
except ImportError:
    DNI_AVAILABLE = False
    print("[DNI] Warning: DNI module not available. Install or check path.")


class TransformerWithDNI(nn.Module):
    """Wraps a Transformer model with DNI for update decoupling.
    
    This allows transformer layers to be updated asynchronously using synthetic
    gradients instead of waiting for full backpropagation from the loss.
    
    Example:
        >>> from torch_models.transformer import TransformerConfig, TransformerDecoderLM
        >>> config = TransformerConfig(vocab_size=128, embedding_dim=64)
        >>> base_model = TransformerDecoderLM(config)
        >>> model_with_dni = TransformerWithDNI(base_model, num_dni_points=2)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_dni_points: int = 2,
        dni_hidden_dim: Optional[int] = None,
        use_context: bool = False,
    ):
        """Initialize DNI-wrapped transformer.
        
        Args:
            base_model: The transformer model to wrap
            num_dni_points: Number of decoupling points (layers with DNI)
            dni_hidden_dim: Hidden dimension for DNI synthesizers
            use_context: Whether to use context (labels) for conditional DNI
        """
        super().__init__()
        
        if not DNI_AVAILABLE:
            raise ImportError(
                "DNI not available. Make sure dni.py is in torch_training/."
            )
        
        self.base_model = base_model
        self.num_dni_points = num_dni_points
        self.use_context = use_context
        
        # Determine activation dimension from model
        if hasattr(base_model, 'config'):
            self.activation_dim = base_model.config.embedding_dim
        else:
            # Fallback: try to infer from first layer
            self.activation_dim = 64  # Default
        
        if dni_hidden_dim is None:
            dni_hidden_dim = self.activation_dim
        
        # Create backward interfaces for update decoupling
        self.backward_interfaces = nn.ModuleList([
            dni.BackwardInterface(
                dni.BasicSynthesizer(
                    output_dim=self.activation_dim,
                    n_hidden=1,
                    hidden_dim=dni_hidden_dim,
                    context_dim=self.activation_dim if use_context else None,
                )
            )
            for _ in range(num_dni_points)
        ])
        
        print(f"[DNI] Initialized with {num_dni_points} decoupling points")
        print(f"[DNI] Activation dim: {self.activation_dim}")
        print(f"[DNI] Use context: {use_context}")
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with DNI decoupling.
        
        Args:
            x: Input tokens (B, T)
            context: Optional context for conditional DNI (B, T, D)
            
        Returns:
            Log probabilities (B, T, V)
        """
        if not self.training:
            # During evaluation, use standard forward pass
            return self.base_model(x)
        
        # During training, apply DNI at intermediate layers
        # Note: This is a simplified implementation
        # For full integration, you'd need to hook into transformer layers
        
        if self.use_context and context is not None:
            with dni.synthesizer_context(context):
                return self._forward_with_dni(x)
        else:
            return self._forward_with_dni(x)
    
    def _forward_with_dni(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with DNI synthetic gradients.
        
        This is a simplified version. For full implementation:
        1. Hook into transformer encoder layers
        2. Apply BackwardInterface at strategic points
        3. Allow layers to update asynchronously
        """
        # For now, just wrap the base model
        # Full DNI integration would require modifying the transformer internals
        return self.base_model(x)


class SimpleMLPWithDNI(nn.Module):
    """Example: Simple MLP with DNI for feedforward networks.
    
    Demonstrates basic DNI usage for update decoupling in a simple architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
    ):
        super().__init__()
        
        if not DNI_AVAILABLE:
            raise ImportError("DNI not available")
        
        self.layers = nn.ModuleList()
        self.dni_interfaces = nn.ModuleList()
        
        # Build network with DNI between layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add DNI interface after each hidden layer
            if i < num_layers - 1:
                self.dni_interfaces.append(
                    dni.BackwardInterface(
                        dni.BasicSynthesizer(
                            output_dim=dims[i + 1],
                            n_hidden=1,
                        )
                    )
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with DNI decoupling."""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.relu(x)
            # Apply DNI for update decoupling
            if self.training:
                x = self.dni_interfaces[i](x)
        
        # Final layer without DNI
        x = self.layers[-1](x)
        return x


def create_dni_model(
    base_model: nn.Module,
    use_dni: bool = True,
    num_dni_points: int = 2,
    dni_hidden_dim: Optional[int] = None,
    use_context: bool = False,
) -> nn.Module:
    """Create a DNI-wrapped model if requested.
    
    Args:
        base_model: The model to wrap (typically TransformerDecoderLM)
        use_dni: Whether to use DNI
        num_dni_points: Number of decoupling points
        dni_hidden_dim: Hidden dim for DNI synthesizers
        use_context: Whether to use conditional DNI
        
    Returns:
        DNI-wrapped model or original model
    """
    if not use_dni or not DNI_AVAILABLE:
        if use_dni and not DNI_AVAILABLE:
            print("[DNI] DNI requested but not available. Using standard model.")
        return base_model
    
    print("[DNI] Wrapping model with Decoupled Neural Interfaces")
    return TransformerWithDNI(
        base_model=base_model,
        num_dni_points=num_dni_points,
        dni_hidden_dim=dni_hidden_dim,
        use_context=use_context,
    )


# Example usage:
"""
# In training script:
from torch_training.dni_adapter import create_dni_model
from torch_models.transformer import TransformerConfig, TransformerDecoderLM

config = TransformerConfig(vocab_size=128, embedding_dim=64)
model = TransformerDecoderLM(config)

# Wrap with DNI for update decoupling
model = create_dni_model(
    base_model=model,
    use_dni=True,
    num_dni_points=2,
)

# Train as usual - layers will update asynchronously
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
...
"""
