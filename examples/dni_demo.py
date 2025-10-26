#!/usr/bin/env python3
"""Demo: Decoupled Neural Interfaces (DNI) with Sequential Finetuning

This demonstrates how DNI enables asynchronous layer updates using synthetic gradients.

Benefits of DNI:
1. Update decoupling - layers can update without waiting for backprop
2. Faster training - reduced dependency on sequential updates
3. Parallelization - layers can be trained on different devices
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn

# Import DNI modules
from torch_training import dni
from torch_training.dni_adapter import SimpleMLPWithDNI, TransformerWithDNI, create_dni_model
from torch_models.transformer import TransformerConfig, TransformerDecoderLM


def demo_basic_dni():
    """Demonstrate basic DNI with a simple MLP."""
    print("="*70)
    print("Demo 1: Basic DNI with Simple MLP")
    print("="*70)
    print()
    
    # Create simple data
    batch_size = 32
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    
    X = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))
    
    # Create model with DNI
    model = SimpleMLPWithDNI(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
    )
    
    print(f"Model: MLP with DNI")
    print(f"Architecture: {input_dim} → {hidden_dim} → {hidden_dim} → {output_dim}")
    print(f"DNI interfaces: 2 (between layers)")
    print()
    
    # Train for a few steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass with DNI - synthetic gradients are used internally
        output = model(X)
        loss = criterion(output, y)
        
        # Backward pass - DNI updates synthetic gradient predictors
        loss.backward()
        optimizer.step()
        
        print(f"Step {step}: loss = {loss.item():.6f}")
    
    print()
    print("✓ Basic DNI demo complete")
    print("  Layers updated using synthetic gradients!")
    print()


def demo_transformer_with_dni():
    """Demonstrate DNI with a Transformer model."""
    print("="*70)
    print("Demo 2: DNI with Transformer (Sequential Finetuning)")
    print("="*70)
    print()
    
    # Create transformer
    config = TransformerConfig(
        vocab_size=128,
        embedding_dim=64,
        num_layers=4,
        num_heads=8,
        widening_factor=4,
    )
    base_model = TransformerDecoderLM(config)
    
    print(f"Base model: Transformer")
    print(f"Layers: {config.num_layers}")
    print(f"Embedding dim: {config.embedding_dim}")
    print()
    
    # Wrap with DNI
    model = create_dni_model(
        base_model=base_model,
        use_dni=True,
        num_dni_points=2,
        dni_hidden_dim=64,
        use_context=False,
    )
    
    print()
    
    # Create sample data
    batch_size = 8
    seq_length = 32
    
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Train for a few steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass with DNI
        log_probs = model(tokens)
        
        # Simple loss (just for demo)
        loss = -log_probs.mean()
        
        # Backward pass with DNI
        loss.backward()
        optimizer.step()
        
        print(f"Step {step}: loss = {loss.item():.6f}")
    
    print()
    print("✓ Transformer DNI demo complete")
    print("  Transformer layers decoupled with synthetic gradients!")
    print()


def demo_dni_context():
    """Demonstrate conditional DNI (cDNI) using context."""
    print("="*70)
    print("Demo 3: Conditional DNI with Context")
    print("="*70)
    print()
    
    # Create a simple network with context
    activation_dim = 64
    context_dim = 10
    batch_size = 16
    
    # Create data with labels as context
    X = torch.randn(batch_size, activation_dim)
    labels = torch.randint(0, context_dim, (batch_size,))
    labels_one_hot = torch.nn.functional.one_hot(labels, context_dim).float()
    
    # Create DNI synthesizer with context
    synthesizer = dni.BasicSynthesizer(
        output_dim=activation_dim,
        n_hidden=1,
        hidden_dim=64,
        context_dim=context_dim,
    )
    
    backward_interface = dni.BackwardInterface(synthesizer)
    
    print(f"Conditional DNI (cDNI) setup:")
    print(f"  Activation dim: {activation_dim}")
    print(f"  Context dim: {context_dim}")
    print(f"  Context: one-hot labels")
    print()
    
    # Use DNI with context
    backward_interface.train()
    
    for step in range(5):
        # Use context manager to pass labels
        with dni.synthesizer_context(labels_one_hot):
            # Apply DNI - synthetic gradients conditioned on labels
            X_dni = backward_interface(X)
        
        # Compute a simple loss
        loss = (X_dni ** 2).mean()
        loss.backward()
        
        print(f"Step {step}: loss = {loss.item():.6f}")
    
    print()
    print("✓ Conditional DNI demo complete")
    print("  Synthetic gradients conditioned on label context!")
    print()


def demo_sequential_finetune_with_dni():
    """Show how to use DNI in sequential finetuning."""
    print("="*70)
    print("Demo 4: Sequential Finetuning with DNI")
    print("="*70)
    print()
    
    print("To use DNI in sequential finetuning, run:")
    print()
    print("  python torch_training/train_sequential_finetune.py \\")
    print("    --stage all \\")
    print("    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \\")
    print("    --use_hf \\")
    print("    --use_dni \\")
    print("    --dni_num_points 2 \\")
    print("    --dni_hidden_dim 64 \\")
    print("    --utm_steps 10000 \\")
    print("    --ctw_steps 5000")
    print()
    print("Benefits:")
    print("  ✓ Layers update asynchronously")
    print("  ✓ Faster convergence (in theory)")
    print("  ✓ Enables distributed training")
    print()


def main():
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " "*12 + "Decoupled Neural Interfaces (DNI) Demo" + " "*18 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    try:
        # Demo 1: Basic MLP with DNI
        demo_basic_dni()
        
        # Demo 2: Transformer with DNI
        demo_transformer_with_dni()
        
        # Demo 3: Conditional DNI
        demo_dni_context()
        
        # Demo 4: Usage in sequential finetuning
        demo_sequential_finetune_with_dni()
        
        print("="*70)
        print("✓ All DNI demos complete!")
        print("="*70)
        print()
        print("Key Takeaways:")
        print("  1. DNI enables asynchronous layer updates")
        print("  2. Works with any PyTorch model")
        print("  3. Can be conditioned on context (cDNI)")
        print("  4. Integrated into sequential finetuning pipeline")
        print()
        print("Next Steps:")
        print("  • Try DNI with your own models")
        print("  • Experiment with different decoupling points")
        print("  • Use conditional DNI with labels")
        print("  • Apply to sequential finetuning (UTM → CTW)")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
