#!/usr/bin/env python3
"""Quick-start example for FastAI LSTM sequential finetuning.

This example demonstrates how to use the FastAI-based implementation
for training an LSTM model on UTM and CTW data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch_models.lstm import LSTMConfig, LSTMDecoderLM
from torch_training.fastai_dataloaders import create_utm_dataloaders, create_ctw_dataloaders
from torch_training.fastai_callbacks import (
    PerplexityMetric,
    GradNormCallback,
    ModelCheckpointCallback,
    LSTMLogProbLoss,
)
from fastai.learner import Learner
from fastai.optimizer import Adam

# Data generators
from data import utm_data_generator as utm_dg
from data import ctw_data_generator as ctw_dg
from data import utms as utms_lib


def main():
    """Run FastAI LSTM training example."""
    
    print("="*60)
    print("FastAI LSTM Sequential Finetuning Example")
    print("="*60)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 128  # ASCII tokenizer
    batch_size = 32
    seq_length = 256
    
    print(f"\nDevice: {device}")
    print(f"Vocab size: {vocab_size}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    
    # Create LSTM model
    print("\n1. Creating LSTM model...")
    config = LSTMConfig(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
    )
    model = LSTMDecoderLM(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Create UTM data generator
    print("\n2. Creating UTM data generator...")
    rng = np.random.default_rng(seed=42)
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler)
    
    utm_generator = utm_dg.UTMDataGenerator(
        batch_size=batch_size,
        seq_length=seq_length,
        rng=rng,
        utm=utm,
        memory_size=10,
        maximum_steps=100,
        tokenizer=utm_dg.Tokenizer.ASCII,
        maximum_program_length=100,
    )
    
    # Create FastAI DataLoaders
    print("\n3. Creating FastAI DataLoaders...")
    dls_utm = create_utm_dataloaders(
        utm_generator,
        batches_per_epoch=50,  # Small for example
        device=device,
    )
    
    # Create FastAI Learner
    print("\n4. Creating FastAI Learner...")
    learn = Learner(
        dls=dls_utm,
        model=model,
        loss_func=LSTMLogProbLoss(),
        opt_func=Adam,
        metrics=[PerplexityMetric()],
    )
    
    # Train Stage 1: UTM
    print("\n5. Training on UTM data (Stage 1)...")
    print("-" * 60)
    
    cbs = [
        GradNormCallback(),
        ModelCheckpointCallback(
            output_dir="./checkpoints/fastai_example",
            stage="utm",
            save_every=25,  # Save every 25 steps
        ),
    ]
    
    learn.fit_one_cycle(
        n_epoch=3,  # Small number for example
        lr_max=1e-3,
        cbs=cbs,
    )
    
    print("\n✓ Stage 1 (UTM) training complete!")
    
    # Create CTW data generator
    print("\n6. Creating CTW data generator...")
    rng_ctw = np.random.default_rng(seed=43)
    ctw_generator = ctw_dg.CTWGenerator(
        batch_size=batch_size,
        seq_length=seq_length,
        rng=rng_ctw,
        max_depth=5,
        with_contexts=False,
    )
    
    # Create CTW DataLoaders
    print("\n7. Creating CTW DataLoaders...")
    dls_ctw = create_ctw_dataloaders(
        ctw_generator,
        batches_per_epoch=50,
        device=device,
    )
    
    # Update learner with new data
    learn.dls = dls_ctw
    
    # Train Stage 2: CTW
    print("\n8. Training on CTW data (Stage 2)...")
    print("-" * 60)
    
    cbs = [
        GradNormCallback(),
        ModelCheckpointCallback(
            output_dir="./checkpoints/fastai_example",
            stage="ctw",
            save_every=25,
        ),
    ]
    
    learn.fit_one_cycle(
        n_epoch=2,  # Smaller for Stage 2
        lr_max=5e-4,  # Lower LR for fine-tuning
        cbs=cbs,
    )
    
    print("\n✓ Stage 2 (CTW) training complete!")
    
    # Save final model
    print("\n9. Saving final model...")
    final_path = Path("./checkpoints/fastai_example/final_model.pth")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }, final_path)
    print(f"   Saved to: {final_path}")
    
    print("\n" + "="*60)
    print("✓ FastAI LSTM Sequential Finetuning Example Complete!")
    print("="*60)
    print("\nCheckpoints saved to: ./checkpoints/fastai_example/")
    print("\nNext steps:")
    print("  - Run full training with more epochs")
    print("  - Experiment with different hyperparameters")
    print("  - Try transformer architecture for comparison")
    print("  - Use learning rate finder: learn.lr_find()")


if __name__ == "__main__":
    main()
