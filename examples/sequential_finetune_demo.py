#!/usr/bin/env python3
"""Demo script showing sequential finetuning workflow.

This demonstrates the complete pipeline:
1. Load pretrained model (HuggingFace or custom)
2. Finetune on UTM data
3. Finetune on CTW data
4. Evaluate on both tasks
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from data import utm_data_generator as utm_dg
from data import ctw_data_generator as ctw_dg
from data import utms as utms_lib
from torch_models.transformer import TransformerConfig, TransformerDecoderLM


def main():
    print("="*70)
    print("Sequential Finetuning Demo: UTM → CTW")
    print("="*70)
    print()
    
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    OUTPUT_DIR = "./demo_output"
    
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Step 1: Create or load base model
    print("Step 1: Creating base model...")
    config = TransformerConfig(
        vocab_size=128,  # ASCII
        embedding_dim=64,
        num_layers=4,
        num_heads=8,
        widening_factor=4,
    )
    model = TransformerDecoderLM(config).to(DEVICE)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Step 2: Create UTM data generator
    print("Step 2: Creating UTM data generator...")
    rng_utm = np.random.default_rng(seed=SEED)
    program_sampler = utms_lib.FastSampler(rng=rng_utm)
    utm = utms_lib.BrainPhoqueUTM(program_sampler)
    
    utm_generator = utm_dg.UTMDataGenerator(
        batch_size=16,
        seq_length=64,
        rng=rng_utm,
        utm=utm,
        memory_size=10,
        maximum_steps=50,
        tokenizer=utm_dg.Tokenizer.ASCII,
        maximum_program_length=50,
    )
    print(f"✓ UTM generator ready (vocab_size={utm_generator.feature_size})")
    print()
    
    # Step 3: Quick UTM training (demo - just a few steps)
    print("Step 3: Training on UTM data (demo - 20 steps)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    for step in range(20):
        sequences, log_dict = utm_generator.sample()
        sequences = np.asarray(sequences)
        tokens = np.argmax(sequences, axis=-1).astype(np.int64)
        
        x = torch.from_numpy(tokens).to(DEVICE)
        y = x.clone()
        
        optimizer.zero_grad()
        log_probs = model(x)
        B, T, V = log_probs.shape
        
        loss = torch.nn.functional.nll_loss(
            log_probs.reshape(B * T, V),
            y.reshape(B * T),
            reduction="mean",
        )
        
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"  UTM step {step}: loss={loss.item():.6f}")
    
    print("✓ UTM training complete (demo)")
    print()
    
    # Step 4: Create CTW data generator
    print("Step 4: Creating CTW data generator...")
    rng_ctw = np.random.default_rng(seed=SEED + 1)
    
    ctw_generator = ctw_dg.CTWGenerator(
        batch_size=16,
        seq_length=64,
        rng=rng_ctw,
        max_depth=5,
        with_contexts=False,
    )
    print(f"✓ CTW generator ready (vocab_size={ctw_generator.feature_size})")
    print()
    
    # Step 5: Quick CTW training (demo - just a few steps)
    print("Step 5: Training on CTW data (demo - 20 steps)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    model.train()
    for step in range(20):
        sequences, log_dict = ctw_generator.sample()
        sequences = np.asarray(sequences)
        tokens = np.argmax(sequences, axis=-1).astype(np.int64)
        
        x = torch.from_numpy(tokens).to(DEVICE)
        y = x.clone()
        
        optimizer.zero_grad()
        log_probs = model(x)
        B, T, V = log_probs.shape
        
        loss = torch.nn.functional.nll_loss(
            log_probs.reshape(B * T, V),
            y.reshape(B * T),
            reduction="mean",
        )
        
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            tree_depths = log_dict.get('tree_depths', None)
            depth_str = f", avg_depth={np.mean(tree_depths):.2f}" if tree_depths is not None else ""
            print(f"  CTW step {step}: loss={loss.item():.6f}{depth_str}")
    
    print("✓ CTW training complete (demo)")
    print()
    
    # Step 6: Evaluation
    print("Step 6: Evaluating final model...")
    model.eval()
    
    # Evaluate on UTM
    with torch.no_grad():
        sequences, _ = utm_generator.sample()
        sequences = np.asarray(sequences)
        tokens = np.argmax(sequences, axis=-1).astype(np.int64)
        x = torch.from_numpy(tokens).to(DEVICE)
        log_probs = model(x)
        B, T, V = log_probs.shape
        loss_utm = torch.nn.functional.nll_loss(
            log_probs.reshape(B * T, V),
            x.reshape(B * T),
            reduction="mean",
        )
        print(f"  UTM eval loss: {loss_utm.item():.6f}")
    
    # Evaluate on CTW
    with torch.no_grad():
        sequences, _ = ctw_generator.sample()
        sequences = np.asarray(sequences)
        tokens = np.argmax(sequences, axis=-1).astype(np.int64)
        x = torch.from_numpy(tokens).to(DEVICE)
        log_probs = model(x)
        B, T, V = log_probs.shape
        loss_ctw = torch.nn.functional.nll_loss(
            log_probs.reshape(B * T, V),
            x.reshape(B * T),
            reduction="mean",
        )
        print(f"  CTW eval loss: {loss_ctw.item():.6f}")
    
    print()
    
    # Summary
    print("="*70)
    print("Demo complete! 🎉")
    print("="*70)
    print()
    print("This was a minimal demo with very few training steps.")
    print("For actual training, use:")
    print()
    print("  python torch_training/train_sequential_finetune.py \\")
    print("    --stage all \\")
    print("    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \\")
    print("    --use_hf \\")
    print("    --utm_steps 10000 \\")
    print("    --ctw_steps 5000")
    print()
    print("See QUICKSTART_SEQUENTIAL_FINETUNE.md for more details.")
    print()


if __name__ == "__main__":
    main()
