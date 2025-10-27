#!/usr/bin/env python3
"""
Minimal Colab-optimized training script for Skywork sequential fine-tuning.
Designed to work within Google Colab's 15GB GPU memory limit.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class MemoryEfficientTrainer:
    """Memory-efficient trainer for Colab."""
    
    def __init__(self, model, device, output_dir, log_every=25):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.global_step = 0
        
        # Setup logging
        self.log_file = self.output_dir / "train.log"
        self.metrics = []
        
    def log(self, message):
        """Log to console and file."""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
    
    def train_step(self, batch, optimizer):
        """Memory-efficient training step."""
        self.model.train()
        optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
        
        # Move to device
        x = torch.from_numpy(batch).long().to(self.device)
        
        # Forward pass using HuggingFace model API
        outputs = self.model(x, labels=x)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Get loss value before cleanup
        loss_val = float(loss.detach().cpu().item())
        perplexity = math.exp(loss_val) if loss_val < 10 else float('inf')
        
        # Aggressive memory cleanup
        del outputs, loss, x
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return {"loss": loss_val, "perplexity": perplexity}
    
    def train(self, data_fn, steps, lr, save_every):
        """Training loop."""
        # Use AdamW with lower memory overhead
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
        
        for step in range(steps):
            # Generate batch
            batch = data_fn()
            
            # Training step
            metrics = self.train_step(batch, optimizer)
            self.metrics.append(metrics)
            self.global_step += 1
            
            # Logging
            if self.log_every > 0 and step % self.log_every == 0:
                self.log(
                    f"Step {step}/{steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"PPL: {metrics['perplexity']:.2f}"
                )
            
            # Checkpointing
            if save_every > 0 and step > 0 and step % save_every == 0:
                checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
                self.save_checkpoint(checkpoint_path, step)
                
                # Cleanup old checkpoints to save space
                self.cleanup_old_checkpoints(keep_last=2)
        
        # Final checkpoint
        final_path = self.output_dir / "final.pt"
        self.save_checkpoint(final_path, steps)
        self.log(f"✓ Training complete! Saved to {final_path}")
    
    def save_checkpoint(self, path, step):
        """Save checkpoint (model only, not optimizer to save space)."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "step": step,
            "global_step": self.global_step,
        }
        torch.save(checkpoint, path)
        self.log(f"Saved checkpoint: {path}")
    
    def cleanup_old_checkpoints(self, keep_last=2):
        """Remove old checkpoints to save disk space."""
        checkpoints = sorted(self.output_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > keep_last:
            for ckpt in checkpoints[:-keep_last]:
                ckpt.unlink()
    
    def save_metrics(self):
        """Save metrics to JSON."""
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)


def load_model_for_colab(model_name, device, use_8bit=True):
    """Load model with Colab-optimized settings."""
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        print(f"Loading {model_name}...")
        
        if use_8bit and device == 'cuda':
            # 8-bit quantization saves ~4x memory
            print("Using 8-bit quantization for memory efficiency...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map='auto',
                trust_remote_code=True,
            )
        else:
            # FP16 for some memory savings
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map='auto' if device == 'cuda' else None,
                trust_remote_code=True,
            )
            
            if device == 'cpu':
                model = model.to(device)
        
        # Enable gradient checkpointing for more memory savings
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
        
    except ImportError:
        print("Error: transformers library not installed")
        print("Run: pip install transformers bitsandbytes")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def create_dummy_data_generator(batch_size, seq_length, vocab_size=1000):
    """Create dummy data generator (replace with real UTM/CTW data)."""
    def generate():
        return np.random.randint(0, vocab_size, (batch_size, seq_length))
    return generate


def main():
    parser = argparse.ArgumentParser(
        description="Colab-optimized Skywork sequential fine-tuning"
    )
    
    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        default=True,
        help="Use 8-bit quantization (recommended for Colab)",
    )
    
    # Training
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    
    # Logging
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/colab")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*60)
    print("Colab Sequential Fine-tuning")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"8-bit: {args.use_8bit}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Steps: {args.steps}")
    print(f"Learning rate: {args.lr}")
    print("="*60)
    
    # Memory check
    if args.device == 'cuda':
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {total_mem:.2f} GB")
        
        if total_mem < 12:
            print("⚠️  Warning: Low GPU memory detected!")
            print("   Recommended: batch_size=2, seq_length=64")
    
    # Clear memory
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # Load model
    model = load_model_for_colab(args.model_name, args.device, args.use_8bit)
    
    # Create data generator (replace with real data)
    data_fn = create_dummy_data_generator(args.batch_size, args.seq_length)
    print("\n⚠️  Using dummy data generator - replace with real UTM/CTW data!")
    
    # Create trainer
    trainer = MemoryEfficientTrainer(
        model=model,
        device=args.device,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        data_fn=data_fn,
        steps=args.steps,
        lr=args.lr,
        save_every=args.save_every,
    )
    
    # Save metrics
    trainer.save_metrics()
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    
    # Memory stats
    if args.device == 'cuda':
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
