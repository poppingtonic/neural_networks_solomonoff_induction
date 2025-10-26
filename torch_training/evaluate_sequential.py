#!/usr/bin/env python3
"""Evaluate sequentially finetuned models on UTM and CTW tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from data import utm_data_generator as utm_dg
from data import ctw_data_generator as ctw_dg
from data import utms as utms_lib
from torch_models.transformer import TransformerConfig, TransformerDecoderLM
from utm_dataset import IGNORE_INDEX


def load_checkpoint(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "config" in checkpoint:
        config = TransformerConfig(**checkpoint["config"])
        model = TransformerDecoderLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try loading as HuggingFace model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    
    model.to(device)
    model.eval()
    return model


def evaluate_utm(
    model: torch.nn.Module,
    data_generator,
    num_eval_steps: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on UTM data.
    
    Args:
        model: Model to evaluate
        data_generator: UTM data generator
        num_eval_steps: Number of evaluation steps
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    losses = []
    perplexities = []
    
    with torch.no_grad():
        for _ in range(num_eval_steps):
            sequences, log_dict = data_generator.sample()
            sequences = np.asarray(sequences)
            tokens = np.argmax(sequences, axis=-1).astype(np.int64)
            
            if 'loss_mask' in log_dict:
                loss_mask = np.asarray(log_dict['loss_mask']).astype(bool)
            else:
                loss_mask = np.zeros(tokens.shape, dtype=bool)
            
            targets = tokens.copy()
            targets[loss_mask] = IGNORE_INDEX
            
            x = torch.from_numpy(tokens).to(device)
            y = torch.from_numpy(targets).to(device)
            
            log_probs = model(x)
            B, T, V = log_probs.shape
            
            loss = F.nll_loss(
                log_probs.reshape(B * T, V),
                y.reshape(B * T),
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )
            
            loss_val = float(loss.cpu().item())
            losses.append(loss_val)
            perplexities.append(math.exp(loss_val))
    
    return {
        "loss": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "perplexity": float(np.mean(perplexities)),
        "perplexity_std": float(np.std(perplexities)),
    }


def evaluate_ctw(
    model: torch.nn.Module,
    data_generator,
    num_eval_steps: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on CTW data.
    
    Args:
        model: Model to evaluate
        data_generator: CTW data generator
        num_eval_steps: Number of evaluation steps
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    losses = []
    perplexities = []
    tree_depths = []
    
    with torch.no_grad():
        for _ in range(num_eval_steps):
            sequences, log_dict = data_generator.sample()
            sequences = np.asarray(sequences)
            tokens = np.argmax(sequences, axis=-1).astype(np.int64)
            
            x = torch.from_numpy(tokens).to(device)
            y = x.clone()
            
            log_probs = model(x)
            B, T, V = log_probs.shape
            
            loss = F.nll_loss(
                log_probs.reshape(B * T, V),
                y.reshape(B * T),
                reduction="mean",
            )
            
            loss_val = float(loss.cpu().item())
            losses.append(loss_val)
            perplexities.append(math.exp(loss_val))
            
            if 'tree_depths' in log_dict:
                tree_depths.extend(log_dict['tree_depths'].tolist())
    
    metrics = {
        "loss": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "perplexity": float(np.mean(perplexities)),
        "perplexity_std": float(np.std(perplexities)),
    }
    
    if tree_depths:
        metrics["avg_tree_depth"] = float(np.mean(tree_depths))
        metrics["max_tree_depth"] = float(np.max(tree_depths))
    
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate sequentially finetuned models"
    )
    
    # Model checkpoints
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Evaluation tasks
    parser.add_argument(
        "--eval_utm",
        action="store_true",
        help="Evaluate on UTM task",
    )
    parser.add_argument(
        "--eval_ctw",
        action="store_true",
        help="Evaluate on CTW task",
    )
    parser.add_argument(
        "--num_eval_steps",
        type=int,
        default=100,
        help="Number of evaluation steps per task",
    )
    
    # Data configuration
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    
    # UTM config
    parser.add_argument("--memory_size", type=int, default=10)
    parser.add_argument("--maximum_steps", type=int, default=100)
    parser.add_argument("--tokenizer", type=str, default="ascii")
    parser.add_argument("--maximum_program_length", type=int, default=100)
    
    # CTW config
    parser.add_argument("--ctw_max_depth", type=int, default=5)
    
    # Output
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save evaluation results to JSON file",
    )
    
    args = parser.parse_args(argv)
    
    # Load model
    print(f"Loading model from: {args.checkpoint_path}")
    model = load_checkpoint(args.checkpoint_path, args.device)
    print(f"✓ Model loaded on {args.device}")
    
    results = {
        "checkpoint": args.checkpoint_path,
        "device": args.device,
        "num_eval_steps": args.num_eval_steps,
    }
    
    # Evaluate on UTM
    if args.eval_utm:
        print("\n" + "="*60)
        print("Evaluating on UTM task...")
        print("="*60)
        
        rng = np.random.default_rng(seed=args.seed)
        program_sampler = utms_lib.FastSampler(rng=rng)
        utm = utms_lib.BrainPhoqueUTM(program_sampler)
        tokenizer_enum = (
            utm_dg.Tokenizer.ASCII if args.tokenizer.lower() == "ascii"
            else utm_dg.Tokenizer.SEQ_POSITION
        )
        
        utm_generator = utm_dg.UTMDataGenerator(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            rng=rng,
            utm=utm,
            memory_size=args.memory_size,
            maximum_steps=args.maximum_steps,
            tokenizer=tokenizer_enum,
            maximum_program_length=args.maximum_program_length,
        )
        
        utm_metrics = evaluate_utm(
            model=model,
            data_generator=utm_generator,
            num_eval_steps=args.num_eval_steps,
            device=args.device,
        )
        
        results["utm"] = utm_metrics
        
        print(f"\nUTM Results:")
        print(f"  Loss: {utm_metrics['loss']:.6f} ± {utm_metrics['loss_std']:.6f}")
        print(f"  Perplexity: {utm_metrics['perplexity']:.3f} ± {utm_metrics['perplexity_std']:.3f}")
    
    # Evaluate on CTW
    if args.eval_ctw:
        print("\n" + "="*60)
        print("Evaluating on CTW task...")
        print("="*60)
        
        rng = np.random.default_rng(seed=args.seed + 1)
        ctw_generator = ctw_dg.CTWGenerator(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            rng=rng,
            max_depth=args.ctw_max_depth,
            with_contexts=False,
        )
        
        ctw_metrics = evaluate_ctw(
            model=model,
            data_generator=ctw_generator,
            num_eval_steps=args.num_eval_steps,
            device=args.device,
        )
        
        results["ctw"] = ctw_metrics
        
        print(f"\nCTW Results:")
        print(f"  Loss: {ctw_metrics['loss']:.6f} ± {ctw_metrics['loss_std']:.6f}")
        print(f"  Perplexity: {ctw_metrics['perplexity']:.3f} ± {ctw_metrics['perplexity_std']:.3f}")
        if "avg_tree_depth" in ctw_metrics:
            print(f"  Avg Tree Depth: {ctw_metrics['avg_tree_depth']:.2f}")
            print(f"  Max Tree Depth: {ctw_metrics['max_tree_depth']:.0f}")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output_file}")
    
    print("\n" + "="*60)
    print("✓ Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
