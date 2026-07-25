#!/usr/bin/env python3
"""
Cost calculator for Tinker model training.

Usage:
    python cost_calculator.py --config config.json
    python cost_calculator.py --model Qwen/Qwen3-30B-A3B --batch 8 --seq 256 --steps 400
"""

import json
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

# Rate card (USD per million tokens) - update as needed
RATE_CARD = {
    "Qwen/Qwen3-4B-Instruct-2507": {"train": 0.22},
    "Qwen/Qwen3-8B": {"train": 0.40},
    "Qwen/Qwen3-30B-A3B": {"train": 0.36},
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {"train": 0.53},
    "Qwen/Qwen3-32B": {"train": 1.47},
    "meta-llama/Llama-3.2-1B": {"train": 0.09},
    "meta-llama/Llama-3.2-3B": {"train": 0.18},
    "meta-llama/Llama-3.1-8B": {"train": 0.40},
    # Add other models as needed
}

@dataclass
class TrainingConfig:
    model: str
    batch_size: int
    seq_length: int
    total_steps: int

    @classmethod
    def from_config_file(cls, config_path: str) -> 'TrainingConfig':
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Calculate total steps across all stages
        total_steps = sum(stage.get('steps', 0) for stage in config.get('stages', []))
        
        return cls(
            model=config['base_model'],
            batch_size=config['batch_size'],
            seq_length=config['seq_length'],
            total_steps=total_steps
        )

def calculate_cost(config: TrainingConfig) -> Dict[str, float]:
    """Calculate estimated training cost."""
    if config.model not in RATE_CARD:
        available_models = '\n  - ' + '\n  - '.join(RATE_CARD.keys())
        raise ValueError(f"Model {config.model} not in rate card. Available models:{available_models}")
    
    rate_per_million = RATE_CARD[config.model]["train"]
    
    # Calculate total tokens (batch_size * seq_length * steps)
    total_tokens = config.batch_size * config.seq_length * config.total_steps
    tokens_in_millions = total_tokens / 1_000_000
    
    estimated_cost = tokens_in_millions * rate_per_million
    
    return {
        "model": config.model,
        "batch_size": config.batch_size,
        "seq_length": config.seq_length,
        "total_steps": config.total_steps,
        "total_tokens": total_tokens,
        "rate_per_million_tokens": rate_per_million,
        "estimated_cost_usd": estimated_cost
    }

def format_cost(cost_info: Dict[str, float]) -> str:
    """Format cost information into a human-readable string."""
    return (
        f"\nTraining Cost Estimate\n"
        f"{'='*40}\n"
        f"Model:               {cost_info['model']}\n"
        f"Batch size:          {cost_info['batch_size']}\n"
        f"Sequence length:     {cost_info['seq_length']}\n"
        f"Total steps:         {cost_info['total_steps']:,}\n"
        f"Total tokens:        {cost_info['total_tokens']:,} ({cost_info['total_tokens']/1_000_000:.2f}M)\n"
        f"Rate per M tokens:   ${cost_info['rate_per_million_tokens']:.2f}\n"
        f"{'='*40}\n"
        f"Estimated cost:      ${cost_info['estimated_cost_usd']:.2f} USD\n"
        f"{'='*40}"
    )

def main():
    parser = argparse.ArgumentParser(description='Estimate training costs for Tinker models.')
    
    # Either provide a config file OR individual parameters
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help='Path to config.json')
    group.add_argument('--model', type=str, help='Model name (e.g., Qwen/Qwen3-30B-A3B)')
    
    # Additional parameters if not using a config file
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--seq', type=int, default=256, help='Sequence length')
    parser.add_argument('--steps', type=int, default=100, help='Total training steps')
    
    args = parser.parse_args()
    
    if args.config:
        config = TrainingConfig.from_config_file(args.config)
    else:
        config = TrainingConfig(
            model=args.model,
            batch_size=args.batch,
            seq_length=args.seq,
            total_steps=args.steps
        )
    
    try:
        cost_info = calculate_cost(config)
        print(format_cost(cost_info))
    except ValueError as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
