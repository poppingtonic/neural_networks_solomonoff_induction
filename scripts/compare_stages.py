#!/usr/bin/env python3
"""Compare performance across different training stages.

Loads checkpoints from different stages and compares their performance
on both UTM and CTW tasks.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(checkpoint_dir: str) -> Dict[str, Any]:
    """Load training metrics from checkpoint directory."""
    metrics_path = Path(checkpoint_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}


def plot_training_curves(metrics: Dict[str, Any], output_dir: Path):
    """Plot training loss curves for both stages."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # UTM stage
    if "utm" in metrics and metrics["utm"]:
        utm_losses = [m["loss"] for m in metrics["utm"]]
        axes[0].plot(utm_losses, label="UTM Loss", color="blue", linewidth=2)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Stage 1: UTM Training")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # CTW stage
    if "ctw" in metrics and metrics["ctw"]:
        ctw_losses = [m["loss"] for m in metrics["ctw"]]
        axes[1].plot(ctw_losses, label="CTW Loss", color="green", linewidth=2)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Stage 2: CTW Training")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {output_path}")
    plt.close()


def plot_comparison(all_metrics: Dict[str, Dict], output_dir: Path):
    """Plot comparison across multiple experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_names = []
    final_utm_losses = []
    final_ctw_losses = []
    
    for exp_name, metrics in all_metrics.items():
        exp_names.append(exp_name)
        
        if "utm" in metrics and metrics["utm"]:
            final_utm_losses.append(metrics["utm"][-1]["loss"])
        else:
            final_utm_losses.append(None)
        
        if "ctw" in metrics and metrics["ctw"]:
            final_ctw_losses.append(metrics["ctw"][-1]["loss"])
        else:
            final_ctw_losses.append(None)
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    utm_bars = [l if l is not None else 0 for l in final_utm_losses]
    ctw_bars = [l if l is not None else 0 for l in final_ctw_losses]
    
    ax.bar(x - width/2, utm_bars, width, label="UTM Final Loss", color="blue", alpha=0.7)
    ax.bar(x + width/2, ctw_bars, width, label="CTW Final Loss", color="green", alpha=0.7)
    
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss Comparison Across Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to {output_path}")
    plt.close()


def print_summary(metrics: Dict[str, Any], name: str):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"Summary: {name}")
    print(f"{'='*60}")
    
    if "utm" in metrics and metrics["utm"]:
        utm_losses = [m["loss"] for m in metrics["utm"]]
        print(f"\nUTM Stage:")
        print(f"  Steps: {len(utm_losses)}")
        print(f"  Initial Loss: {utm_losses[0]:.6f}")
        print(f"  Final Loss: {utm_losses[-1]:.6f}")
        print(f"  Improvement: {utm_losses[0] - utm_losses[-1]:.6f}")
        print(f"  Avg Loss: {np.mean(utm_losses):.6f}")
    
    if "ctw" in metrics and metrics["ctw"]:
        ctw_losses = [m["loss"] for m in metrics["ctw"]]
        print(f"\nCTW Stage:")
        print(f"  Steps: {len(ctw_losses)}")
        print(f"  Initial Loss: {ctw_losses[0]:.6f}")
        print(f"  Final Loss: {ctw_losses[-1]:.6f}")
        print(f"  Improvement: {ctw_losses[0] - ctw_losses[-1]:.6f}")
        print(f"  Avg Loss: {np.mean(ctw_losses):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance across training stages"
    )
    parser.add_argument(
        "checkpoint_dirs",
        nargs="+",
        help="Checkpoint directories to compare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="Output directory for plots",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading metrics from checkpoint directories...")
    all_metrics = {}
    
    for checkpoint_dir in args.checkpoint_dirs:
        path = Path(checkpoint_dir)
        name = path.name
        metrics = load_metrics(checkpoint_dir)
        
        if metrics:
            all_metrics[name] = metrics
            print(f"✓ Loaded metrics from {checkpoint_dir}")
            print_summary(metrics, name)
            
            # Plot individual training curves
            plot_training_curves(metrics, output_dir / name)
        else:
            print(f"⚠ No metrics found in {checkpoint_dir}")
    
    # Plot comparison across experiments
    if len(all_metrics) > 1:
        print(f"\nGenerating comparison plots...")
        plot_comparison(all_metrics, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ Comparison complete! Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
