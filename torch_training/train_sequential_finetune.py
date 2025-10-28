#!/usr/bin/env python3
"""Sequential Finetuning Pipeline for Universal Predictors.

Stage 1: Finetune on UTM data
Stage 2: Finetune on CTW data

Supports:
- HuggingFace pretrained models (e.g., Skywork-Reward-V2-Qwen3-0.6B)
- Custom LSTM (default - better length generalization, https://arxiv.org/html/2401.14953v1)
- Custom transformers
- Optional MultiSWAG, DNI, BitNet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Local imports
from data import utm_data_generator as utm_dg
from data import ctw_data_generator as ctw_dg
from data import utms as utms_lib
from torch_models.transformer import TransformerConfig, TransformerDecoderLM
from torch_models.lstm import LSTMConfig, LSTMDecoderLM
from utm_dataset import IGNORE_INDEX
from dni_adapter import create_dni_model
from swag_lora_adapter import create_swag_lora_model, update_swag_model

# LLC Estimation (optional)
try:
    from devinterp.optim import SGLD
    from devinterp.slt.sampler import estimate_learning_coeff_with_summary
    from devinterp.utils import plot_trace, default_nbeta
    DEVINTERP_AVAILABLE = True
except ImportError:
    DEVINTERP_AVAILABLE = False

class SequentialFinetuner:
    """Manages sequential finetuning across UTM and CTW stages."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        output_dir: str,
        log_every: int = 100,
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.global_step = 0
        
        # Initialize logging
        self.log_file = self.output_dir / "train.log"
        self.metrics = {"utm": [], "ctw": []}
        
    def log(self, message: str):
        """Log to both console and file."""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
    
    def train_step_utm(
        self,
        data_generator,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """Single training step on UTM data."""
        self.model.train()
        
        sequences, log_dict = data_generator.sample()
        sequences = np.asarray(sequences)
        tokens = np.argmax(sequences, axis=-1).astype(np.int64)
        
        # Handle loss mask for padding
        if 'loss_mask' in log_dict:
            loss_mask = np.asarray(log_dict['loss_mask']).astype(bool)
        else:
            loss_mask = np.zeros(tokens.shape, dtype=bool)
        
        targets = tokens.copy()
        targets[loss_mask] = IGNORE_INDEX
        
        x = torch.from_numpy(tokens).to(self.device)
        y = torch.from_numpy(targets).to(self.device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = self.model(x)
        
        # Handle both HuggingFace and custom model outputs
        if hasattr(outputs, 'logits'):
            # HuggingFace model output
            logits = outputs.logits
        else:
            # Custom model output (already logits)
            logits = outputs
        
        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        
        loss = F.nll_loss(
            log_probs.reshape(B * T, V),
            y.reshape(B * T),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        
        loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += float(param_norm.item() ** 2)
        grad_norm = math.sqrt(grad_norm)
        
        optimizer.step()
        
        return {
            "loss": float(loss.detach().cpu().item()),
            "perplexity": float(math.exp(loss.detach().cpu().item())),
            "grad_norm": grad_norm,
        }
    
    def train_step_ctw(
        self,
        data_generator,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """Single training step on CTW data."""
        self.model.train()
        
        sequences, log_dict = data_generator.sample()
        sequences = np.asarray(sequences)
        tokens = np.argmax(sequences, axis=-1).astype(np.int64)
        
        x = torch.from_numpy(tokens).to(self.device)
        y = x.clone()  # For CTW, we predict the sequence itself
        
        optimizer.zero_grad(set_to_none=True)
        outputs = self.model(x)
        
        # Handle both HuggingFace and custom model outputs
        if hasattr(outputs, 'logits'):
            # HuggingFace model output
            logits = outputs.logits
        else:
            # Custom model output (already logits)
            logits = outputs
        
        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        
        loss = F.nll_loss(
            log_probs.reshape(B * T, V),
            y.reshape(B * T),
            reduction="mean",
        )
        
        loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += float(param_norm.item() ** 2)
        grad_norm = math.sqrt(grad_norm)
        
        optimizer.step()
        
        # Track tree depth if available
        tree_depth = log_dict.get('tree_depths', None)
        avg_tree_depth = float(np.mean(tree_depth)) if tree_depth is not None else None
        
        return {
            "loss": float(loss.detach().cpu().item()),
            "perplexity": float(math.exp(loss.detach().cpu().item())),
            "grad_norm": grad_norm,
            "avg_tree_depth": avg_tree_depth,
        }
    
    def train_stage_utm(
        self,
        data_generator,
        steps: int,
        lr: float,
        save_every: int,
    ):
        """Train Stage 1: UTM data."""
        self.log(f"\n{'='*60}")
        self.log(f"STAGE 1: UTM Finetuning ({steps} steps)")
        self.log(f"{'='*60}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for step in range(steps):
            metrics = self.train_step_utm(data_generator, optimizer)
            self.metrics["utm"].append(metrics)
            self.global_step += 1
            
            if self.log_every > 0 and step % self.log_every == 0:
                self.log(
                    f"[UTM] step={step}/{steps} "
                    f"loss={metrics['loss']:.6f} "
                    f"ppl={metrics['perplexity']:.3f} "
                    f"grad_norm={metrics['grad_norm']:.3f}"
                )
            
            if save_every > 0 and step > 0 and step % save_every == 0:
                checkpoint_path = self.output_dir / f"stage1_utm_step_{step}.pt"
                self.save_checkpoint(checkpoint_path, stage="utm", step=step)
        
        # Save final Stage 1 checkpoint
        final_path = self.output_dir / "stage1_utm_final.pt"
        self.save_checkpoint(final_path, stage="utm", step=steps)
        self.log(f"✓ Stage 1 complete. Saved to {final_path}")
    
    def train_stage_ctw(
        self,
        data_generator,
        steps: int,
        lr: float,
        save_every: int,
    ):
        """Train Stage 2: CTW data."""
        self.log(f"\n{'='*60}")
        self.log(f"STAGE 2: CTW Finetuning ({steps} steps)")
        self.log(f"{'='*60}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for step in range(steps):
            metrics = self.train_step_ctw(data_generator, optimizer)
            self.metrics["ctw"].append(metrics)
            self.global_step += 1
            
            if self.log_every > 0 and step % self.log_every == 0:
                depth_str = f" tree_depth={metrics['avg_tree_depth']:.2f}" if metrics['avg_tree_depth'] else ""
                self.log(
                    f"[CTW] step={step}/{steps} "
                    f"loss={metrics['loss']:.6f} "
                    f"ppl={metrics['perplexity']:.3f} "
                    f"grad_norm={metrics['grad_norm']:.3f}"
                    f"{depth_str}"
                )
            
            if save_every > 0 and step > 0 and step % save_every == 0:
                checkpoint_path = self.output_dir / f"stage2_ctw_step_{step}.pt"
                self.save_checkpoint(checkpoint_path, stage="ctw", step=step)
        
        # Save final Stage 2 checkpoint
        final_path = self.output_dir / "stage2_ctw_final.pt"
        self.save_checkpoint(final_path, stage="ctw", step=steps)
        self.log(f"✓ Stage 2 complete. Saved to {final_path}")
    
    def save_checkpoint(self, path: Path, stage: str, step: int):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "stage": stage,
            "step": step,
            "global_step": self.global_step,
        }
        torch.save(checkpoint, path)
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        self.log(f"Metrics saved to {metrics_path}")


def load_pretrained_model(
    model_name_or_path: str,
    device: str,
    use_hf: bool = False,
    architecture: str = "lstm",
) -> Tuple[torch.nn.Module, Optional[int]]:
    """Load pretrained model from HuggingFace or local checkpoint.
    
    Args:
        model_name_or_path: Path or HF model name
        device: Device to load model on
        use_hf: Force HuggingFace loading
        architecture: Model architecture ('lstm' or 'transformer')
    
    Returns:
        model: Loaded model
        vocab_size: Vocabulary size (None if using HF model)
    """
    if use_hf or "/" in model_name_or_path:
        # Load from HuggingFace
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading HuggingFace model: {model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            vocab_size = len(tokenizer)
            print(f"✓ Loaded model with vocab_size={vocab_size}")
            return model, vocab_size
        except Exception as e:
            print(f"Failed to load HuggingFace model: {e}")
            print("Falling back to custom transformer...")
    
    # Load local checkpoint or create new model
    if os.path.exists(model_name_or_path):
        print(f"Loading checkpoint from: {model_name_or_path}")
        checkpoint = torch.load(model_name_or_path, map_location=device)
        
        # Detect architecture from config
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
            if "num_heads" in config_dict:
                config = TransformerConfig(**config_dict)
                model = TransformerDecoderLM(config)
                vocab_size = config.vocab_size
            else:
                config = LSTMConfig(**config_dict)
                model = LSTMDecoderLM(config)
                vocab_size = config.vocab_size
            
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✓ Loaded checkpoint with vocab_size={vocab_size}")
            return model, vocab_size
    
    # Model path doesn't exist
    raise ValueError(f"Model not found: {model_name_or_path}")


def build_utm_generator(
    batch_size: int,
    seq_length: int,
    memory_size: int,
    maximum_steps: int,
    tokenizer: str,
    maximum_program_length: int,
    seed: int,
):
    """Build UTM data generator."""
    rng = np.random.default_rng(seed=seed)
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler)
    tokenizer_enum = (
        utm_dg.Tokenizer.ASCII if tokenizer.lower() == "ascii"
        else utm_dg.Tokenizer.SEQ_POSITION
    )
    return utm_dg.UTMDataGenerator(
        batch_size=batch_size,
        seq_length=seq_length,
        rng=rng,
        utm=utm,
        memory_size=memory_size,
        maximum_steps=maximum_steps,
        tokenizer=tokenizer_enum,
        maximum_program_length=maximum_program_length,
    )


def build_ctw_generator(
    batch_size: int,
    seq_length: int,
    max_depth: int,
    seed: int,
):
    """Build CTW data generator."""
    rng = np.random.default_rng(seed=seed)
    return ctw_dg.CTWGenerator(
        batch_size=batch_size,
        seq_length=seq_length,
        rng=rng,
        max_depth=max_depth,
        with_contexts=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sequential finetuning: UTM → CTW"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lstm",
        help="HuggingFace model name, path to checkpoint, or 'lstm'/'transformer' for custom models",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["lstm", "transformer"],
        default="lstm",
        help="Model architecture (lstm has better length generalization)",
    )
    parser.add_argument(
        "--use_hf",
        action="store_true",
        help="Force HuggingFace model loading",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for LSTM/Transformer",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of layers",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Training stages
    parser.add_argument(
        "--stage",
        type=str,
        choices=["utm", "ctw", "all"],
        default="all",
        help="Training stage to run",
    )
    parser.add_argument("--utm_steps", type=int, default=10000)
    parser.add_argument("--ctw_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--utm_lr", type=float, default=None, help="Override LR for UTM stage")
    parser.add_argument("--ctw_lr", type=float, default=None, help="Override LR for CTW stage")
    
    # UTM data configuration
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--memory_size", type=int, default=10)
    parser.add_argument("--maximum_steps", type=int, default=100)
    parser.add_argument("--tokenizer", type=str, default="ascii", choices=["ascii", "seq_position"])
    parser.add_argument("--maximum_program_length", type=int, default=100)
    
    # CTW data configuration
    parser.add_argument("--ctw_max_depth", type=int, default=5)
    parser.add_argument("--ctw_batch_size", type=int, default=None, help="Override batch size for CTW")
    parser.add_argument("--ctw_seq_length", type=int, default=None, help="Override seq length for CTW")
    
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sequential")
    
    # Advanced options
    parser.add_argument("--use_dni", action="store_true", help="Use Decoupled Neural Interfaces")
    parser.add_argument("--dni_num_points", type=int, default=2, help="Number of DNI decoupling points")
    parser.add_argument("--dni_hidden_dim", type=int, default=None, help="DNI synthesizer hidden dim")
    parser.add_argument("--use_bitnet", action="store_true", help="Use BitNet quantization")
    parser.add_argument("--enable_mswag", action="store_true", help="Enable MultiSWAG")
    
    # SWAG-LoRA options
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (lower = fewer parameters)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--use_swag", action="store_true", help="Use SWAG for uncertainty quantification")
    parser.add_argument("--swag_start_epoch", type=int, default=10, help="Epoch to start SWAG")
    parser.add_argument("--swag_update_freq", type=int, default=100, help="SWAG update frequency")
    parser.add_argument("--swag_lr", type=float, default=1e-5, help="SWAG learning rate")
    
    args = parser.parse_args(argv)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load or create model
    try:
        model, hf_vocab_size = load_pretrained_model(
            args.model_name_or_path,
            args.device,
            use_hf=args.use_hf,
            architecture=args.architecture,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Creating custom {args.architecture} from scratch...")
        
        # Determine vocab size based on tokenizer
        vocab_size = 128 if args.tokenizer == "ascii" else 512
        
        if args.architecture == "lstm":
            print(f"Using LSTM (better length generalization, see https://arxiv.org/html/2401.14953v1)")
            config = LSTMConfig(
                vocab_size=vocab_size,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=0.2,
            )
            model = LSTMDecoderLM(config)
        else:
            config = TransformerConfig(
                vocab_size=vocab_size,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                num_heads=8,
                widening_factor=4,
            )
            model = TransformerDecoderLM(config)
        
        hf_vocab_size = None
    
    # Apply SWAG-LoRA if requested (before DNI for max efficiency)
    swag_model = None
    if args.use_lora or args.use_swag:
        model, swag_model = create_swag_lora_model(
            model=model,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_swag=args.use_swag,
            swag_start_epoch=args.swag_start_epoch,
            swag_lr=args.swag_lr,
            swag_update_freq=args.swag_update_freq,
        )
    
    # Apply DNI if requested
    if args.use_dni:
        print(f"[DNI] Applying Decoupled Neural Interfaces...")
        model = create_dni_model(
            base_model=model,
            use_dni=True,
            num_dni_points=args.dni_num_points,
            dni_hidden_dim=args.dni_hidden_dim,
            use_context=False,
        )
    
    # Create finetuner
    finetuner = SequentialFinetuner(
        model=model,
        device=args.device,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )
    
    finetuner.log(f"Sequential Finetuning Pipeline")
    finetuner.log(f"Model: {args.model_name_or_path}")
    finetuner.log(f"Architecture: {args.architecture}")
    finetuner.log(f"Device: {args.device}")
    finetuner.log(f"Output: {args.output_dir}")
    if args.architecture == "lstm":
        finetuner.log(f"Note: LSTMs generalize better to longer sequences (https://arxiv.org/html/2401.14953v1)")
    
    # Determine learning rates
    utm_lr = args.utm_lr if args.utm_lr is not None else args.lr
    ctw_lr = args.ctw_lr if args.ctw_lr is not None else args.lr * 0.5
    
    # Stage 1: UTM
    if args.stage in ["utm", "all"]:
        utm_generator = build_utm_generator(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            memory_size=args.memory_size,
            maximum_steps=args.maximum_steps,
            tokenizer=args.tokenizer,
            maximum_program_length=args.maximum_program_length,
            seed=args.seed,
        )
        
        finetuner.train_stage_utm(
            data_generator=utm_generator,
            steps=args.utm_steps,
            lr=utm_lr,
            save_every=args.save_every,
        )
    
    # Stage 2: CTW
    if args.stage in ["ctw", "all"]:
        ctw_batch_size = args.ctw_batch_size if args.ctw_batch_size is not None else args.batch_size
        ctw_seq_length = args.ctw_seq_length if args.ctw_seq_length is not None else args.seq_length
        
        ctw_generator = build_ctw_generator(
            batch_size=ctw_batch_size,
            seq_length=ctw_seq_length,
            max_depth=args.ctw_max_depth,
            seed=args.seed + 1,  # Different seed for CTW
        )
        
        finetuner.train_stage_ctw(
            data_generator=ctw_generator,
            steps=args.ctw_steps,
            lr=ctw_lr,
            save_every=args.save_every,
        )
    
    # Save final metrics
    finetuner.save_metrics()
    finetuner.log(f"\n{'='*60}")
    finetuner.log("✓ Sequential finetuning complete!")
    finetuner.log(f"{'='*60}")


if __name__ == "__main__":
    main()
