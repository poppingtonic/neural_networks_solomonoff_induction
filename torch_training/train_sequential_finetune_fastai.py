#!/usr/bin/env python3
"""FastAI-based Sequential Finetuning Pipeline for Universal Predictors.

Stage 1: Finetune on UTM data
Stage 2: Finetune on CTW data

This implementation uses fastai primitives for:
- DataLoaders
- Learner with callbacks
- One-cycle training policy
- Progressive learning rate scheduling
- Advanced metrics and logging

Supports:
- Custom LSTM (default - better length generalization, https://arxiv.org/html/2401.14953v1)
- Custom transformers
- HuggingFace pretrained models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from fastai.callback.schedule import fit_one_cycle
from fastai.learner import Learner
from fastai.optimizer import Adam

# Local imports
from data import utm_data_generator as utm_dg
from data import ctw_data_generator as ctw_dg
from data import utms as utms_lib
from torch_models.transformer import TransformerConfig, TransformerDecoderLM
from torch_models.lstm import LSTMConfig, LSTMDecoderLM
from fastai_dataloaders import create_utm_dataloaders, create_ctw_dataloaders
from fastai_callbacks import (
    PerplexityMetric,
    GradNormCallback,
    SequentialStageCallback,
    ModelCheckpointCallback,
    LSTMLogProbLoss,
    TransformerLogitsLoss,
)


class SequentialFastAIFinetuner:
    """FastAI-based sequential finetuning manager."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        output_dir: str,
        device: str,
        is_lstm: bool = True,
    ):
        """Initialize finetuner.
        
        Args:
            model: PyTorch model to train
            output_dir: Output directory for checkpoints and logs
            device: Device to use ('cuda' or 'cpu')
            is_lstm: Whether model outputs log probs (True) or logits (False)
        """
        self.model = model.to(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.is_lstm = is_lstm
        
        # Select appropriate loss function
        self.loss_func = LSTMLogProbLoss() if is_lstm else TransformerLogitsLoss()
        
        # Initialize logging
        self.log_file = self.output_dir / "train.log"
        self.metrics_history = {"utm": [], "ctw": []}
    
    def log(self, message: str):
        """Log to both console and file."""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
    
    def train_stage_utm(
        self,
        data_generator,
        epochs: int,
        lr: float,
        batches_per_epoch: int = 100,
        save_every: int = 1000,
    ):
        """Train Stage 1: UTM data using fastai.
        
        Args:
            data_generator: UTM data generator
            epochs: Number of epochs to train
            lr: Maximum learning rate for one-cycle policy
            batches_per_epoch: Batches per epoch
            save_every: Save checkpoint every N steps
        """
        self.log(f"\n{'='*60}")
        self.log(f"STAGE 1: UTM Finetuning ({epochs} epochs)")
        self.log(f"{'='*60}")
        
        # Create dataloaders
        dls = create_utm_dataloaders(
            data_generator,
            batches_per_epoch=batches_per_epoch,
            device=self.device,
        )
        
        # Create learner
        learn = Learner(
            dls=dls,
            model=self.model,
            loss_func=self.loss_func,
            opt_func=Adam,
            metrics=[PerplexityMetric()],
        )
        
        # Add callbacks
        cbs = [
            SequentialStageCallback(stage="utm"),
            GradNormCallback(),
            ModelCheckpointCallback(
                output_dir=str(self.output_dir),
                stage="utm",
                save_every=save_every,
            ),
        ]
        
        # Train with one-cycle policy
        learn.fit_one_cycle(
            n_epoch=epochs,
            lr_max=lr,
            cbs=cbs,
        )
        
        # Save final checkpoint
        final_path = self.output_dir / "stage1_utm_final.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "stage": "utm",
            "epochs": epochs,
        }, final_path)
        self.log(f"✓ Stage 1 complete. Saved to {final_path}")
        
        return learn
    
    def train_stage_ctw(
        self,
        data_generator,
        epochs: int,
        lr: float,
        batches_per_epoch: int = 100,
        save_every: int = 1000,
    ):
        """Train Stage 2: CTW data using fastai.
        
        Args:
            data_generator: CTW data generator
            epochs: Number of epochs to train
            lr: Maximum learning rate for one-cycle policy
            batches_per_epoch: Batches per epoch
            save_every: Save checkpoint every N steps
        """
        self.log(f"\n{'='*60}")
        self.log(f"STAGE 2: CTW Finetuning ({epochs} epochs)")
        self.log(f"{'='*60}")
        
        # Create dataloaders
        dls = create_ctw_dataloaders(
            data_generator,
            batches_per_epoch=batches_per_epoch,
            device=self.device,
        )
        
        # Create learner
        learn = Learner(
            dls=dls,
            model=self.model,
            loss_func=self.loss_func,
            opt_func=Adam,
            metrics=[PerplexityMetric()],
        )
        
        # Add callbacks
        cbs = [
            SequentialStageCallback(stage="ctw"),
            GradNormCallback(),
            ModelCheckpointCallback(
                output_dir=str(self.output_dir),
                stage="ctw",
                save_every=save_every,
            ),
        ]
        
        # Train with one-cycle policy
        learn.fit_one_cycle(
            n_epoch=epochs,
            lr_max=lr,
            cbs=cbs,
        )
        
        # Save final checkpoint
        final_path = self.output_dir / "stage2_ctw_final.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "stage": "ctw",
            "epochs": epochs,
        }, final_path)
        self.log(f"✓ Stage 2 complete. Saved to {final_path}")
        
        return learn
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        self.log(f"Metrics saved to {metrics_path}")


def load_pretrained_model(
    model_name_or_path: str,
    device: str,
    use_hf: bool = False,
    architecture: str = "lstm",
) -> Tuple[torch.nn.Module, bool]:
    """Load pretrained model from HuggingFace or local checkpoint.
    
    Args:
        model_name_or_path: Path or HF model name
        device: Device to load model on
        use_hf: Force HuggingFace loading
        architecture: Model architecture ('lstm' or 'transformer')
    
    Returns:
        model: Loaded model
        is_lstm: Whether model outputs log probs (True) or logits (False)
    """
    if use_hf or "/" in model_name_or_path:
        # Load from HuggingFace
        try:
            from transformers import AutoModelForCausalLM
            
            print(f"Loading HuggingFace model: {model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            print(f"✓ Loaded HuggingFace model")
            return model, False  # HF models output logits
        except Exception as e:
            print(f"Failed to load HuggingFace model: {e}")
            print("Falling back to custom model...")
    
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
        description="FastAI Sequential Finetuning: UTM → CTW"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lstm",
        help="HuggingFace model name or 'lstm'/'transformer' for custom models",
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
    parser.add_argument("--utm_epochs", type=int, default=10)
    parser.add_argument("--ctw_epochs", type=int, default=5)
    parser.add_argument("--batches_per_epoch", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Max learning rate for one-cycle")
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
    parser.add_argument("--ctw_batch_size", type=int, default=None)
    parser.add_argument("--ctw_seq_length", type=int, default=None)
    
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sequential_fastai")
    
    args = parser.parse_args(argv)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load or create model
    try:
        model, is_lstm = load_pretrained_model(
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
            is_lstm = True
        else:
            config = TransformerConfig(
                vocab_size=vocab_size,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                num_heads=8,
                widening_factor=4,
            )
            model = TransformerDecoderLM(config)
            is_lstm = False
    
    # Create finetuner
    finetuner = SequentialFastAIFinetuner(
        model=model,
        output_dir=args.output_dir,
        device=args.device,
        is_lstm=is_lstm,
    )
    
    finetuner.log(f"FastAI Sequential Finetuning Pipeline")
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
            epochs=args.utm_epochs,
            lr=utm_lr,
            batches_per_epoch=args.batches_per_epoch,
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
            seed=args.seed + 1,
        )
        
        finetuner.train_stage_ctw(
            data_generator=ctw_generator,
            epochs=args.ctw_epochs,
            lr=ctw_lr,
            batches_per_epoch=args.batches_per_epoch,
            save_every=args.save_every,
        )
    
    # Save final metrics
    finetuner.save_metrics()
    finetuner.log(f"\n{'='*60}")
    finetuner.log("✓ FastAI Sequential finetuning complete!")
    finetuner.log(f"{'='*60}")


if __name__ == "__main__":
    main()
