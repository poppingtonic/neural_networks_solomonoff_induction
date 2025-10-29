#!/usr/bin/env python3
"""FastAI callbacks and utilities for LSTM sequential finetuning.

Provides custom callbacks for:
- Sequential stage tracking (UTM -> CTW)
- Gradient norm logging
- Custom metrics (perplexity)
- Model checkpointing
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastai.callback.core import Callback
from fastai.learner import Learner
from fastai.metrics import Metric

from utm_dataset import IGNORE_INDEX


class PerplexityMetric(Metric):
    """Perplexity metric for language modeling."""
    
    def __init__(self):
        self.total_loss = 0.0
        self.count = 0
    
    def reset(self):
        """Reset metric state."""
        self.total_loss = 0.0
        self.count = 0
    
    def accumulate(self, learn: Learner):
        """Accumulate metric over batch."""
        if learn.loss is not None:
            self.total_loss += float(learn.loss.detach().cpu().item())
            self.count += 1
    
    @property
    def value(self) -> float:
        """Compute perplexity."""
        if self.count == 0:
            return 0.0
        avg_loss = self.total_loss / self.count
        return math.exp(avg_loss)
    
    @property
    def name(self) -> str:
        return "perplexity"


class GradNormCallback(Callback):
    """Callback to log gradient norms during training."""
    
    def __init__(self):
        super().__init__()
        self.grad_norm = 0.0
    
    def after_backward(self):
        """Compute gradient norm after backward pass."""
        grad_norm = 0.0
        for p in self.learn.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += float(param_norm.item() ** 2)
        self.grad_norm = math.sqrt(grad_norm)
    
    def after_step(self):
        """Log gradient norm after optimizer step."""
        if self.learn.training:
            self.learn.smooth_loss = getattr(self.learn, 'smooth_loss', self.learn.loss)


class SequentialStageCallback(Callback):
    """Callback to track sequential finetuning stages (UTM -> CTW)."""
    
    def __init__(self, stage: str = "utm"):
        """Initialize stage tracker.
        
        Args:
            stage: Current training stage ('utm' or 'ctw')
        """
        super().__init__()
        self.stage = stage
    
    def before_epoch(self):
        """Log stage information before each epoch."""
        if self.epoch == 0:
            print(f"\n{'='*60}")
            print(f"Stage: {self.stage.upper()}")
            print(f"{'='*60}")


class ModelCheckpointCallback(Callback):
    """Enhanced checkpoint callback with stage tracking."""
    
    def __init__(
        self,
        output_dir: str,
        stage: str,
        save_every: int = 1000,
        monitor: str = 'valid_loss',
        mode: str = 'min',
    ):
        """Initialize checkpoint callback.
        
        Args:
            output_dir: Directory to save checkpoints
            stage: Current stage ('utm' or 'ctw')
            save_every: Save checkpoint every N steps
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for metric monitoring
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.save_every = save_every
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.step = 0
    
    def after_batch(self):
        """Save checkpoint periodically."""
        if not self.learn.training:
            return
        
        self.step += 1
        if self.save_every > 0 and self.step % self.save_every == 0:
            checkpoint_path = self.output_dir / f"stage_{self.stage}_step_{self.step}.pth"
            self._save_checkpoint(checkpoint_path)
    
    def after_epoch(self):
        """Save checkpoint after each epoch and track best model."""
        # Check if this is the best model
        current_metric = getattr(self.learn, self.monitor, None)
        if current_metric is not None:
            is_better = (
                (self.mode == 'min' and current_metric < self.best_metric) or
                (self.mode == 'max' and current_metric > self.best_metric)
            )
            if is_better:
                self.best_metric = current_metric
                best_path = self.output_dir / f"stage_{self.stage}_best.pth"
                self._save_checkpoint(best_path)
                print(f"✓ New best model saved: {self.monitor}={current_metric:.6f}")
        
        # Save epoch checkpoint
        epoch_path = self.output_dir / f"stage_{self.stage}_epoch_{self.epoch}.pth"
        self._save_checkpoint(epoch_path)
    
    def _save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.learn.model.state_dict(),
            "optimizer_state_dict": self.learn.opt.state_dict(),
            "stage": self.stage,
            "step": self.step,
            "epoch": self.epoch,
        }
        torch.save(checkpoint, path)


class LSTMLogProbLoss:
    """Custom loss function for LSTM models that output log probabilities."""
    
    def __init__(self):
        self.name = "lstm_log_prob_loss"
    
    def __call__(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NLL loss from log probabilities.
        
        Args:
            log_probs: Log probabilities of shape (B, T, V)
            targets: Target tokens of shape (B, T)
        
        Returns:
            Loss tensor
        """
        B, T, V = log_probs.shape
        
        # Reshape for NLL loss
        loss = F.nll_loss(
            log_probs.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        
        return loss
    
    def decodes(self, x):
        """Decode predictions (required by fastai)."""
        return x.argmax(dim=-1)


class TransformerLogitsLoss:
    """Loss function for transformer models that output logits."""
    
    def __init__(self):
        self.name = "transformer_logits_loss"
    
    def __call__(
        self,
        outputs,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NLL loss from model outputs.
        
        Args:
            outputs: Model outputs (logits or log_probs)
            targets: Target tokens of shape (B, T)
        
        Returns:
            Loss tensor
        """
        # Handle both HuggingFace and custom model outputs
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        
        loss = F.nll_loss(
            log_probs.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        
        return loss
    
    def decodes(self, x):
        """Decode predictions (required by fastai)."""
        if hasattr(x, 'logits'):
            return x.logits.argmax(dim=-1)
        return x.argmax(dim=-1)
