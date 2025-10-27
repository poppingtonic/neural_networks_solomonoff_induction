#!/usr/bin/env python3
"""
SWAG-LoRA Integration for Memory-Efficient Sequential Fine-tuning.

Combines:
- LoRA (Low-Rank Adaptation): Parameter-efficient fine-tuning
- SWAG (Stochastic Weight Averaging-Gaussian): Uncertainty quantification

References:
- SWAG-LoRA: https://github.com/fortuinlab/swag-lora
- LoRA: https://arxiv.org/abs/2106.09685
- SWAG: https://arxiv.org/abs/1902.02476
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import torch
import torch.nn as nn


def check_swag_lora_available():
    """Check if SWAG-LoRA is available."""
    try:
        import swag_lora
        return True
    except ImportError:
        return False


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list[str]] = None,
    bias: str = "none",
) -> nn.Module:
    """
    Apply LoRA to a model for parameter-efficient fine-tuning.
    
    Args:
        model: Base model to apply LoRA to
        r: LoRA rank (lower = fewer parameters)
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout for LoRA layers
        target_modules: Which modules to apply LoRA to (None = auto-detect)
        bias: How to handle biases ("none", "all", "lora_only")
    
    Returns:
        Model with LoRA applied
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Auto-detect target modules if not specified
        if target_modules is None:
            # Common patterns for transformers
            target_modules = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Extract layer type (q_proj, k_proj, v_proj, etc.)
                    layer_name = name.split('.')[-1]
                    if any(key in layer_name for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                           'gate_proj', 'up_proj', 'down_proj',
                                                           'qkv', 'dense', 'fc']):
                        if layer_name not in target_modules:
                            target_modules.append(layer_name)
            
            # Fallback to common defaults
            if not target_modules:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        print(f"Applying LoRA to modules: {target_modules}")
        print(f"LoRA rank: {r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
        
        # Create LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        
        # Print parameter counts
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ LoRA applied successfully")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Memory reduction: ~{(1 - trainable_params/total_params)*100:.1f}%")
        
        return model
        
    except ImportError:
        print("Error: peft library not installed")
        print("Install with: pip install peft")
        raise


def create_swag_lora_model(
    model: nn.Module,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    use_swag: bool = True,
    swag_start_epoch: int = 10,
    swag_lr: float = 1e-5,
    swag_update_freq: int = 100,
    **kwargs
) -> tuple[nn.Module, Optional[Any]]:
    """
    Create a model with LoRA and/or SWAG.
    
    Args:
        model: Base model
        use_lora: Whether to apply LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_swag: Whether to use SWAG
        swag_start_epoch: When to start SWAG sampling
        swag_lr: Learning rate for SWAG
        swag_update_freq: How often to update SWAG statistics
    
    Returns:
        (model, swag_model) tuple
    """
    swag_model = None
    
    # Apply LoRA first (reduces parameters)
    if use_lora:
        print("\n" + "="*60)
        print("Applying LoRA for Parameter-Efficient Fine-tuning")
        print("="*60)
        model = apply_lora_to_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=kwargs.get('lora_target_modules'),
            bias=kwargs.get('lora_bias', 'none'),
        )
    
    # Setup SWAG if requested
    if use_swag:
        if check_swag_lora_available():
            print("\n" + "="*60)
            print("Initializing SWAG for Uncertainty Quantification")
            print("="*60)
            try:
                from swag_lora import SWAG
                
                swag_model = SWAG(
                    base_model=model,
                    no_cov_mat=False,
                    max_num_models=20,
                )
                
                print(f"✓ SWAG initialized")
                print(f"  Start epoch: {swag_start_epoch}")
                print(f"  Update frequency: {swag_update_freq}")
                print(f"  Learning rate: {swag_lr}")
                
            except Exception as e:
                print(f"Warning: Could not initialize SWAG: {e}")
                print("Continuing without SWAG...")
        else:
            print("\nWarning: swag-lora not installed")
            print("Install with: pip install git+https://github.com/fortuinlab/swag-lora.git")
            print("or use local: pip install -e ~/src/swag-lora")
            print("Continuing without SWAG...")
    
    return model, swag_model


def update_swag_model(
    swag_model: Any,
    base_model: nn.Module,
    current_step: int,
    swag_start_step: int,
    swag_update_freq: int,
) -> bool:
    """
    Update SWAG model statistics if conditions are met.
    
    Args:
        swag_model: SWAG model instance
        base_model: Current base model
        current_step: Current training step
        swag_start_step: Step to start SWAG updates
        swag_update_freq: Frequency of updates
    
    Returns:
        True if SWAG was updated
    """
    if swag_model is None:
        return False
    
    if current_step < swag_start_step:
        return False
    
    if (current_step - swag_start_step) % swag_update_freq == 0:
        try:
            swag_model.collect_model(base_model)
            return True
        except Exception as e:
            print(f"Warning: SWAG update failed: {e}")
            return False
    
    return False


def sample_swag_predictions(
    swag_model: Any,
    input_data: torch.Tensor,
    num_samples: int = 10,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Sample predictions from SWAG posterior.
    
    Args:
        swag_model: SWAG model instance
        input_data: Input tensor
        num_samples: Number of samples to draw
        scale: Scaling factor for covariance
    
    Returns:
        Tensor of shape (num_samples, batch_size, seq_len, vocab_size)
    """
    if swag_model is None:
        raise ValueError("SWAG model is None")
    
    predictions = []
    
    for _ in range(num_samples):
        swag_model.sample(scale=scale)
        with torch.no_grad():
            pred = swag_model(input_data)
        predictions.append(pred)
    
    return torch.stack(predictions)


class LoRAConfig:
    """Configuration for LoRA application."""
    
    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[list[str]] = None,
        bias: str = "none",
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.bias = bias


class SWAGConfig:
    """Configuration for SWAG."""
    
    def __init__(
        self,
        start_epoch: int = 10,
        update_freq: int = 100,
        lr: float = 1e-5,
        no_cov_mat: bool = False,
        max_num_models: int = 20,
    ):
        self.start_epoch = start_epoch
        self.update_freq = update_freq
        self.lr = lr
        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models


def get_lora_memory_savings(model: nn.Module, lora_r: int = 8) -> Dict[str, float]:
    """
    Estimate memory savings from LoRA.
    
    Args:
        model: Base model
        lora_r: LoRA rank
    
    Returns:
        Dictionary with memory statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Rough estimate: LoRA adds 2*r*d parameters per adapted layer
    # where d is the layer dimension
    # For a transformer with attention, this is typically 0.1-1% of original params
    estimated_lora_params = total_params * (lora_r / 100)  # Very rough estimate
    
    return {
        'original_params': total_params,
        'estimated_lora_params': estimated_lora_params,
        'reduction_ratio': estimated_lora_params / total_params,
        'memory_savings_percent': (1 - estimated_lora_params / total_params) * 100,
    }
