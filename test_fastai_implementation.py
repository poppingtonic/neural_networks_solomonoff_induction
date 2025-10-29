#!/usr/bin/env python3
"""Test script for FastAI LSTM implementation.

Runs basic smoke tests to verify the implementation works correctly.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Test imports
try:
    from torch_training.fastai_dataloaders import (
        UTMFastAIDataset,
        CTWFastAIDataset,
        create_utm_dataloaders,
        create_ctw_dataloaders,
    )
    from torch_training.fastai_callbacks import (
        PerplexityMetric,
        GradNormCallback,
        SequentialStageCallback,
        ModelCheckpointCallback,
        LSTMLogProbLoss,
        TransformerLogitsLoss,
    )
    from torch_models.lstm import LSTMConfig, LSTMDecoderLM
    from data import utm_data_generator as utm_dg
    from data import ctw_data_generator as ctw_dg
    from data import utms as utms_lib
    
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_utm_dataset():
    """Test UTM dataset creation."""
    print("\n1. Testing UTM dataset...")
    
    rng = np.random.default_rng(seed=42)
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler)
    
    generator = utm_dg.UTMDataGenerator(
        batch_size=4,
        seq_length=32,
        rng=rng,
        utm=utm,
        memory_size=5,
        maximum_steps=50,
        tokenizer=utm_dg.Tokenizer.ASCII,
        maximum_program_length=50,
    )
    
    dataset = UTMFastAIDataset(generator, batches_per_epoch=2)
    
    for i, (x, y) in enumerate(dataset):
        assert x.shape[0] == 4, f"Expected batch size 4, got {x.shape[0]}"
        assert x.shape[1] == 32, f"Expected seq length 32, got {x.shape[1]}"
        assert y.shape == x.shape, f"Shape mismatch: x={x.shape}, y={y.shape}"
        if i >= 1:  # Only test 2 batches
            break
    
    print("   ✓ UTM dataset working correctly")


def test_ctw_dataset():
    """Test CTW dataset creation."""
    print("\n2. Testing CTW dataset...")
    
    rng = np.random.default_rng(seed=42)
    generator = ctw_dg.CTWGenerator(
        batch_size=4,
        seq_length=32,
        rng=rng,
        max_depth=3,
        with_contexts=False,
    )
    
    dataset = CTWFastAIDataset(generator, batches_per_epoch=2)
    
    for i, (x, y) in enumerate(dataset):
        assert x.shape[0] == 4, f"Expected batch size 4, got {x.shape[0]}"
        assert x.shape[1] == 32, f"Expected seq length 32, got {x.shape[1]}"
        assert y.shape == x.shape, f"Shape mismatch: x={x.shape}, y={y.shape}"
        if i >= 1:
            break
    
    print("   ✓ CTW dataset working correctly")


def test_lstm_model():
    """Test LSTM model forward pass."""
    print("\n3. Testing LSTM model...")
    
    config = LSTMConfig(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    )
    model = LSTMDecoderLM(config)
    
    # Test forward pass
    x = torch.randint(0, 128, (2, 16))  # batch=2, seq_len=16
    output = model(x)
    
    assert output.shape == (2, 16, 128), f"Expected (2, 16, 128), got {output.shape}"
    
    # Check output is log probs (should be negative)
    assert (output <= 0).all(), "LSTM should output log probabilities (<=0)"
    
    print("   ✓ LSTM model working correctly")


def test_loss_functions():
    """Test custom loss functions."""
    print("\n4. Testing loss functions...")
    
    # Test LSTM loss (log probs)
    lstm_loss = LSTMLogProbLoss()
    log_probs = torch.randn(2, 10, 128).log_softmax(dim=-1)
    targets = torch.randint(0, 128, (2, 10))
    loss = lstm_loss(log_probs, targets)
    
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss > 0, "Loss should be positive"
    
    print("   ✓ LSTM loss working correctly")
    
    # Test Transformer loss (logits)
    transformer_loss = TransformerLogitsLoss()
    logits = torch.randn(2, 10, 128)
    loss = transformer_loss(logits, targets)
    
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss > 0, "Loss should be positive"
    
    print("   ✓ Transformer loss working correctly")


def test_metrics():
    """Test custom metrics."""
    print("\n5. Testing metrics...")
    
    metric = PerplexityMetric()
    metric.reset()
    
    # Simulate some losses
    class FakeLearner:
        def __init__(self, loss):
            self.loss = loss
    
    for loss_val in [2.0, 1.5, 1.0]:
        learn = FakeLearner(torch.tensor(loss_val))
        metric.accumulate(learn)
    
    ppl = metric.value
    expected_ppl = np.exp(np.mean([2.0, 1.5, 1.0]))
    
    assert abs(ppl - expected_ppl) < 0.01, f"Expected ppl≈{expected_ppl}, got {ppl}"
    
    print(f"   ✓ Perplexity metric working correctly (value: {ppl:.2f})")


def test_dataloaders():
    """Test dataloader creation."""
    print("\n6. Testing dataloader creation...")
    
    # Create UTM generator
    rng = np.random.default_rng(seed=42)
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler)
    
    utm_generator = utm_dg.UTMDataGenerator(
        batch_size=4,
        seq_length=32,
        rng=rng,
        utm=utm,
        memory_size=5,
        maximum_steps=50,
        tokenizer=utm_dg.Tokenizer.ASCII,
        maximum_program_length=50,
    )
    
    dls = create_utm_dataloaders(utm_generator, batches_per_epoch=5, device="cpu")
    
    assert dls.train is not None, "Train dataloader missing"
    assert dls.valid is not None, "Valid dataloader missing"
    
    print("   ✓ DataLoaders created successfully")


def main():
    """Run all tests."""
    print("="*60)
    print("FastAI LSTM Implementation - Smoke Tests")
    print("="*60)
    
    try:
        test_utm_dataset()
        test_ctw_dataset()
        test_lstm_model()
        test_loss_functions()
        test_metrics()
        test_dataloaders()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        print("\nFastAI LSTM implementation is working correctly.")
        print("You can now run:")
        print("  - python examples/fastai_lstm_example.py")
        print("  - python torch_training/train_sequential_finetune_fastai.py")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
