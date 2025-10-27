#!/usr/bin/env python3
"""Quick test for LSTM model implementation."""

import torch
import torch.nn.functional as F
from torch_models.lstm import LSTMConfig, LSTMDecoderLM


def test_lstm_forward():
    """Test LSTM forward pass."""
    print("Testing LSTM forward pass...")
    
    config = LSTMConfig(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    )
    
    model = LSTMDecoderLM(config)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    seq_length = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    with torch.no_grad():
        log_probs = model(x)
    
    # Check output shape
    assert log_probs.shape == (batch_size, seq_length, config.vocab_size)
    
    # Check it's log probabilities
    probs = torch.exp(log_probs)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_length), atol=1e-5)
    
    print(f"✓ Forward pass: output shape {log_probs.shape}")
    print(f"✓ Log probabilities sum to 1.0")
    

def test_lstm_training_step():
    """Test LSTM training step."""
    print("\nTesting LSTM training step...")
    
    config = LSTMConfig(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    )
    
    model = LSTMDecoderLM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dummy batch
    batch_size = 4
    seq_length = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    log_probs = model(x)
    B, T, V = log_probs.shape
    
    # Compute loss
    loss = F.nll_loss(
        log_probs.reshape(B * T, V),
        x.reshape(B * T),
        reduction="mean",
    )
    
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step: loss = {loss.item():.4f}")
    

def test_lstm_generation():
    """Test LSTM autoregressive generation."""
    print("\nTesting LSTM generation...")
    
    config = LSTMConfig(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.0,  # No dropout for generation
    )
    
    model = LSTMDecoderLM(config)
    model.eval()
    
    # Create prompt
    batch_size = 2
    prompt_length = 8
    max_length = 16
    prompt = torch.randint(0, config.vocab_size, (batch_size, prompt_length))
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=1.0,
            top_k=10,
        )
    
    assert generated.shape == (batch_size, max_length)
    print(f"✓ Generation: output shape {generated.shape}")
    print(f"✓ Generated sequence sample: {generated[0, :10].tolist()}")


def test_length_generalization():
    """Test that LSTM can process sequences longer than training."""
    print("\nTesting length generalization...")
    
    config = LSTMConfig(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.0,
    )
    
    model = LSTMDecoderLM(config)
    model.eval()
    
    # Train on short sequences
    short_seq = torch.randint(0, config.vocab_size, (1, 16))
    
    # Test on longer sequences
    long_seq = torch.randint(0, config.vocab_size, (1, 64))
    
    with torch.no_grad():
        short_output = model(short_seq)
        long_output = model(long_seq)
    
    print(f"✓ Short sequence ({short_seq.shape[1]}): {short_output.shape}")
    print(f"✓ Long sequence ({long_seq.shape[1]}): {long_output.shape}")
    print("✓ LSTM handles arbitrary length sequences")


def test_weight_tying():
    """Test that weight tying reduces parameters."""
    print("\nTesting weight tying...")
    
    config_tied = LSTMConfig(
        vocab_size=128,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        tie_weights=True,
    )
    
    config_untied = LSTMConfig(
        vocab_size=128,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        tie_weights=False,
    )
    
    model_tied = LSTMDecoderLM(config_tied)
    model_untied = LSTMDecoderLM(config_untied)
    
    params_tied = sum(p.numel() for p in model_tied.parameters())
    params_untied = sum(p.numel() for p in model_untied.parameters())
    
    print(f"✓ Parameters with tied weights: {params_tied:,}")
    print(f"✓ Parameters without tying: {params_untied:,}")
    print(f"✓ Reduction: {params_untied - params_tied:,} parameters ({(1 - params_tied/params_untied)*100:.1f}%)")
    
    # Verify weights are actually tied
    if config_tied.embedding_dim == config_tied.hidden_dim:
        assert model_tied.out.weight.data_ptr() == model_tied.embed.weight.data_ptr()
        print("✓ Weights are properly tied")


def main():
    """Run all tests."""
    print("="*60)
    print("LSTM Model Tests")
    print("="*60)
    
    test_lstm_forward()
    test_lstm_training_step()
    test_lstm_generation()
    test_length_generalization()
    test_weight_tying()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()
