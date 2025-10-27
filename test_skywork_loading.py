#!/usr/bin/env python3
"""Test script to verify Skywork model can be loaded and used."""

import torch
from torch_training.train_sequential_finetune import load_pretrained_model


def test_skywork_loading():
    """Test loading Skywork model from HuggingFace."""
    print("="*60)
    print("Testing Skywork Model Loading")
    print("="*60)
    
    model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    print("\nAttempting to load model from HuggingFace...")
    
    try:
        model, vocab_size = load_pretrained_model(
            model_name_or_path=model_name,
            device=device,
            use_hf=True,
        )
        
        print(f"✓ Model loaded successfully!")
        print(f"✓ Vocabulary size: {vocab_size}")
        print(f"✓ Model type: {type(model).__name__}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        seq_length = 16
        
        # Create dummy input tokens
        dummy_input = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_length))
        dummy_input = dummy_input.to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful!")
        print(f"✓ Input shape: {dummy_input.shape}")
        print(f"✓ Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\nYou can now run sequential fine-tuning:")
        print(f"  python torch_training/train_sequential_finetune.py \\")
        print(f"    --model_name_or_path '{model_name}' \\")
        print(f"    --use_hf \\")
        print(f"    --stage all \\")
        print(f"    --device {device}")
        print()
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nPlease install transformers library:")
        print("  pip install transformers")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("3. Verify model name is correct")
        return False


if __name__ == "__main__":
    success = test_skywork_loading()
    exit(0 if success else 1)
