#!/usr/bin/env python3
"""
Simple test script for ResNet32 CIFAR models
"""

import torch
import torch.nn.functional as F
from imb_cll.utils.resnet32_cifar100 import resnet32_cifar10, resnet32_cifar100

def test_resnet32():
    print("Testing ResNet32 models...")
    
    # Test device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test CIFAR-10 model
    print("\n--- Testing CIFAR-10 ResNet32 ---")
    try:
        model_cifar10 = resnet32_cifar10(pretrained=True, device=device)
        print(f"Model created successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model_cifar10.parameters()):,}")
        
        # Test forward pass
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
        
        model_cifar10.eval()
        with torch.no_grad():
            output = model_cifar10(dummy_input)
            probabilities = F.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Predictions: {predictions.cpu().numpy()}")
        print(f"Max probabilities: {probabilities.max(dim=1)[0].cpu().numpy()}")
        
    except Exception as e:
        print(f"CIFAR-10 model failed: {e}")
    
    # Test CIFAR-100 model
    print("\n--- Testing CIFAR-100 ResNet32 ---")
    try:
        model_cifar100 = resnet32_cifar100(pretrained=True, device=device)
        print(f"Model created successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model_cifar100.parameters()):,}")
        
        # Test forward pass
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
        
        model_cifar100.eval()
        with torch.no_grad():
            output = model_cifar100(dummy_input)
            probabilities = F.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Predictions: {predictions.cpu().numpy()}")
        print(f"Max probabilities: {probabilities.max(dim=1)[0].cpu().numpy()}")
        
    except Exception as e:
        print(f"CIFAR-100 model failed: {e}")

    print("\n--- Test completed ---")

if __name__ == "__main__":
    test_resnet32()
