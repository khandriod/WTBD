#!/usr/bin/env python3
"""
Test script to verify new attention mechanisms work correctly
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to path
sys.path.insert(0, '.')

try:
    from ultralytics.nn.modules.conv import (
        CoordinateAttention, 
        CBAMv2, 
        SelectiveKernelAttention, 
        EfficientMultiScaleAttention
    )
    print("✅ Successfully imported all new attention mechanisms")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_attention_mechanism(attention_class, input_tensor, *args, **kwargs):
    """Test a specific attention mechanism"""
    try:
        # Create attention module
        attention = attention_class(*args, **kwargs)
        
        # Forward pass
        output = attention(input_tensor)
        
        # Check output shape
        assert output.shape == input_tensor.shape, f"Shape mismatch: input {input_tensor.shape} vs output {output.shape}"
        
        # Check if output is different from input (attention is applied)
        assert not torch.allclose(input_tensor, output), "Attention mechanism doesn't modify input"
        
        print(f"✅ {attention_class.__name__} test passed")
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in attention.parameters())}")
        
        return True
    except Exception as e:
        print(f"❌ {attention_class.__name__} test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Latest Attention Mechanisms for Wind Turbine Damage Detection")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    channels = 256
    height = 32
    width = 32
    
    # Create test input
    input_tensor = torch.randn(batch_size, channels, height, width)
    print(f"Test input shape: {input_tensor.shape}")
    print("-" * 60)
    
    # Test results
    results = []
    
    # Test 1: Coordinate Attention
    print("1. Testing Coordinate Attention (CA)")
    results.append(test_attention_mechanism(
        CoordinateAttention, input_tensor, channels, channels
    ))
    print()
    
    # Test 2: CBAM v2
    print("2. Testing CBAM v2")
    results.append(test_attention_mechanism(
        CBAMv2, input_tensor, channels
    ))
    print()
    
    # Test 3: Selective Kernel Attention
    print("3. Testing Selective Kernel Attention (SKA)")
    results.append(test_attention_mechanism(
        SelectiveKernelAttention, input_tensor, channels, channels
    ))
    print()
    
    # Test 4: Efficient Multi-Scale Attention
    print("4. Testing Efficient Multi-Scale Attention (EMA)")
    results.append(test_attention_mechanism(
        EfficientMultiScaleAttention, input_tensor, channels
    ))
    print()
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary:")
    mechanisms = ['Coordinate Attention', 'CBAM v2', 'Selective Kernel Attention', 'Efficient Multi-Scale Attention']
    
    for i, (mechanism, result) in enumerate(zip(mechanisms, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {i+1}. {mechanism}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All attention mechanisms are working correctly!")
        print("🚀 Ready for wind turbine damage detection training!")
    else:
        print("⚠️  Some attention mechanisms failed. Check implementation.")
        sys.exit(1)

def test_with_different_input_sizes():
    """Test attention mechanisms with different input sizes"""
    print("\n🔍 Testing with different input sizes (simulating multi-scale detection)")
    print("-" * 60)
    
    test_sizes = [
        (1, 128, 80, 80),   # P3 feature map
        (1, 256, 40, 40),   # P4 feature map  
        (1, 512, 20, 20),   # P5 feature map
    ]
    
    for i, (b, c, h, w) in enumerate(test_sizes):
        print(f"Testing P{i+3} feature map: ({b}, {c}, {h}, {w})")
        input_tensor = torch.randn(b, c, h, w)
        
        # Test Coordinate Attention (most important for cracks)
        try:
            ca = CoordinateAttention(c, c)
            output = ca(input_tensor)
            print(f"  ✅ Coordinate Attention: {input_tensor.shape} -> {output.shape}")
        except Exception as e:
            print(f"  ❌ Coordinate Attention failed: {e}")
            
        # Test EMA (for real-time processing)
        try:
            ema = EfficientMultiScaleAttention(c)
            output = ema(input_tensor)
            print(f"  ✅ EMA: {input_tensor.shape} -> {output.shape}")
        except Exception as e:
            print(f"  ❌ EMA failed: {e}")
        
        print()

if __name__ == "__main__":
    main()
    test_with_different_input_sizes()
    
    print("\n💡 Next steps:")
    print("1. Train models with: python train_wind_turbine_with_attention.py")
    print("2. Compare performance across different attention mechanisms")
    print("3. Test on real wind turbine damage images")
    print("4. Deploy best performing model for inspection drones")