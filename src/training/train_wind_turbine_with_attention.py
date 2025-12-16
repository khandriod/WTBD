#!/usr/bin/env python3
"""
Wind Turbine Damage Detection with Latest Attention Mechanisms

This script demonstrates training YOLOv8 with the latest attention mechanisms
specifically designed for wind turbine blade damage detection.

New Attention Mechanisms:
1. Coordinate Attention (CA) - Perfect for detecting linear cracks and edge erosion
2. CBAM v2 - Enhanced spatial attention for surface damage patterns
3. Selective Kernel Attention (SKA) - Multi-scale damage detection
4. Efficient Multi-Scale Attention (EMA) - Real-time inspection systems

Usage:
    python train_wind_turbine_with_attention.py --model yolov8_CA --data ./GRAZPEDWRI-DX/data/data.yaml
    python train_wind_turbine_with_attention.py --model yolov8_CBAMv2 --data ./GRAZPEDWRI-DX/data/data.yaml
    python train_wind_turbine_with_attention.py --model yolov8_SKA --data ./GRAZPEDWRI-DX/data/data.yaml
    python train_wind_turbine_with_attention.py --model yolov8_EMA --data ./GRAZPEDWRI-DX/data/data.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 with Latest Attention Mechanisms for Wind Turbine Damage Detection')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['yolov8_CA', 'yolov8_CBAMv2', 'yolov8_SKA', 'yolov8_EMA'],
                       help='Model configuration with attention mechanism')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, 
                       help='Image size for training')
    parser.add_argument('--device', type=str, default='0', 
                       help='Device to use (0, 1, 2, etc. or cpu)')
    parser.add_argument('--project', type=str, default='wind_turbine_attention', 
                       help='Project name for saving results')
    parser.add_argument('--name', type=str, default=None, 
                       help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Set experiment name based on model if not provided
    if args.name is None:
        args.name = f"{args.model}_experiment"
    
    # Model configuration paths
    model_configs = {
        'yolov8_CA': './ultralytics/cfg/models/v8/yolov8_CA.yaml',
        'yolov8_CBAMv2': './ultralytics/cfg/models/v8/yolov8_CBAMv2.yaml',
        'yolov8_SKA': './ultralytics/cfg/models/v8/yolov8_SKA.yaml',
        'yolov8_EMA': './ultralytics/cfg/models/v8/yolov8_EMA.yaml'
    }
    
    # Verify files exist
    model_config_path = model_configs[args.model]
    if not Path(model_config_path).exists():
        print(f"Error: Model configuration file not found: {model_config_path}")
        sys.exit(1)
    
    if not Path(args.data).exists():
        print(f"Error: Dataset configuration file not found: {args.data}")
        sys.exit(1)
    
    # Print configuration
    print(f"🚀 Starting Wind Turbine Damage Detection Training")
    print(f"📊 Model: {args.model}")
    print(f"📁 Dataset: {args.data}")
    print(f"🔧 Configuration: {model_config_path}")
    print(f"⚙️  Device: {args.device}")
    print(f"🎯 Target Classes: Crack, Leading Edge Erosion, Lightning Strike, Surface Damage, Surface Dust")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📏 Image Size: {args.img_size}")
    print("-" * 60)
    
    # Initialize model
    try:
        model = YOLO(model_config_path)
        print(f"✅ Model initialized successfully: {args.model}")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        sys.exit(1)
    
    # Set up proper save directory in current working directory
    current_dir = os.getcwd()
    project_dir = os.path.join(current_dir, args.project)
    
    # Start training
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=args.device,
            project=project_dir,  # Use absolute path to current directory
            name=args.name,
            resume=args.resume,
            save=True,
            val=True,
            plots=True,
            verbose=True,
            exist_ok=True  # Allow overwriting existing directories
        )
        
        print(f"✅ Training completed successfully!")
        print(f"📊 Results saved to: {results.save_dir}")
        
        # Print performance summary
        if hasattr(results, 'results_dict'):
            print(f"📈 Final mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"📈 Final mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


def compare_attention_mechanisms():
    """
    Helper function to compare different attention mechanisms
    """
    print("🔍 Attention Mechanisms Comparison for Wind Turbine Damage Detection:")
    print("\n1. Coordinate Attention (CA):")
    print("   - Best for: Linear cracks, edge erosion patterns")
    print("   - Advantages: Lightweight, captures spatial relationships")
    print("   - Use case: Detecting crack propagation paths")
    
    print("\n2. CBAM v2:")
    print("   - Best for: Surface damage patterns, texture anomalies")
    print("   - Advantages: Improved gradient flow, better feature extraction")
    print("   - Use case: Identifying surface roughness changes")
    
    print("\n3. Selective Kernel Attention (SKA):")
    print("   - Best for: Multi-scale damage detection")
    print("   - Advantages: Adaptive receptive fields, handles various damage sizes")
    print("   - Use case: Detecting both fine cracks and large erosion areas")
    
    print("\n4. Efficient Multi-Scale Attention (EMA):")
    print("   - Best for: Real-time inspection systems")
    print("   - Advantages: Computational efficiency, suitable for deployment")
    print("   - Use case: Drone-based inspection with real-time processing")


if __name__ == "__main__":
    # Display comparison if no arguments provided
    if len(sys.argv) == 1:
        compare_attention_mechanisms()
        print("\n" + "="*60)
        print("Usage examples:")
        print("python train_wind_turbine_with_attention.py --model yolov8_CA --data ./GRAZPEDWRI-DX/data/data.yaml")
        print("python train_wind_turbine_with_attention.py --model yolov8_CBAMv2 --data ./GRAZPEDWRI-DX/data/data.yaml")
        print("python train_wind_turbine_with_attention.py --model yolov8_SKA --data ./GRAZPEDWRI-DX/data/data.yaml")
        print("python train_wind_turbine_with_attention.py --model yolov8_EMA --data ./GRAZPEDWRI-DX/data/data.yaml")
    else:
        main()