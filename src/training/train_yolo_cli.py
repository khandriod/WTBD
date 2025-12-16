#!/usr/bin/env python3
"""
YOLO Package CLI Training Script for Wind Turbine Damage Detection

This script uses the proper YOLO CLI format to train models with attention mechanisms
and ensures results are saved in the current working directory.

Usage:
    python train_yolo_cli.py --model yolov8_CA --data ./GRAZPEDWRI-DX/data/data.yaml
    python train_yolo_cli.py --model yolov8_CBAMv2 --data ./GRAZPEDWRI-DX/data/data.yaml
    python train_yolo_cli.py --model yolov8_SKA --data ./GRAZPEDWRI-DX/data/data.yaml
    python train_yolo_cli.py --model yolov8_EMA --data ./GRAZPEDWRI-DX/data/data.yaml
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 with CLI format for Wind Turbine Damage Detection')
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
    parser.add_argument('--resume', action='store_true', 
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
    
    # Get current working directory for saving results
    current_dir = os.getcwd()
    project_path = os.path.join(current_dir, args.project)
    
    # Build YOLO command
    cmd = [
        'yolo', 'train',
        f'model={model_config_path}',
        f'data={args.data}',
        f'epochs={args.epochs}',
        f'batch={args.batch_size}',
        f'imgsz={args.img_size}',
        f'device={args.device}',
        f'project={project_path}',
        f'name={args.name}',
        'save=true',
        'val=true',
        'plots=true',
        'verbose=true',
        'exist_ok=true'
    ]
    
    if args.resume:
        cmd.append('resume=true')
    
    # Print configuration
    print(f"🚀 Starting Wind Turbine Damage Detection Training (YOLO CLI)")
    print(f"📊 Model: {args.model}")
    print(f"📁 Dataset: {args.data}")
    print(f"🔧 Configuration: {model_config_path}")
    print(f"⚙️  Device: {args.device}")
    print(f"🎯 Target Classes: Crack, Leading Edge Erosion, Lightning Strike, Surface Damage, Surface Dust")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📏 Image Size: {args.img_size}")
    print(f"💾 Results will be saved to: {project_path}/{args.name}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Execute training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ Training completed successfully!")
        print(f"📊 Results saved to: {project_path}/{args.name}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with return code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error executing training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()