import os
# Fix MKL threading layer incompatibility issue for multi-GPU training
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU YOLO Training')
    parser.add_argument('--model', type=str, required=True, help='Path to model config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data config file')
    parser.add_argument('--project', type=str, default='runs', help='Project name for saving results')
    parser.add_argument('--name', type=str, default='train', help='Experiment name')
    parser.add_argument('--device', type=str, default='0,1', help='GPU devices to use (e.g., 0,1 or 0,1,2,3)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse device string to list of integers
    if hasattr(args, 'device') and args.device:
        device_list = [int(d.strip()) for d in args.device.split(',')]
    else:
        device_list = [0, 1]  # Default to first two GPUs
    
    # Get current working directory
    current_dir = os.getcwd()
    expected_save_path = Path(current_dir) / args.project / args.name
    
    print(f"🚀 Starting Multi-GPU YOLO training")
    print(f"📁 Model config: {args.model}")
    print(f"📊 Data config: {args.data_dir}")
    print(f"💾 Results will be saved to: {expected_save_path}")
    print(f"🔧 GPU devices: {device_list}")
    print(f"⚡ Training mode: Multi-GPU DDP")
    print(f"🏃 Epochs: {args.epochs}, Batch size: {args.batch}, Image size: {args.imgsz}")
    print("=" * 60)
    
    # Initialize YOLO model
    model = YOLO(args.model)
    
    # Start training with multi-GPU configuration
    model.train(
        data=args.data_dir,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device_list,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Additional recommended settings for multi-GPU training
        workers=8,  # Number of data loading workers
        patience=50,  # Early stopping patience
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,  # Validate during training
        plots=True,  # Generate training plots
        verbose=True  # Verbose output
    )
    
    print("✅ Training completed successfully!")

if __name__ == '__main__':
    main() 