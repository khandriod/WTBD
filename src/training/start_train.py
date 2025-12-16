import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parse = argparse.ArgumentParser(description='Data Postprocess')
    parse.add_argument('--model', type=str, default=None, help='load the model')
    parse.add_argument('--data_dir', type=str, default=None, help='the dir to data')
    parse.add_argument('--project', type=str, default='runs', help='project name for saving results')
    parse.add_argument('--name', type=str, default='train', help='experiment name')
    args = parse.parse_args()
    return args

def main():
    args = parse_args()
    
    # Get current working directory
    current_dir = os.getcwd()
    expected_save_path = Path(current_dir) / args.project / args.name
    
    print(f"🚀 Starting YOLO training")
    print(f"📁 Model config: {args.model}")
    print(f"📊 Data config: {args.data_dir}")
    print(f"💾 Results will be saved to: {expected_save_path}")
    print("=" * 50)
    
    # Change to current directory to ensure relative paths work correctly
    os.chdir(current_dir)
    
    model = YOLO(args.model)
    model.train(
        data=args.data_dir,
        project=args.project,
        name=args.name,
        device=[1],
        exist_ok=True
    )

if __name__ == '__main__':
    main()
