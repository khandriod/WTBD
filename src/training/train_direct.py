#!/usr/bin/env python3
"""
Direct training script that bypasses shell scripts completely.
This avoids any line ending issues and works reliably across all systems.
"""

import os
import sys
import subprocess
from pathlib import Path

def train_model(model_name, model_config, description):
    """Train a model with the given configuration"""
    
    # Get absolute paths
    current_dir = os.path.abspath(os.getcwd())
    abs_model_config = os.path.abspath(model_config)
    abs_data_config = os.path.abspath("./GRAZPEDWRI-DX/data/data.yaml")
    abs_project_dir = os.path.abspath(os.path.join(current_dir, "wind_turbine_attention"))
    
    print(f"\n🚀 Training {model_name}")
    print(f"📋 Description: {description}")
    print(f"🔧 Current directory: {current_dir}")
    print(f"📁 Model config: {abs_model_config}")
    print(f"📊 Data config: {abs_data_config}")
    print(f"📁 Results will be saved to: {abs_project_dir}/{model_name}")
    print("=" * 50)
    
    # Check if model config exists
    if not Path(abs_model_config).exists():
        print(f"❌ Error: Model configuration not found: {abs_model_config}")
        return False
    
    # Check if data config exists
    if not Path(abs_data_config).exists():
        print(f"❌ Error: Dataset configuration not found: {abs_data_config}")
        return False
    
    # Create project directory if it doesn't exist
    Path(abs_project_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command with absolute paths
    cmd = [
        'yolo', 'train',
        f'model={abs_model_config}',
        f'data={abs_data_config}',
        'epochs=100',
        'batch=16',
        'imgsz=640',
        'device=0',
        f'project={abs_project_dir}',
        f'name={model_name}',
        'save=true',
        'val=true',
        'plots=true',
        'verbose=true',
        'exist_ok=true'
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        # Execute training
        result = subprocess.run(cmd, check=True)
        
        print(f"✅ {model_name} training completed successfully!")
        print(f"📊 Results saved to: {abs_project_dir}/{model_name}")
        print(f"🔍 Check the following files:")
        print(f"   - {abs_project_dir}/{model_name}/weights/best.pt")
        print(f"   - {abs_project_dir}/{model_name}/results.png")
        print(f"   - {abs_project_dir}/{model_name}/confusion_matrix.png")
        
        # Verify results are in the correct location
        if Path(f"{abs_project_dir}/{model_name}/weights/best.pt").exists():
            print(f"✅ Model weights confirmed in current directory!")
        else:
            print(f"⚠️  Model weights not found in expected location")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name} training failed!")
        return False
    except Exception as e:
        print(f"❌ Error executing {model_name}: {e}")
        return False

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("🎯 Wind Turbine Damage Detection Training")
        print("Usage: python train_direct.py <model_name>")
        print("\nAvailable models:")
        print("  ca      - Coordinate Attention (best for linear cracks)")
        print("  cbamv2  - CBAM v2 (best for surface damage)")
        print("  ska     - Selective Kernel Attention (best for multi-scale)")
        print("  ema     - Efficient Multi-Scale Attention (best for real-time)")
        print("  all     - Train all models sequentially")
        print("\nExamples:")
        print("  python train_direct.py ca")
        print("  python train_direct.py cbamv2")
        print("  python train_direct.py all")
        return
    
    model_choice = sys.argv[1].lower()
    
    # Model configurations
    models = {
        'ca': {
            'config': './ultralytics/cfg/models/v8/yolov8_CA.yaml',
            'name': 'coordinate_attention',
            'description': 'Perfect for linear cracks and edge erosion'
        },
        'cbamv2': {
            'config': './ultralytics/cfg/models/v8/yolov8_CBAMv2.yaml',
            'name': 'cbamv2',
            'description': 'Enhanced spatial attention for surface damage'
        },
        'ska': {
            'config': './ultralytics/cfg/models/v8/yolov8_SKA.yaml',
            'name': 'ska',
            'description': 'Multi-scale damage detection'
        },
        'ema': {
            'config': './ultralytics/cfg/models/v8/yolov8_EMA.yaml',
            'name': 'ema',
            'description': 'Real-time inspection systems'
        }
    }
    
    if model_choice == 'all':
        print("🚀 Training ALL YOLOv8 Attention Mechanisms")
        print("🕐 This will take several hours...")
        
        results = []
        for key, model_info in models.items():
            success = train_model(
                model_info['name'],
                model_info['config'],
                model_info['description']
            )
            results.append((key, success))
        
        print("\n" + "=" * 60)
        print("🎉 ALL TRAINING COMPLETED!")
        print("📊 Results Summary:")
        for key, success in results:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {key.upper()}: {status}")
        
    elif model_choice in models:
        model_info = models[model_choice]
        success = train_model(
            model_info['name'],
            model_info['config'],
            model_info['description']
        )
        
        if success:
            print(f"\n🎉 Training completed successfully!")
        else:
            print(f"\n❌ Training failed!")
            sys.exit(1)
    
    else:
        print(f"❌ Unknown model choice: {model_choice}")
        print("Available options: ca, cbamv2, ska, ema, all")
        sys.exit(1)

if __name__ == "__main__":
    main()