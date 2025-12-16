#!/bin/bash

# Force training to save results in current working directory
# This script ensures YOLO saves results where we want them

echo "🚀 YOLOv8 Training with Fixed Directory Saving"
echo "📁 Current working directory: $(pwd)"
echo "🎯 Results will be saved in: $(pwd)/wind_turbine_attention/"
echo "=========================================="

# Function to train a model with absolute paths
train_with_fixed_dir() {
    local model_name=$1
    local model_config=$2
    local description=$3
    
    # Get absolute paths
    local current_dir=$(pwd)
    local abs_model_config="$current_dir/$model_config"
    local abs_data_config="$current_dir/GRAZPEDWRI-DX/data/data.yaml"
    local abs_project_dir="$current_dir/wind_turbine_attention"
    
    echo ""
    echo "🔄 Training $model_name"
    echo "📋 Description: $description"
    echo "🔧 Model config: $abs_model_config"
    echo "📊 Data config: $abs_data_config"
    echo "📁 Project dir: $abs_project_dir"
    echo "⏰ Started at: $(date)"
    
    # Check if files exist
    if [ ! -f "$abs_model_config" ]; then
        echo "❌ Error: Model configuration not found: $abs_model_config"
        return 1
    fi
    
    if [ ! -f "$abs_data_config" ]; then
        echo "❌ Error: Dataset configuration not found: $abs_data_config"
        return 1
    fi
    
    # Create project directory if it doesn't exist
    mkdir -p "$abs_project_dir"
    
    # Train with absolute paths
    yolo train \
        model="$abs_model_config" \
        data="$abs_data_config" \
        epochs=100 \
        batch=16 \
        imgsz=640 \
        device=0 \
        project="$abs_project_dir" \
        name="$model_name" \
        save=true \
        val=true \
        plots=true \
        verbose=true \
        exist_ok=true
    
    if [ $? -eq 0 ]; then
        echo "✅ $model_name training completed successfully!"
        echo "📊 Results saved to: $abs_project_dir/$model_name"
        echo "⏰ Completed at: $(date)"
        
        # Verify results are in the correct location
        if [ -f "$abs_project_dir/$model_name/weights/best.pt" ]; then
            echo "✅ Model weights confirmed: $abs_project_dir/$model_name/weights/best.pt"
        else
            echo "⚠️  Model weights not found in expected location"
        fi
        
        return 0
    else
        echo "❌ $model_name training failed!"
        return 1
    fi
}

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_type>"
    echo ""
    echo "Available models:"
    echo "  ca      - Coordinate Attention (best for linear cracks)"
    echo "  cbamv2  - CBAM v2 (best for surface damage)"
    echo "  ska     - Selective Kernel Attention (best for multi-scale)"
    echo "  ema     - Efficient Multi-Scale Attention (best for real-time)"
    echo "  all     - Train all models sequentially"
    echo ""
    echo "Examples:"
    echo "  $0 ca"
    echo "  $0 cbamv2"
    echo "  $0 all"
    exit 1
fi

# Main execution
case $1 in
    ca)
        train_with_fixed_dir "coordinate_attention" "ultralytics/cfg/models/v8/yolov8_CA.yaml" "Perfect for linear cracks and edge erosion"
        ;;
    cbamv2)
        train_with_fixed_dir "cbamv2" "ultralytics/cfg/models/v8/yolov8_CBAMv2.yaml" "Enhanced spatial attention for surface damage"
        ;;
    ska)
        train_with_fixed_dir "ska" "ultralytics/cfg/models/v8/yolov8_SKA.yaml" "Multi-scale damage detection"
        ;;
    ema)
        train_with_fixed_dir "ema" "ultralytics/cfg/models/v8/yolov8_EMA.yaml" "Real-time inspection systems"
        ;;
    all)
        echo "🚀 Training ALL models with fixed directory saving"
        echo "🕐 This will take several hours..."
        echo ""
        
        train_with_fixed_dir "coordinate_attention" "ultralytics/cfg/models/v8/yolov8_CA.yaml" "Perfect for linear cracks and edge erosion"
        train_with_fixed_dir "cbamv2" "ultralytics/cfg/models/v8/yolov8_CBAMv2.yaml" "Enhanced spatial attention for surface damage"
        train_with_fixed_dir "ska" "ultralytics/cfg/models/v8/yolov8_SKA.yaml" "Multi-scale damage detection"
        train_with_fixed_dir "ema" "ultralytics/cfg/models/v8/yolov8_EMA.yaml" "Real-time inspection systems"
        
        echo ""
        echo "🎉 ALL TRAINING COMPLETED!"
        echo "📊 Results saved in: $(pwd)/wind_turbine_attention/"
        echo "📁 Available experiments: coordinate_attention, cbamv2, ska, ema"
        ;;
    *)
        echo "❌ Unknown model type: $1"
        echo "Available options: ca, cbamv2, ska, ema, all"
        exit 1
        ;;
esac