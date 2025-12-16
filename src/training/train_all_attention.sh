#!/bin/bash

# Train all YOLOv8 attention mechanisms for Wind Turbine Damage Detection
# This script trains all 4 attention mechanisms sequentially and saves results in current directory.

echo "🚀 Training ALL YOLOv8 Attention Mechanisms for Wind Turbine Damage Detection"
echo "📁 Results will be saved in: $(pwd)/wind_turbine_attention/"
echo "🕐 This will take several hours to complete..."
echo "=========================================="

# Function to train a model
train_model() {
    local model_name=$1
    local model_config=$2
    local description=$3
    
    echo ""
    echo "🔄 Training $model_name..."
    echo "📋 Description: $description"
    echo "⏰ Started at: $(date)"
    
    yolo train \
        model="$model_config" \
        data="./GRAZPEDWRI-DX/data/data.yaml" \
        epochs=100 \
        batch=16 \
        imgsz=640 \
        device=0 \
        project="$(pwd)/wind_turbine_attention" \
        name="$model_name" \
        save=true \
        val=true \
        plots=true \
        verbose=false \
        exist_ok=true
    
    if [ $? -eq 0 ]; then
        echo "✅ $model_name training completed successfully!"
        echo "⏰ Completed at: $(date)"
    else
        echo "❌ $model_name training failed!"
        return 1
    fi
}

# Check if data config exists
if [ ! -f "./GRAZPEDWRI-DX/data/data.yaml" ]; then
    echo "❌ Error: Dataset configuration not found: ./GRAZPEDWRI-DX/data/data.yaml"
    exit 1
fi

# Train all models
echo "🎯 Training sequence:"
echo "1. Coordinate Attention (CA) - Best for linear cracks"
echo "2. CBAM v2 - Best for surface damage"
echo "3. Selective Kernel Attention (SKA) - Best for multi-scale"
echo "4. Efficient Multi-Scale Attention (EMA) - Best for real-time"
echo ""

# Model 1: Coordinate Attention
train_model "coordinate_attention" "./ultralytics/cfg/models/v8/yolov8_CA.yaml" "Perfect for linear cracks and edge erosion"

# Model 2: CBAM v2
train_model "cbamv2" "./ultralytics/cfg/models/v8/yolov8_CBAMv2.yaml" "Enhanced spatial attention for surface damage"

# Model 3: Selective Kernel Attention
train_model "ska" "./ultralytics/cfg/models/v8/yolov8_SKA.yaml" "Multi-scale damage detection"

# Model 4: Efficient Multi-Scale Attention
train_model "ema" "./ultralytics/cfg/models/v8/yolov8_EMA.yaml" "Real-time inspection systems"

echo ""
echo "🎉 ALL TRAINING COMPLETED!"
echo "📊 Results saved in: $(pwd)/wind_turbine_attention/"
echo "📁 Available experiments:"
echo "   - coordinate_attention/ (CA)"
echo "   - cbamv2/ (CBAM v2)"
echo "   - ska/ (Selective Kernel Attention)"
echo "   - ema/ (Efficient Multi-Scale Attention)"
echo ""
echo "🔍 Each experiment contains:"
echo "   - weights/best.pt (trained model)"
echo "   - results.png (training curves)"
echo "   - confusion_matrix.png (validation results)"
echo "   - labels.jpg (ground truth visualization)"
echo "   - val_batch*.jpg (validation predictions)"
echo ""
echo "📈 Compare results to choose the best model for your use case!"