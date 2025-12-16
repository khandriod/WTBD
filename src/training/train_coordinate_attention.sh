#!/bin/bash

# Train YOLOv8 with Coordinate Attention for Wind Turbine Damage Detection
# This script trains the model using YOLO CLI format and saves results in current directory.
# Perfect for detecting linear cracks and edge erosion patterns.

echo "🚀 Training YOLOv8 with Coordinate Attention for Wind Turbine Damage Detection"
echo "🎯 Best for: Linear cracks, edge erosion patterns, structural damage"
echo "📁 Results will be saved in: $(pwd)/wind_turbine_attention/coordinate_attention"
echo "=========================================="

# Set variables
MODEL_CONFIG="./ultralytics/cfg/models/v8/yolov8_CA.yaml"
DATA_CONFIG="./GRAZPEDWRI-DX/data/data.yaml"
CURRENT_DIR="$(pwd)"
PROJECT_DIR="$CURRENT_DIR/wind_turbine_attention"
EXPERIMENT_NAME="coordinate_attention"

echo "🔧 Current directory: $CURRENT_DIR"
echo "📁 Project directory: $PROJECT_DIR"

# Check if model config exists
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "❌ Error: Model configuration not found: $MODEL_CONFIG"
    exit 1
fi

# Check if data config exists
if [ ! -f "$DATA_CONFIG" ]; then
    echo "❌ Error: Dataset configuration not found: $DATA_CONFIG"
    exit 1
fi

# Train the model
yolo train \
    model="$CURRENT_DIR/$MODEL_CONFIG" \
    data="$CURRENT_DIR/$DATA_CONFIG" \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    device=0 \
    project="$PROJECT_DIR" \
    name="$EXPERIMENT_NAME" \
    save=true \
    val=true \
    plots=true \
    verbose=true \
    exist_ok=true

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📊 Results saved to: $PROJECT_DIR/$EXPERIMENT_NAME"
    echo "🔍 Check the following files:"
    echo "   - $PROJECT_DIR/$EXPERIMENT_NAME/weights/best.pt"
    echo "   - $PROJECT_DIR/$EXPERIMENT_NAME/results.png"
    echo "   - $PROJECT_DIR/$EXPERIMENT_NAME/confusion_matrix.png"
else
    echo "❌ Training failed!"
    exit 1
fi