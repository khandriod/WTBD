#!/bin/bash

# =============================================================================
# YOLOv8 Single Attention Model Training Script
# =============================================================================
# Usage: ./train_single_model.sh [model_name] [experiment_name]
# 
# Available models:
# - ca       : Coordinate Attention
# - cbamv2   : CBAM v2 
# - ska      : Selective Kernel Attention
# - ema      : Efficient Multi-Scale Attention
# =============================================================================

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DATA_DIR="./GRAZPEDWRI-DX/data/data.yaml"
PROJECT_DIR="experiments"

# Model configurations
declare -A models=(
    ["ca"]="./ultralytics/cfg/models/v8/yolov8_CA.yaml"
    ["cbamv2"]="./ultralytics/cfg/models/v8/yolov8_CBAMv2.yaml"
    ["ska"]="./ultralytics/cfg/models/v8/yolov8_SKA.yaml"
    ["ema"]="./ultralytics/cfg/models/v8/yolov8_EMA.yaml"
)

declare -A descriptions=(
    ["ca"]="Coordinate Attention - Perfect for linear cracks and edge erosion"
    ["cbamv2"]="CBAM v2 - Enhanced spatial attention for surface damage"
    ["ska"]="Selective Kernel Attention - Multi-scale damage detection"
    ["ema"]="Efficient Multi-Scale Attention - Real-time inspection systems"
)

# Function to show usage
show_usage() {
    echo -e "${BLUE}Usage: $0 [model_name] [experiment_name]${NC}"
    echo ""
    echo "Available models:"
    for model in "${!models[@]}"; do
        echo -e "  ${GREEN}$model${NC} - ${descriptions[$model]}"
    done
    echo ""
    echo "Examples:"
    echo -e "  ${YELLOW}$0 ca my_ca_experiment${NC}"
    echo -e "  ${YELLOW}$0 cbamv2 surface_damage_test${NC}"
    echo -e "  ${YELLOW}$0 ska multi_scale_test${NC}"
    echo -e "  ${YELLOW}$0 ema realtime_test${NC}"
}

# Check arguments
if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

MODEL_NAME=$1
EXPERIMENT_NAME=${2:-"${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"}

# Validate model name
if [[ ! " ${!models[@]} " =~ " ${MODEL_NAME} " ]]; then
    echo -e "${RED}Error: Invalid model name '${MODEL_NAME}'${NC}"
    echo ""
    show_usage
    exit 1
fi

# Get model config
MODEL_CONFIG=${models[$MODEL_NAME]}
DESCRIPTION=${descriptions[$MODEL_NAME]}

# Check if files exist
if [ ! -f "$MODEL_CONFIG" ]; then
    echo -e "${RED}Error: Model configuration not found: $MODEL_CONFIG${NC}"
    exit 1
fi

if [ ! -f "$DATA_DIR" ]; then
    echo -e "${RED}Error: Dataset configuration not found: $DATA_DIR${NC}"
    exit 1
fi

# Print training info
echo -e "${BLUE}🚀 Starting YOLOv8 Training${NC}"
echo -e "${BLUE}════════════════════════════${NC}"
echo -e "📊 Model: ${GREEN}$MODEL_NAME${NC} - $DESCRIPTION"
echo -e "🔧 Config: $MODEL_CONFIG"
echo -e "📁 Data: $DATA_DIR"
echo -e "💾 Results: ./${PROJECT_DIR}/${EXPERIMENT_NAME}/"
echo -e "🎯 Classes: Crack, Leading Edge Erosion, Lightning Strike, Surface Damage, Surface Dust"
echo ""

# Ask for confirmation
read -p "Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled.${NC}"
    exit 0
fi

# Run training
echo -e "${BLUE}🚀 Training started...${NC}"
python start_train.py \
    --model "$MODEL_CONFIG" \
    --data_dir "$DATA_DIR" \
    --project "$PROJECT_DIR" \
    --name "$EXPERIMENT_NAME"

# Check result
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "${GREEN}📊 Results saved to: ./${PROJECT_DIR}/${EXPERIMENT_NAME}/${NC}"
    echo -e "${GREEN}🎉 Check the results and happy analyzing!${NC}"
else
    echo -e "${RED}❌ Training failed!${NC}"
    echo -e "${RED}Check the output above for error details.${NC}"
    exit 1
fi 