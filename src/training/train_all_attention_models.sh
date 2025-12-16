#!/bin/bash

# =============================================================================
# YOLOv8 Attention Models Training Script
# =============================================================================
# This script trains all fixed attention mechanisms for wind turbine damage detection
# 
# Fixed Models:
# - CoordinateAttention (CA) - Best for linear cracks and edge erosion
# - CBAMv2 - Enhanced spatial attention for surface damage  
# - SelectiveKernelAttention (SKA) - Multi-scale damage detection
# - EfficientMultiScaleAttention (EMA) - Real-time inspection systems
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="./GRAZPEDWRI-DX/data/data.yaml"
PROJECT_DIR="wind_turbine_attention_experiments_600epochs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs_${TIMESTAMP}"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

print_header() {
    echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║$(printf "%126s" | tr ' ' ' ')║${NC}"
    echo -e "${PURPLE}║$(printf "%*s" $(($(echo "$1" | wc -c) + 63)) "$1" | tr ' ' ' ')║${NC}"
    echo -e "${PURPLE}║$(printf "%126s" | tr ' ' ' ')║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
}

# Function to train a model
train_model() {
    local model_name=$1
    local model_config=$2
    local description=$3
    local log_file="${LOG_DIR}/${model_name}_training.log"
    
    print_header "Training $model_name - $description"
    
    print_status "📊 Model: $model_name"
    print_status "🔧 Config: $model_config"
    print_status "📁 Data: $DATA_DIR"
    print_status "💾 Results: ./${PROJECT_DIR}/${model_name}_${TIMESTAMP}/"
    print_status "📋 Log: $log_file"
    print_status "🚀 Starting training..."
    
    # Check if model config exists
    if [ ! -f "$model_config" ]; then
        print_error "Model configuration not found: $model_config"
        return 1
    fi
    
    # Check if data config exists
    if [ ! -f "$DATA_DIR" ]; then
        print_error "Dataset configuration not found: $DATA_DIR"
        return 1
    fi
    
    # Run training
    local start_time=$(date +%s)
    
    if python start_train.py \
        --model "$model_config" \
        --data_dir "$DATA_DIR" \
        --project "$PROJECT_DIR" \
        --name "${model_name}_${TIMESTAMP}" \
        2>&1 | tee "$log_file"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        print_success "Training completed successfully!"
        print_success "Duration: ${hours}h ${minutes}m ${seconds}s"
        print_success "Results saved to: ./${PROJECT_DIR}/${model_name}_${TIMESTAMP}/"
        print_success "Log file: $log_file"
        
        return 0
    else
        print_error "Training failed!"
        print_error "Check log file: $log_file"
        return 1
    fi
}

# Function to show training summary
show_summary() {
    local results_summary="${LOG_DIR}/training_summary.txt"
    
    print_header "Training Summary"
    
    echo "YOLOv8 Attention Models Training Summary" > "$results_summary"
    echo "=========================================" >> "$results_summary"
    echo "Timestamp: $(date)" >> "$results_summary"
    echo "Project Directory: ./${PROJECT_DIR}/" >> "$results_summary"
    echo "" >> "$results_summary"
    
    echo "Model Results:" >> "$results_summary"
    for result in "${training_results[@]}"; do
        echo "  $result" >> "$results_summary"
    done
    
    echo "" >> "$results_summary"
    echo "Next Steps:" >> "$results_summary"
    echo "1. Check individual model results in ./${PROJECT_DIR}/" >> "$results_summary"
    echo "2. Compare model performance using validation metrics" >> "$results_summary"
    echo "3. Use best model for inference: python start_train.py --model path/to/best.pt --mode predict" >> "$results_summary"
    
    cat "$results_summary"
}

# Main execution
main() {
    print_header "YOLOv8 Attention Models Training Pipeline"
    
    print_status "🎯 Target: Wind Turbine Damage Detection"
    print_status "📦 Classes: Crack, Leading Edge Erosion, Lightning Strike, Surface Damage, Surface Dust"
    print_status "🔧 Framework: YOLOv8 with Advanced Attention Mechanisms"
    print_status "💻 Saving results to current directory"
    echo ""
    
    # Array to store training results
    training_results=()
    
    # Model definitions
    declare -A models=(
        #["coordinate_attention"]="./ultralytics/cfg/models/v8/yolov8_CA.yaml|Perfect for linear cracks and edge erosion"
        #["cbamv2"]="./ultralytics/cfg/models/v8/yolov8_CBAMv2.yaml|Enhanced spatial attention for surface damage"
        ["selective_kernel_attention"]="./ultralytics/cfg/models/v8/yolov8_SKA.yaml|Multi-scale damage detection"
        ["MSPA"]="./ultralytics/cfg/models/v8/yolov8_MSPA.yaml|Multi-scale Pyramidal Attention"
        ["EEA"]="./ultralytics/cfg/models/v8/yolov8_EEA.yaml|Efficient Enhanced Attention"
        #["base_yolov8"]="./ultralytics/cfg/models/v8/yolov8.yaml"
        ["efficient_multiscale_attention"]="./ultralytics/cfg/models/v8/yolov8_EMA.yaml|Real-time inspection systems"
    )
    
    # Train each model
    for model_name in "${!models[@]}"; do
        IFS='|' read -r model_config description <<< "${models[$model_name]}"
        
        if train_model "$model_name" "$model_config" "$description"; then
            training_results+=("✅ $model_name: SUCCESS")
        else
            training_results+=("❌ $model_name: FAILED")
        fi
        
        echo ""
    done
    
    # Show summary
    show_summary
    
    print_header "All Training Sessions Completed!"
    print_status "📊 Check results in: ./${PROJECT_DIR}/"
    print_status "📋 Check logs in: $LOG_DIR/"
    print_status "🎉 Happy analyzing!"
}

# Check if start_train.py exists
if [ ! -f "start_train.py" ]; then
    print_error "start_train.py not found in current directory!"
    print_error "Please run this script from the project root directory."
    exit 1
fi

# Run main function
main 
