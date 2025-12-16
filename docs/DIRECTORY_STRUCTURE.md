# Repository Directory Structure

This document describes the organized directory structure of the FDI_Yolov8 wind turbine damage detection repository.

## 📁 Overview

The repository has been reorganized following ML project best practices for better maintainability, clarity, and collaboration.

```
FDI_Yolov8/
├── 📂 src/                    # Source code (all Python scripts)
├── 📂 data/                   # Data files and datasets
├── 📂 models/                 # Trained models and experiments
├── 📂 results/                # All outputs and results
├── 📂 configs/                # Configuration files
├── 📂 docs/                   # Documentation and figures
├── 📂 logs/                   # Training logs
├── 📂 ultralytics/            # YOLO framework (modified)
├── 📂 runs/                   # YOLO training runs
├── 📂 wandb/                  # Weights & Biases tracking
├── 📄 README.md               # Main documentation
├── 📄 LICENSE                 # License file
└── 📄 *.py                    # Wrapper scripts for backward compatibility
```

## 🔧 Source Code (`src/`)

Organized by functionality for better code management:

```
src/
├── detection/                 # Detection and inference scripts
│   ├── wind_turbine_detector.py       # Main detection system
│   ├── enhanced_tiled_detection.py    # Advanced tiled processing
│   └── tiled_damage_detection.py      # Original tiled detection
├── training/                  # Training and model creation
│   ├── start_train.py                 # Main training script
│   ├── start_train_multi_gpu.py       # Multi-GPU training
│   ├── train_wind_turbine_with_attention.py
│   ├── train_yolo_cli.py              # CLI training interface
│   └── train_direct.py                # Direct training script
├── evaluation/                # Model evaluation and testing
│   ├── yolo_model_evaluator.py        # Comprehensive model evaluation
│   ├── model_evaluation.py            # Basic evaluation
│   ├── test_attention_mechanisms.py   # Attention testing
│   └── test_tiled_detection.py        # Tiled detection testing
└── utils/                     # Utility functions and helpers
    ├── tile_utils.py                  # Tile processing utilities
    ├── split.py                       # Data splitting
    └── imgaug.py                      # Image augmentation
```

## 📊 Data (`data/`)

Centralized data management:

```
data/
├── raw/                       # Original, immutable data
│   └── GRAZPEDWRI-DX/        # Main dataset
├── processed/                 # Cleaned and processed data
└── samples/                   # Sample images and test data
    ├── avingrid_data/        # Avingrid test images
    └── img/                  # Sample images
```

## 🤖 Models (`models/`)

Organized experiment results and trained models:

```
models/
├── experiments_200/           # 200-epoch training experiments
│   ├── selective_kernel_attention_20250716_174959/
│   ├── efficient_multiscale_attention_20250716_174959/
│   ├── coordinate_attention_20250716_174959/
│   ├── cbamv2_20250716_174959/
│   ├── EEA_20250718_142109/
│   ├── MSPA_20250718_142245/
│   └── wind_turbine_attention_experiments/
├── experiments_600/           # 600-epoch training experiments  
│   ├── selective_kernel_attention_20250719_021254/
│   ├── efficient_multiscale_attention_20250719_021254/
│   ├── EEA_20250719_020920/
│   └── MSPA_20250719_020920/
├── legacy/                    # Legacy attention mechanisms
│   ├── efficient_channel_attention_20250717_153446/
│   ├── global_attention_mechanism_20250717_224121/
│   ├── resblock_cbam_20250717_224505/
│   └── yolov8_baseline_*/
└── yolov8n.pt                # Base YOLO model
```

## 📈 Results (`results/`)

Consolidated output management:

```
results/
├── evaluations/               # Model evaluation results
│   └── YOLOv8_Attention_Performance_Analysis_*.xlsx
├── predictions/               # Model prediction outputs
│   └── SKA_600_epochs/
├── validation/                # Validation results
│   └── ska_validation_results/
├── detection_results/         # Detection system outputs
├── quick_test_results/        # Quick test outputs
└── tiled_results/            # Tiled detection results
```

## ⚙️ Configuration (`configs/`)

Configuration files and requirements:

```
configs/
└── requirements.txt           # Python dependencies
```

## 📚 Documentation (`docs/`)

All documentation and visual resources:

```
docs/
├── README_sliding_window.md   # Sliding window documentation
├── specstory_generated.md     # Generated documentation
├── figures/                   # Charts and performance plots
└── mds/                      # Markdown documentation files
```

## 🔄 Backward Compatibility

**Wrapper Scripts**: The root directory contains wrapper scripts that maintain backward compatibility:

- `wind_turbine_detector.py` → `src/detection/wind_turbine_detector.py`
- `yolo_model_evaluator.py` → `src/evaluation/yolo_model_evaluator.py`

**Symlinks**: For model access compatibility:
- `modelsandweights_200/` → `models/experiments_200/`
- `modelsandweights_600/` → `models/experiments_600/`

## 🚀 Usage Examples

### Using the New Structure

```bash
# Run detection from anywhere in the repo
python wind_turbine_detector.py --model models/experiments_600/selective_kernel_attention_*/weights/best.pt --input data/samples/avingrid_data/image.jpg

# Run evaluation
python yolo_model_evaluator.py --model models/experiments_600/selective_kernel_attention_*/weights/best.pt

# Train a new model
python src/training/start_train.py --attention SKA --epochs 200
```

### Direct Module Usage

```bash
# Use modules directly
python -m src.detection.wind_turbine_detector --help
python -m src.evaluation.yolo_model_evaluator --help
python -m src.training.start_train --help
```

## ✨ Benefits of New Structure

1. **🎯 Clear Separation**: Code, data, models, and results are clearly separated
2. **🔍 Easy Navigation**: Find files quickly based on functionality
3. **🤝 Better Collaboration**: Standard structure familiar to ML practitioners
4. **📦 Modular Code**: Easy to import and reuse components
5. **🛡️ Backward Compatibility**: Existing scripts still work
6. **📊 Organized Results**: All outputs in predictable locations
7. **🧪 Experiment Tracking**: Clear organization of different model experiments

## 📋 Migration Notes

- All Python scripts moved to `src/` subdirectories
- Data consolidated under `data/`
- Results consolidated under `results/`
- Model experiments organized by training duration
- Wrapper scripts maintain compatibility with existing workflows
- Documentation centralized in `docs/`

## 🔧 Maintenance

To maintain this structure:

1. Add new scripts to appropriate `src/` subdirectories
2. Store new data in `data/` with proper categorization
3. Place experiment results in organized `models/` structure
4. Save outputs to appropriate `results/` subdirectories
5. Update documentation in `docs/` when making changes 