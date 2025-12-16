# YOLOv8 Attention Models Training Scripts

This directory contains bash scripts to train all fixed attention mechanisms for wind turbine damage detection.

## 🚀 Quick Start

### Make Scripts Executable
```bash
chmod +x train_all_attention_models.sh
chmod +x train_single_model.sh
```

## 📋 Available Scripts

### 1. Train All Models (Recommended)
**Script**: `train_all_attention_models.sh`

Trains all 4 attention models sequentially with comprehensive logging.

```bash
./train_all_attention_models.sh
```

**Features:**
- ✅ Trains all 4 attention models automatically
- 📊 Detailed progress tracking with timestamps
- 📋 Individual log files for each model
- 📁 Organized results with timestamped directories
- 🎯 Training summary at the end
- ⏱️ Training duration tracking
- 🔍 Error handling and validation

**Output Structure:**
```
wind_turbine_attention_experiments/
├── coordinate_attention_20240716_123456/
├── cbamv2_20240716_130245/
├── selective_kernel_attention_20240716_133012/
└── efficient_multiscale_attention_20240716_140156/

logs_20240716_123456/
├── coordinate_attention_training.log
├── cbamv2_training.log
├── selective_kernel_attention_training.log
├── efficient_multiscale_attention_training.log
└── training_summary.txt
```

### 2. Train Single Model (Interactive)
**Script**: `train_single_model.sh`

Train individual models with more control and confirmation prompts.

```bash
# Show usage and available models
./train_single_model.sh

# Train specific model
./train_single_model.sh ca my_ca_experiment
./train_single_model.sh cbamv2 surface_damage_test
./train_single_model.sh ska multi_scale_test
./train_single_model.sh ema realtime_test
```

**Available Models:**
- `ca` - Coordinate Attention (Perfect for linear cracks and edge erosion)
- `cbamv2` - CBAM v2 (Enhanced spatial attention for surface damage)
- `ska` - Selective Kernel Attention (Multi-scale damage detection)
- `ema` - Efficient Multi-Scale Attention (Real-time inspection systems)

## 🎯 Model Descriptions

| Model | Attention Type | Best For | Parameters | FLOPs |
|-------|----------------|----------|------------|-------|
| **CA** | Coordinate Attention | Linear cracks, edge erosion | 3.0M | 8.2G |
| **CBAMv2** | Enhanced Channel+Spatial | Surface damage | 3.0M | 8.2G |
| **SKA** | Selective Kernel | Multi-scale damage | 6.5M | 15.4G |
| **EMA** | Efficient Multi-Scale | Real-time systems | 3.1M | 8.4G |

## 📊 Training Configuration

### Default Parameters
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640x640
- **Learning Rate**: 0.01 (linear decay to 0.0001)
- **Warmup**: 3 epochs
- **Optimizer**: SGD (momentum=0.937)
- **Device**: GPU (CUDA:0)

### Dataset
- **Classes**: 5 (Crack, Leading Edge Erosion, Lightning Strike, Surface Damage, Surface Dust)
- **Data Format**: YOLO format with YAML configuration
- **Augmentation**: Automatic (blur, median blur, grayscale, CLAHE)

## 📁 File Structure

```
├── start_train.py                    # Main training script
├── train_all_attention_models.sh    # Batch training script
├── train_single_model.sh            # Individual model training
├── TRAINING_SCRIPTS_README.md       # This file
├── ultralytics/cfg/models/v8/
│   ├── yolov8_CA.yaml              # Coordinate Attention
│   ├── yolov8_CBAMv2.yaml          # CBAM v2
│   ├── yolov8_SKA.yaml             # Selective Kernel Attention
│   └── yolov8_EMA.yaml             # Efficient Multi-Scale Attention
└── GRAZPEDWRI-DX/data/data.yaml    # Dataset configuration
```

## 🔧 Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x train_all_attention_models.sh
   chmod +x train_single_model.sh
   ```

2. **Dataset Not Found**
   - Ensure `./GRAZPEDWRI-DX/data/data.yaml` exists
   - Check dataset paths in the YAML file

3. **CUDA Out of Memory**
   - Reduce batch size in training script
   - Use smaller model variant (if available)

4. **Model Config Not Found**
   - Verify model YAML files exist in `./ultralytics/cfg/models/v8/`
   - Check file permissions

### Log Files
- Check individual model logs in `logs_[timestamp]/`
- Review training summary for overall results
- Monitor GPU usage with `nvidia-smi`

## 🎉 Next Steps

After training completion:

1. **Compare Results**: Check validation metrics in each model's directory
2. **Best Model Selection**: Use model with highest mAP@0.5:0.95
3. **Inference**: Use trained model for prediction on new images
4. **Fine-tuning**: Adjust hyperparameters based on initial results

## 💡 Pro Tips

- **Resource Management**: Training all models will take several hours
- **Monitor Progress**: Check individual log files for real-time updates
- **Early Stopping**: Training includes patience=50 for automatic stopping
- **Validation**: Models are validated during training for performance tracking
- **Reproducibility**: Fixed seed (0) ensures consistent results

## 🔗 Related Commands

```bash
# Manual training (alternative to scripts)
python start_train.py --model ./ultralytics/cfg/models/v8/yolov8_CA.yaml --data_dir ./GRAZPEDWRI-DX/data/data.yaml --project experiments --name ca_manual

# Resume training from checkpoint
python start_train.py --model ./experiments/ca_manual/weights/last.pt --data_dir ./GRAZPEDWRI-DX/data/data.yaml --project experiments --name ca_resume

# Validate trained model
python start_train.py --model ./experiments/ca_manual/weights/best.pt --data_dir ./GRAZPEDWRI-DX/data/data.yaml --project experiments --name ca_validation --mode val
```

---

**Happy Training! 🚀** 