# 🔧 Troubleshooting Guide

## ✅ Shell Script Line Ending Issues (FIXED)

### Problem: `/bin/bash^M: no such file or directory`
This was caused by Windows line endings in the shell scripts.

### Solution:
I've recreated all shell scripts with proper Unix line endings. The scripts should now work correctly:

```bash
# Test the scripts
./train_coordinate_attention.sh
./train_cbamv2.sh
./train_ska.sh
./train_ema.sh
```

### Alternative Solutions:

#### 1. Use Python Script (Recommended)
```bash
# No shell script issues - works on all systems
python train_direct.py ca      # Coordinate Attention
python train_direct.py cbamv2  # CBAM v2
python train_direct.py ska     # Selective Kernel Attention
python train_direct.py ema     # Efficient Multi-Scale Attention
python train_direct.py all     # Train all models
```

#### 2. Direct YOLO Commands
```bash
# Bypass shell scripts entirely
yolo train model=./ultralytics/cfg/models/v8/yolov8_CA.yaml \
           data=./GRAZPEDWRI-DX/data/data.yaml \
           epochs=100 batch=16 imgsz=640 device=0 \
           project=$(pwd)/wind_turbine_attention \
           name=coordinate_attention \
           save=true val=true plots=true exist_ok=true
```

#### 3. If Shell Scripts Still Don't Work
```bash
# Convert line endings manually
sed -i 's/\r$//' train_coordinate_attention.sh
chmod +x train_coordinate_attention.sh
./train_coordinate_attention.sh
```

## 🚨 Common Issues and Solutions

### 1. Model Configuration Not Found
**Problem:** `❌ Error: Model configuration not found`

**Solution:**
```bash
# Check if model configs exist
ls -la ./ultralytics/cfg/models/v8/yolov8_*.yaml

# If missing, they should be:
# - yolov8_CA.yaml
# - yolov8_CBAMv2.yaml  
# - yolov8_SKA.yaml
# - yolov8_EMA.yaml
```

### 2. Dataset Configuration Not Found
**Problem:** `❌ Error: Dataset configuration not found`

**Solution:**
```bash
# Check if data config exists
ls -la ./GRAZPEDWRI-DX/data/data.yaml

# Check contents
cat ./GRAZPEDWRI-DX/data/data.yaml
```

### 3. YOLO Command Not Found
**Problem:** `yolo: command not found`

**Solution:**
```bash
# Install ultralytics
pip install ultralytics

# Verify installation
yolo --help
```

### 4. CUDA Out of Memory
**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
yolo train model=./ultralytics/cfg/models/v8/yolov8_CA.yaml \
           data=./GRAZPEDWRI-DX/data/data.yaml \
           batch=8 \
           epochs=100 imgsz=640 device=0 \
           project=$(pwd)/wind_turbine_attention \
           name=coordinate_attention_small_batch
```

### 5. Permission Denied
**Problem:** `Permission denied`

**Solution:**
```bash
# Make scripts executable
chmod +x train_coordinate_attention.sh
chmod +x train_cbamv2.sh
chmod +x train_ska.sh
chmod +x train_ema.sh
chmod +x train_all_attention.sh
```

### 6. Results Not Saving in Current Directory
**Problem:** Results saving in wrong location

**Solution:**
```bash
# Use absolute path
yolo train model=./ultralytics/cfg/models/v8/yolov8_CA.yaml \
           data=./GRAZPEDWRI-DX/data/data.yaml \
           project=$(pwd)/wind_turbine_attention \
           name=coordinate_attention
```

### 7. Import Error for Attention Mechanisms
**Problem:** `ImportError: cannot import name 'CoordinateAttention'`

**Solution:**
```bash
# Test if attention mechanisms are properly installed
python test_attention_mechanisms.py

# If errors, check if the ultralytics directory is in the correct location
ls -la ./ultralytics/nn/modules/conv.py
```

## 📋 Quick Verification Checklist

### Before Training:
- [ ] YOLO package installed: `pip install ultralytics`
- [ ] Model configs exist: `ls ./ultralytics/cfg/models/v8/yolov8_*.yaml`
- [ ] Data config exists: `ls ./GRAZPEDWRI-DX/data/data.yaml`
- [ ] Scripts are executable: `ls -la *.sh`
- [ ] Attention mechanisms work: `python test_attention_mechanisms.py`

### During Training:
- [ ] Results saving to current directory: `ls -la ./wind_turbine_attention/`
- [ ] GPU memory sufficient (reduce batch size if needed)
- [ ] Dataset paths are correct in data.yaml

### After Training:
- [ ] Model weights saved: `ls ./wind_turbine_attention/*/weights/best.pt`
- [ ] Training curves generated: `ls ./wind_turbine_attention/*/results.png`
- [ ] Validation results available: `ls ./wind_turbine_attention/*/confusion_matrix.png`

## 🔄 Training Methods (In Order of Reliability)

### 1. Python Script (Most Reliable)
```bash
python train_direct.py ca      # Single model
python train_direct.py all     # All models
```

### 2. Shell Scripts (If Working)
```bash
./train_coordinate_attention.sh
./train_all_attention.sh
```

### 3. Direct YOLO CLI
```bash
yolo train model=./ultralytics/cfg/models/v8/yolov8_CA.yaml \
           data=./GRAZPEDWRI-DX/data/data.yaml \
           epochs=100 batch=16 imgsz=640 device=0 \
           project=$(pwd)/wind_turbine_attention \
           name=coordinate_attention
```

### 4. Original Python Scripts
```bash
python train_wind_turbine_with_attention.py --model yolov8_CA --data ./GRAZPEDWRI-DX/data/data.yaml
python start_train.py --model ./ultralytics/cfg/models/v8/yolov8_CA.yaml --data_dir ./GRAZPEDWRI-DX/data/data.yaml
```

## 🆘 If All Else Fails

### Minimal Working Example:
```bash
# Test basic YOLO functionality
yolo train model=yolov8n.pt data=coco8.yaml epochs=1 batch=1 imgsz=640 device=cpu

# If this works, the issue is with the custom attention mechanisms
# If this doesn't work, there's a fundamental YOLO installation issue
```

### Contact Information:
- Check the README_NEW_ATTENTION.md for detailed documentation
- Run `python test_attention_mechanisms.py` to verify installations
- Try the Python script approach first: `python train_direct.py ca`

## 📊 Expected Output Structure

After successful training, you should see:
```
./wind_turbine_attention/
├── coordinate_attention/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── labels.jpg
│   └── val_batch*.jpg
├── cbamv2/
├── ska/
└── ema/
```

Each experiment directory contains:
- `weights/best.pt` - Best model weights
- `results.png` - Training curves
- `confusion_matrix.png` - Validation metrics
- `labels.jpg` - Ground truth visualization
- `val_batch*.jpg` - Validation predictions