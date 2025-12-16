g 🎯 Fixed YOLO Commands for Current Directory Saving

These commands are guaranteed to save results in your current working directory using **absolute paths**.

## 🚀 Ready-to-Use Commands

### Current Directory: `/home/sdasl/improve_yolo_7162025/FDI_Yolov8`

Copy and paste these commands directly:

### 1. Coordinate Attention (Best for Linear Cracks)
```bash
yolo train \
    model=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/ultralytics/cfg/models/v8/yolov8_CA.yaml \
    data=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/GRAZPEDWRI-DX/data/data.yaml \
    epochs=100 batch=16 imgsz=640 device=0 \
    project=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/wind_turbine_attention \
    name=coordinate_attention \
    save=true val=true plots=true verbose=true exist_ok=true
```

### 2. CBAM v2 (Best for Surface Damage)
```bash
yolo train \
    model=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/ultralytics/cfg/models/v8/yolov8_CBAMv2.yaml \
    data=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/GRAZPEDWRI-DX/data/data.yaml \
    epochs=100 batch=16 imgsz=640 device=0 \
    project=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/wind_turbine_attention \
    name=cbamv2 \
    save=true val=true plots=true verbose=true exist_ok=true
```

### 3. Selective Kernel Attention (Best for Multi-Scale)
```bash
yolo train \
    model=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/ultralytics/cfg/models/v8/yolov8_SKA.yaml \
    data=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/GRAZPEDWRI-DX/data/data.yaml \
    epochs=100 batch=16 imgsz=640 device=0 \
    project=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/wind_turbine_attention \
    name=ska \
    save=true val=true plots=true verbose=true exist_ok=true
```

### 4. Efficient Multi-Scale Attention (Best for Real-Time)
```bash
yolo train \
    model=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/ultralytics/cfg/models/v8/yolov8_EMA.yaml \
    data=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/GRAZPEDWRI-DX/data/data.yaml \
    epochs=100 batch=16 imgsz=640 device=0 \
    project=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/wind_turbine_attention \
    name=ema \
    save=true val=true plots=true verbose=true exist_ok=true
```

## 🔧 Alternative Scripts (Fixed)

### Option 1: Shell Script with Absolute Paths
```bash
# Train single model
./train_fixed_dir.sh ca      # Coordinate Attention
./train_fixed_dir.sh cbamv2  # CBAM v2
./train_fixed_dir.sh ska     # Selective Kernel Attention
./train_fixed_dir.sh ema     # Efficient Multi-Scale Attention

# Train all models
./train_fixed_dir.sh all
```

### Option 2: Python Script with Absolute Paths
```bash
# Train single model
python train_direct.py ca      # Coordinate Attention
python train_direct.py cbamv2  # CBAM v2
python train_direct.py ska     # Selective Kernel Attention
python train_direct.py ema     # Efficient Multi-Scale Attention

# Train all models
python train_direct.py all
```

## 📁 Expected Results Location

All results will be saved to:
```
/home/sdasl/improve_yolo_7162025/FDI_Yolov8/wind_turbine_attention/
├── coordinate_attention/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.png
│   ├── confusion_matrix.png
│   └── val_batch*.jpg
├── cbamv2/
├── ska/
└── ema/
```

## 🧪 Test Before Full Training

Run this test to verify directory saving works:
```bash
python test_directory_fix.py
```

This will run 1 epoch to confirm results save in the current directory.

## 🔧 Troubleshooting

### If results still save to wrong location:

1. **Check current directory**:
   ```bash
   pwd
   # Should show: /home/sdasl/improve_yolo_7162025/FDI_Yolov8
   ```

2. **Use absolute paths in commands**:
   ```bash
   # Replace $(pwd) with absolute path
   yolo train project=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/wind_turbine_attention ...
   ```

3. **Set YOLO configuration**:
   ```bash
   # Override default settings
   export YOLO_CONFIG_DIR=/home/sdasl/improve_yolo_7162025/FDI_Yolov8
   ```

4. **Check for conflicting YOLO installations**:
   ```bash
   which yolo
   pip show ultralytics
   ```

## 📊 Quick Start (30 seconds)

```bash
# Test that directory saving works
python test_directory_fix.py

# If test passes, run training
./train_fixed_dir.sh ca

# Or use direct command
yolo train model=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/ultralytics/cfg/models/v8/yolov8_CA.yaml \
           data=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/GRAZPEDWRI-DX/data/data.yaml \
           epochs=100 batch=16 imgsz=640 device=0 \
           project=/home/sdasl/improve_yolo_7162025/FDI_Yolov8/wind_turbine_attention \
           name=coordinate_attention \
           save=true val=true plots=true exist_ok=true
```

Results will be saved in: `/home/sdasl/improve_yolo_7162025/FDI_Yolov8/wind_turbine_attention/`
