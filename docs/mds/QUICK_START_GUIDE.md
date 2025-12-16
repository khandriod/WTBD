# 🚀 Quick Start Guide: Wind Turbine Damage Detection with Latest Attention Mechanisms

## ✅ Prerequisites
- YOLO package installed (`pip install ultralytics`)
- Dataset in `./GRAZPEDWRI-DX/data/` directory
- GPU recommended for training

## 🎯 Choose Your Attention Mechanism

| Attention | Best For | Command |
|-----------|----------|---------|
| **Coordinate Attention** | Linear cracks, edge erosion | `./train_coordinate_attention.sh` |
| **CBAM v2** | Surface damage, texture issues | `./train_cbamv2.sh` |
| **Selective Kernel** | Multi-scale damage (cracks + erosion) | `./train_ska.sh` |
| **Efficient Multi-Scale** | Real-time drone inspection | `./train_ema.sh` |

## 🏃‍♂️ Quick Start (30 seconds)

### Option 1: Train Single Model (Recommended)
```bash
# Make scripts executable (first time only)
chmod +x *.sh

# Train best model for crack detection
./train_coordinate_attention.sh
```

### Option 2: Train All Models (Takes 6-8 hours)
```bash
# Train all 4 attention mechanisms
./train_all_attention.sh
```

### Option 3: Direct YOLO Command
```bash
# Train with Coordinate Attention
yolo train model=./ultralytics/cfg/models/v8/yolov8_CA.yaml \
           data=./GRAZPEDWRI-DX/data/data.yaml \
           epochs=100 batch=16 imgsz=640 device=0 \
           project=$(pwd)/wind_turbine_attention \
           name=coordinate_attention \
           save=true val=true plots=true exist_ok=true
```

## 📁 Results Location
All results are saved in your **current directory**:
```
./wind_turbine_attention/
├── coordinate_attention/    # Coordinate Attention results
│   ├── weights/
│   │   ├── best.pt         # Best model weights
│   │   └── last.pt         # Last epoch weights
│   ├── results.png         # Training curves
│   ├── confusion_matrix.png # Validation accuracy
│   └── val_batch*.jpg      # Validation predictions
├── cbamv2/                 # CBAM v2 results
├── ska/                    # Selective Kernel Attention results
└── ema/                    # Efficient Multi-Scale Attention results
```

## 🔍 Quick Validation
```bash
# Test if attention mechanisms work
python test_attention_mechanisms.py

# Validate trained model
yolo val model=./wind_turbine_attention/coordinate_attention/weights/best.pt \
          data=./GRAZPEDWRI-DX/data/data.yaml
```

## 📊 Performance Comparison
After training, compare models:
```bash
# View training curves
open ./wind_turbine_attention/coordinate_attention/results.png
open ./wind_turbine_attention/cbamv2/results.png
open ./wind_turbine_attention/ska/results.png
open ./wind_turbine_attention/ema/results.png

# Check validation accuracy
cat ./wind_turbine_attention/*/results.csv
```

## 🚨 Troubleshooting

### Problem: "Model configuration not found"
```bash
# Check if model configs exist
ls -la ./ultralytics/cfg/models/v8/yolov8_*.yaml
```

### Problem: "Dataset configuration not found"
```bash
# Check if data config exists
ls -la ./GRAZPEDWRI-DX/data/data.yaml
cat ./GRAZPEDWRI-DX/data/data.yaml
```

### Problem: Out of memory
```bash
# Reduce batch size
yolo train model=./ultralytics/cfg/models/v8/yolov8_CA.yaml \
           data=./GRAZPEDWRI-DX/data/data.yaml \
           batch=8 \
           project=$(pwd)/wind_turbine_attention \
           name=coordinate_attention_small_batch
```

### Problem: Want to resume training
```bash
# Resume from last checkpoint
yolo train model=./ultralytics/cfg/models/v8/yolov8_CA.yaml \
           data=./GRAZPEDWRI-DX/data/data.yaml \
           resume=true \
           project=$(pwd)/wind_turbine_attention \
           name=coordinate_attention
```

## 🎯 Next Steps
1. ✅ Train your preferred attention mechanism
2. 📊 Compare results with baseline models
3. 🔍 Test on real wind turbine images
4. 🚁 Deploy best model for drone inspection

## 📝 Wind Turbine Damage Classes
Your models will detect:
- **Crack** - Linear fractures in blade material
- **Leading Edge Erosion** - Wear on front edge
- **Lightning Strike** - Electrical discharge damage
- **Surface Damage** - General irregularities
- **Surface Dust** - Aerodynamic-affecting accumulation

## 💡 Tips for Best Results
- **Start with Coordinate Attention** for crack detection
- **Use CBAM v2** for surface damage focus
- **Try Selective Kernel** for mixed damage types
- **Choose EMA** for real-time deployment
- **Compare all models** for your specific dataset

## 🆘 Need Help?
- Check `README_NEW_ATTENTION.md` for detailed documentation
- Run `python test_attention_mechanisms.py` to verify installation
- View example outputs in `./wind_turbine_attention/` after training