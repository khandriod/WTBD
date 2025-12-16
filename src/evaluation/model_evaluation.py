#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Evaluates all trained YOLO models on test dataset and saves results
"""

import os
import json
import random
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import YOLO from ultralytics
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import torch
import cv2

def setup_paths():
    """Setup all necessary paths"""
    base_dir = Path("/home/sdasl/improve_yolo_7162025/FDI_extra_attn")
    experiments_dir = base_dir / "all_training_experiments"
    test_images_dir = base_dir / "GRAZPEDWRI-DX" / "data" / "images" / "test"
    data_yaml = base_dir / "GRAZPEDWRI-DX" / "data" / "data.yaml"
    
    return base_dir, experiments_dir, test_images_dir, data_yaml

def get_all_models(experiments_dir):
    """Find all trained models in the experiments directory"""
    models = []
    for model_dir in experiments_dir.iterdir():
        if model_dir.is_dir():
            weights_dir = model_dir / "weights"
            best_pt = weights_dir / "best.pt"
            if best_pt.exists():
                models.append({
                    'name': model_dir.name,
                    'path': str(model_dir),
                    'weights': str(best_pt),
                    'args_yaml': str(model_dir / "args.yaml")
                })
    return models

def select_test_images(test_images_dir, num_images=50):
    """Randomly select test images for evaluation"""
    all_images = list(test_images_dir.glob("*.jpg"))
    if len(all_images) < num_images:
        print(f"Warning: Only {len(all_images)} images available, using all of them")
        return all_images
    
    # Set seed for reproducible results
    random.seed(42)
    selected_images = random.sample(all_images, num_images)
    return sorted(selected_images)

def load_model_info(args_yaml_path):
    """Load model configuration information"""
    try:
        import yaml
        with open(args_yaml_path, 'r') as f:
            args = yaml.safe_load(f)
        return args
    except Exception as e:
        print(f"Error loading {args_yaml_path}: {e}")
        return {}

def evaluate_single_model(model_info, test_images, data_yaml, class_names):
    """Evaluate a single model on test images"""
    print(f"\n🔍 Evaluating model: {model_info['name']}")
    print(f"📁 Loading weights from: {model_info['weights']}")
    
    try:
        # Load the model
        model = YOLO(model_info['weights'])
        
        # Create results directory for this model
        model_dir = Path(model_info['path'])
        results_dir = model_dir / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        predictions_dir = results_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        # Run predictions on test images
        results = []
        predictions_data = []
        
        print(f"📊 Running inference on {len(test_images)} test images...")
        
        for i, img_path in enumerate(test_images):
            print(f"Processing {i+1}/{len(test_images)}: {img_path.name}")
            
            # Run prediction
            pred_results = model.predict(
                source=str(img_path),
                conf=0.25,
                iou=0.45,
                save=False,
                verbose=False
            )
            
            # Save prediction visualization
            if pred_results:
                result = pred_results[0]
                
                # Save annotated image
                annotated_img = result.plot()
                save_path = predictions_dir / f"{img_path.stem}_pred.jpg"
                cv2.imwrite(str(save_path), annotated_img)
                
                # Extract prediction data
                if result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        predictions_data.append({
                            'image': img_path.name,
                            'class': int(cls),
                            'class_name': class_names[int(cls)],
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
        
        # Run validation to get comprehensive metrics
        print("📈 Computing detailed metrics...")
        val_results = model.val(
            data=str(data_yaml),
            split='test',
            batch=1,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            save_json=True,
            save=False,
            plots=True,
            verbose=False
        )
        
        # Extract key metrics
        metrics = {
            'model_name': model_info['name'],
            'total_parameters': sum(p.numel() for p in model.model.parameters()),
            'test_images_count': len(test_images),
            'total_predictions': len(predictions_data),
            'precision': float(val_results.box.mp) if hasattr(val_results.box, 'mp') else 0.0,
            'recall': float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 0.0,
            'mAP50': float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0.0,
            'mAP50_95': float(val_results.box.map) if hasattr(val_results.box, 'map') else 0.0,
            'fitness': float(val_results.fitness) if hasattr(val_results, 'fitness') else 0.0,
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Add per-class metrics if available
        if hasattr(val_results.box, 'ap_class_index') and val_results.box.ap_class_index is not None:
            for i, class_idx in enumerate(val_results.box.ap_class_index):
                if i < len(class_names):
                    metrics[f'{class_names[class_idx]}_AP50'] = float(val_results.box.ap50[i])
                    metrics[f'{class_names[class_idx]}_AP'] = float(val_results.box.ap[i])
        
        # Save metrics to JSON
        metrics_file = results_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions data
        predictions_file = results_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        # Load model configuration
        model_config = load_model_info(model_info['args_yaml'])
        config_file = results_dir / "model_config.json"
        with open(config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✅ Model {model_info['name']} evaluation completed!")
        print(f"📊 mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50_95']:.4f}")
        print(f"💾 Results saved in: {results_dir}")
        
        return metrics, predictions_data
        
    except Exception as e:
        print(f"❌ Error evaluating model {model_info['name']}: {e}")
        return None, None

def create_comparison_report(all_metrics, output_dir):
    """Create a comprehensive comparison report of all models"""
    print("\n📈 Creating comparison report...")
    
    if not all_metrics:
        print("No metrics available for comparison")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Save detailed CSV report
    csv_file = output_dir / "model_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"📊 Detailed comparison saved to: {csv_file}")
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # mAP50 comparison
    df_sorted = df.sort_values('mAP50', ascending=True)
    axes[0,0].barh(range(len(df_sorted)), df_sorted['mAP50'])
    axes[0,0].set_yticks(range(len(df_sorted)))
    axes[0,0].set_yticklabels(df_sorted['model_name'], fontsize=8)
    axes[0,0].set_xlabel('mAP@0.5')
    axes[0,0].set_title('mAP@0.5 Comparison')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # mAP50-95 comparison
    df_sorted = df.sort_values('mAP50_95', ascending=True)
    axes[0,1].barh(range(len(df_sorted)), df_sorted['mAP50_95'])
    axes[0,1].set_yticks(range(len(df_sorted)))
    axes[0,1].set_yticklabels(df_sorted['model_name'], fontsize=8)
    axes[0,1].set_xlabel('mAP@0.5:0.95')
    axes[0,1].set_title('mAP@0.5:0.95 Comparison')
    axes[0,1].grid(axis='x', alpha=0.3)
    
    # Precision vs Recall scatter
    axes[1,0].scatter(df['recall'], df['precision'], s=100, alpha=0.7)
    for i, row in df.iterrows():
        axes[1,0].annotate(row['model_name'], 
                          (row['recall'], row['precision']), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision vs Recall')
    axes[1,0].grid(True, alpha=0.3)
    
    # Model size vs Performance
    axes[1,1].scatter(df['total_parameters'], df['mAP50'], s=100, alpha=0.7)
    for i, row in df.iterrows():
        axes[1,1].annotate(row['model_name'], 
                          (row['total_parameters'], row['mAP50']), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1,1].set_xlabel('Total Parameters')
    axes[1,1].set_ylabel('mAP@0.5')
    axes[1,1].set_title('Model Size vs Performance')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "model_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    summary_file = output_dir / "summary_report.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Models Evaluated: {len(df)}\n")
        f.write(f"Test Images Used: {df.iloc[0]['test_images_count'] if len(df) > 0 else 0}\n\n")
        
        f.write("TOP PERFORMING MODELS:\n")
        f.write("-" * 40 + "\n")
        
        # Top 3 by mAP50
        top_map50 = df.nlargest(3, 'mAP50')
        f.write("Top 3 by mAP@0.5:\n")
        for i, (_, row) in enumerate(top_map50.iterrows(), 1):
            f.write(f"{i}. {row['model_name']}: {row['mAP50']:.4f}\n")
        
        f.write("\nTop 3 by mAP@0.5:0.95:\n")
        top_map95 = df.nlargest(3, 'mAP50_95')
        for i, (_, row) in enumerate(top_map95.iterrows(), 1):
            f.write(f"{i}. {row['model_name']}: {row['mAP50_95']:.4f}\n")
        
        f.write("\nALL MODELS DETAILED METRICS:\n")
        f.write("-" * 60 + "\n")
        for _, row in df.iterrows():
            f.write(f"\nModel: {row['model_name']}\n")
            f.write(f"  Parameters: {row['total_parameters']:,}\n")
            f.write(f"  Precision: {row['precision']:.4f}\n")
            f.write(f"  Recall: {row['recall']:.4f}\n")
            f.write(f"  mAP@0.5: {row['mAP50']:.4f}\n")
            f.write(f"  mAP@0.5:0.95: {row['mAP50_95']:.4f}\n")
            f.write(f"  Fitness: {row['fitness']:.4f}\n")
    
    print(f"📋 Summary report saved to: {summary_file}")
    print(f"📈 Visualization saved to: {plot_file}")

def main():
    """Main evaluation function"""
    print("🚀 Starting Comprehensive Model Evaluation")
    print("=" * 60)
    
    # Setup paths
    base_dir, experiments_dir, test_images_dir, data_yaml = setup_paths()
    
    # Verify paths exist
    if not experiments_dir.exists():
        print(f"❌ Experiments directory not found: {experiments_dir}")
        return
    
    if not test_images_dir.exists():
        print(f"❌ Test images directory not found: {test_images_dir}")
        return
    
    # Get all trained models
    models = get_all_models(experiments_dir)
    if not models:
        print("❌ No trained models found!")
        return
    
    print(f"✅ Found {len(models)} trained models:")
    for model in models:
        print(f"   - {model['name']}")
    
    # Select test images
    test_images = select_test_images(test_images_dir, 50)
    print(f"✅ Selected {len(test_images)} test images for evaluation")
    
    # Load class names
    class_names = ['Crack', 'Leading Edge Erosion', 'Lightning Strike', 'Surface Damage', 'Surface Dust']
    
    # Create output directory for comparison results
    output_dir = base_dir / "model_evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    # Evaluate each model
    all_metrics = []
    all_predictions = []
    
    for model_info in models:
        metrics, predictions = evaluate_single_model(
            model_info, test_images, data_yaml, class_names
        )
        if metrics:
            all_metrics.append(metrics)
        if predictions:
            all_predictions.extend(predictions)
    
    # Create comparison report
    if all_metrics:
        create_comparison_report(all_metrics, output_dir)
        
        # Save all predictions combined
        combined_predictions_file = output_dir / "all_predictions.json"
        with open(combined_predictions_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 All results saved in: {output_dir}")
        print(f"📊 {len(all_metrics)} models evaluated on {len(test_images)} test images")
        
        # Show quick summary
        df = pd.DataFrame(all_metrics)
        best_model = df.loc[df['mAP50'].idxmax()]
        print(f"🏆 Best performing model (mAP50): {best_model['model_name']} ({best_model['mAP50']:.4f})")
        
    else:
        print("❌ No models were successfully evaluated")

if __name__ == "__main__":
    main() 