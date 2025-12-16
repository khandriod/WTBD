#!/usr/bin/env python3
"""
Simple YOLO Model Evaluator with Prediction Capabilities
Evaluates and compares YOLO models on test datasets and runs predictions on images
"""

import os
import json
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YOLOModelEvaluator:
    def __init__(self, test_data_path):
        """
        Initialize the YOLO model evaluator.
        
        Args:
            test_data_path (str): Path to the test dataset YAML file
        """
        self.test_data_path = test_data_path
        self.results = []
        self.loaded_models = {}  # Cache for loaded models
        
        # Color palette for visualizations
        self.colors = {
            'Crack': '#FF4444',                # Red
            'Leading Edge Erosion': '#FF8800', # Orange
            'Lightning Strike': '#8A2BE2',     # Purple
            'Surface Damage': '#FFD700',       # Gold
            'Surface Dust': '#32CD32'          # Green
        }
        
    def load_model(self, model_path, model_name):
        """Load and cache a YOLO model."""
        if model_name not in self.loaded_models:
            print(f"🚀 Loading {model_name}...")
            self.loaded_models[model_name] = YOLO(model_path)
            print(f"✅ {model_name} loaded successfully!")
        return self.loaded_models[model_name]
        
    def predict_on_images(self, model_path, model_name, image_paths, conf_threshold=0.25, 
                         save_predictions=True, output_dir="predictions"):
        """
        Run predictions on images using a YOLO model.
        
        Args:
            model_path (str): Path to model weights
            model_name (str): Model identifier
            image_paths (list): List of image file paths
            conf_threshold (float): Confidence threshold
            save_predictions (bool): Save prediction results
            output_dir (str): Output directory for results
            
        Returns:
            list: Prediction results for each image
        """
        model = self.load_model(model_path, model_name)
        
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
            
        output_dir = Path(output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔍 Running predictions with {model_name}...")
        print(f"📁 Images: {len(image_paths)}")
        print(f"💾 Output: {output_dir}")
        
        all_predictions = []
        
        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)
            print(f"📸 Processing [{i+1}/{len(image_paths)}]: {image_path.name}")
            
            try:
                # Run prediction
                results = model(str(image_path), conf=conf_threshold)
                
                # Extract predictions
                predictions = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = result.names[class_id]
                            
                            prediction = {
                                'image': image_path.name,
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'center_x': float((x1 + x2) / 2),
                                'center_y': float((y1 + y2) / 2),
                                'width': float(x2 - x1),
                                'height': float(y2 - y1)
                            }
                            predictions.append(prediction)
                
                # Store results
                image_result = {
                    'image_path': str(image_path),
                    'model_name': model_name,
                    'predictions': predictions,
                    'detection_count': len(predictions),
                    'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                all_predictions.append(image_result)
                
                print(f"  ✅ {len(predictions)} detections found")
                
                # Save individual results
                if save_predictions:
                    # Save JSON
                    json_file = output_dir / f"{image_path.stem}_predictions.json"
                    with open(json_file, 'w') as f:
                        json.dump(image_result, f, indent=2)
                    
                    # Create visualization
                    self._create_prediction_visualization(
                        image_path, predictions, 
                        output_dir / f"{image_path.stem}_predicted.jpg",
                        model_name
                    )
                
            except Exception as e:
                print(f"  ❌ Error processing {image_path.name}: {e}")
                continue
        
        # Save batch results
        if save_predictions and all_predictions:
            batch_file = output_dir / f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(batch_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            
            self._create_prediction_summary(all_predictions, output_dir)
            
        print(f"🎉 Prediction complete! {len(all_predictions)} images processed")
        return all_predictions
    
    def _create_prediction_visualization(self, image_path, predictions, output_path, model_name):
        """Create a visual prediction result image."""
        
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Add title
        ax.set_title(f'{model_name} Predictions - {Path(image_path).name}', 
                    fontsize=14, fontweight='bold')
        
        # Draw predictions
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            class_name = pred['class_name']
            confidence = pred['confidence']
            
            # Get color
            color = self.colors.get(class_name, '#FFFFFF')
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = f'{class_name}\n{confidence:.1%}'
            ax.text(x1, y1-10, label_text, 
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
        
        # Add summary
        summary_text = f'Detections: {len(predictions)}'
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
               fontsize=12, fontweight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7),
               verticalalignment='top')
        
        # Clean up axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_prediction_summary(self, all_predictions, output_dir):
        """Create a summary report of predictions."""
        
        summary_file = output_dir / "prediction_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# Prediction Summary\n\n")
            f.write(f"**Model**: {all_predictions[0]['model_name']}  \n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Images Processed**: {len(all_predictions)}  \n\n")
            
            # Statistics
            total_detections = sum(p['detection_count'] for p in all_predictions)
            avg_detections = total_detections / len(all_predictions) if all_predictions else 0
            
            f.write(f"## 📊 Summary Statistics\n\n")
            f.write(f"- **Total Detections**: {total_detections}\n")
            f.write(f"- **Average per Image**: {avg_detections:.1f}\n")
            
            # Per-image results
            f.write(f"\n## 📸 Per-Image Results\n\n")
            f.write(f"| Image | Detections | Classes Detected |\n")
            f.write(f"|-------|------------|------------------|\n")
            
            for pred in all_predictions:
                classes = set(p['class_name'] for p in pred['predictions'])
                classes_str = ', '.join(sorted(classes)) if classes else 'None'
                f.write(f"| {Path(pred['image_path']).name} | {pred['detection_count']} | {classes_str} |\n")
        
        print(f"📝 Summary saved: {summary_file}")
    
    def predict_directory(self, model_path, model_name, images_dir, 
                         conf_threshold=0.25, max_images=None):
        """
        Run predictions on all images in a directory.
        
        Args:
            model_path (str): Path to model weights  
            model_name (str): Model identifier
            images_dir (str): Directory containing images
            conf_threshold (float): Confidence threshold
            max_images (int): Maximum number of images to process
            
        Returns:
            list: Prediction results
        """
        images_dir = Path(images_dir)
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No images found in {images_dir}")
            return []
        
        # Limit images if specified
        if max_images and len(image_files) > max_images:
            print(f"📊 Limiting to {max_images} images (found {len(image_files)})")
            image_files = image_files[:max_images]
        
        return self.predict_on_images(
            model_path, model_name, image_files, conf_threshold
        )
    
    def compare_predictions(self, model_configs, image_paths, conf_threshold=0.25):
        """
        Compare predictions from multiple models on the same images.
        
        Args:
            model_configs (list): List of model config dicts
            image_paths (list): List of image paths
            conf_threshold (float): Confidence threshold
            
        Returns:
            dict: Comparison results
        """
        print(f"🔄 Running model comparison on {len(image_paths)} images...")
        
        comparison_results = {}
        
        for config in model_configs:
            model_name = config['name']
            model_path = config['path']
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_name}")
                continue
                
            results = self.predict_on_images(
                model_path, model_name, image_paths, 
                conf_threshold, save_predictions=True,
                output_dir=f"comparison_predictions"
            )
            
            comparison_results[model_name] = results
        
        # Create comparison summary
        self._create_comparison_summary(comparison_results, image_paths)
        
        return comparison_results
    
    def _create_comparison_summary(self, comparison_results, image_paths):
        """Create a comparison summary across models."""
        
        output_dir = Path("comparison_predictions")
        output_dir.mkdir(exist_ok=True)
        
        summary_file = output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        
        with open(summary_file, 'w') as f:
            f.write("# Model Prediction Comparison\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Images**: {len(image_paths)}  \n")
            f.write(f"**Models**: {len(comparison_results)}  \n\n")
            
            # Summary table
            f.write("## 📊 Detection Summary\n\n")
            f.write("| Model | Total Detections | Avg per Image | Images with Detections |\n")
            f.write("|-------|------------------|---------------|------------------------|\n")
            
            for model_name, results in comparison_results.items():
                total_det = sum(r['detection_count'] for r in results)
                avg_det = total_det / len(results) if results else 0
                images_with_det = sum(1 for r in results if r['detection_count'] > 0)
                
                f.write(f"| {model_name} | {total_det} | {avg_det:.1f} | {images_with_det}/{len(results)} |\n")
        
        print(f"📊 Comparison summary saved: {summary_file}")

    def evaluate_model(self, model_path, model_name, conf_threshold=0.25):
        """
        Evaluate a single YOLO model.
        
        Args:
            model_path (str): Path to the model weights
            model_name (str): Name identifier for the model
            conf_threshold (float): Confidence threshold for predictions
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n🔍 Evaluating {model_name}...")
        print(f"📁 Model: {model_path}")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(
                data=self.test_data_path,
                conf=conf_threshold,
                iou=0.5,
                verbose=False
            )
            
            # Extract metrics with correct attributes
            metrics = {
                'model_name': model_name,
                'model_path': model_path,
                'conf_threshold': conf_threshold,
                'precision': float(results.box.mp) if hasattr(results.box, 'mp') and results.box.mp is not None else 0.0,
                'recall': float(results.box.mr) if hasattr(results.box, 'mr') and results.box.mr is not None else 0.0,
                'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') and results.box.map50 is not None else 0.0,
                'mAP50_95': float(results.box.map) if hasattr(results.box, 'map') and results.box.map is not None else 0.0,
                'total_images': len(getattr(results, 'names', {})) if hasattr(results, 'names') else 0,
                'inference_time': float(results.speed.get('inference', 0.0)) if hasattr(results, 'speed') and isinstance(results.speed, dict) else 0.0,
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add per-class metrics if available
            if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
                class_names = list(results.names.values()) if hasattr(results, 'names') else ['Crack', 'Leading_Edge_Erosion', 'Lightning_Strike', 'Surface_Damage', 'Surface_Dust']
                ap_values = results.box.ap50
                if ap_values is not None and len(ap_values) > 0:
                    for i, class_name in enumerate(class_names):
                        if i < len(ap_values):
                            metrics[f'{class_name}_AP50'] = float(ap_values[i]) if ap_values[i] is not None else 0.0
            
            print(f"✅ Evaluation complete!")
            print(f"   📊 Precision: {metrics['precision']:.1%}")
            print(f"   📊 Recall: {metrics['recall']:.1%}")
            print(f"   📊 mAP50: {metrics['mAP50']:.1%}")
            print(f"   📊 mAP50-95: {metrics['mAP50_95']:.1%}")
            
            self.results.append(metrics)
            return metrics
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return None
    
    def evaluate_multiple_models(self, model_configs):
        """
        Evaluate multiple YOLO models.
        
        Args:
            model_configs (list): List of dicts with 'path' and 'name' keys
            
        Returns:
            list: List of evaluation results
        """
        print(f"🚀 Starting batch evaluation of {len(model_configs)} models...")
        
        all_results = []
        
        for i, config in enumerate(model_configs):
            print(f"\n📈 Progress: [{i+1}/{len(model_configs)}]")
            
            result = self.evaluate_model(
                model_path=config['path'],
                model_name=config['name'],
                conf_threshold=config.get('conf_threshold', 0.25)
            )
            
            if result:
                all_results.append(result)
        
        print(f"\n🎉 Batch evaluation complete! {len(all_results)} models evaluated successfully.")
        return all_results
    
    def generate_comparison_report(self, output_dir="evaluation_results"):
        """Generate comparison report of all evaluated models."""
        
        if not self.results:
            print("❌ No evaluation results available!")
            return
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Sort by mAP50-95 (descending)
        df = df.sort_values('mAP50_95', ascending=False)
        
        # Generate CSV report
        csv_file = output_dir / f"model_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(csv_file, index=False)
        
        # Generate summary report
        report_file = output_dir / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# YOLO Model Evaluation Report\n\n")
            f.write(f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Dataset**: {self.test_data_path}  \n")
            f.write(f"**Models Evaluated**: {len(self.results)}  \n\n")
            
            # Overall Rankings
            f.write("## 🏆 Model Rankings\n\n")
            f.write("| Rank | Model | mAP50 | mAP50-95 | Precision | Recall | Inference (ms) |\n")
            f.write("|------|-------|-------|----------|-----------|--------|----------------|\n")
            
            for i, row in df.iterrows():
                rank = df.index.get_loc(i) + 1
                f.write(f"| {rank} | {row['model_name']} | {row['mAP50']:.1%} | {row['mAP50_95']:.1%} | "
                       f"{row['precision']:.1%} | {row['recall']:.1%} | {row['inference_time']:.1f}ms |\n")
            
            # Best performers
            f.write(f"\n## 🥇 Best Performers\n\n")
            
            best_map = df.iloc[0]
            best_precision = df.loc[df['precision'].idxmax()]
            best_recall = df.loc[df['recall'].idxmax()]
            best_speed = df.loc[df['inference_time'].idxmin()]
            
            f.write(f"- **Best Overall (mAP50-95)**: {best_map['model_name']} - {best_map['mAP50_95']:.1%}\n")
            f.write(f"- **Best Precision**: {best_precision['model_name']} - {best_precision['precision']:.1%}\n")
            f.write(f"- **Best Recall**: {best_recall['model_name']} - {best_recall['recall']:.1%}\n")
            f.write(f"- **Fastest Inference**: {best_speed['model_name']} - {best_speed['inference_time']:.1f}ms\n")
            
            # Statistics
            f.write(f"\n## 📊 Statistics\n\n")
            f.write(f"- **Average mAP50**: {df['mAP50'].mean():.1%}\n")
            f.write(f"- **Average mAP50-95**: {df['mAP50_95'].mean():.1%}\n")
            f.write(f"- **Average Precision**: {df['precision'].mean():.1%}\n")
            f.write(f"- **Average Recall**: {df['recall'].mean():.1%}\n")
            f.write(f"- **Average Inference Time**: {df['inference_time'].mean():.1f}ms\n")
        
        print(f"📊 Reports generated:")
        print(f"   📄 CSV: {csv_file}")
        print(f"   📋 Summary: {report_file}")
        
        return df
    
    def print_comparison_table(self):
        """Print a formatted comparison table to console."""
        
        if not self.results:
            print("❌ No evaluation results available!")
            return
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('mAP50_95', ascending=False)
        
        print("\n" + "="*80)
        print("🏆 YOLO MODEL EVALUATION RESULTS")
        print("="*80)
        
        print(f"{'Rank':<4} {'Model':<25} {'mAP50':<8} {'mAP50-95':<8} {'Precision':<9} {'Recall':<8} {'Speed(ms)':<9}")
        print("-"*80)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            rank = i + 1
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            
            print(f"{emoji} {rank:<2} {row['model_name']:<25} {row['mAP50']:<8.1%} {row['mAP50_95']:<8.1%} "
                  f"{row['precision']:<9.1%} {row['recall']:<8.1%} {row['inference_time']:<9.1f}")
        
        print("="*80)


def main():
    """Main function to run model evaluation and predictions."""
    
    # Configuration
    test_data_path = "./GRAZPEDWRI-DX/data/data.yaml"
    
    # Define models to evaluate
    models_to_evaluate = [
        {
            'name': 'SKA_600_epochs',
            'path': 'training_experiments_600/selective_kernel_attention_20250719_021254/weights/best.pt',
            'conf_threshold': 0.25
        },
        {
            'name': 'EMA_600_epochs', 
            'path': 'training_experiments_600/efficient_multiscale_attention_20250719_021254/weights/best.pt',
            'conf_threshold': 0.25
        },
        {
            'name': 'MSPA_600_epochs',
            'path': 'training_experiments_600/MSPA_20250719_020920/weights/best.pt',
            'conf_threshold': 0.25
        },
        {
            'name': 'EEA_600_epochs',
            'path': 'training_experiments_600/EEA_20250719_020920/weights/best.pt',
            'conf_threshold': 0.25
        }
    ]
    
    # Filter models that exist
    available_models = []
    for model in models_to_evaluate:
        if os.path.exists(model['path']):
            available_models.append(model)
            print(f"✅ Found: {model['name']}")
        else:
            print(f"❌ Missing: {model['name']} at {model['path']}")
    
    if not available_models:
        print("❌ No model weights found! Please check the paths.")
        return
    
    # Initialize evaluator
    evaluator = YOLOModelEvaluator(test_data_path)
    
    print("\n" + "="*60)
    print("🚀 YOLO MODEL EVALUATOR & PREDICTOR")
    print("="*60)
    print("Choose an option:")
    print("1. 📊 Run full model evaluation")
    print("2. 🔍 Run predictions on sample images")
    print("3. 📁 Run predictions on directory")
    print("4. 🔄 Compare models on same images")
    print("5. 🎯 Run both evaluation and predictions")
    print("="*60)
    
    # For demo purposes, let's run predictions
    choice = input("Enter choice (1-5, or press Enter for full evaluation): ").strip()
    
    try:
        if choice == "2" or choice == "":
            # Run predictions on sample images
            print("\n🔍 Running predictions on sample images...")
            
            # Use the champion model (SKA)
            champion_model = available_models[0]  # Assuming SKA is first
            
            # Sample images (you can modify these paths)
            sample_images = []
            
            # Check for validation dataset images
            test_images_path = Path("./GRAZPEDWRI-DX/data/images/test")
            if test_images_path.exists():
                sample_images = list(test_images_path.glob("*.jpg"))[:5]  # First 5 images
                print(f"📸 Using 5 sample images from test dataset")
            
            # Fallback to other directories
            if not sample_images:
                for img_dir in ["avingrid_data", "ska_validation_results_dataset"]:
                    img_path = Path(img_dir)
                    if img_path.exists():
                        jpg_files = list(img_path.glob("*.jpg"))
                        png_files = list(img_path.glob("*.png"))
                        sample_images = (jpg_files + png_files)[:5]
                        if sample_images:
                            print(f"📸 Using 5 sample images from {img_dir}")
                            break
            
            if sample_images:
                results = evaluator.predict_on_images(
                    champion_model['path'],
                    champion_model['name'],
                    sample_images,
                    conf_threshold=0.25
                )
                
                print(f"\n🎉 Predictions complete!")
                print(f"📊 Results saved in: predictions/{champion_model['name']}/")
                print(f"🖼️  Visualizations created with bounding boxes and labels")
                print(f"📄 JSON results saved for each image")
                
            else:
                print("❌ No sample images found!")
        
        elif choice == "3":
            # Run predictions on directory
            images_dir = input("Enter images directory path: ").strip()
            if not images_dir:
                images_dir = "avingrid_data"
            
            max_images = input("Max images to process (or Enter for all): ").strip()
            max_images = int(max_images) if max_images.isdigit() else None
            
            champion_model = available_models[0]
            results = evaluator.predict_directory(
                champion_model['path'],
                champion_model['name'],
                images_dir,
                conf_threshold=0.25,
                max_images=max_images
            )
            
            if results:
                print(f"\n🎉 Directory prediction complete!")
                print(f"📊 Processed {len(results)} images")
                print(f"🎯 Total detections: {sum(r['detection_count'] for r in results)}")
            
        elif choice == "4":
            # Compare models
            print("\n🔄 Running model comparison...")
            
            # Get sample images
            test_images_path = Path("./GRAZPEDWRI-DX/data/images/test")
            if test_images_path.exists():
                sample_images = list(test_images_path.glob("*.jpg"))[:3]
            else:
                sample_images = list(Path("avingrid_data").glob("*.jpg"))[:3] if Path("avingrid_data").exists() else []
            
            if sample_images:
                results = evaluator.compare_predictions(
                    available_models[:2],  # Compare top 2 models
                    sample_images,
                    conf_threshold=0.25
                )
                
                print(f"\n🎉 Model comparison complete!")
                print(f"📊 Results saved in: comparison_predictions/")
                
            else:
                print("❌ No sample images found for comparison!")
                
        elif choice == "5":
            # Run both evaluation and predictions
            print("\n📊 Running full evaluation...")
            
            # Run evaluation
            eval_results = evaluator.evaluate_multiple_models(available_models)
            
            if eval_results:
                evaluator.print_comparison_table()
                df = evaluator.generate_comparison_report()
                
                # Run predictions with best model
                best_model = available_models[0]
                print(f"\n🔍 Running predictions with best model: {best_model['name']}")
                
                test_images_path = Path("./GRAZPEDWRI-DX/data/images/test")
                if test_images_path.exists():
                    sample_images = list(test_images_path.glob("*.jpg"))[:5]
                    
                    pred_results = evaluator.predict_on_images(
                        best_model['path'],
                        best_model['name'],
                        sample_images,
                        conf_threshold=0.25
                    )
                    
                    print(f"\n🎉 Complete analysis finished!")
                    print(f"📊 Best model: {df.iloc[0]['model_name']} ({df.iloc[0]['mAP50_95']:.1%} mAP50-95)")
                    print(f"🔍 Predictions: {sum(r['detection_count'] for r in pred_results)} detections on {len(pred_results)} images")
                
        else:
            # Default: Run evaluation only
            print("\n📊 Running model evaluation...")
            results = evaluator.evaluate_multiple_models(available_models)
            
            if results:
                evaluator.print_comparison_table()
                df = evaluator.generate_comparison_report()
                
                print(f"\n🎉 Evaluation complete! Best model: {df.iloc[0]['model_name']} "
                      f"with {df.iloc[0]['mAP50_95']:.1%} mAP50-95")
            else:
                print("❌ No successful evaluations!")
                
    except Exception as e:
        print(f"❌ Operation failed: {e}")


# Additional utility functions
def predict_single_image():
    """Standalone function to predict on a single image."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python yolo_model_evaluator.py predict <image_path> [model_name]")
        return
    
    image_path = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "SKA_600_epochs"
    
    # Model paths
    model_configs = {
        'SKA_600_epochs': 'training_experiments_600/selective_kernel_attention_20250719_021254/weights/best.pt',
        'EMA_600_epochs': 'training_experiments_600/efficient_multiscale_attention_20250719_021254/weights/best.pt',
        'MSPA_600_epochs': 'training_experiments_600/MSPA_20250719_020920/weights/best.pt',
        'EEA_600_epochs': 'training_experiments_600/EEA_20250719_020920/weights/best.pt'
    }
    
    if model_name not in model_configs:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {list(model_configs.keys())}")
        return
    
    model_path = model_configs[model_name]
    if not os.path.exists(model_path):
        print(f"❌ Model weights not found: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Run prediction
    evaluator = YOLOModelEvaluator("./GRAZPEDWRI-DX/data/data.yaml")
    results = evaluator.predict_on_images(
        model_path, model_name, [image_path], conf_threshold=0.25
    )
    
    if results:
        detections = results[0]['detection_count']
        print(f"\n🎉 Prediction complete!")
        print(f"📊 Found {detections} damage detections")
        print(f"📁 Results saved in: predictions/{model_name}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_single_image()
    else:
        main() 