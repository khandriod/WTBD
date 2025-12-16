#!/usr/bin/env python3
"""
Unified Wind Turbine Damage Detection System
Combines model evaluation, tiled detection, and batch processing
"""

import os
import sys
import json
import cv2
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime
import time

# Import enhanced components
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.detection.enhanced_tiled_detection import EnhancedTiledDetector, ProcessingConfig
    from src.utils.tile_utils import AutoTileSizer, TileConfig
    enhanced_available = True
except ImportError as e:
    print(f"⚠️  Enhanced modules not found ({e}), falling back to basic functionality")
    EnhancedTiledDetector = None
    ProcessingConfig = None
    enhanced_available = False

from ultralytics import YOLO


class WindTurbineDetectionSystem:
    """Unified detection system with multiple processing modes"""
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.class_names = ['Crack', 'Leading Edge Erosion', 'Lightning Strike', 
                           'Surface Damage', 'Surface Dust']
        
        # Default configuration
        self.config = {
            'confidence_threshold': 0.25,
            'tile_size': None,  # Auto-detect
            'overlap_ratio': 0.25,
            'use_tiled_detection': True,
            'parallel_processing': True,
            'max_workers': 4,
            'high_res_threshold': 2000,  # Switch to tiled for images larger than this
            'output_format': 'comprehensive'  # 'simple', 'detailed', 'comprehensive'
        }
        
        if config:
            self.config.update(config)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced detector if available
        self.enhanced_detector = None
        if EnhancedTiledDetector is not None:
            try:
                processing_config = ProcessingConfig(
                    tile_size=self.config['tile_size'],
                    overlap_ratio=self.config['overlap_ratio'],
                    confidence_threshold=self.config['confidence_threshold'],
                    max_workers=1,  # Always use single worker for GPU safety
                    device='0'  # Force single GPU device
                )
                self.enhanced_detector = EnhancedTiledDetector(model_path, processing_config)
            except Exception as e:
                self.logger.warning(f"Enhanced detector initialization failed: {e}")
        
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'processing_time': 0.0,
            'tiled_images': 0,
            'simple_images': 0
        }
    
    def detect_single_image(self, image_path: str, output_dir: str = "results") -> Dict:
        """Detect damages in a single image with intelligent mode selection"""
        start_time = time.time()
        
        # Load image to check resolution
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        # Decide processing mode
        use_tiled = (self.config['use_tiled_detection'] and 
                    max_dim > self.config['high_res_threshold'] and 
                    self.enhanced_detector is not None)
        
        print(f"🔍 Processing {Path(image_path).name} ({width}x{height})")
        print(f"📋 Mode: {'🧩 Tiled' if use_tiled else '⚡ Simple'} detection")
        
        self.logger.info(f"Processing {Path(image_path).name} ({width}x{height}) "
                        f"with {'tiled' if use_tiled else 'simple'} detection")
        
        if use_tiled:
            results = self._detect_tiled(image_path, output_dir)
            self.stats['tiled_images'] += 1
        else:
            results = self._detect_simple(image_path, output_dir)
            self.stats['simple_images'] += 1
        
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        results['processing_mode'] = 'tiled' if use_tiled else 'simple'
        
        # Update stats
        self.stats['total_images'] += 1
        self.stats['total_detections'] += len(results.get('detections', []))
        self.stats['processing_time'] += processing_time
        
        return results
    
    def _detect_tiled(self, image_path: str, output_dir: str) -> Dict:
        """Use enhanced tiled detection for high-resolution images"""
        try:
            return self.enhanced_detector.detect_damages_single(image_path, output_dir)
        except Exception as e:
            self.logger.warning(f"Tiled detection failed, falling back to simple: {e}")
            return self._detect_simple(image_path, output_dir)
    
    def _detect_simple(self, image_path: str, output_dir: str) -> Dict:
        """Simple detection for smaller images or fallback"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run inference with single GPU
        results = self.model.predict(
            image_path, 
            conf=self.config['confidence_threshold'],
            device='0',  # Force single GPU
            verbose=False,
            half=False  # Disable half precision for stability
        )
        
        # Extract detections
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                detection = {
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id],
                    'confidence': float(conf),
                    'bbox': [float(x) for x in box],
                    'area': float((box[2] - box[0]) * (box[3] - box[1]))
                }
                detections.append(detection)
        
        # Create results
        image = cv2.imread(image_path)
        image_shape = image.shape if image is not None else (0, 0, 0)
        
        results_dict = {
            'image_path': str(image_path),
            'image_shape': image_shape,
            'detections': detections,
            'damage_summary': self._create_damage_summary(detections),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results based on output format
        self._save_results(results_dict, output_dir, 'simple')
        
        return results_dict
    
    def _create_damage_summary(self, detections: List[Dict]) -> Dict:
        """Create damage summary from detections"""
        damage_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            damage_counts[class_name] = damage_counts.get(class_name, 0) + 1
        
        return {
            'total_damages': len(detections),
            'damage_breakdown': damage_counts,
            'confidence_stats': {
                'mean': np.mean([d['confidence'] for d in detections]) if detections else 0.0,
                'max': max([d['confidence'] for d in detections]) if detections else 0.0
            }
        }
    
    def _save_results(self, results: Dict, output_dir: str, mode: str):
        """Save results in specified format"""
        image_name = Path(results['image_path']).stem
        output_path = Path(output_dir)
        
        # Save JSON results
        json_file = output_path / f"{image_name}_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create annotated image
        if results['detections'] and self.config['output_format'] != 'simple':
            self._create_annotated_image(results, output_path / f"{image_name}_annotated.jpg")
    
    def _create_annotated_image(self, results: Dict, output_path: Path):
        """Create annotated image with detections"""
        image = cv2.imread(results['image_path'])
        if image is None:
            return
        
        # Draw detections
        for detection in results['detections']:
            bbox = [int(x) for x in detection['bbox']]
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            
            # Get color for class
            color_map = {
                0: (0, 0, 255),    # Crack - Red
                1: (0, 165, 255),  # Leading Edge Erosion - Orange
                2: (255, 0, 255),  # Lightning Strike - Magenta
                3: (0, 255, 255),  # Surface Damage - Yellow
                4: (0, 255, 0)     # Surface Dust - Green
            }
            color = color_map.get(detection['class_id'], (255, 255, 255))
            
            # Draw rectangle and label
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(image, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(str(output_path), image)
    
    def process_batch(self, input_path: str, output_dir: str = "results", 
                     max_images: Optional[int] = None) -> Dict:
        """Process multiple images efficiently"""
        input_path = Path(input_path)
        
        if input_path.is_file():
            # Single image
            results = self.detect_single_image(str(input_path), output_dir)
            return {
                'processed_images': 1,
                'total_detections': len(results.get('detections', [])),
                'processing_time': results.get('processing_time', 0),
                'results': [results]
            }
        
        elif input_path.is_dir():
            # Directory processing
            if self.enhanced_detector is not None and self.config['use_tiled_detection']:
                # Use enhanced batch processing if available
                return self.enhanced_detector.process_directory(
                    str(input_path), output_dir, max_images
                )
            else:
                # Simple batch processing
                return self._process_directory_simple(input_path, output_dir, max_images)
        
        else:
            raise ValueError(f"Invalid input path: {input_path}")
    
    def _process_directory_simple(self, images_dir: Path, output_dir: str, 
                                 max_images: Optional[int]) -> Dict:
        """Simple directory processing without enhanced features"""
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = [f for f in images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            raise ValueError(f"No images found in {images_dir}")
        
        # Process images
        results = []
        total_detections = 0
        
        for i, image_file in enumerate(image_files):
            self.logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            try:
                result = self.detect_single_image(str(image_file), output_dir)
                results.append(result)
                total_detections += len(result.get('detections', []))
            except Exception as e:
                self.logger.error(f"Failed to process {image_file.name}: {e}")
        
        return {
            'processed_images': len(results),
            'total_detections': total_detections,
            'processing_time': sum(r.get('processing_time', 0) for r in results),
            'results': results,
            'stats': self.stats.copy()
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_path': str(self.model_path),
            'model_type': 'YOLO',
            'classes': self.class_names,
            'enhanced_features_available': self.enhanced_detector is not None,
            'current_config': self.config.copy()
        }


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Unified Wind Turbine Damage Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with auto-detection
  python wind_turbine_detector.py --model path/to/model.pt --input image.jpg
  
  # Directory with tiled processing
  python wind_turbine_detector.py --model path/to/model.pt --input images_dir/ --tiled
  
  # High-resolution with custom settings
  python wind_turbine_detector.py --model path/to/model.pt --input large_image.jpg --tile-size 1024 --overlap 0.3
        """
    )
    
    # Model and I/O
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image file or directory')
    parser.add_argument('--output', type=str, default='detection_results',
                       help='Output directory (default: detection_results)')
    
    # Detection parameters
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--tile-size', type=int, default=None,
                       help='Tile size for high-res images (auto if not specified)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Tile overlap ratio (default: 0.25)')
    
    # Processing modes
    parser.add_argument('--tiled', action='store_true',
                       help='Force tiled processing (auto-detect by default)')
    parser.add_argument('--simple', action='store_true',
                       help='Force simple processing (no tiling)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    
    # Batch processing
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to process')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    # Output options
    parser.add_argument('--format', choices=['simple', 'detailed', 'comprehensive'], 
                       default='comprehensive', help='Output detail level')
    parser.add_argument('--no-annotate', action='store_true',
                       help='Skip creating annotated images')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode (errors only)')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Validate arguments
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    if not Path(args.input).exists():
        print(f"❌ Input path not found: {args.input}")
        return 1
    
    # Configure system
    config = {
        'confidence_threshold': args.confidence,
        'tile_size': args.tile_size,
        'overlap_ratio': args.overlap,
        'use_tiled_detection': not args.simple,
        'parallel_processing': not args.no_parallel,
        'max_workers': args.workers,
        'output_format': args.format
    }
    
    if args.tiled:
        config['use_tiled_detection'] = True
        config['high_res_threshold'] = 0  # Force tiling for all images
    
    try:
        # Initialize detection system
        detector = WindTurbineDetectionSystem(args.model, config)
        
        print("🚀 Wind Turbine Damage Detection System")
        print(f"📁 Model: {Path(args.model).name}")
        print(f"🎯 Input: {args.input}")
        print(f"💾 Output: {args.output}")
        
        # Get model info
        model_info = detector.get_model_info()
        print(f"✨ Enhanced features: {'✅ Available' if model_info['enhanced_features_available'] else '❌ Not available'}")
        
        # Process input
        start_time = time.time()
        results = detector.process_batch(args.input, args.output, args.max_images)
        total_time = time.time() - start_time
        
        # Print results
        print(f"\n🎉 Processing Complete!")
        print(f"📊 Images processed: {results['processed_images']}")
        print(f"🔍 Total detections: {results['total_detections']}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"⚡ Average per image: {total_time/results['processed_images']:.2f}s")
        
        # Save batch summary
        summary_file = Path(args.output) / "processing_summary.json"
        summary = {
            'command_args': vars(args),
            'model_info': model_info,
            'results': results,
            'total_processing_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary saved: {summary_file}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 