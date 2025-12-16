#!/usr/bin/env python3
"""
Enhanced Tiled Damage Detection System
Improved version with better performance, modularization, and integration
"""

import os
import json
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import time
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from collections import defaultdict, Counter

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


@dataclass
class ProcessingConfig:
    """Configuration for tiled processing"""
    tile_size: Optional[int] = None
    overlap_ratio: float = 0.25
    confidence_threshold: float = 0.15
    iou_threshold: float = 0.4
    auto_tile: bool = True
    smart_merge: bool = True
    crack_filtering: bool = True
    max_workers: int = 4
    batch_size: int = 8
    device: str = "0"
    
    # Resolution-based tile sizes
    resolution_thresholds: Dict[Tuple[int, int], int] = field(default_factory=lambda: {
        (0, 1000): 512,
        (1000, 2500): 640,
        (2500, 4000): 768,
        (4000, 6000): 1024,
        (6000, float('inf')): 1280
    })


@dataclass
class TileInfo:
    """Lightweight tile information"""
    x: int
    y: int
    width: int
    height: int
    global_x: int
    global_y: int
    tile_id: str = field(default="")
    
    def __post_init__(self):
        if not self.tile_id:
            self.tile_id = f"tile_{self.global_x}_{self.global_y}"


@dataclass
class Detection:
    """Enhanced detection with properties"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in global coordinates
    tile_id: str = ""
    area: float = 0.0
    aspect_ratio: float = 0.0
    
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        width, height = x2 - x1, y2 - y1
        self.area = width * height
        self.aspect_ratio = max(width, height) / max(min(width, height), 1)


class TileProcessor:
    """Optimized tile processing with memory efficiency"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_tiles(self, image_shape: Tuple[int, int]) -> List[TileInfo]:
        """Calculate optimal tile positions"""
        height, width = image_shape[:2]
        tile_size = self._get_tile_size(image_shape)
        step_size = int(tile_size * (1 - self.config.overlap_ratio))
        
        tiles = []
        y_positions = range(0, height, step_size)
        x_positions = range(0, width, step_size)
        
        for y in y_positions:
            for x in x_positions:
                tile_width = min(tile_size, width - x)
                tile_height = min(tile_size, height - y)
                
                # Skip tiles that are too small
                if tile_width < tile_size * 0.5 or tile_height < tile_size * 0.5:
                    continue
                
                tiles.append(TileInfo(
                    x=0, y=0, width=tile_width, height=tile_height,
                    global_x=x, global_y=y
                ))
        
        # Add edge coverage tiles
        tiles.extend(self._add_edge_tiles(image_shape, tiles, tile_size))
        
        self.logger.info(f"Generated {len(tiles)} tiles for {width}x{height} image")
        return tiles
    
    def _get_tile_size(self, image_shape: Tuple[int, int]) -> int:
        """Get optimal tile size based on image resolution"""
        if self.config.tile_size is not None:
            return self.config.tile_size
        
        max_dim = max(image_shape[:2])
        for (min_res, max_res), size in self.config.resolution_thresholds.items():
            if min_res <= max_dim < max_res:
                return size
        return 640
    
    def _add_edge_tiles(self, image_shape: Tuple[int, int], 
                       existing_tiles: List[TileInfo], tile_size: int) -> List[TileInfo]:
        """Add tiles to ensure complete edge coverage"""
        height, width = image_shape[:2]
        edge_tiles = []
        
        # Right edge
        if not any(tile.global_x + tile.width >= width * 0.9 for tile in existing_tiles):
            edge_x = max(0, width - tile_size)
            edge_tiles.append(TileInfo(
                x=0, y=0, width=min(tile_size, width - edge_x), height=tile_size,
                global_x=edge_x, global_y=0
            ))
        
        # Bottom edge
        if not any(tile.global_y + tile.height >= height * 0.9 for tile in existing_tiles):
            edge_y = max(0, height - tile_size)
            edge_tiles.append(TileInfo(
                x=0, y=0, width=tile_size, height=min(tile_size, height - edge_y),
                global_x=0, global_y=edge_y
            ))
        
        return edge_tiles
    
    def extract_tile(self, image: np.ndarray, tile_info: TileInfo) -> np.ndarray:
        """Extract tile with padding if needed"""
        tile = image[tile_info.global_y:tile_info.global_y + tile_info.height,
                    tile_info.global_x:tile_info.global_x + tile_info.width]
        
        # Pad to square if needed
        target_size = self._get_tile_size(image.shape)
        if tile.shape[0] != target_size or tile.shape[1] != target_size:
            padded = np.zeros((target_size, target_size, 3), dtype=tile.dtype)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            return padded
        
        return tile


class DetectionMerger:
    """Optimized detection merging with smart NMS"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.class_names = ['Crack', 'Leading Edge Erosion', 'Lightning Strike', 
                           'Surface Damage', 'Surface Dust']
        
        # Class-specific IoU thresholds
        self.class_iou_thresholds = {
            'Crack': 0.3,
            'Leading Edge Erosion': 0.3,
            'Lightning Strike': 0.5,
            'Surface Damage': 0.4,
            'Surface Dust': 0.6
        }
    
    def merge_detections(self, detections: List[Detection]) -> List[Detection]:
        """Merge overlapping detections with smart NMS"""
        if not detections:
            return []
        
        # Group by class and merge each class separately
        merged_all = []
        for class_id in range(len(self.class_names)):
            class_detections = [d for d in detections if d.class_id == class_id]
            if class_detections:
                merged_class = self._merge_class_detections(class_detections)
                merged_all.extend(merged_class)
        
        return self._post_process(merged_all)
    
    def _merge_class_detections(self, detections: List[Detection]) -> List[Detection]:
        """Merge detections for a single class"""
        if len(detections) <= 1:
            return detections
        
        class_name = detections[0].class_name
        iou_threshold = self.class_iou_thresholds.get(class_name, self.config.iou_threshold)
        
        # Convert to OpenCV format for NMS
        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            score_threshold=0.01, nms_threshold=iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Return filtered detections
        merged = []
        for i in indices.flatten():
            detection = detections[i]
            # Apply confidence boost for elongated damage (cracks/erosion)
            if self._is_elongated_damage(detection):
                detection.confidence = min(1.0, detection.confidence + 0.1)
            merged.append(detection)
        
        return merged
    
    def _is_elongated_damage(self, detection: Detection) -> bool:
        """Check if damage appears elongated (typical of cracks)"""
        return (detection.aspect_ratio > 3.0 and 
                detection.class_name in ['Crack', 'Leading Edge Erosion'])
    
    def _post_process(self, detections: List[Detection]) -> List[Detection]:
        """Final post-processing and filtering"""
        filtered = []
        for detection in detections:
            # Filter by confidence and area
            if detection.confidence >= 0.05 and detection.area >= 50:
                filtered.append(detection)
        return filtered


class EnhancedTiledDetector:
    """Main enhanced tiled detection system"""
    
    def __init__(self, model_path: str, config: Optional[ProcessingConfig] = None):
        self.model_path = model_path
        self.config = config or ProcessingConfig()
        
        # Initialize components
        self.tile_processor = TileProcessor(self.config)
        self.detection_merger = DetectionMerger(self.config)
        
        # Load model with single GPU device
        self.model = YOLO(model_path)
        
        # Ensure model uses single GPU device
        if self.config.device != 'cpu':
            # Force single GPU usage (device 0)
            self.config.device = '0'
            
        self.class_names = ['Crack', 'Leading Edge Erosion', 'Lightning Strike', 
                           'Surface Damage', 'Surface Dust']
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.stats = {
            'images_processed': 0,
            'total_tiles': 0,
            'total_detections': 0,
            'processing_time': 0.0
        }
    
    def detect_damages_single(self, image_path: str, output_dir: str = "results") -> Dict:
        """Process a single image with enhanced efficiency"""
        start_time = time.time()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load and validate image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_shape = image.shape
        self.logger.info(f"Processing {image_path} ({image_shape[1]}x{image_shape[0]})")
        
        # Generate tiles
        tiles = self.tile_processor.calculate_tiles(image_shape)
        
        # Process tiles (use optimized batching for GPU safety)
        use_batched = len(tiles) > 8  # Use batched processing for larger tile counts
        processing_mode = "batched" if use_batched else "sequential"
        self.logger.info(f"Processing {len(tiles)} tiles using {processing_mode} mode...")
        
        if use_batched:
            all_detections = self._process_tiles_parallel(image, tiles)  # Now safe batched processing
        else:
            all_detections = self._process_tiles_sequential(image, tiles)
        
        # Merge detections
        merged_detections = self.detection_merger.merge_detections(all_detections)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create results
        results = self._create_results_dict(
            image_path, image_shape, tiles, all_detections, 
            merged_detections, processing_time
        )
        
        # Save outputs
        self._save_results(results, output_dir)
        
        # Update stats
        self._update_stats(results)
        
        self.logger.info(f"Processed {Path(image_path).name} in {processing_time:.2f}s: "
                        f"{len(merged_detections)} detections")
        
        return results
    
    def _process_tiles_sequential(self, image: np.ndarray, 
                                 tiles: List[TileInfo]) -> List[Detection]:
        """Process tiles sequentially"""
        all_detections = []
        
        for i, tile_info in enumerate(tiles):
            if i % 10 == 0:
                self.logger.info(f"Processing tile {i+1}/{len(tiles)}")
            
            tile_image = self.tile_processor.extract_tile(image, tile_info)
            detections = self._process_single_tile(tile_image, tile_info)
            all_detections.extend(detections)
        
        return all_detections
    
    def _process_tiles_parallel(self, image: np.ndarray, 
                               tiles: List[TileInfo]) -> List[Detection]:
        """Process tiles with optimized batching for single GPU"""
        all_detections = []
        
        # For GPU inference, process tiles sequentially but with batch optimization
        # to avoid GPU memory conflicts and segmentation faults
        batch_size = min(self.config.batch_size, 4)  # Limit batch size for GPU safety
        
        self.logger.info(f"Using safe sequential processing with batch size {batch_size}")
        
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            batch_images = []
            
            # Extract all tiles in batch first (CPU operation)
            for tile_info in batch_tiles:
                tile_image = self.tile_processor.extract_tile(image, tile_info)
                batch_images.append((tile_image, tile_info))
            
            # Process batch sequentially on single GPU
            for tile_image, tile_info in batch_images:
                detections = self._process_single_tile(tile_image, tile_info)
                all_detections.extend(detections)
            
            self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(tiles)-1)//batch_size + 1}")
        
        return all_detections
    
    def _process_single_tile(self, tile_image: np.ndarray, 
                           tile_info: TileInfo) -> List[Detection]:
        """Process a single tile and return detections in global coordinates"""
        try:
            # Run inference with explicit device and single GPU settings
            results = self.model.predict(
                tile_image, 
                conf=self.config.confidence_threshold,
                device=self.config.device,
                verbose=False,
                half=False,  # Disable half precision for stability
                imgsz=tile_image.shape[0]  # Use actual tile size
            )
            
            # Convert to global coordinates
            detections = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    # Convert to global coordinates
                    global_bbox = [
                        float(box[0] + tile_info.global_x),
                        float(box[1] + tile_info.global_y),
                        float(box[2] + tile_info.global_x),
                        float(box[3] + tile_info.global_y)
                    ]
                    
                    detection = Detection(
                        class_id=int(class_id),
                        class_name=self.class_names[class_id],
                        confidence=float(conf),
                        bbox=global_bbox,
                        tile_id=tile_info.tile_id
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"Error processing tile {tile_info.tile_id}: {e}")
            return []
    
    def _create_results_dict(self, image_path: str, image_shape: tuple,
                           tiles: List[TileInfo], all_detections: List[Detection],
                           merged_detections: List[Detection], 
                           processing_time: float) -> Dict:
        """Create comprehensive results dictionary"""
        damage_counts = Counter(d.class_name for d in merged_detections)
        
        return {
            'image_path': str(image_path),
            'image_shape': image_shape,
            'processing_config': {
                'tile_size': getattr(self.tile_processor, '_get_tile_size', lambda x: 'auto')(image_shape),
                'overlap_ratio': self.config.overlap_ratio,
                'confidence_threshold': self.config.confidence_threshold,
                'parallel_processing': self.config.max_workers > 1
            },
            'performance': {
                'processing_time': processing_time,
                'tiles_processed': len(tiles),
                'tiles_per_second': len(tiles) / processing_time,
                'raw_detections': len(all_detections),
                'final_detections': len(merged_detections),
                'merge_efficiency': len(merged_detections) / max(len(all_detections), 1)
            },
            'detections': [self._detection_to_dict(d) for d in merged_detections],
            'damage_summary': {
                'total_damages': len(merged_detections),
                'damage_breakdown': dict(damage_counts)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _detection_to_dict(self, detection: Detection) -> Dict:
        """Convert Detection to dictionary"""
        return {
            'class_id': detection.class_id,
            'class_name': detection.class_name,
            'confidence': detection.confidence,
            'bbox': detection.bbox,
            'area': detection.area,
            'aspect_ratio': detection.aspect_ratio,
            'tile_id': detection.tile_id
        }
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save results to files"""
        image_name = Path(results['image_path']).stem
        output_path = Path(output_dir)
        
        # Save JSON results
        json_file = output_path / f"{image_name}_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create annotated image
        if results['detections']:
            annotated_file = output_path / f"{image_name}_annotated.jpg"
            self._create_annotated_image(results, annotated_file)
    
    def _create_annotated_image(self, results: Dict, output_path: Path):
        """Create annotated image with detections"""
        image = cv2.imread(results['image_path'])
        if image is None:
            return
        
        annotator = Annotator(image)
        
        for detection in results['detections']:
            bbox = [int(x) for x in detection['bbox']]
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            color = colors(detection['class_id'], True)
            annotator.box_label(bbox, label, color=color)
        
        cv2.imwrite(str(output_path), image)
    
    def _update_stats(self, results: Dict):
        """Update global statistics"""
        self.stats['images_processed'] += 1
        self.stats['total_tiles'] += results['performance']['tiles_processed']
        self.stats['total_detections'] += results['performance']['final_detections']
        self.stats['processing_time'] += results['performance']['processing_time']
    
    def process_directory(self, images_dir: str, output_dir: str = "results", 
                         max_images: Optional[int] = None) -> Dict:
        """Process all images in a directory"""
        images_path = Path(images_dir)
        if not images_path.exists():
            raise ValueError(f"Directory not found: {images_dir}")
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = [f for f in images_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")
        
        self.logger.info(f"Processing {len(image_files)} images from {images_dir}")
        
        # Process all images
        all_results = []
        for i, image_file in enumerate(image_files):
            self.logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
            try:
                results = self.detect_damages_single(str(image_file), output_dir)
                all_results.append(results)
            except Exception as e:
                self.logger.error(f"Failed to process {image_file.name}: {e}")
        
        # Create batch summary
        batch_summary = self._create_batch_summary(all_results, output_dir)
        
        return batch_summary
    
    def _create_batch_summary(self, all_results: List[Dict], output_dir: str) -> Dict:
        """Create batch processing summary"""
        summary = {
            'processed_images': len(all_results),
            'total_detections': sum(r['performance']['final_detections'] for r in all_results),
            'total_processing_time': sum(r['performance']['processing_time'] for r in all_results),
            'global_stats': self.stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save batch summary
        summary_file = Path(output_dir) / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    """Enhanced command line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced Tiled Damage Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model and input/output
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model weights (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--output', type=str, default='enhanced_results',
                       help='Output directory')
    
    # Processing parameters
    parser.add_argument('--tile-size', type=int, default=None,
                       help='Tile size (auto-detected if not specified)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap ratio (default: 0.25)')
    parser.add_argument('--confidence', type=float, default=0.15,
                       help='Confidence threshold (default: 0.15)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for parallel processing (default: 8)')
    
    # Feature toggles
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to process')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = ProcessingConfig(
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        confidence_threshold=args.confidence,
        max_workers=1 if args.no_parallel else args.workers,
        batch_size=args.batch_size
    )
    
    # Initialize detector
    detector = EnhancedTiledDetector(args.model, config)
    
    # Process input
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            # Single image
            results = detector.detect_damages_single(str(input_path), args.output)
            print(f"✅ Processed 1 image: {results['performance']['final_detections']} detections")
            
        elif input_path.is_dir():
            # Directory
            batch_results = detector.process_directory(
                str(input_path), args.output, args.max_images
            )
            print(f"✅ Processed {batch_results['processed_images']} images: "
                  f"{batch_results['total_detections']} total detections")
            
        else:
            raise ValueError(f"Invalid input path: {input_path}")
        
        print(f"📁 Results saved to: {args.output}")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 