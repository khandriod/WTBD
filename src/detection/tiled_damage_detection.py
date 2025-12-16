#!/usr/bin/env python3
"""
Advanced Tiled Damage Detection for High-Resolution Wind Turbine Blade Images

This system implements an intelligent sliding window approach to detect multiple types of damage
(Crack, Leading Edge Erosion, Lightning Strike, Surface Damage, Surface Dust)
on high-resolution drone images of wind turbine blades.

### Key Features
- **Multi-resolution support**: Handles images of any size with automatic resolution detection
- **Intelligent tiling**: Automatic tile generation with optimal overlap and dynamic sizing
- **Smart merging**: Advanced Non-Maximum Suppression with confidence weighting
- **Crack-specific filtering**: Optimized morphological analysis for crack detection patterns
- **Comprehensive output**: Annotated images, JSON data, batch statistics, and performance metrics

Usage:
    python tiled_damage_detection.py --model selective_kernel_attention_20250716_174959 --images ./avingrid_data/ --output ./results/
    python tiled_damage_detection.py --model-path ./modelsandweights/cbamv2_20250716_174959/weights/best.pt --images ./avingrid_data/ --auto-tile --smart-merge
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import logging
import time
import math
from collections import defaultdict, Counter
from scipy import ndimage
from skimage import morphology, measure
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


@dataclass
class TileInfo:
    """Information about a single tile position"""
    x: int
    y: int
    width: int
    height: int
    global_x: int
    global_y: int
    tile_id: str = field(default="")
    
    def __post_init__(self):
        if not self.tile_id:
            self.tile_id = f"tile_{self.global_x}_{self.global_y}_{self.width}x{self.height}"


@dataclass 
class Detection:
    """Enhanced damage detection with morphological properties"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in global image coordinates
    tile_id: str = ""
    aspect_ratio: float = 0.0
    area: float = 0.0
    crack_probability: float = 0.0
    merge_count: int = 1  # Number of detections merged into this one
    
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        width = x2 - x1
        height = y2 - y1
        self.area = width * height
        self.aspect_ratio = max(width, height) / max(min(width, height), 1)
        
        # Calculate crack probability based on aspect ratio and class
        if self.class_name in ['Crack', 'Leading Edge Erosion']:
            self.crack_probability = min(1.0, self.aspect_ratio / 5.0)
        else:
            self.crack_probability = 0.1
    

class IntelligentTileProcessor:
    """Advanced tile processor with multi-resolution support and intelligent tiling"""
    
    def __init__(self, tile_size: Optional[int] = None, overlap_ratio: float = 0.25, auto_tile: bool = False):
        """
        Initialize intelligent tile processor
        
        Args:
            tile_size: Size of square tiles (None for auto-detection)
            overlap_ratio: Overlap between adjacent tiles (0.25 = 25%)
            auto_tile: Enable automatic tile size selection based on image resolution
        """
        self.base_tile_size = tile_size
        self.overlap_ratio = min(overlap_ratio, 0.5)  # Cap at 50% for performance
        self.auto_tile = auto_tile
        
        # Multi-resolution settings
        self.resolution_thresholds = {
            (0, 1000): 512,      # Small images: 512x512 tiles
            (1000, 2500): 640,   # Medium images: 640x640 tiles (training size)
            (2500, 4000): 768,   # Large images: 768x768 tiles
            (4000, 6000): 1024,  # Very large images: 1024x1024 tiles
            (6000, float('inf')): 1280  # Ultra high-res: 1280x1280 tiles
        }
        
        # Damage-specific settings
        self.min_tile_coverage = 0.6  # Minimum coverage for edge tiles
        self.damage_classes = ['Crack', 'Leading Edge Erosion', 'Lightning Strike', 'Surface Damage', 'Surface Dust']
        
        logging.info(f"IntelligentTileProcessor: auto_tile={auto_tile}, overlap={overlap_ratio:.1%}")
    
    def _determine_optimal_tile_size(self, image_shape: Tuple[int, int]) -> int:
        """Automatically determine optimal tile size based on image resolution"""
        height, width = image_shape[:2]
        max_dimension = max(height, width)
        
        if self.base_tile_size is not None:
            return self.base_tile_size
        
        # Find optimal tile size based on resolution
        for (min_res, max_res), tile_size in self.resolution_thresholds.items():
            if min_res <= max_dimension < max_res:
                logging.info(f"Auto-selected tile size: {tile_size}x{tile_size} for {width}x{height} image")
                return tile_size
        
        # Default fallback
        return 640
    
    def calculate_tile_positions(self, image_shape: Tuple[int, int]) -> List[TileInfo]:
        """
        Calculate optimal tile positions with intelligent sizing
        
        Args:
            image_shape: (height, width) of the source image
            
        Returns:
            List of TileInfo objects with positions and dimensions
        """
        height, width = image_shape[:2]
        
        # Determine optimal tile size
        self.tile_size = self._determine_optimal_tile_size(image_shape)
        self.step_size = int(self.tile_size * (1 - self.overlap_ratio))
        
        tiles = []
        
        logging.info(f"Processing image: {width}x{height}px with {self.tile_size}x{self.tile_size} tiles, {self.overlap_ratio:.1%} overlap")
        
        # Calculate intelligent grid positions with edge handling
        y_positions = self._calculate_positions(height, self.tile_size, self.step_size)
        x_positions = self._calculate_positions(width, self.tile_size, self.step_size)
        
        for y in y_positions:
            for x in x_positions:
                # Calculate tile dimensions with boundary handling
                tile_width = min(self.tile_size, width - x)
                tile_height = min(self.tile_size, height - y)
                
                # Ensure minimum tile size (important for damage detection)
                min_size = int(self.tile_size * self.min_tile_coverage)
                if tile_width < min_size or tile_height < min_size:
                    continue
                
                # Create tile info
                tile_info = TileInfo(
                    x=0, y=0,  # Local coordinates (always 0,0 for extracted tile)
                    width=tile_width,
                    height=tile_height,
                    global_x=x,
                    global_y=y
                )
                
                # Check for significant overlap with existing tiles
                if not self._tiles_overlap_significantly(tile_info, tiles):
                    tiles.append(tile_info)
        
        # Add edge refinement tiles if needed
        edge_tiles = self._generate_edge_tiles(image_shape, tiles)
        tiles.extend(edge_tiles)
        
        logging.info(f"Generated {len(tiles)} tiles for processing (including {len(edge_tiles)} edge tiles)")
        return tiles
    
    def _calculate_positions(self, dimension: int, tile_size: int, step_size: int) -> List[int]:
        """Calculate optimal tile positions with edge handling"""
        positions = list(range(0, dimension, step_size))
        
        # Ensure we cover the entire dimension
        if positions and positions[-1] + tile_size < dimension:
            # Add final position to cover the edge
            final_pos = max(0, dimension - tile_size)
            if final_pos not in positions:
                positions.append(final_pos)
        
        return positions
    
    def _generate_edge_tiles(self, image_shape: Tuple[int, int], existing_tiles: List[TileInfo]) -> List[TileInfo]:
        """Generate additional tiles to ensure complete edge coverage"""
        height, width = image_shape[:2]
        edge_tiles = []
        
        # Check coverage near edges and add tiles if needed
        edge_margin = int(self.tile_size * 0.1)  # 10% margin from edges
        
        # Right edge coverage
        right_coverage = any(tile.global_x + tile.width >= width - edge_margin for tile in existing_tiles)
        if not right_coverage:
            edge_x = max(0, width - self.tile_size)
            for tile in existing_tiles:
                if abs(tile.global_y - 0) < self.tile_size:  # Near top
                    edge_tile = TileInfo(
                        x=0, y=0,
                        width=min(self.tile_size, width - edge_x),
                        height=min(self.tile_size, height),
                        global_x=edge_x,
                        global_y=0
                    )
                    if not self._tiles_overlap_significantly(edge_tile, existing_tiles, threshold=0.7):
                        edge_tiles.append(edge_tile)
                    break
        
        return edge_tiles
    
    def _tiles_overlap_significantly(self, new_tile: TileInfo, existing_tiles: List[TileInfo], 
                                   threshold: float = 0.8) -> bool:
        """Check if new tile overlaps significantly with existing ones"""
        for existing in existing_tiles:
            # Calculate intersection area
            x1 = max(new_tile.global_x, existing.global_x)
            y1 = max(new_tile.global_y, existing.global_y)
            x2 = min(new_tile.global_x + new_tile.width, existing.global_x + existing.width)
            y2 = min(new_tile.global_y + new_tile.height, existing.global_y + existing.height)
            
            if x2 > x1 and y2 > y1:
                intersection_area = (x2 - x1) * (y2 - y1)
                new_tile_area = new_tile.width * new_tile.height
                
                if intersection_area / new_tile_area > threshold:
                    return True
        return False
    
    def extract_tile(self, image: np.ndarray, tile_info: TileInfo) -> np.ndarray:
        """Extract a single tile from the image with padding if needed"""
        extracted = image[tile_info.global_y:tile_info.global_y + tile_info.height,
                         tile_info.global_x:tile_info.global_x + tile_info.width]
        
        # Pad tile if it's smaller than expected (edge tiles)
        if extracted.shape[0] < self.tile_size or extracted.shape[1] < self.tile_size:
            padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=extracted.dtype)
            padded[:extracted.shape[0], :extracted.shape[1]] = extracted
            return padded
        
        return extracted


class AdvancedDetectionMerger:
    """Advanced detection merger with smart NMS and crack-specific filtering"""
    
    def __init__(self, iou_threshold: float = 0.4, confidence_boost: float = 0.15, 
                 smart_merge: bool = True, crack_filtering: bool = True):
        """
        Initialize advanced detection merger
        
        Args:
            iou_threshold: IoU threshold for considering detections as duplicates
            confidence_boost: Boost factor for elongated damage (typical of cracks/erosion)
            smart_merge: Enable intelligent merging based on detection properties
            crack_filtering: Enable morphological analysis for crack validation
        """
        self.iou_threshold = iou_threshold
        self.confidence_boost = confidence_boost
        self.smart_merge = smart_merge
        self.crack_filtering = crack_filtering
        self.damage_classes = ['Crack', 'Leading Edge Erosion', 'Lightning Strike', 'Surface Damage', 'Surface Dust']
        
        # Advanced merging parameters
        self.class_specific_iou = {
            'Crack': 0.3,                   # Lower IoU for cracks (can be elongated)
            'Leading Edge Erosion': 0.3,    # Lower IoU for erosion
            'Lightning Strike': 0.5,        # Higher IoU for compact damage
            'Surface Damage': 0.4,          # Standard IoU
            'Surface Dust': 0.6             # Higher IoU for dust (often large areas)
        }
        
        logging.info(f"AdvancedDetectionMerger: smart_merge={smart_merge}, crack_filtering={crack_filtering}")
        
    def merge_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Advanced merging with smart NMS and crack-specific filtering
        
        Args:
            detections: List of Detection objects from all tiles
            
        Returns:
            List of merged Detection objects with enhanced properties
        """
        if not detections:
            return []
        
        # Pre-processing: enhance detection properties
        enhanced_detections = self._enhance_detections(detections)
        
        # Apply crack-specific filtering if enabled
        if self.crack_filtering:
            enhanced_detections = self._apply_crack_filtering(enhanced_detections)
        
        # Group detections by class and apply smart merging
        merged_all = []
        for class_id in range(len(self.damage_classes)):
            class_detections = [d for d in enhanced_detections if d.class_id == class_id]
            if class_detections:
                if self.smart_merge:
                    merged_class = self._smart_merge_class_detections(class_detections)
                else:
                    merged_class = self._merge_class_detections(class_detections)
                merged_all.extend(merged_class)
        
        # Post-processing: final validation and confidence adjustment
        final_detections = self._post_process_detections(merged_all)
        
        logging.info(f"Advanced merge: {len(detections)} → {len(enhanced_detections)} → {len(merged_all)} → {len(final_detections)} detections")
        return final_detections
    
    def _enhance_detections(self, detections: List[Detection]) -> List[Detection]:
        """Enhance detection properties for better merging"""
        enhanced = []
        for detection in detections:
            # Detection properties are automatically calculated in __post_init__
            enhanced.append(detection)
        return enhanced
    
    def _apply_crack_filtering(self, detections: List[Detection]) -> List[Detection]:
        """Apply morphological analysis for crack validation"""
        filtered = []
        for detection in detections:
            if detection.class_name in ['Crack', 'Leading Edge Erosion']:
                # Validate crack characteristics
                if self._validate_crack_morphology(detection):
                    # Boost confidence for valid cracks
                    detection.confidence = min(1.0, detection.confidence + self.confidence_boost)
                    filtered.append(detection)
            else:
                filtered.append(detection)
        
        return filtered
    
    def _validate_crack_morphology(self, detection: Detection) -> bool:
        """Validate crack based on morphological properties"""
        # Aspect ratio validation (cracks are typically elongated)
        if detection.aspect_ratio < 1.5:
            return False
        
        # Area validation (reject very small detections that might be noise)
        min_crack_area = 100  # Minimum pixel area for a valid crack
        if detection.area < min_crack_area:
            return False
        
        # Confidence validation
        min_crack_confidence = 0.05
        if detection.confidence < min_crack_confidence:
            return False
        
        return True
    
    def _smart_merge_class_detections(self, detections: List[Detection]) -> List[Detection]:
        """Smart merging with class-specific IoU thresholds and confidence weighting"""
        if len(detections) <= 1:
            return detections
        
        class_name = detections[0].class_name
        class_iou = self.class_specific_iou.get(class_name, self.iou_threshold)
        
        # Sort by confidence (descending)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Advanced NMS with confidence weighting
        merged = []
        while detections:
            # Take highest confidence detection
            best = detections.pop(0)
            merged.append(best)
            
            # Remove overlapping detections with lower confidence
            remaining = []
            for detection in detections:
                iou = self._calculate_iou(best.bbox, detection.bbox)
                
                if iou > class_iou:
                    # Merge properties instead of just removing
                    best = self._merge_detection_properties(best, detection)
                else:
                    remaining.append(detection)
            
            detections = remaining
        
        return merged
    
    def _merge_detection_properties(self, primary: Detection, secondary: Detection) -> Detection:
        """Merge properties of overlapping detections"""
        # Weighted average of bounding boxes based on confidence
        primary_weight = primary.confidence
        secondary_weight = secondary.confidence
        total_weight = primary_weight + secondary_weight
        
        # Merge bounding box (confidence-weighted average)
        p_bbox = primary.bbox
        s_bbox = secondary.bbox
        
        merged_bbox = [
            (p_bbox[0] * primary_weight + s_bbox[0] * secondary_weight) / total_weight,
            (p_bbox[1] * primary_weight + s_bbox[1] * secondary_weight) / total_weight,
            (p_bbox[2] * primary_weight + s_bbox[2] * secondary_weight) / total_weight,
            (p_bbox[3] * primary_weight + s_bbox[3] * secondary_weight) / total_weight
        ]
        
        # Update primary detection
        primary.bbox = merged_bbox
        primary.confidence = max(primary.confidence, secondary.confidence)  # Take higher confidence
        primary.merge_count += secondary.merge_count
        
        # Recalculate derived properties
        primary.__post_init__()
        
        return primary
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _post_process_detections(self, detections: List[Detection]) -> List[Detection]:
        """Final post-processing and validation"""
        processed = []
        
        for detection in detections:
            # Final confidence thresholding
            min_final_confidence = 0.05
            if detection.confidence < min_final_confidence:
                continue
            
            # Final area validation
            min_final_area = 50
            if detection.area < min_final_area:
                continue
            
            processed.append(detection)
        
        return processed
    
    def _merge_class_detections(self, detections: List[Detection]) -> List[Detection]:
        """Standard merge for backward compatibility"""
        if len(detections) <= 1:
            return detections
        
        # Convert to format suitable for NMS
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                  score_threshold=0.05, 
                                  nms_threshold=self.iou_threshold)
        
        if len(indices) == 0:
            return []
        
        # Return merged detections
        merged = []
        for i in indices.flatten():
            detection = detections[i]
            # Apply confidence boost for elongated damage (typical of cracks/erosion)
            if self._is_elongated_damage(detection):
                detection.confidence = min(1.0, detection.confidence + self.confidence_boost)
            merged.append(detection)
        
        return merged
    
    def _is_elongated_damage(self, detection: Detection) -> bool:
        """Check if damage is elongated (typical of cracks or erosion)"""
        # Use the aspect_ratio already calculated in Detection.__post_init__
        return detection.aspect_ratio > 3.0 and detection.class_name in ['Crack', 'Leading Edge Erosion']


class AdvancedTiledDamageDetector:
    """Advanced tiled damage detector with comprehensive features"""
    
    def __init__(self, model_path: str, tile_size: Optional[int] = None, overlap_ratio: float = 0.25,
                 confidence: float = 0.15, device: str = "0", auto_tile: bool = True, 
                 smart_merge: bool = True, crack_filtering: bool = True):
        """
        Initialize the advanced tiled damage detector
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            tile_size: Size of sliding window tiles (None for auto-detection)
            overlap_ratio: Overlap between adjacent tiles
            confidence: Confidence threshold for detections
            device: Device to run inference on
            auto_tile: Enable automatic tile size selection
            smart_merge: Enable intelligent detection merging
            crack_filtering: Enable morphological crack filtering
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.auto_tile = auto_tile
        self.smart_merge = smart_merge
        self.crack_filtering = crack_filtering
        
        # Initialize advanced components
        self.tile_processor = IntelligentTileProcessor(tile_size, overlap_ratio, auto_tile)
        self.detection_merger = AdvancedDetectionMerger(
            confidence_boost=0.15, 
            smart_merge=smart_merge, 
            crack_filtering=crack_filtering
        )
        
        # Load model
        logging.info(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.damage_classes = ['Crack', 'Leading Edge Erosion', 'Lightning Strike', 'Surface Damage', 'Surface Dust']
        
        # Performance tracking
        self.processing_stats = {
            'total_images': 0,
            'total_tiles': 0,
            'total_detections': 0,
            'processing_time': 0.0,
            'damage_breakdown': Counter()
        }
        
        logging.info(f"AdvancedTiledDamageDetector: auto_tile={auto_tile}, smart_merge={smart_merge}, crack_filtering={crack_filtering}")
    
    def detect_damages(self, image_path: str, output_dir: str = "results") -> Dict:
        """
        Detect damages with comprehensive analysis and output
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            
        Returns:
            Dictionary with detailed detection results and performance metrics
        """
        start_time = time.time()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_shape = image.shape
        image_resolution = self._classify_image_resolution(original_shape)
        logging.info(f"Processing {image_resolution} image: {image_path} ({original_shape[1]}x{original_shape[0]})")
        
        # Generate intelligent tiles
        tiles = self.tile_processor.calculate_tile_positions(original_shape)
        
        # Process each tile with progress tracking
        all_detections = []
        processed_tiles = []
        tile_processing_times = []
        
        for i, tile_info in enumerate(tiles):
            tile_start = time.time()
            logging.info(f"Processing tile {i+1}/{len(tiles)}: {tile_info.tile_id}")
            
            # Extract tile
            tile_image = self.tile_processor.extract_tile(image, tile_info)
            
            # Run inference
            results = self.model.predict(tile_image, conf=self.confidence, verbose=False)
            
            # Convert detections to global coordinates
            tile_detections = self._convert_tile_detections(results[0], tile_info)
            all_detections.extend(tile_detections)
            
            tile_time = time.time() - tile_start
            tile_processing_times.append(tile_time)
            
            # Store detailed tile info
            processed_tiles.append({
                'tile_id': tile_info.tile_id,
                'global_position': [tile_info.global_x, tile_info.global_y],
                'size': [tile_info.width, tile_info.height],
                'detections': len(tile_detections),
                'processing_time': tile_time,
                'damage_types': [d.class_name for d in tile_detections]
            })
        
        # Advanced merging with comprehensive analysis
        merge_start = time.time()
        merged_detections = self.detection_merger.merge_detections(all_detections)
        merge_time = time.time() - merge_start
        
        # Calculate comprehensive statistics
        damage_stats = self._calculate_damage_statistics(merged_detections)
        processing_time = time.time() - start_time
        
        # Create comprehensive results
        results_data = {
            'image_path': str(image_path),
            'image_shape': original_shape,
            'image_resolution': image_resolution,
            'model_path': str(self.model_path),
            'processing_params': {
                'tile_size': getattr(self.tile_processor, 'tile_size', 'auto'),
                'overlap_ratio': self.tile_processor.overlap_ratio,
                'confidence': self.confidence,
                'auto_tile': self.auto_tile,
                'smart_merge': self.smart_merge,
                'crack_filtering': self.crack_filtering
            },
            'performance_metrics': {
                'total_processing_time': processing_time,
                'average_tile_time': np.mean(tile_processing_times),
                'merge_time': merge_time,
                'tiles_per_second': len(tiles) / processing_time,
                'detections_per_tile': len(all_detections) / len(tiles) if tiles else 0
            },
            'detection_summary': {
                'tiles_processed': len(tiles),
                'raw_detections': len(all_detections),
                'merged_detections': len(merged_detections),
                'merge_ratio': len(merged_detections) / len(all_detections) if all_detections else 0,
                'damage_statistics': damage_stats
            },
            'detections': [self._detection_to_dict(d) for d in merged_detections],
            'tile_details': processed_tiles,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update global statistics
        self._update_global_stats(results_data)
        
        # Save comprehensive results
        image_name = Path(image_path).stem
        results_file = Path(output_dir) / f"{image_name}_detections.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create enhanced annotated visualization
        annotated_image = self._create_enhanced_annotated_image(image, merged_detections, damage_stats)
        annotated_file = Path(output_dir) / f"{image_name}_annotated.jpg"
        cv2.imwrite(str(annotated_file), annotated_image)
        
        # Create performance summary
        summary_file = Path(output_dir) / f"{image_name}_summary.txt"
        self._create_performance_summary(results_data, summary_file)
        
        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Annotated image saved to: {annotated_file}")
        logging.info(f"Performance summary saved to: {summary_file}")
        logging.info(f"Processing completed in {processing_time:.2f}s with {len(merged_detections)} detections")
        
        return results_data
    
    def _convert_tile_detections(self, results, tile_info: TileInfo) -> List[Detection]:
        """Convert tile-local detections to global coordinates with enhanced properties"""
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                # Convert to global coordinates
                global_x1 = box[0] + tile_info.global_x
                global_y1 = box[1] + tile_info.global_y
                global_x2 = box[2] + tile_info.global_x
                global_y2 = box[3] + tile_info.global_y
                
                detection = Detection(
                    class_id=int(class_id),
                    class_name=self.damage_classes[class_id],
                    confidence=float(conf),
                    bbox=[global_x1, global_y1, global_x2, global_y2],
                    tile_id=tile_info.tile_id
                )
                detections.append(detection)
        
        return detections
    
    def _classify_image_resolution(self, image_shape: Tuple[int, int, int]) -> str:
        """Classify image resolution category"""
        height, width = image_shape[:2]
        max_dim = max(height, width)
        
        if max_dim < 1000:
            return "Low-resolution"
        elif max_dim < 2500:
            return "Medium-resolution"
        elif max_dim < 4000:
            return "High-resolution"
        elif max_dim < 6000:
            return "Very high-resolution"
        else:
            return "Ultra high-resolution"
    
    def _calculate_damage_statistics(self, detections: List[Detection]) -> Dict:
        """Calculate comprehensive damage statistics"""
        if not detections:
            return {
                'total_damages': 0, 
                'damage_breakdown': {}, 
                'confidence_stats': {
                    'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0
                },
                'area_stats': {
                    'mean': 0.0, 'median': 0.0, 'total': 0.0
                },
                'aspect_ratio_stats': {
                    'mean': 0.0, 'elongated_count': 0
                },
                'crack_analysis': {
                    'potential_cracks': 0, 'high_confidence_cracks': 0
                }
            }
        
        damage_counts = Counter(d.class_name for d in detections)
        confidences = [d.confidence for d in detections]
        areas = [d.area for d in detections]
        aspect_ratios = [d.aspect_ratio for d in detections]
        
        return {
            'total_damages': len(detections),
            'damage_breakdown': dict(damage_counts),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'median': float(np.median(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'std': float(np.std(confidences))
            },
            'area_stats': {
                'mean': float(np.mean(areas)),
                'median': float(np.median(areas)),
                'total': float(np.sum(areas))
            },
            'aspect_ratio_stats': {
                'mean': float(np.mean(aspect_ratios)),
                'elongated_count': sum(1 for ar in aspect_ratios if ar > 3.0)
            },
            'crack_analysis': {
                'potential_cracks': sum(1 for d in detections if d.crack_probability > 0.5),
                'high_confidence_cracks': sum(1 for d in detections 
                                            if d.class_name == 'Crack' and d.confidence > 0.7)
            }
        }
    
    def _update_global_stats(self, results_data: Dict):
        """Update global processing statistics"""
        self.processing_stats['total_images'] += 1
        self.processing_stats['total_tiles'] += results_data['detection_summary']['tiles_processed']
        self.processing_stats['total_detections'] += results_data['detection_summary']['merged_detections']
        self.processing_stats['processing_time'] += results_data['performance_metrics']['total_processing_time']
        
        # Update damage breakdown
        for damage_type, count in results_data['detection_summary']['damage_statistics']['damage_breakdown'].items():
            self.processing_stats['damage_breakdown'][damage_type] += count
    
    def _detection_to_dict(self, detection: Detection) -> Dict:
        """Convert Detection object to enhanced dictionary for JSON serialization"""
        return {
            'class_id': detection.class_id,
            'class_name': detection.class_name,
            'confidence': detection.confidence,
            'bbox': detection.bbox,
            'tile_id': detection.tile_id,
            'area': detection.area,
            'aspect_ratio': detection.aspect_ratio,
            'crack_probability': detection.crack_probability,
            'merge_count': detection.merge_count
        }
    
    def _create_enhanced_annotated_image(self, image: np.ndarray, detections: List[Detection], 
                                       damage_stats: Dict) -> np.ndarray:
        """Create enhanced annotated image with comprehensive visualization"""
        annotated = image.copy()
        annotator = Annotator(annotated)
        
        # Draw detections with enhanced labels
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Use different colors for different damage types
            color = colors(detection.class_id, True)
            
            # Enhanced label with additional info
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.merge_count > 1:
                label += f" (×{detection.merge_count})"
            if detection.crack_probability > 0.5:
                label += " [CRACK]"
            
            annotator.box_label([x1, y1, x2, y2], label, color=color)
        
        # Add statistics overlay
        self._add_stats_overlay(annotated, damage_stats)
        
        return annotated
    
    def _add_stats_overlay(self, image: np.ndarray, damage_stats: Dict):
        """Add statistics overlay to annotated image"""
        height, width = image.shape[:2]
        overlay_height = min(200, height // 4)
        overlay_width = min(400, width // 3)
        
        # Create semi-transparent overlay
        overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)  # Black background
        
        # Add text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        
        # Title
        cv2.putText(overlay, "Detection Summary:", (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25
        
        # Add damage breakdown
        for damage_type, count in damage_stats['damage_breakdown'].items():
            text = f"{damage_type}: {count}"
            cv2.putText(overlay, text, (10, y_offset), font, font_scale - 0.1, color, thickness)
            y_offset += 20
        
        # Blend overlay with original image
        alpha = 0.7
        roi = image[10:10+overlay_height, 10:10+overlay_width]
        cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
    
    def _create_performance_summary(self, results_data: Dict, summary_file: Path):
        """Create human-readable performance summary"""
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ADVANCED TILED DAMAGE DETECTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Image info
            f.write(f"Image: {Path(results_data['image_path']).name}\n")
            f.write(f"Resolution: {results_data['image_resolution']} "
                   f"({results_data['image_shape'][1]}x{results_data['image_shape'][0]})\n")
            f.write(f"Processed: {results_data['timestamp']}\n\n")
            
            # Processing info
            params = results_data['processing_params']
            f.write("PROCESSING PARAMETERS:\n")
            f.write(f"  Tile Size: {params['tile_size']}\n")
            f.write(f"  Overlap: {params['overlap_ratio']:.1%}\n")
            f.write(f"  Confidence: {params['confidence']}\n")
            f.write(f"  Auto Tile: {params['auto_tile']}\n")
            f.write(f"  Smart Merge: {params['smart_merge']}\n")
            f.write(f"  Crack Filtering: {params['crack_filtering']}\n\n")
            
            # Performance metrics
            perf = results_data['performance_metrics']
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"  Total Time: {perf['total_processing_time']:.2f}s\n")
            f.write(f"  Avg Tile Time: {perf['average_tile_time']:.3f}s\n")
            f.write(f"  Merge Time: {perf['merge_time']:.3f}s\n")
            f.write(f"  Tiles/Second: {perf['tiles_per_second']:.1f}\n")
            f.write(f"  Detections/Tile: {perf['detections_per_tile']:.1f}\n\n")
            
            # Detection summary
            summary = results_data['detection_summary']
            f.write("DETECTION SUMMARY:\n")
            f.write(f"  Tiles Processed: {summary['tiles_processed']}\n")
            f.write(f"  Raw Detections: {summary['raw_detections']}\n")
            f.write(f"  Final Detections: {summary['merged_detections']}\n")
            f.write(f"  Merge Efficiency: {summary['merge_ratio']:.1%}\n\n")
            
            # Damage breakdown
            damage_stats = summary['damage_statistics']
            f.write("DAMAGE ANALYSIS:\n")
            for damage_type, count in damage_stats['damage_breakdown'].items():
                f.write(f"  {damage_type}: {count}\n")
            
            f.write(f"\n  Confidence (avg): {damage_stats['confidence_stats']['mean']:.3f}\n")
            f.write(f"  Potential Cracks: {damage_stats['crack_analysis']['potential_cracks']}\n")
            f.write(f"  High-Conf Cracks: {damage_stats['crack_analysis']['high_confidence_cracks']}\n")
            
    def get_global_statistics(self) -> Dict:
        """Get global processing statistics across all processed images"""
        stats = self.processing_stats.copy()
        if stats['total_images'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_images']
            stats['avg_tiles_per_image'] = stats['total_tiles'] / stats['total_images']
            stats['avg_detections_per_image'] = stats['total_detections'] / stats['total_images']
        
        return stats


def main():
    """Advanced command line interface"""
    parser = argparse.ArgumentParser(description="Advanced Tiled Damage Detection for High-Resolution Wind Turbine Images")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=str,
                            help='Model name from modelsandweights directory (e.g., MSPA_20250718_142245)')
    model_group.add_argument('--model-path', type=str,
                            help='Direct path to model weights (.pt file)')
    
    # Input/output
    parser.add_argument('--images', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--output', type=str, default='tiled_results',
                       help='Output directory for results')
    
    # Processing parameters
    parser.add_argument('--tile-size', type=int, default=None,
                       help='Size of sliding window tiles (None for auto-detection)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap ratio between tiles (default: 0.25)')
    parser.add_argument('--confidence', type=float, default=0.15,
                       help='Confidence threshold (default: 0.15 for high-res)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use for inference (default: 0)')
    
    # Advanced features
    parser.add_argument('--auto-tile', action='store_true', default=True,
                       help='Enable automatic tile size selection (default: True)')
    parser.add_argument('--smart-merge', action='store_true', default=True,
                       help='Enable intelligent detection merging (default: True)')
    parser.add_argument('--crack-filtering', action='store_true', default=True,
                       help='Enable morphological crack filtering (default: True)')
    parser.add_argument('--batch-stats', action='store_true',
                       help='Generate batch statistics for multiple images')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Determine model path
    if args.model:
        model_path = f"modelsandweights/{args.model}/weights/best.pt"
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
    else:
        model_path = args.model_path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize advanced detector
    detector = AdvancedTiledDamageDetector(
        model_path=model_path,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        confidence=args.confidence,
        device=args.device,
        auto_tile=args.auto_tile,
        smart_merge=args.smart_merge,
        crack_filtering=args.crack_filtering
    )
    
    # Process images
    images_path = Path(args.images)
    if images_path.is_file():
        # Single image
        print(f"Processing single image: {images_path}")
        results = detector.detect_damages(str(images_path), args.output)
        print(f"Found {results['detection_summary']['merged_detections']} damage detections")
    
    elif images_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = [f for f in images_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise ValueError(f"No image files found in: {images_path}")
        
        print(f"Processing {len(image_files)} images from: {images_path}")
        
        total_detections = 0
        for image_file in image_files:
            print(f"Processing: {image_file.name}")
            try:
                results = detector.detect_damages(str(image_file), args.output)
                detections = results['detection_summary']['merged_detections']
                total_detections += detections
                print(f"  Found {detections} damage detections")
            except Exception as e:
                print(f"  Error processing {image_file.name}: {e}")
        
        print(f"Total detections across all images: {total_detections}")
        
        # Generate batch statistics if requested
        if args.batch_stats:
            batch_stats = detector.get_global_statistics()
            batch_file = Path(args.output) / "batch_statistics.json"
            with open(batch_file, 'w') as f:
                json.dump(batch_stats, f, indent=2)
            print(f"Batch statistics saved to: {batch_file}")
    
    else:
        raise ValueError(f"Invalid path: {images_path}")
    
    print(f"\nResults saved to: {args.output}")
    print(f"Enhanced features used: auto_tile={args.auto_tile}, smart_merge={args.smart_merge}, crack_filtering={args.crack_filtering}")


# Legacy compatibility class
class TiledDamageDetector(AdvancedTiledDamageDetector):
    """Legacy compatibility wrapper"""
    def __init__(self, model_path: str, tile_size: int = 640, overlap_ratio: float = 0.25,
                 confidence: float = 0.15, device: str = "0"):
        super().__init__(model_path, tile_size, overlap_ratio, confidence, device, 
                        auto_tile=False, smart_merge=True, crack_filtering=True)

if __name__ == "__main__":
    main() 