#!/usr/bin/env python3
"""
Tile Processing Utilities
Optimized functions for image tiling and memory management
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class TileConfig:
    """Configuration for tile processing"""
    size: int = 640
    overlap: float = 0.25
    min_coverage: float = 0.5
    
    
class MemoryOptimizedTileGenerator:
    """Memory-efficient tile generator for large images"""
    
    def __init__(self, config: TileConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_tile_positions(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Generate tile positions without loading full image"""
        height, width = image_shape[:2]
        step_size = int(self.config.size * (1 - self.config.overlap))
        
        positions = []
        
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                # Calculate tile boundaries
                x2 = min(x + self.config.size, width)
                y2 = min(y + self.config.size, height)
                
                # Skip tiles that are too small
                if ((x2 - x) < self.config.size * self.config.min_coverage or 
                    (y2 - y) < self.config.size * self.config.min_coverage):
                    continue
                
                positions.append((x, y, x2, y2))
        
        # Add edge tiles to ensure full coverage
        positions.extend(self._add_edge_coverage(image_shape, positions))
        
        return positions
    
    def _add_edge_coverage(self, image_shape: Tuple[int, int], 
                          existing_positions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Add tiles to ensure edge coverage"""
        height, width = image_shape[:2]
        edge_tiles = []
        
        # Check right edge coverage
        max_x = max(pos[2] for pos in existing_positions) if existing_positions else 0
        if max_x < width * 0.95:
            x = max(0, width - self.config.size)
            edge_tiles.append((x, 0, width, min(self.config.size, height)))
        
        # Check bottom edge coverage
        max_y = max(pos[3] for pos in existing_positions) if existing_positions else 0
        if max_y < height * 0.95:
            y = max(0, height - self.config.size)
            edge_tiles.append((0, y, min(self.config.size, width), height))
        
        return edge_tiles
    
    def extract_tile_from_file(self, image_path: str, position: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract tile directly from file to save memory"""
        try:
            # For now, load full image (can be optimized with libraries like tifffile for large TIFF images)
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            x, y, x2, y2 = position
            tile = image[y:y2, x:x2]
            
            # Pad to expected size if needed
            if tile.shape[0] != self.config.size or tile.shape[1] != self.config.size:
                padded = np.zeros((self.config.size, self.config.size, 3), dtype=tile.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                return padded
            
            return tile
            
        except Exception as e:
            self.logger.error(f"Failed to extract tile from {image_path}: {e}")
            return None


class AutoTileSizer:
    """Automatic tile size selection based on image characteristics"""
    
    RESOLUTION_MAPPINGS = {
        (0, 1000): 512,
        (1000, 2500): 640,
        (2500, 4000): 768,
        (4000, 6000): 1024,
        (6000, float('inf')): 1280
    }
    
    @classmethod
    def get_optimal_size(cls, image_shape: Tuple[int, int], 
                        base_size: Optional[int] = None) -> int:
        """Get optimal tile size based on image dimensions"""
        if base_size is not None:
            return base_size
        
        max_dim = max(image_shape[:2])
        
        for (min_res, max_res), size in cls.RESOLUTION_MAPPINGS.items():
            if min_res <= max_dim < max_res:
                return size
        
        return 640  # Default fallback
    
    @classmethod
    def estimate_tile_count(cls, image_shape: Tuple[int, int], 
                           tile_size: int, overlap: float = 0.25) -> int:
        """Estimate number of tiles that will be generated"""
        height, width = image_shape[:2]
        step_size = int(tile_size * (1 - overlap))
        
        x_tiles = (width + step_size - 1) // step_size
        y_tiles = (height + step_size - 1) // step_size
        
        return x_tiles * y_tiles


def calculate_overlap_area(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate overlap area between two bounding boxes"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 > x1 and y2 > y1:
        return (x2 - x1) * (y2 - y1)
    return 0.0


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    intersection = calculate_overlap_area(bbox1, bbox2)
    if intersection == 0:
        return 0.0
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def validate_detection_quality(bbox: List[float], confidence: float, 
                             min_area: float = 50, min_confidence: float = 0.05) -> bool:
    """Validate detection meets minimum quality requirements"""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    
    return (area >= min_area and 
            confidence >= min_confidence and
            width > 0 and height > 0)


def calculate_aspect_ratio(bbox: List[float]) -> float:
    """Calculate aspect ratio of bounding box"""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    if width <= 0 or height <= 0:
        return 1.0
    
    return max(width, height) / min(width, height)


def is_elongated_damage(bbox: List[float], class_name: str, threshold: float = 3.0) -> bool:
    """Check if damage appears elongated (typical for cracks/erosion)"""
    if class_name not in ['Crack', 'Leading Edge Erosion']:
        return False
    
    aspect_ratio = calculate_aspect_ratio(bbox)
    return aspect_ratio > threshold 