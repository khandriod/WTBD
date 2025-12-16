#!/usr/bin/env python3
"""
Test Script for Tiled Damage Detection System

This script demonstrates the sliding window damage detection approach
on high-resolution wind turbine blade images using different trained models.

Usage:
    python test_tiled_detection.py
    python test_tiled_detection.py --quick-test    # Test with single image
    python test_tiled_detection.py --compare-models    # Compare different attention mechanisms
"""

import os
import sys
import json
import time
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Import our tiled detection system
from tiled_damage_detection import TiledDamageDetector


def setup_logging(verbose: bool = True):
    """Setup logging configuration"""
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def find_available_models():
    """Find all available trained models"""
    models_dir = Path("modelsandweights")
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return []
    
    available_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            weights_file = model_dir / "weights" / "best.pt"
            if weights_file.exists():
                available_models.append({
                    'name': model_dir.name,
                    'path': str(weights_file),
                    'dir': str(model_dir)
                })
    
    return available_models


def find_test_images():
    """Find available test images"""
    images_dir = Path("avingrid_data")
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    test_images = [f for f in images_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    return sorted(test_images)


def run_single_test(model_info, image_path, output_dir="test_results"):
    """Run detection on a single image with a specific model"""
    print(f"\n{'='*50}")
    print(f"Testing Model: {model_info['name']}")
    print(f"Image: {image_path.name}")
    print(f"{'='*50}")
    
    # Create model-specific output directory
    model_output_dir = Path(output_dir) / model_info['name']
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector with optimal settings for high-resolution images
    detector = TiledDamageDetector(
        model_path=model_info['path'],
        tile_size=1024,      # Good balance for 5K images
        overlap_ratio=0.25,   # 25% overlap for thorough coverage
        confidence=0.15,      # Lower threshold for high-res images
        device="0"
    )
    
    # Run detection
    start_time = time.time()
    try:
        results = detector.detect_damages(str(image_path), str(model_output_dir))
        processing_time = time.time() - start_time
        
        # Print results summary
        print(f"✓ Processing completed in {processing_time:.1f}s")
        print(f"✓ Image dimensions: {results['image_shape'][1]}x{results['image_shape'][0]}")
        print(f"✓ Tiles processed: {results['tiles_processed']}")
        print(f"✓ Total detections: {results['total_detections']}")
        print(f"✓ Final detections (after merging): {results['final_detections']}")
        
        # Print damage breakdown
        if results['detections']:
            damage_count = {}
            for detection in results['detections']:
                damage_type = detection['class_name']
                damage_count[damage_type] = damage_count.get(damage_type, 0) + 1
            
            print("\nDamage Breakdown:")
            for damage_type, count in damage_count.items():
                print(f"  {damage_type}: {count}")
        else:
            print("  No damages detected")
        
        print(f"✓ Results saved to: {model_output_dir}")
        return results
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def quick_test():
    """Run a quick test with the first available model and image"""
    print("🚀 Running Quick Test")
    print("This will test the sliding window approach with one model and one image")
    
    # Find available resources
    models = find_available_models()
    images = find_test_images()
    
    if not models:
        print("❌ No trained models found in modelsandweights/")
        return
    
    if not images:
        print("❌ No test images found in avingrid_data/")
        return
    
    # Use first available model and image
    model = models[0]
    image = images[0]
    
    # Try to use a working model first (avoid MSPA for now)
    working_models = [m for m in models if 'efficient_multiscale' in m['name'].lower() or 'cbam' in m['name'].lower()]
    if working_models:
        model = working_models[0]
        print(f"Using working model: {model['name']} (avoiding MSPA for now)")
    
    print(f"📊 Using model: {model['name']}")
    print(f"🖼️  Using image: {image.name}")
    
    results = run_single_test(model, image, "quick_test_results")
    
    if results:
        print("\n🎉 Quick test completed successfully!")
        print("💡 Try running with --compare-models to test multiple models")
    else:
        print("\n❌ Quick test failed")


def compare_models():
    """Compare performance of different attention mechanism models"""
    print("🔬 Comparing Models with Different Attention Mechanisms")
    print("This will test all available models on the same image")
    
    # Find available resources  
    models = find_available_models()
    images = find_test_images()
    
    if not models:
        print("❌ No trained models found in modelsandweights/")
        return
    
    if not images:
        print("❌ No test images found in avingrid_data/")
        return
    
    # Use the largest image for comparison (most challenging)
    test_image = max(images, key=lambda x: x.stat().st_size)
    
    print(f"🖼️  Test image: {test_image.name}")
    print(f"📊 Found {len(models)} models to compare")
    
    comparison_results = []
    
    for i, model in enumerate(models, 1):
        print(f"\n🔄 Testing model {i}/{len(models)}: {model['name']}")
        
        results = run_single_test(model, test_image, "model_comparison")
        
        if results:
            # Extract key metrics for comparison
            comparison_results.append({
                'model_name': model['name'],
                'processing_time': time.time(),  # This would be captured in run_single_test
                'tiles_processed': results['tiles_processed'],
                'total_detections': results['total_detections'],
                'final_detections': results['final_detections'],
                'damage_breakdown': {}
            })
            
            # Calculate damage breakdown
            for detection in results['detections']:
                damage_type = detection['class_name']
                comparison_results[-1]['damage_breakdown'][damage_type] = \
                    comparison_results[-1]['damage_breakdown'].get(damage_type, 0) + 1
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print("📈 MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Tiles':<8} {'Detections':<12} {'Final':<8} {'Key Damages'}")
    print(f"{'-'*70}")
    
    for result in comparison_results:
        key_damages = []
        for damage_type, count in result['damage_breakdown'].items():
            if count > 0:
                key_damages.append(f"{damage_type[:8]}:{count}")
        
        damages_str = ", ".join(key_damages) if key_damages else "None"
        
        print(f"{result['model_name']:<25} "
              f"{result['tiles_processed']:<8} "
              f"{result['total_detections']:<12} "
              f"{result['final_detections']:<8} "
              f"{damages_str}")
    
    # Save detailed comparison
    comparison_file = Path("model_comparison") / "comparison_summary.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            'test_image': str(test_image),
            'test_date': datetime.now().isoformat(),
            'results': comparison_results
        }, f, indent=2)
    
    print(f"\n💾 Detailed comparison saved to: {comparison_file}")


def full_test():
    """Run comprehensive test on all available models and images"""
    print("🔬 Running Comprehensive Test Suite")
    print("This will test all available models on all test images")
    
    models = find_available_models()
    images = find_test_images()
    
    if not models:
        print("❌ No trained models found")
        return
    
    if not images:
        print("❌ No test images found")
        return
    
    print(f"📊 Found {len(models)} models and {len(images)} images")
    print(f"⏱️  Estimated time: ~{len(models) * len(images) * 2} minutes")
    
    total_tests = len(models) * len(images)
    current_test = 0
    
    for model in models:
        for image in images:
            current_test += 1
            print(f"\n🔄 Test {current_test}/{total_tests}: {model['name']} on {image.name}")
            
            results = run_single_test(model, image, "full_test_results")
            
            if results:
                print(f"✅ Completed successfully")
            else:
                print(f"❌ Failed")
    
    print(f"\n🎉 Full test suite completed!")
    print(f"📁 Results saved in: full_test_results/")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Test Tiled Damage Detection System")
    
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--quick-test', action='store_true',
                           help='Run quick test with first available model and image')
    test_group.add_argument('--compare-models', action='store_true',
                           help='Compare all available models on the same image')
    test_group.add_argument('--full-test', action='store_true',
                           help='Test all models on all images (comprehensive)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("🔍 Tiled Damage Detection Test Suite")
    print("=" * 50)
    
    # Check for required directories and files
    if not Path("modelsandweights").exists():
        print("❌ modelsandweights directory not found!")
        print("💡 Make sure you have trained models in modelsandweights/")
        return
    
    if not Path("avingrid_data").exists():
        print("❌ avingrid_data directory not found!")
        print("💡 Make sure you have test images in avingrid_data/")
        return
    
    # Run selected test
    if args.quick_test:
        quick_test()
    elif args.compare_models:
        compare_models()
    elif args.full_test:
        full_test()
    else:
        # Default: show available options
        models = find_available_models()
        images = find_test_images()
        
        print(f"📊 Available Models: {len(models)}")
        for model in models:
            print(f"  - {model['name']}")
        
        print(f"\n🖼️  Available Test Images: {len(images)}")
        for image in images:
            # Get image size info
            size_mb = image.stat().st_size / (1024 * 1024)
            print(f"  - {image.name} ({size_mb:.1f} MB)")
        
        print(f"\n💡 Usage Options:")
        print(f"  python {sys.argv[0]} --quick-test        # Quick test with one model/image")
        print(f"  python {sys.argv[0]} --compare-models    # Compare all models")
        print(f"  python {sys.argv[0]} --full-test         # Comprehensive test suite")
        
        # Run quick test by default
        print(f"\n🚀 Running quick test (default)...")
        quick_test()


if __name__ == "__main__":
    main() 