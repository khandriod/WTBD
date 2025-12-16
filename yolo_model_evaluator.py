#!/usr/bin/env python3
"""
YOLO Model Evaluator - Main Entry Point
Wrapper script for backward compatibility
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the evaluator
if __name__ == "__main__":
    # Change working directory to maintain relative path compatibility
    original_cwd = os.getcwd()
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        from evaluation.yolo_model_evaluator import main
        sys.exit(main())
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("⚠️  Please ensure you're running from the repository root directory")
        print("💡 Try: python -m src.evaluation.yolo_model_evaluator")
        sys.exit(1)
    except AttributeError:
        # If no main function, import the module for interactive use
        try:
            from evaluation.yolo_model_evaluator import YOLOModelEvaluator
            print("✅ YOLOModelEvaluator imported successfully")
            print("💡 Usage: evaluator = YOLOModelEvaluator()")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)
    finally:
        os.chdir(original_cwd) 