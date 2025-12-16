#!/usr/bin/env python3
"""
Wind Turbine Detector - Main Entry Point
Wrapper script for backward compatibility
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main detector
if __name__ == "__main__":
    # Change working directory to maintain relative path compatibility
    original_cwd = os.getcwd()
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        from detection.wind_turbine_detector import main
        sys.exit(main())
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("⚠️  Please ensure you're running from the repository root directory")
        sys.exit(1)
    finally:
        os.chdir(original_cwd) 