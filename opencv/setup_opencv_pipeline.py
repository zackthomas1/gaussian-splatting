#!/usr/bin/env python
"""
Setup script for OpenCV calibration pipeline.
Installs dependencies and validates the installation.
"""

import subprocess
import sys
import os

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def install_dependencies():
    """Install required packages."""
    print_header("Installing Dependencies")
    
    print("\nInstalling packages from requirements-opencv.txt...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements-opencv.txt"
        ])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def run_tests():
    """Run test suite."""
    print_header("Running Tests")
    
    print("\nValidating installation...")
    
    try:
        result = subprocess.call([sys.executable, "test_opencv_pipeline.py"])
        return result == 0
    except Exception as e:
        print(f"✗ Tests failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print_header("Setup Complete!")
    
    print("\n🎉 OpenCV calibration pipeline is ready to use!\n")
    
    print("Quick Start:")
    print("  1. Place images in: data/your_dataset/input/")
    print("  2. Run: python opencv_convert.py -s data/your_dataset")
    print("  3. Train: python train.py -s data/your_dataset -m output/model\n")
    
    print("Documentation:")
    print("  • Quick Start: docs/OPENCV_QUICKSTART.md")
    print("  • Full Guide:  docs/OPENCV_CALIBRATION_README.md")
    print("  • Tech Details: docs/OPENCV_CALIBRATION_PLAN.md")
    print("  • Summary:     docs/OPENCV_IMPLEMENTATION_SUMMARY.md\n")
    
    print("Example Commands:")
    print("  # Basic usage")
    print("  python opencv_convert.py -s data/silverlake\n")
    
    print("  # High quality")
    print("  python opencv_convert.py -s data/silverlake \\")
    print("    --feature_detector sift --max_features 10000 \\")
    print("    --match_threshold 0.6 --min_matches 50\n")
    
    print("  # Fast (ORB)")
    print("  python opencv_convert.py -s data/silverlake --feature_detector orb\n")

def main():
    print_header("OpenCV Calibration Pipeline Setup")
    
    print("\nThis script will:")
    print("  1. Install required dependencies")
    print("  2. Run validation tests")
    print("  3. Provide usage instructions")
    
    # Check if requirements file exists
    if not os.path.exists("requirements-opencv.txt"):
        print("\n✗ Error: requirements-opencv.txt not found")
        print("   Make sure you're running from the project root directory")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed during dependency installation")
        return 1
    
    # Run tests
    if not run_tests():
        print("\n⚠ Warning: Some tests failed")
        print("   The pipeline may still work, but check the errors above")
    
    # Print usage
    print_usage_instructions()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
