#!/bin/bash
# Quick setup and test script for OpenCV calibration pipeline
# Run this after cloning/updating the repository

echo "========================================"
echo "OpenCV Calibration Pipeline - Setup"
echo "========================================"
echo ""

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
python setup_opencv_pipeline.py

if [ $? -ne 0 ]; then
    echo "Setup failed! Please check errors above."
    exit 1
fi

echo ""
echo "========================================"
echo "✓ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Test on silverlake dataset:"
echo "   python opencv/opencv_convert.py -s data/silverlake"
echo ""
echo "2. Or prepare your own dataset:"
echo "   mkdir -p data/my_scene/input"
echo "   # Copy images to data/my_scene/input/"
echo "   python opencv/opencv_convert.py -s data/my_scene"
echo ""
echo "3. Train Gaussian Splatting:"
echo "   python train.py -s data/my_scene -m output/my_scene"
echo ""
echo "For more info, see:"
echo "  • OPENCV_IMPLEMENTATION.md (start here!)"
echo "  • docs/OPENCV_QUICKSTART.md"
echo "  • docs/OPENCV_CALIBRATION_README.md"
echo ""
