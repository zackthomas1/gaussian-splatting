# OpenCV Camera Calibration Pipeline

Custom implementation of camera calibration and sparse point cloud generation using OpenCV.

## Location
All OpenCV-related files are now in the `opencv/` directory for better organization.

## Quick Start

### From Project Root
```bash
# Install dependencies
cd opencv
python setup_opencv_pipeline.py
cd ..

# Run pipeline
python opencv/opencv_convert.py -s data/silverlake

# Train Gaussian Splatting
python train.py -s data/silverlake -m output/silverlake
```

### From OpenCV Directory
```bash
# Change to opencv directory
cd opencv

# Run calibration
python opencv_calibration.py -s ../data/silverlake/input

# Run reconstruction
python opencv_sparse_reconstruction.py \
  -s ../data/silverlake/input \
  --calibration ../data/silverlake/calibration.json
```

## Files in this Directory

- **opencv_convert.py** - Main pipeline script
- **opencv_calibration.py** - Camera calibration module
- **opencv_sparse_reconstruction.py** - Sparse reconstruction module
- **opencv_utils.py** - Helper functions and COLMAP format converters
- **test_opencv_pipeline.py** - Test suite
- **setup_opencv_pipeline.py** - Installation script
- **requirements-opencv.txt** - Python dependencies
- **setup_opencv.bat/sh** - Quick setup scripts
- **README.md** - Complete documentation

## Documentation

For detailed documentation, see:
- **README.md** - Complete user guide (this directory)
- **../docs/OPENCV_QUICKSTART.md** - Quick start tutorial
- **../docs/OPENCV_CALIBRATION_README.md** - Full reference
- **../docs/OPENCV_CALIBRATION_PLAN.md** - Technical details

## Example Usage

```bash
# Basic usage (from project root)
python opencv/opencv_convert.py -s data/your_scene

# High quality SIFT
python opencv/opencv_convert.py -s data/your_scene \
  --feature_detector sift \
  --max_features 10000 \
  --match_threshold 0.6

# Fast ORB
python opencv/opencv_convert.py -s data/your_scene \
  --feature_detector orb
```

## Output

The pipeline generates COLMAP-compatible files:
```
data/your_scene/
├── calibration.json
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    └── points3D.ply
```

These files can be used directly with Gaussian Splatting training.
