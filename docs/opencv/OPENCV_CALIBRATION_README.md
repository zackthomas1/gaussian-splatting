# Custom Camera Calibration and Sparse Reconstruction

Custom implementation of camera calibration and sparse point cloud generation using OpenCV as an alternative to COLMAP for the Gaussian Splatting pipeline.

## Overview

This implementation provides a complete Structure-from-Motion (SfM) pipeline that:
- Performs camera calibration (intrinsic parameters)
- Generates sparse 3D point clouds from images
- Outputs COLMAP-compatible binary files for Gaussian Splatting

## Files

- **`opencv_convert.py`** - Main script integrating calibration and reconstruction
- **`opencv_calibration.py`** - Camera calibration module
- **`opencv_sparse_reconstruction.py`** - Sparse reconstruction module
- **`utils/opencv_utils.py`** - Helper functions and COLMAP format converters
- **`docs/OPENCV_CALIBRATION_PLAN.md`** - Detailed technical documentation

## Installation

### Prerequisites

```bash
# Install required packages
pip install opencv-python opencv-contrib-python numpy scipy tqdm
```

Or using the existing environment:

```bash
conda activate gaussian_splatting
pip install opencv-python opencv-contrib-python scipy tqdm
```

### Verify Installation

```bash
python -c "import cv2; print(cv2.__version__)"
```

## Usage

### Quick Start

Process a dataset with automatic calibration and reconstruction:

```bash
python opencv_convert.py -s data/silverlake
```

This will:
1. Calibrate camera from `data/silverlake/input/` images
2. Generate sparse reconstruction
3. Save outputs to `data/silverlake/sparse/0/`

### Command Line Options

```
Required:
  -s, --source_path PATH        Source directory with input/ folder

Calibration:
  --skip_calibration            Use existing calibration.json
  --calibration_method METHOD   auto|checkerboard|images (default: auto)
  --pattern_size W H            Checkerboard size (default: 9 6)
  --square_size SIZE            Checkerboard square size (default: 1.0)

Feature Detection:
  --feature_detector TYPE       sift|orb (default: sift)
  --max_features N              Max features per image (default: 8000)

Matching:
  --match_threshold RATIO       Ratio test threshold (default: 0.7)
  --min_matches N               Min matches per pair (default: 30)

Reconstruction:
  --skip_reconstruction         Calibration only

Other:
  --visualize                   Generate visualizations
```

### Examples

#### Basic Usage
```bash
# Full pipeline with default settings
python opencv_convert.py -s data/silverlake
```

#### Use ORB Features (faster)
```bash
python opencv_convert.py -s data/silverlake --feature_detector orb
```

#### Calibration Only
```bash
python opencv_convert.py -s data/silverlake --skip_reconstruction
```

#### Use Existing Calibration
```bash
python opencv_convert.py -s data/silverlake --skip_calibration
```

#### High-Quality Reconstruction
```bash
python opencv_convert.py -s data/silverlake \
  --feature_detector sift \
  --max_features 10000 \
  --match_threshold 0.6 \
  --min_matches 50
```

### Individual Modules

You can also run calibration and reconstruction separately:

#### Camera Calibration
```bash
python opencv_calibration.py \
  -s data/silverlake/input \
  --output data/silverlake/calibration.json \
  --method images
```

#### Sparse Reconstruction
```bash
python opencv_sparse_reconstruction.py \
  -s data/silverlake/input \
  --calibration data/silverlake/calibration.json \
  --output data/silverlake/sparse/0
```

## Input/Output Structure

### Expected Input Structure
```
data/silverlake/
├── input/              # Place your images here
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
```

### Generated Output Structure
```
data/silverlake/
├── input/              # Your original images
├── calibration.json    # Camera calibration parameters
└── sparse/
    └── 0/
        ├── cameras.bin     # Camera intrinsics (COLMAP format)
        ├── images.bin      # Camera poses (COLMAP format)
        ├── points3D.bin    # 3D points (COLMAP format)
        └── points3D.ply    # Point cloud for visualization
```

## Output Formats

### COLMAP Binary Format
The output files are fully compatible with COLMAP and can be used directly with Gaussian Splatting:

- **cameras.bin** - Camera models and intrinsic parameters
- **images.bin** - Camera poses (rotation + translation)
- **points3D.bin** - Sparse 3D point cloud with tracks

### Visualization
- **points3D.ply** - ASCII PLY file for visualization in MeshLab, CloudCompare, or similar tools

## Training Gaussian Splatting

After running the pipeline, train Gaussian Splatting as usual:

```bash
python train.py -s data/silverlake -m output/silverlake
```

## Comparison with COLMAP

### Advantages
- **No COLMAP dependency** - Pure Python/OpenCV implementation
- **Customizable** - Easy to modify and extend
- **Educational** - Clear, readable code for learning SfM
- **Integrated** - Single pipeline from images to sparse reconstruction

### When to Use COLMAP
- Need dense reconstruction
- Require highly optimized bundle adjustment
- Working with very large datasets (1000+ images)
- Need loop closure detection

### Performance Comparison
On a typical dataset (50-100 images):
- **COLMAP**: More accurate, better bundle adjustment, slower
- **OpenCV Implementation**: Faster feature extraction, good for prototyping, adequate for Gaussian Splatting

## Algorithm Overview

### 1. Camera Calibration
- **Auto mode**: Tries checkerboard first, falls back to self-calibration
- **Checkerboard**: Uses cv2.calibrateCamera() with detected corners
- **Self-calibration**: Estimates focal length from feature matches

### 2. Feature Extraction
- **SIFT**: Scale-invariant, robust (recommended)
- **ORB**: Faster, patent-free alternative

### 3. Feature Matching
- Brute-force matcher with ratio test (Lowe's ratio)
- RANSAC-based geometric verification
- Fundamental matrix filtering

### 4. Incremental Reconstruction
1. Select initial image pair (most matches + good baseline)
2. Estimate relative pose via essential matrix
3. Triangulate initial 3D points
4. Register next images via PnP (Perspective-n-Point)
5. Triangulate new points
6. Repeat until all images processed

### 5. Output Generation
- Convert poses to quaternions
- Write COLMAP binary format
- Generate PLY for visualization

## Troubleshooting

### Not Enough Matches
- Try lower `--match_threshold` (e.g., 0.6)
- Increase `--max_features` (e.g., 10000)
- Use SIFT instead of ORB
- Ensure images have sufficient overlap

### Poor Calibration
- Use checkerboard calibration if available
- Ensure images show scene from diverse viewpoints
- Check that images are sharp and well-lit

### Few 3D Points
- Lower `--min_matches` threshold
- Use more images
- Ensure good image coverage and overlap

### Registration Fails
- Check that calibration is reasonable
- Verify image quality
- Ensure sufficient feature matches

## Technical Details

See `docs/OPENCV_CALIBRATION_PLAN.md` for:
- Detailed algorithm descriptions
- Mathematical foundations
- COLMAP format specifications
- Implementation notes

## Testing

Compare with COLMAP output:

```bash
# Generate COLMAP reconstruction
python convert.py -s data/silverlake

# Generate OpenCV reconstruction
python opencv_convert.py -s data/silverlake

# Compare point clouds
# Open both sparse/0/points3D.ply files in visualization software
```

## Future Enhancements

- [ ] Bundle adjustment with Ceres or g2o
- [ ] Multi-scale feature extraction
- [ ] GPU-accelerated feature matching
- [ ] Automatic parameter tuning
- [ ] Progress visualization
- [ ] Dense reconstruction support
- [ ] Video input support

## References

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [COLMAP Documentation](https://colmap.github.io/)
- [Structure from Motion Wikipedia](https://en.wikipedia.org/wiki/Structure_from_motion)
- [Multiple View Geometry (Hartley & Zisserman)](https://www.robots.ox.ac.uk/~vgg/hzbook/)

## Credits

Part of the Gaussian Splatting custom calibration project.

Based on:
- OpenCV Structure-from-Motion concepts
- COLMAP format specifications
- Gaussian Splatting pipeline requirements
