# Custom Camera Calibration and Sparse Reconstruction Plan

## Project Overview
Implement custom camera calibration and sparse point cloud generation using OpenCV as an alternative to COLMAP for the Gaussian Splatting pipeline.

## Goals
1. Replace COLMAP's camera calibration with custom OpenCV implementation
2. Generate sparse point cloud from custom feature matching and triangulation
3. Output results in COLMAP-compatible format for seamless integration with existing Gaussian Splatting pipeline

## Implementation Components

### 1. Camera Calibration (`opencv_calibration.py`)
**Purpose**: Estimate camera intrinsic and extrinsic parameters

**Features**:
- **Checkerboard calibration** (optional, if calibration pattern available)
- **Self-calibration from images** using feature detection
- Calculate intrinsic matrix (focal length, principal point, distortion coefficients)
- Support for both single camera and multi-camera setups
- Export calibration parameters in COLMAP format

**Methods**:
- Feature detection (SIFT/ORB)
- Fundamental/Essential matrix estimation
- Camera matrix decomposition
- Bundle adjustment for refinement

### 2. Sparse Reconstruction (`opencv_sparse_reconstruction.py`)
**Purpose**: Generate sparse 3D point cloud from calibrated images

**Features**:
- Feature extraction and matching across multiple images
- Geometric verification (RANSAC for outlier rejection)
- Pose estimation (PnP, essential matrix)
- Triangulation of 3D points
- Track building (linking 2D features across images)
- Bundle adjustment for global optimization

**Pipeline**:
1. Extract features from all images
2. Match features pairwise
3. Select image pairs for initialization
4. Incremental reconstruction (add images one by one)
5. Triangulate new points as images are added
6. Refine with bundle adjustment
7. Filter outliers

### 3. Integration Script (`opencv_convert.py`)
**Purpose**: Main script that orchestrates the entire process

**Features**:
- Command-line interface matching `convert.py`
- Automatic image loading and preprocessing
- Calls calibration and reconstruction modules
- Outputs COLMAP-compatible binary files:
  - `cameras.bin` - camera intrinsics
  - `images.bin` - camera poses and 2D features
  - `points3D.bin` - 3D point cloud
- Optional visualization of results

### 4. Utility Module (`utils/opencv_utils.py`)
**Purpose**: Helper functions for format conversion and I/O

**Features**:
- COLMAP binary format writers
- Quaternion/rotation matrix conversions
- Visualization utilities
- Image undistortion
- Data validation

## Technical Specifications

### Camera Model
- Start with **PINHOLE** model (matches COLMAP default)
- Intrinsic parameters: fx, fy, cx, cy
- Distortion: k1, k2, p1, p2 (OpenCV model)

### Feature Detector
- Primary: **SIFT** (scale-invariant, robust)
- Alternative: **ORB** (faster, patent-free)
- Configurable via command-line arguments

### Matching Strategy
- Brute-force matcher with ratio test (Lowe's ratio)
- Geometric verification with fundamental matrix (RANSAC)
- Minimum matches threshold: 30-50 per pair

### Bundle Adjustment
- Use **scipy.optimize** for Levenberg-Marquardt
- Optimize camera poses and 3D points jointly
- Minimize reprojection error

### Incremental Reconstruction
1. Select initial pair (highest matches, good baseline)
2. Estimate relative pose and triangulate
3. Register next image via PnP
4. Triangulate new points
5. Bundle adjustment every N images
6. Repeat until all images processed

## Output Format (COLMAP Compatible)

### cameras.bin
```
# Binary format:
num_cameras (uint64)
For each camera:
  camera_id (uint32)
  model_id (int) - 1 for PINHOLE
  width (uint64)
  height (uint64)
  params (double[]) - fx, fy, cx, cy
```

### images.bin
```
num_images (uint64)
For each image:
  image_id (uint32)
  qw, qx, qy, qz (double[4]) - rotation quaternion
  tx, ty, tz (double[3]) - translation vector
  camera_id (uint32)
  name (string)
  num_points2D (uint64)
  For each point2D:
    x, y (double[2])
    point3D_id (uint64) - -1 if not triangulated
```

### points3D.bin
```
num_points (uint64)
For each point:
  point3D_id (uint64)
  x, y, z (double[3])
  r, g, b (uint8[3])
  error (double) - reprojection error
  track_length (uint64)
  For each track element:
    image_id (uint32)
    point2D_idx (uint32)
```

## Dependencies
```
numpy
opencv-python (cv2)
scipy
struct (for binary I/O)
argparse
tqdm (progress bars)
```

## Usage Examples

### Basic Usage
```bash
python opencv_convert.py -s data/silverlake
```

### With Custom Parameters
```bash
python opencv_convert.py -s data/silverlake \
  --feature_detector sift \
  --max_features 8000 \
  --match_threshold 0.7 \
  --min_matches 30 \
  --visualize
```

### Calibration Only
```bash
python opencv_calibration.py -s data/silverlake/input \
  --output data/silverlake/calibration.json
```

### Reconstruction Only (with existing calibration)
```bash
python opencv_sparse_reconstruction.py \
  -s data/silverlake/input \
  --calibration data/silverlake/calibration.json \
  --output data/silverlake/sparse/0
```

## Validation
- Compare results with COLMAP output
- Visualize sparse point cloud
- Check reprojection errors
- Verify camera poses
- Test with Gaussian Splatting training

## Future Enhancements
- Dense reconstruction (optional)
- GPU acceleration for feature extraction
- Support for video input
- Automatic parameter tuning
- Multi-threaded matching
- Graph-based image selection
- Loop closure detection

## References
- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Structure from Motion: https://en.wikipedia.org/wiki/Structure_from_motion
- COLMAP Format: https://colmap.github.io/format.html
- Bundle Adjustment: http://ceres-solver.org/nnls_tutorial.html
