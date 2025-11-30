# Quick Start Guide: Custom Camera Calibration with OpenCV

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements-opencv.txt
   ```

2. **Verify installation:**
   ```bash
   python test_opencv_pipeline.py
   ```

## Run on Your Dataset

### Step 1: Prepare Your Data

Organize your images:
```
data/your_dataset/
└── input/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

### Step 2: Run the Pipeline

```bash
python opencv_convert.py -s data/your_dataset
```

This will:
- ✓ Calibrate camera from your images
- ✓ Extract and match features
- ✓ Build sparse 3D reconstruction
- ✓ Save COLMAP-compatible outputs

### Step 3: Train Gaussian Splatting

```bash
python train.py -s data/your_dataset -m output/your_dataset
```

## Example: Silverlake Dataset

```bash
# Run custom OpenCV pipeline
python opencv_convert.py -s data/silverlake

# Expected output:
# data/silverlake/
#   ├── calibration.json
#   └── sparse/0/
#       ├── cameras.bin
#       ├── images.bin
#       ├── points3D.bin
#       └── points3D.ply
```

## Customization

### Use ORB (faster, less accurate)
```bash
python opencv_convert.py -s data/your_dataset --feature_detector orb
```

### High-quality reconstruction
```bash
python opencv_convert.py -s data/your_dataset \
  --max_features 10000 \
  --match_threshold 0.6 \
  --min_matches 50
```

### Calibration only (no reconstruction)
```bash
python opencv_convert.py -s data/your_dataset --skip_reconstruction
```

## Troubleshooting

### "Not enough matches"
- Use more images (50+ recommended)
- Ensure 50-70% overlap between consecutive images
- Try: `--match_threshold 0.6 --max_features 10000`

### "Calibration failed"
- Check image quality (sharp, well-lit)
- Ensure varied camera positions/angles
- Try: `--calibration_method images`

### "Few 3D points"
- Increase image overlap
- Use better lighting
- Try: `--min_matches 20`

## Compare with COLMAP

Run both pipelines and compare:

```bash
# COLMAP
python convert.py -s data/silverlake

# OpenCV (custom)
python opencv_convert.py -s data/silverlake

# Visualize both point clouds
# Open sparse/0/points3D.ply files in MeshLab
```

## Next Steps

After successful reconstruction:

1. **Visualize sparse point cloud**
   - Open `sparse/0/points3D.ply` in MeshLab or CloudCompare

2. **Train Gaussian Splatting**
   - `python train.py -s data/your_dataset -m output/model_name`

3. **Render trained model**
   - `python render.py -m output/model_name`

4. **View interactively**
   - `./SIBR_viewers/bin/SIBR_gaussianViewer_app -m output/model_name`

## Key Files

- **opencv_convert.py** - Main pipeline script
- **opencv_calibration.py** - Camera calibration
- **opencv_sparse_reconstruction.py** - 3D reconstruction
- **utils/opencv_utils.py** - Helper functions
- **docs/OPENCV_CALIBRATION_PLAN.md** - Technical details
- **docs/OPENCV_CALIBRATION_README.md** - Full documentation

## Support

For detailed information, see:
- `docs/OPENCV_CALIBRATION_README.md` - Complete usage guide
- `docs/OPENCV_CALIBRATION_PLAN.md` - Technical documentation
- Original COLMAP: https://colmap.github.io/

## Tips for Best Results

1. **Image capture:**
   - 50-100 images with good overlap
   - Capture from multiple angles
   - Consistent lighting
   - Sharp images (no motion blur)

2. **Feature detection:**
   - SIFT for best quality (slower)
   - ORB for quick tests (faster)
   - More features = better (but slower)

3. **Reconstruction quality:**
   - More images = denser point cloud
   - Better camera calibration = better 3D structure
   - Good feature matches = more 3D points
