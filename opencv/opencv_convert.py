"""
OpenCV-based camera calibration and sparse reconstruction pipeline.
Alternative to COLMAP's convert.py script using custom OpenCV implementation.
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path

# Import our custom modules
from opencv_calibration import CameraCalibrator
from opencv_sparse_reconstruction import SparseReconstructor


def setup_directories(source_path: str) -> dict:
    """
    Set up directory structure similar to COLMAP.
    
    Args:
        source_path: Base directory containing input images
        
    Returns:
        Dictionary with directory paths
    """
    paths = {
        'source': Path(source_path),
        'input': Path(source_path) / 'input',
        'sparse': Path(source_path) / 'sparse' / '0',
        'calibration': Path(source_path) / 'calibration.json'
    }
    
    # Create directories
    paths['sparse'].mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists
    if not paths['input'].exists():
        logging.error(f"Input directory not found: {paths['input']}")
        logging.info("Please place your images in <source_path>/input/")
        sys.exit(1)
    
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV-based camera calibration and sparse reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-s", "--source_path", required=True, type=str,
                       help="Path to source directory (should contain 'input' folder with images)")
    
    # Calibration arguments
    parser.add_argument("--skip_calibration", action='store_true',
                       help="Skip calibration (use existing calibration.json)")
    parser.add_argument("--calibration_method", type=str, default="auto",
                       choices=["auto", "checkerboard", "images"],
                       help="Calibration method")
    parser.add_argument("--pattern_size", type=int, nargs=2, default=[9, 6],
                       help="Checkerboard pattern size (width height)")
    parser.add_argument("--square_size", type=float, default=1.0,
                       help="Checkerboard square size")
    
    # Feature detection arguments
    parser.add_argument("--feature_detector", type=str, default="sift",
                       choices=["sift", "orb"],
                       help="Feature detector (SIFT or ORB)")
    parser.add_argument("--max_features", type=int, default=8000,
                       help="Maximum features per image")
    
    # Matching arguments
    parser.add_argument("--match_threshold", type=float, default=0.7,
                       help="Feature matching ratio test threshold (0.0-1.0)")
    parser.add_argument("--min_matches", type=int, default=30,
                       help="Minimum matches required per image pair")
    
    # Reconstruction arguments
    parser.add_argument("--skip_reconstruction", action='store_true',
                       help="Skip sparse reconstruction (calibration only)")
    
    # Visualization
    parser.add_argument("--visualize", action='store_true',
                       help="Generate visualization outputs")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("OpenCV Camera Calibration and Sparse Reconstruction")
    print("=" * 70)
    print(f"\nSource path: {args.source_path}")
    print(f"Feature detector: {args.feature_detector.upper()}")
    print(f"Max features: {args.max_features}")
    print(f"Match threshold: {args.match_threshold}")
    print(f"Min matches: {args.min_matches}")
    
    # Setup directories
    paths = setup_directories(args.source_path)
    
    # ========== STEP 1: Camera Calibration ==========
    if not args.skip_calibration:
        print("\n" + "=" * 70)
        print("STEP 1: Camera Calibration")
        print("=" * 70)
        
        calibrator = CameraCalibrator(
            str(paths['input']),
            feature_detector=args.feature_detector,
            max_features=args.max_features
        )
        
        # Perform calibration
        success = False
        
        if args.calibration_method == "checkerboard":
            success = calibrator.calibrate_from_checkerboard(
                pattern_size=tuple(args.pattern_size),
                square_size=args.square_size
            )
        elif args.calibration_method == "images":
            success = calibrator.calibrate_from_images(
                match_threshold=args.match_threshold,
                min_matches=args.min_matches
            )
        else:  # auto
            success = calibrator.calibrate_from_checkerboard(
                pattern_size=tuple(args.pattern_size),
                square_size=args.square_size
            )
            
            if not success:
                print("\nCheckerboard calibration failed. Trying image-based calibration...")
                success = calibrator.calibrate_from_images(
                    match_threshold=args.match_threshold,
                    min_matches=args.min_matches
                )
        
        if not success:
            logging.error("Calibration failed!")
            return 1
        
        # Save calibration
        calibrator.save_calibration(str(paths['calibration']))
        print(f"\n✓ Calibration complete!")
    
    else:
        print("\n" + "=" * 70)
        print("STEP 1: Camera Calibration - SKIPPED")
        print("=" * 70)
        
        if not paths['calibration'].exists():
            logging.error(f"Calibration file not found: {paths['calibration']}")
            logging.info("Run without --skip_calibration to generate calibration")
            return 1
        
        print(f"Using existing calibration: {paths['calibration']}")
    
    # ========== STEP 2: Sparse Reconstruction ==========
    if not args.skip_reconstruction:
        print("\n" + "=" * 70)
        print("STEP 2: Sparse 3D Reconstruction")
        print("=" * 70)
        
        reconstructor = SparseReconstructor(
            str(paths['input']),
            str(paths['calibration']),
            feature_detector=args.feature_detector,
            max_features=args.max_features
        )
        
        # Perform reconstruction
        try:
            reconstructor.reconstruct(
                match_threshold=args.match_threshold,
                min_matches=args.min_matches
            )
        except Exception as e:
            logging.error(f"Reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        # Save reconstruction
        reconstructor.save_reconstruction(str(paths['sparse']))
        print(f"\n✓ Sparse reconstruction complete!")
    
    else:
        print("\n" + "=" * 70)
        print("STEP 2: Sparse Reconstruction - SKIPPED")
        print("=" * 70)
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if paths['calibration'].exists():
        print(f"✓ Calibration: {paths['calibration']}")
    
    if (paths['sparse'] / 'cameras.bin').exists():
        print(f"✓ Cameras: {paths['sparse'] / 'cameras.bin'}")
    
    if (paths['sparse'] / 'images.bin').exists():
        print(f"✓ Images: {paths['sparse'] / 'images.bin'}")
    
    if (paths['sparse'] / 'points3D.bin').exists():
        print(f"✓ Points3D: {paths['sparse'] / 'points3D.bin'}")
    
    if (paths['sparse'] / 'points3D.ply').exists():
        print(f"✓ Point cloud: {paths['sparse'] / 'points3D.ply'}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"\n1. Visualize point cloud:")
    print(f"   Open {paths['sparse'] / 'points3D.ply'} in MeshLab or CloudCompare")
    
    print(f"\n2. Train Gaussian Splatting:")
    print(f"   python train.py -s {args.source_path} -m output/<model_name>")
    
    print(f"\n3. Compare with COLMAP:")
    print(f"   python convert.py -s {args.source_path}")
    print(f"   Compare sparse/0/ folders from both methods")
    
    print("\n" + "=" * 70)
    print("✓ Pipeline complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
