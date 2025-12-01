"""
Camera calibration using OpenCV.
Estimates intrinsic camera parameters from a set of images.
"""

import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm


class CameraCalibrator:
    """
    Camera calibration class supporting both checkerboard and self-calibration methods.
    """
    
    def __init__(self, image_dir: str, feature_detector: str = 'sift',
                 max_features: int = 8000):
        """
        Initialize camera calibrator.
        
        Args:
            image_dir: Directory containing input images
            feature_detector: Feature detector to use ('sift' or 'orb')
            max_features: Maximum number of features to detect
        """
        self.image_dir = Path(image_dir)
        self.image_paths = self._load_images()
        self.feature_detector = feature_detector.lower()
        self.max_features = max_features
        
        # Calibration results
        self.K = None  # Intrinsic matrix
        self.dist_coeffs = None  # Distortion coefficients
        self.image_size = None
        
        print(f"Loaded {len(self.image_paths)} images from {image_dir}")
    
    def _load_images(self) -> List[Path]:
        """Load all image paths from directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in valid_extensions:
            image_paths.extend(self.image_dir.glob(f'*{ext}'))
            image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        return sorted(image_paths)
    
    def calibrate_from_checkerboard(self, pattern_size: Tuple[int, int] = (9, 6),
                                    square_size: float = 1.0) -> bool:
        """
        Calibrate camera using checkerboard pattern.
        
        Args:
            pattern_size: Number of inner corners (width, height)
            square_size: Size of checkerboard square in world units
            
        Returns:
            True if calibration successful
        """
        print(f"\nCalibrating with checkerboard pattern {pattern_size}...")
        
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in world space
        imgpoints = []  # 2D points in image plane
        
        for img_path in tqdm(self.image_paths, desc="Finding checkerboard"):
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.image_size is None:
                self.image_size = gray.shape[::-1]
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_refined)
        
        if len(objpoints) < 3:
            print(f"Error: Found checkerboard in only {len(objpoints)} images. Need at least 3.")
            return False
        
        print(f"Found checkerboard in {len(objpoints)} images")
        
        # Calibrate camera
        ret, self.K, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        
        if ret:
            print(f"Calibration successful! RMS error: {ret:.4f}")
            self._print_calibration_results()
            return True
        
        return False
    
    def calibrate_from_images(self, match_threshold: float = 0.7,
                             min_matches: int = 50) -> bool:
        """
        Self-calibration from images without checkerboard.
        Uses feature matching and fundamental matrix estimation.
        
        Args:
            match_threshold: Ratio test threshold for feature matching
            min_matches: Minimum number of matches required
            
        Returns:
            True if calibration successful
        """
        print(f"\nSelf-calibrating from images using {self.feature_detector.upper()}...")
        
        # Initialize feature detector
        if self.feature_detector == 'sift':
            detector = cv2.SIFT_create(nfeatures=self.max_features)
        elif self.feature_detector == 'orb':
            detector = cv2.ORB_create(nfeatures=self.max_features)
        else:
            raise ValueError(f"Unknown feature detector: {self.feature_detector}")
        
        # Read first image to get size
        img0 = cv2.imread(str(self.image_paths[0]))
        self.image_size = (img0.shape[1], img0.shape[0])
        h, w = img0.shape[:2]
        
        # Initial guess for intrinsic matrix
        focal_length = max(h, w) * 1.2  # Common heuristic
        cx, cy = w / 2, h / 2
        
        self.K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Extract features from all images
        print("Extracting features...")
        all_keypoints = []
        all_descriptors = []
        
        for img_path in tqdm(self.image_paths, desc="Feature extraction"):
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            kp, desc = detector.detectAndCompute(gray, None)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
        
        # Match features between consecutive pairs to refine calibration
        print("Matching features and estimating focal length...")
        
        focal_lengths = []
        
        # Use BFMatcher
        if self.feature_detector == 'sift':
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Try multiple image pairs
        num_pairs = min(10, len(self.image_paths) - 1)
        
        for i in range(num_pairs):
            if all_descriptors[i] is None or all_descriptors[i+1] is None:
                continue
            
            # Match features
            matches = matcher.knnMatch(all_descriptors[i], all_descriptors[i+1], k=2)
            
            # Apply ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < match_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < min_matches:
                continue
            
            # Get matched points
            pts1 = np.float32([all_keypoints[i][m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([all_keypoints[i+1][m.trainIdx].pt for m in good_matches])
            
            # Estimate fundamental matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
            
            if F is None:
                continue
            
            # Estimate focal length from fundamental matrix
            # For calibrated cameras: E = K^T * F * K
            # We can estimate focal length from F
            pts1_inliers = pts1[mask.ravel() == 1]
            pts2_inliers = pts2[mask.ravel() == 1]
            
            if len(pts1_inliers) > 20:
                # Use median focal length estimation
                focal_est = self._estimate_focal_from_fundamental(F, pts1_inliers, pts2_inliers, w, h)
                if focal_est > 0 and focal_est < max(w, h) * 2:
                    focal_lengths.append(focal_est)
        
        # Refine focal length estimate
        if len(focal_lengths) > 0:
            focal_length = np.median(focal_lengths)
            print(f"Estimated focal length: {focal_length:.2f} (from {len(focal_lengths)} pairs)")
            
            self.K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            print("Warning: Could not refine focal length estimate. Using initial guess.")
        
        # Assume no distortion for self-calibration
        self.dist_coeffs = np.zeros(5)
        
        print("Self-calibration complete!")
        self._print_calibration_results()
        
        return True
    
    def _estimate_focal_from_fundamental(self, F: np.ndarray, pts1: np.ndarray,
                                        pts2: np.ndarray, w: int, h: int) -> float:
        """
        Estimate focal length from fundamental matrix.
        
        Args:
            F: Fundamental matrix
            pts1: Points from image 1
            pts2: Points from image 2
            w: Image width
            h: Image height
            
        Returns:
            Estimated focal length
        """
        # Normalize points
        pts1_norm = pts1 - np.array([w/2, h/2])
        pts2_norm = pts2 - np.array([w/2, h/2])
        
        # Estimate focal length using the property that F ~ [t]_x * R for calibrated cameras
        # This is a simplified estimation
        scale = np.median(np.linalg.norm(pts1_norm, axis=1))
        
        if scale > 0:
            focal = max(w, h) / (2 * np.tan(np.arctan(scale / max(w, h)) / 2))
            return focal
        
        return max(w, h) * 1.2
    
    def _print_calibration_results(self):
        """Print calibration results."""
        print("\n=== Calibration Results ===")
        print(f"Image size: {self.image_size}")
        print(f"\nIntrinsic matrix (K):")
        print(self.K)
        print(f"\nDistortion coefficients: {self.dist_coeffs.ravel()}")
        print(f"\nFocal lengths: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}")
        print(f"Principal point: cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")
    
    def save_calibration(self, output_path: str):
        """
        Save calibration results to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        if self.K is None:
            print("Error: No calibration results to save")
            return
        
        calibration_data = {
            'image_size': self.image_size,
            'camera_matrix': self.K.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'model': 'PINHOLE',
            'params': [self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\nCalibration saved to {output_path}")
    
    def load_calibration(self, input_path: str):
        """
        Load calibration from JSON file.
        
        Args:
            input_path: Path to input JSON file
        """
        with open(input_path, 'r') as f:
            calibration_data = json.load(f)
        
        self.image_size = tuple(calibration_data['image_size'])
        self.K = np.array(calibration_data['camera_matrix'])
        self.dist_coeffs = np.array(calibration_data['distortion_coefficients'])
        
        print(f"Calibration loaded from {input_path}")
        self._print_calibration_results()


def main():
    parser = argparse.ArgumentParser(description="Camera calibration using OpenCV")
    parser.add_argument("-s", "--source_path", required=True, type=str,
                       help="Path to directory containing images")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for calibration file (default: source_path/calibration.json)")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "checkerboard", "images"],
                       help="Calibration method (default: auto - tries checkerboard, falls back to images)")
    parser.add_argument("--pattern_size", type=int, nargs=2, default=[9, 6],
                       help="Checkerboard pattern size (width height)")
    parser.add_argument("--square_size", type=float, default=1.0,
                       help="Checkerboard square size in world units")
    parser.add_argument("--feature_detector", type=str, default="sift", choices=["sift", "orb"],
                       help="Feature detector for image-based calibration")
    parser.add_argument("--max_features", type=int, default=8000,
                       help="Maximum number of features to detect")
    parser.add_argument("--match_threshold", type=float, default=0.7,
                       help="Feature matching ratio test threshold")
    parser.add_argument("--min_matches", type=int, default=50,
                       help="Minimum number of matches for calibration")
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        args.output = os.path.join(args.source_path, "calibration.json")
    
    # Initialize calibrator
    calibrator = CameraCalibrator(
        args.source_path,
        feature_detector=args.feature_detector,
        max_features=args.max_features
    )
    
    # Perform calibration
    success = False
    
    if args.method == "checkerboard":
        success = calibrator.calibrate_from_checkerboard(
            pattern_size=tuple(args.pattern_size),
            square_size=args.square_size
        )
    elif args.method == "images":
        success = calibrator.calibrate_from_images(
            match_threshold=args.match_threshold,
            min_matches=args.min_matches
        )
    else:  # auto
        # Try checkerboard first
        success = calibrator.calibrate_from_checkerboard(
            pattern_size=tuple(args.pattern_size),
            square_size=args.square_size
        )
        
        # Fall back to image-based calibration
        if not success:
            print("\nCheckerboard calibration failed. Trying image-based calibration...")
            success = calibrator.calibrate_from_images(
                match_threshold=args.match_threshold,
                min_matches=args.min_matches
            )
    
    # Save results
    if success:
        calibrator.save_calibration(args.output)
        print("\n✓ Calibration complete!")
    else:
        print("\n✗ Calibration failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
