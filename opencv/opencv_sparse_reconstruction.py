"""
Sparse 3D reconstruction using OpenCV.
Performs structure-from-motion to generate sparse point cloud.
"""

import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
from tqdm import tqdm
from collections import defaultdict
from opencv_utils import (
    rotmat_to_quat, triangulate_points, compute_reprojection_error,
    filter_matches_ransac, compute_pose_from_essential,
    write_cameras_binary, write_images_binary, write_points3D_binary,
    save_ply
)


class SparseReconstructor:
    """
    Incremental structure-from-motion for sparse 3D reconstruction.
    """
    
    def __init__(self, image_dir: str, calibration_path: str,
                 feature_detector: str = 'sift', max_features: int = 8000):
        """
        Initialize sparse reconstructor.
        
        Args:
            image_dir: Directory containing input images
            calibration_path: Path to camera calibration JSON file
            feature_detector: Feature detector to use
            max_features: Maximum number of features per image
        """
        self.image_dir = Path(image_dir)
        self.image_paths = self._load_images()
        self.feature_detector = feature_detector.lower()
        self.max_features = max_features
        
        # Load calibration
        self._load_calibration(calibration_path)
        
        # Reconstruction data
        self.images_data = {}  # image_id -> {R, t, qvec, tvec, registered, keypoints, descriptors}
        self.points_3d = {}    # point3d_id -> {xyz, rgb, error, image_ids, point2D_idxs}
        self.matches_graph = defaultdict(dict)  # image_id -> image_id -> matches
        
        # Next IDs
        self.next_point3d_id = 0
        
        print(f"Loaded {len(self.image_paths)} images")
        print(f"Camera calibration loaded")
    
    def _load_images(self) -> List[Path]:
        """Load all image paths from directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in valid_extensions:
            image_paths.extend(self.image_dir.glob(f'*{ext}'))
            image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        return sorted(image_paths)
    
    def _load_calibration(self, calibration_path: str):
        """Load camera calibration from JSON."""
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
        
        self.K = np.array(calib_data['camera_matrix'])
        self.dist_coeffs = np.array(calib_data['distortion_coefficients'])
        self.image_size = tuple(calib_data['image_size'])
        
        print(f"Intrinsic matrix:\n{self.K}")
    
    def extract_features(self):
        """Extract features from all images."""
        print("\n=== Feature Extraction ===")
        
        # Initialize detector
        if self.feature_detector == 'sift':
            detector = cv2.SIFT_create(nfeatures=self.max_features)
        elif self.feature_detector == 'orb':
            detector = cv2.ORB_create(nfeatures=self.max_features)
        else:
            raise ValueError(f"Unknown feature detector: {self.feature_detector}")
        
        for img_id, img_path in enumerate(tqdm(self.image_paths, desc="Extracting features")):
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect and compute features
            kp, desc = detector.detectAndCompute(gray, None)
            
            # Store image data
            self.images_data[img_id] = {
                'name': img_path.name,
                'path': img_path,
                'image': img,
                'keypoints': kp,
                'descriptors': desc,
                'registered': False,
                'R': None,
                't': None,
                'qvec': None,
                'tvec': None,
                'point3D_ids': [-1] * len(kp)  # -1 means not triangulated
            }
        
        print(f"Extracted features from {len(self.images_data)} images")
    
    def match_features(self, match_threshold: float = 0.7, min_matches: int = 30):
        """
        Match features between all image pairs.
        
        Args:
            match_threshold: Ratio test threshold
            min_matches: Minimum matches to keep a pair
        """
        print("\n=== Feature Matching ===")
        
        # Initialize matcher
        if self.feature_detector == 'sift':
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        num_images = len(self.images_data)
        total_pairs = num_images * (num_images - 1) // 2
        
        with tqdm(total=total_pairs, desc="Matching pairs") as pbar:
            for i in range(num_images):
                for j in range(i + 1, num_images):
                    desc1 = self.images_data[i]['descriptors']
                    desc2 = self.images_data[j]['descriptors']
                    
                    if desc1 is None or desc2 is None:
                        pbar.update(1)
                        continue
                    
                    # Match features
                    matches = matcher.knnMatch(desc1, desc2, k=2)
                    
                    # Apply ratio test
                    good_matches = []
                    for m_n in matches:
                        if len(m_n) == 2:
                            m, n = m_n
                            if m.distance < match_threshold * n.distance:
                                good_matches.append(m)
                    
                    # Filter with RANSAC
                    if len(good_matches) >= min_matches:
                        kp1 = self.images_data[i]['keypoints']
                        kp2 = self.images_data[j]['keypoints']
                        
                        pts1, pts2, filtered_matches = filter_matches_ransac(
                            kp1, kp2, good_matches, threshold=3.0
                        )
                        
                        if len(filtered_matches) >= min_matches:
                            self.matches_graph[i][j] = {
                                'matches': filtered_matches,
                                'pts1': pts1,
                                'pts2': pts2
                            }
                    
                    pbar.update(1)
        
        # Count total matches
        total_matches = sum(len(matches) for matches in self.matches_graph.values())
        print(f"Found {total_matches} valid image pairs")
    
    def select_initial_pair(self) -> Tuple[int, int]:
        """
        Select best initial image pair for reconstruction.
        
        Returns:
            Tuple of (img_id1, img_id2)
        """
        print("\n=== Selecting Initial Pair ===")
        
        best_pair = None
        best_score = 0
        
        for img_id1, connections in self.matches_graph.items():
            for img_id2, match_data in connections.items():
                matches = match_data['matches']
                pts1 = match_data['pts1']
                pts2 = match_data['pts2']
                
                # Score based on number of matches and baseline
                num_matches = len(matches)
                
                # Compute median disparity as proxy for baseline
                disparity = np.median(np.linalg.norm(pts1 - pts2, axis=1))
                
                # Score favors many matches and good baseline
                score = num_matches * disparity
                
                if score > best_score:
                    best_score = score
                    best_pair = (img_id1, img_id2)
        
        if best_pair is None:
            raise ValueError("Could not find suitable initial pair")
        
        print(f"Selected initial pair: {self.images_data[best_pair[0]]['name']} - "
              f"{self.images_data[best_pair[1]]['name']}")
        print(f"  Matches: {len(self.matches_graph[best_pair[0]][best_pair[1]]['matches'])}")
        
        return best_pair
    
    def initialize_reconstruction(self, img_id1: int, img_id2: int):
        """
        Initialize reconstruction from two views.
        
        Args:
            img_id1: First image ID
            img_id2: Second image ID
        """
        print("\n=== Initializing Reconstruction ===")
        
        match_data = self.matches_graph[img_id1][img_id2]
        pts1 = match_data['pts1']
        pts2 = match_data['pts2']
        matches = match_data['matches']
        
        # Compute essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Recover pose
        R, t, pose_mask = compute_pose_from_essential(E, pts1, pts2, self.K)
        
        # First camera at origin
        self.images_data[img_id1]['R'] = np.eye(3)
        self.images_data[img_id1]['t'] = np.zeros((3, 1))
        self.images_data[img_id1]['qvec'] = rotmat_to_quat(np.eye(3))
        self.images_data[img_id1]['tvec'] = np.zeros(3)
        self.images_data[img_id1]['registered'] = True
        
        # Second camera with relative pose
        self.images_data[img_id2]['R'] = R
        self.images_data[img_id2]['t'] = t
        self.images_data[img_id2]['qvec'] = rotmat_to_quat(R)
        self.images_data[img_id2]['tvec'] = t.ravel()
        self.images_data[img_id2]['registered'] = True
        
        # Triangulate points
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        # Filter points with pose mask
        mask_combined = (mask.ravel() == 1) & (pose_mask.ravel() > 0)
        pts1_good = pts1[mask_combined]
        pts2_good = pts2[mask_combined]
        matches_good = [m for m, keep in zip(matches, mask_combined) if keep]
        
        points_3d = triangulate_points(P1, P2, pts1_good, pts2_good)
        
        # Add 3D points
        for i, (pt3d, match) in enumerate(zip(points_3d, matches_good)):
            # Check if point is in front of both cameras
            if pt3d[2] > 0 and (R @ pt3d + t.ravel())[2] > 0:
                # Get color from first image
                kp_idx1 = match.queryIdx
                kp_idx2 = match.trainIdx
                
                pt2d = self.images_data[img_id1]['keypoints'][kp_idx1].pt
                color = self.images_data[img_id1]['image'][int(pt2d[1]), int(pt2d[0])]
                color_rgb = color[::-1]  # BGR to RGB
                
                # Compute reprojection error
                pts_2d_arr = np.array([pts1_good[i]])
                pts_3d_arr = np.array([pt3d])
                error1 = compute_reprojection_error(pts_3d_arr, pts_2d_arr, self.K,
                                                   np.eye(3), np.zeros((3, 1)))
                
                pts_2d_arr2 = np.array([pts2_good[i]])
                error2 = compute_reprojection_error(pts_3d_arr, pts_2d_arr2, self.K, R, t)
                
                error = (error1 + error2) / 2
                
                # Add point
                point3d_id = self.next_point3d_id
                self.next_point3d_id += 1
                
                self.points_3d[point3d_id] = {
                    'xyz': pt3d,
                    'rgb': color_rgb,
                    'error': error,
                    'image_ids': [img_id1, img_id2],
                    'point2D_idxs': [kp_idx1, kp_idx2]
                }
                
                # Update image data
                self.images_data[img_id1]['point3D_ids'][kp_idx1] = point3d_id
                self.images_data[img_id2]['point3D_ids'][kp_idx2] = point3d_id
        
        print(f"Initialized reconstruction with {len(self.points_3d)} points")
        print(f"Registered images: {img_id1}, {img_id2}")
    
    def register_next_image(self) -> bool:
        """
        Register next best image to reconstruction.
        
        Returns:
            True if an image was registered
        """
        # Find unregistered image with most 3D point observations
        best_img_id = None
        best_score = 0
        
        for img_id, img_data in self.images_data.items():
            if img_data['registered']:
                continue
            
            # Count how many 2D points correspond to triangulated 3D points
            num_3d_correspondences = 0
            
            for other_id, img_other in self.images_data.items():
                if not img_other['registered']:
                    continue
                
                if img_id in self.matches_graph and other_id in self.matches_graph[img_id]:
                    match_data = self.matches_graph[img_id][other_id]
                    matches = match_data['matches']
                    
                    for match in matches:
                        kp_idx_other = match.trainIdx
                        # Bounds check
                        if kp_idx_other < len(img_other['point3D_ids']) and img_other['point3D_ids'][kp_idx_other] != -1:
                            num_3d_correspondences += 1
                            
                elif other_id in self.matches_graph and img_id in self.matches_graph[other_id]:
                    match_data = self.matches_graph[other_id][img_id]
                    matches = match_data['matches']
                    
                    for match in matches:
                        kp_idx_other = match.queryIdx
                        # Bounds check
                        if kp_idx_other < len(img_other['point3D_ids']) and img_other['point3D_ids'][kp_idx_other] != -1:
                            num_3d_correspondences += 1
            
            if num_3d_correspondences > best_score:
                best_score = num_3d_correspondences
                best_img_id = img_id
        
        if best_img_id is None or best_score < 10:
            return False
        
        # Register image using PnP
        return self._register_image_pnp(best_img_id)
    
    def _register_image_pnp(self, img_id: int) -> bool:
        """
        Register image using PnP (Perspective-n-Point).
        
        Args:
            img_id: Image ID to register
            
        Returns:
            True if registration successful
        """
        # Collect 2D-3D correspondences
        points_2d = []
        points_3d = []
        point2d_idxs = []
        
        for other_id, img_other in self.images_data.items():
            if not img_other['registered'] or img_id == other_id:
                continue
            
            # Get matches
            if img_id in self.matches_graph and other_id in self.matches_graph[img_id]:
                match_data = self.matches_graph[img_id][other_id]
                matches = match_data['matches']
                is_forward = True
            elif other_id in self.matches_graph and img_id in self.matches_graph[other_id]:
                match_data = self.matches_graph[other_id][img_id]
                matches = match_data['matches']
                is_forward = False
            else:
                continue
            
            for match in matches:
                if is_forward:
                    kp_idx_img = match.queryIdx
                    kp_idx_other = match.trainIdx
                else:
                    kp_idx_img = match.trainIdx
                    kp_idx_other = match.queryIdx
                
                # Bounds check
                if kp_idx_other >= len(img_other['point3D_ids']) or kp_idx_img >= len(self.images_data[img_id]['keypoints']):
                    continue
                    
                point3d_id = img_other['point3D_ids'][kp_idx_other]
                
                if point3d_id != -1:
                    pt_2d = self.images_data[img_id]['keypoints'][kp_idx_img].pt
                    pt_3d = self.points_3d[point3d_id]['xyz']
                    
                    points_2d.append(pt_2d)
                    points_3d.append(pt_3d)
                    point2d_idxs.append(kp_idx_img)
        
        if len(points_2d) < 10:
            return False
        
        points_2d = np.array(points_2d, dtype=np.float32)
        points_3d = np.array(points_3d, dtype=np.float32)
        
        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, self.K, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=8.0
        )
        
        if not success or inliers is None or len(inliers) < 10:
            return False
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Store pose
        self.images_data[img_id]['R'] = R
        self.images_data[img_id]['t'] = tvec
        self.images_data[img_id]['qvec'] = rotmat_to_quat(R)
        self.images_data[img_id]['tvec'] = tvec.ravel()
        self.images_data[img_id]['registered'] = True
        
        # Update point3D correspondences for inliers
        for idx in inliers.ravel():
            kp_idx = point2d_idxs[idx]
            # Find corresponding 3D point
            for other_id, img_other in self.images_data.items():
                if not img_other['registered'] or img_id == other_id:
                    continue
                
                if img_id in self.matches_graph and other_id in self.matches_graph[img_id]:
                    matches = self.matches_graph[img_id][other_id]['matches']
                    for match in matches:
                        if match.queryIdx == kp_idx:
                            # Bounds check
                            if match.trainIdx >= len(img_other['point3D_ids']):
                                continue
                            point3d_id = img_other['point3D_ids'][match.trainIdx]
                            if point3d_id != -1:
                                self.images_data[img_id]['point3D_ids'][kp_idx] = point3d_id
                                # Add observation to 3D point
                                if img_id not in self.points_3d[point3d_id]['image_ids']:
                                    self.points_3d[point3d_id]['image_ids'].append(img_id)
                                    self.points_3d[point3d_id]['point2D_idxs'].append(kp_idx)
                                break
        
        print(f"Registered image {img_id}: {self.images_data[img_id]['name']} "
              f"({len(inliers)} inliers)")
        
        return True
    
    def triangulate_new_points(self):
        """Triangulate new 3D points from registered images."""
        new_points = 0
        
        registered_ids = [img_id for img_id, data in self.images_data.items() if data['registered']]
        
        for i, img_id1 in enumerate(registered_ids):
            for img_id2 in registered_ids[i+1:]:
                # Get matches
                if img_id1 in self.matches_graph and img_id2 in self.matches_graph[img_id1]:
                    match_data = self.matches_graph[img_id1][img_id2]
                elif img_id2 in self.matches_graph and img_id1 in self.matches_graph[img_id2]:
                    match_data = self.matches_graph[img_id2][img_id1]
                    img_id1, img_id2 = img_id2, img_id1
                else:
                    continue
                
                matches = match_data['matches']
                pts1 = match_data['pts1']
                pts2 = match_data['pts2']
                
                # Get camera matrices
                R1 = self.images_data[img_id1]['R']
                t1 = self.images_data[img_id1]['t']
                R2 = self.images_data[img_id2]['R']
                t2 = self.images_data[img_id2]['t']
                
                P1 = self.K @ np.hstack([R1, t1])
                P2 = self.K @ np.hstack([R2, t2])
                
                # Triangulate
                points_3d = triangulate_points(P1, P2, pts1, pts2)
                
                for i, (pt3d, match) in enumerate(zip(points_3d, matches)):
                    kp_idx1 = match.queryIdx
                    kp_idx2 = match.trainIdx
                    
                    # Bounds check
                    if (kp_idx1 >= len(self.images_data[img_id1]['point3D_ids']) or
                        kp_idx2 >= len(self.images_data[img_id2]['point3D_ids'])):
                        continue
                    
                    # Skip if already triangulated
                    if (self.images_data[img_id1]['point3D_ids'][kp_idx1] != -1 or
                        self.images_data[img_id2]['point3D_ids'][kp_idx2] != -1):
                        continue
                    
                    # Check if point is valid (in front of both cameras)
                    pt_cam1 = R1 @ pt3d + t1.ravel()
                    pt_cam2 = R2 @ pt3d + t2.ravel()
                    
                    if pt_cam1[2] > 0 and pt_cam2[2] > 0:
                        # Get color
                        pt2d = self.images_data[img_id1]['keypoints'][kp_idx1].pt
                        color = self.images_data[img_id1]['image'][int(pt2d[1]), int(pt2d[0])]
                        color_rgb = color[::-1]
                        
                        # Add point
                        point3d_id = self.next_point3d_id
                        self.next_point3d_id += 1
                        
                        self.points_3d[point3d_id] = {
                            'xyz': pt3d,
                            'rgb': color_rgb,
                            'error': 0.0,
                            'image_ids': [img_id1, img_id2],
                            'point2D_idxs': [kp_idx1, kp_idx2]
                        }
                        
                        self.images_data[img_id1]['point3D_ids'][kp_idx1] = point3d_id
                        self.images_data[img_id2]['point3D_ids'][kp_idx2] = point3d_id
                        
                        new_points += 1
        
        if new_points > 0:
            print(f"Triangulated {new_points} new points")
    
    def reconstruct(self, match_threshold: float = 0.7, min_matches: int = 30):
        """
        Perform full incremental reconstruction.
        
        Args:
            match_threshold: Feature matching threshold
            min_matches: Minimum matches per pair
        """
        # Extract features
        self.extract_features()
        
        # Match features
        self.match_features(match_threshold, min_matches)
        
        # Select initial pair and initialize
        img_id1, img_id2 = self.select_initial_pair()
        self.initialize_reconstruction(img_id1, img_id2)
        
        # Incrementally add images
        print("\n=== Incremental Reconstruction ===")
        num_registered = 2
        max_images = len(self.images_data)
        
        while num_registered < max_images:
            # Try to register next image
            if not self.register_next_image():
                print("No more images can be registered")
                break
            
            num_registered += 1
            
            # Triangulate new points every few images
            if num_registered % 5 == 0:
                self.triangulate_new_points()
        
        # Final triangulation
        print("\n=== Final Triangulation ===")
        self.triangulate_new_points()
        
        print(f"\n=== Reconstruction Complete ===")
        print(f"Registered images: {num_registered}/{max_images}")
        print(f"Total 3D points: {len(self.points_3d)}")
    
    def save_reconstruction(self, output_dir: str):
        """
        Save reconstruction in COLMAP format.
        
        Args:
            output_dir: Output directory for COLMAP files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Saving Reconstruction ===")
        
        # Prepare cameras data (assuming single camera)
        cameras = {
            1: {
                'model': 'PINHOLE',
                'width': self.image_size[0],
                'height': self.image_size[1],
                'params': np.array([self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
            }
        }
        
        # Prepare images data
        images = {}
        for img_id, img_data in self.images_data.items():
            if img_data['registered']:
                images[img_id + 1] = {  # COLMAP uses 1-based indexing
                    'qvec': img_data['qvec'],
                    'tvec': img_data['tvec'],
                    'camera_id': 1,
                    'name': img_data['name'],
                    'xys': np.array([kp.pt for kp in img_data['keypoints']]),
                    'point3D_ids': np.array(img_data['point3D_ids'], dtype=np.int64)
                }
        
        # Prepare points3D data
        points_3d = {}
        for point_id, point_data in self.points_3d.items():
            # Adjust image IDs to 1-based
            image_ids_adjusted = [img_id + 1 for img_id in point_data['image_ids']]
            
            points_3d[point_id] = {
                'xyz': point_data['xyz'],
                'rgb': point_data['rgb'],
                'error': point_data['error'],
                'image_ids': image_ids_adjusted,
                'point2D_idxs': point_data['point2D_idxs']
            }
        
        # Write binary files
        write_cameras_binary(cameras, os.path.join(output_dir, 'cameras.bin'))
        write_images_binary(images, os.path.join(output_dir, 'images.bin'))
        write_points3D_binary(points_3d, os.path.join(output_dir, 'points3D.bin'))
        
        print(f"Saved to {output_dir}:")
        print(f"  - cameras.bin ({len(cameras)} cameras)")
        print(f"  - images.bin ({len(images)} images)")
        print(f"  - points3D.bin ({len(points_3d)} points)")
        
        # Also save PLY for visualization
        if len(self.points_3d) > 0:
            points_xyz = np.array([p['xyz'] for p in self.points_3d.values()])
            points_rgb = np.array([p['rgb'] for p in self.points_3d.values()])
            
            ply_path = os.path.join(output_dir, 'points3D.ply')
            save_ply(points_xyz, points_rgb, ply_path)
            print(f"  - points3D.ply (for visualization)")


def main():
    parser = argparse.ArgumentParser(description="Sparse 3D reconstruction using OpenCV")
    parser.add_argument("-s", "--source_path", required=True, type=str,
                       help="Path to directory containing images")
    parser.add_argument("--calibration", type=str, required=True,
                       help="Path to camera calibration JSON file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: source_path/sparse/0)")
    parser.add_argument("--feature_detector", type=str, default="sift",
                       choices=["sift", "orb"],
                       help="Feature detector to use")
    parser.add_argument("--max_features", type=int, default=8000,
                       help="Maximum number of features per image")
    parser.add_argument("--match_threshold", type=float, default=0.7,
                       help="Feature matching ratio test threshold")
    parser.add_argument("--min_matches", type=int, default=30,
                       help="Minimum number of matches per pair")
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        args.output = os.path.join(args.source_path, "sparse", "0")
    
    # Initialize reconstructor
    reconstructor = SparseReconstructor(
        args.source_path,
        args.calibration,
        feature_detector=args.feature_detector,
        max_features=args.max_features
    )
    
    # Perform reconstruction
    reconstructor.reconstruct(
        match_threshold=args.match_threshold,
        min_matches=args.min_matches
    )
    
    # Save results
    reconstructor.save_reconstruction(args.output)
    
    print("\n✓ Sparse reconstruction complete!")
    return 0


if __name__ == "__main__":
    exit(main())
