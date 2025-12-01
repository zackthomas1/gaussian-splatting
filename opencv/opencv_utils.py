"""
Utility functions for OpenCV-based camera calibration and sparse reconstruction.
Includes COLMAP format converters and helper functions.
"""

import struct
import numpy as np
from typing import Dict, List, Tuple
import cv2


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quat: Quaternion as [w, x, y, z] or [x, y, z, w]
        
    Returns:
        3x3 rotation matrix
    """
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)
    
    # Assume [w, x, y, z] format
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def write_cameras_binary(cameras: Dict, output_path: str):
    """
    Write cameras to COLMAP binary format.
    
    Args:
        cameras: Dict with camera_id as key and camera params as value
                 Each camera: {'model': str, 'width': int, 'height': int, 'params': np.ndarray}
        output_path: Path to output cameras.bin file
    """
    # Camera model mapping
    CAMERA_MODEL_NAMES = {
        'SIMPLE_PINHOLE': 0,
        'PINHOLE': 1,
        'SIMPLE_RADIAL': 2,
        'RADIAL': 3,
        'OPENCV': 4,
        'OPENCV_FISHEYE': 5,
        'FULL_OPENCV': 6,
        'FOV': 7,
        'SIMPLE_RADIAL_FISHEYE': 8,
        'RADIAL_FISHEYE': 9,
        'THIN_PRISM_FISHEYE': 10
    }
    
    with open(output_path, 'wb') as f:
        # Write number of cameras
        f.write(struct.pack('Q', len(cameras)))
        
        for camera_id, camera in cameras.items():
            model_id = CAMERA_MODEL_NAMES.get(camera['model'], 1)
            
            # Write camera properties
            f.write(struct.pack('i', camera_id))
            f.write(struct.pack('i', model_id))
            f.write(struct.pack('Q', camera['width']))
            f.write(struct.pack('Q', camera['height']))
            
            # Write camera parameters
            params = camera['params']
            for param in params:
                f.write(struct.pack('d', param))


def write_images_binary(images: Dict, output_path: str):
    """
    Write images to COLMAP binary format.
    
    Args:
        images: Dict with image_id as key and image data as value
                Each image: {'qvec': np.ndarray, 'tvec': np.ndarray, 'camera_id': int,
                           'name': str, 'xys': np.ndarray, 'point3D_ids': np.ndarray}
        output_path: Path to output images.bin file
    """
    with open(output_path, 'wb') as f:
        # Write number of images
        f.write(struct.pack('Q', len(images)))
        
        for image_id, image in images.items():
            # Write image properties
            f.write(struct.pack('i', image_id))
            
            # Write quaternion (qw, qx, qy, qz)
            qvec = image['qvec']
            for q in qvec:
                f.write(struct.pack('d', q))
            
            # Write translation vector
            tvec = image['tvec']
            for t in tvec:
                f.write(struct.pack('d', t))
            
            # Write camera_id
            f.write(struct.pack('i', image['camera_id']))
            
            # Write image name (null-terminated string)
            name_bytes = image['name'].encode('utf-8') + b'\x00'
            f.write(name_bytes)
            
            # Write 2D points
            xys = image['xys']
            point3D_ids = image['point3D_ids']
            
            f.write(struct.pack('Q', len(xys)))
            
            for xy, point3D_id in zip(xys, point3D_ids):
                f.write(struct.pack('d', xy[0]))
                f.write(struct.pack('d', xy[1]))
                f.write(struct.pack('q', point3D_id))


def write_points3D_binary(points3D: Dict, output_path: str):
    """
    Write 3D points to COLMAP binary format.
    
    Args:
        points3D: Dict with point3D_id as key and point data as value
                  Each point: {'xyz': np.ndarray, 'rgb': np.ndarray, 'error': float,
                             'image_ids': list, 'point2D_idxs': list}
        output_path: Path to output points3D.bin file
    """
    with open(output_path, 'wb') as f:
        # Write number of points
        f.write(struct.pack('Q', len(points3D)))
        
        for point3D_id, point in points3D.items():
            # Write point properties
            f.write(struct.pack('Q', point3D_id))
            
            # Write XYZ coordinates
            xyz = point['xyz']
            for coord in xyz:
                f.write(struct.pack('d', coord))
            
            # Write RGB color
            rgb = point['rgb'].astype(np.uint8)
            for color in rgb:
                f.write(struct.pack('B', color))
            
            # Write error
            f.write(struct.pack('d', point['error']))
            
            # Write track
            track_length = len(point['image_ids'])
            f.write(struct.pack('Q', track_length))
            
            for img_id, point2D_idx in zip(point['image_ids'], point['point2D_idxs']):
                f.write(struct.pack('i', img_id))
                f.write(struct.pack('i', point2D_idx))


def triangulate_points(P1: np.ndarray, P2: np.ndarray, 
                       pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D points from two views.
    
    Args:
        P1: 3x4 projection matrix for camera 1
        P2: 3x4 projection matrix for camera 2
        pts1: Nx2 array of 2D points in image 1
        pts2: Nx2 array of 2D points in image 2
        
    Returns:
        Nx3 array of 3D points
    """
    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    
    return points_3d.T


def compute_reprojection_error(points_3d: np.ndarray, points_2d: np.ndarray,
                               K: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """
    Compute mean reprojection error.
    
    Args:
        points_3d: Nx3 array of 3D points
        points_2d: Nx2 array of 2D points
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
    Returns:
        Mean reprojection error in pixels
    """
    # Project 3D points to 2D
    points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    
    # World to camera transformation
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt
    
    # Project points
    points_2d_proj = (P @ points_3d_h.T).T
    points_2d_proj = points_2d_proj[:, :2] / points_2d_proj[:, 2:3]
    
    # Compute error
    errors = np.linalg.norm(points_2d - points_2d_proj, axis=1)
    
    return np.mean(errors)


def filter_matches_ransac(kp1: List, kp2: List, matches: List,
                          threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Filter matches using RANSAC with fundamental matrix.
    
    Args:
        kp1: Keypoints from image 1
        kp2: Keypoints from image 2
        matches: List of matches
        threshold: RANSAC threshold in pixels
        
    Returns:
        pts1, pts2, filtered_matches
    """
    if len(matches) < 8:
        return np.array([]), np.array([]), []
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Compute fundamental matrix with RANSAC
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold)
    
    if mask is None:
        return np.array([]), np.array([]), []
    
    # Filter matches
    mask = mask.ravel().astype(bool)
    pts1_filtered = pts1[mask]
    pts2_filtered = pts2[mask]
    matches_filtered = [m for m, keep in zip(matches, mask) if keep]
    
    return pts1_filtered, pts2_filtered, matches_filtered


def undistort_image(image: np.ndarray, K: np.ndarray, 
                    dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undistort image using camera calibration parameters.
    
    Args:
        image: Input distorted image
        K: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3, ...]
        
    Returns:
        Undistorted image and new camera matrix
    """
    h, w = image.shape[:2]
    
    # Get optimal new camera matrix
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    
    # Undistort image
    undistorted = cv2.undistort(image, K, dist_coeffs, None, new_K)
    
    return undistorted, new_K


def visualize_matches(img1: np.ndarray, img2: np.ndarray,
                     kp1: List, kp2: List, matches: List,
                     num_display: int = 50) -> np.ndarray:
    """
    Visualize feature matches between two images.
    
    Args:
        img1: First image
        img2: Second image
        kp1: Keypoints from image 1
        kp2: Keypoints from image 2
        matches: List of matches
        num_display: Number of matches to display
        
    Returns:
        Image showing the matches
    """
    # Select subset of matches to display
    matches_display = matches[:min(num_display, len(matches))]
    
    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches_display,
                                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return match_img


def compute_pose_from_essential(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray,
                                K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover camera pose from essential matrix.
    
    Args:
        E: 3x3 essential matrix
        pts1: Nx2 points in image 1
        pts2: Nx2 points in image 2
        K: 3x3 camera intrinsic matrix
        
    Returns:
        R (rotation), t (translation), mask (inlier mask)
    """
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, t, mask


def save_ply(points_3d: np.ndarray, colors: np.ndarray, output_path: str):
    """
    Save point cloud to PLY format for visualization.
    
    Args:
        points_3d: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (0-255)
        output_path: Path to output PLY file
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points_3d)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        # Write points
        for point, color in zip(points_3d, colors):
            f.write(f'{point[0]} {point[1]} {point[2]} ')
            f.write(f'{int(color[0])} {int(color[1])} {int(color[2])}\n')
