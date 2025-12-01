"""
Test script for OpenCV calibration and reconstruction pipeline.
Validates the installation and basic functionality.
"""

import sys
import os

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"  ✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import scipy
        print(f"  ✓ SciPy version: {scipy.__version__}")
    except ImportError as e:
        print(f"  ✗ SciPy import failed: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print(f"  ✓ tqdm installed")
    except ImportError as e:
        print(f"  ✗ tqdm import failed: {e}")
        return False
    
    return True


def test_opencv_features():
    """Test that OpenCV feature detectors are available."""
    print("\nTesting OpenCV feature detectors...")
    
    import cv2
    import numpy as np
    
    # Create a dummy image
    img = np.zeros((480, 640), dtype=np.uint8)
    
    try:
        sift = cv2.SIFT_create()
        kp, desc = sift.detectAndCompute(img, None)
        print(f"  ✓ SIFT detector available")
    except Exception as e:
        print(f"  ✗ SIFT detector failed: {e}")
        return False
    
    try:
        orb = cv2.ORB_create()
        kp, desc = orb.detectAndCompute(img, None)
        print(f"  ✓ ORB detector available")
    except Exception as e:
        print(f"  ✗ ORB detector failed: {e}")
        return False
    
    return True


def test_modules():
    """Test that custom modules can be imported."""
    print("\nTesting custom modules...")
    
    try:
        import opencv_utils
        print(f"  ✓ opencv_utils module")
    except ImportError as e:
        print(f"  ✗ opencv_utils import failed: {e}")
        print(f"     Make sure you're running from the opencv directory")
        return False
    
    try:
        import opencv_calibration
        print(f"  ✓ opencv_calibration module")
    except ImportError as e:
        print(f"  ✗ opencv_calibration import failed: {e}")
        return False
    
    try:
        import opencv_sparse_reconstruction
        print(f"  ✓ opencv_sparse_reconstruction module")
    except ImportError as e:
        print(f"  ✗ opencv_sparse_reconstruction import failed: {e}")
        return False
    
    try:
        import opencv_convert
        print(f"  ✓ opencv_convert module")
    except ImportError as e:
        print(f"  ✗ opencv_convert import failed: {e}")
        return False
    
    return True


def test_utilities():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    import numpy as np
    from opencv_utils import rotmat_to_quat, quat_to_rotmat
    
    # Test rotation conversion
    R = np.eye(3)
    quat = rotmat_to_quat(R)
    R_back = quat_to_rotmat(quat)
    
    if np.allclose(R, R_back):
        print(f"  ✓ Rotation/quaternion conversion")
    else:
        print(f"  ✗ Rotation/quaternion conversion failed")
        return False
    
    # Test with non-identity rotation
    angle = np.pi / 4
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    quat = rotmat_to_quat(R)
    R_back = quat_to_rotmat(quat)
    
    if np.allclose(R, R_back, atol=1e-6):
        print(f"  ✓ Non-trivial rotation conversion")
    else:
        print(f"  ✗ Non-trivial rotation conversion failed")
        return False
    
    return True


def main():
    print("=" * 60)
    print("OpenCV Calibration Pipeline - Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_opencv_features()
    all_passed &= test_modules()
    all_passed &= test_utilities()
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYou can now run the pipeline:")
        print("  python opencv_convert.py -s data/silverlake")
    else:
        print("✗ Some tests failed!")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements-opencv.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
