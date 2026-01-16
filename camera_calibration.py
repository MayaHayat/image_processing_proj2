import cv2
import numpy as np
import glob
import os


def calibrate_camera(calibration_path='chess', 
                     chessboard_size=(8, 6), 
                     square_size=1.0,
                     verbose=True):
    
    # Prepare object points based on the actual chessboard dimensions
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get all calibration images
    images = glob.glob(os.path.join(calibration_path, '*.jpg'))
    
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {calibration_path}")
    
    h, w = cv2.imread(images[0]).shape[:2]
    
    if verbose:
        print(f"Found {len(images)} calibration images")
        print(f"Looking for chessboard pattern: {chessboard_size}")
    
    for i, fn in enumerate(images):
        if verbose:
            print("processing %s... " % fn)
        
        imgBGR = cv2.imread(fn)
        
        if imgBGR is None:
            if verbose:
                print("Failed to load", fn)
            continue
        
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
        
        found, corners = cv2.findChessboardCorners(img, chessboard_size)
        
        if not found:
            if verbose:
                print("chessboard not found")
            continue
        
        if verbose:
            print(f"{fn}... OK")
        
        imgpoints.append(corners.reshape(-1, 2))
        objpoints.append(objp)
    
    if len(objpoints) == 0:
        raise ValueError("No chessboard patterns found in any images")
    
    # Perform camera calibration
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )
    if verbose:
        print(f"\nCalibration successful!")
        print(f"RMS reprojection error: {rms}")
        print(f"Camera matrix (K):\n{camera_matrix}")
        print(f"Distortion coefficients:\n{dist_coefs}")
    
    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coefs,
        'rms': rms,
        'image_size': (w, h),
        'rvecs': rvecs,
        'tvecs': tvecs
    }


if __name__ == "__main__":
    # Run calibration when script is executed directly
    result = calibrate_camera()
    
    # Optionally save results
    np.savez('camera_calibration.npz', 
             camera_matrix=result['camera_matrix'],
             dist_coeffs=result['dist_coeffs'],
             rms=result['rms'],
             image_size=result['image_size'])
    print("\nCalibration saved to camera_calibration.npz")