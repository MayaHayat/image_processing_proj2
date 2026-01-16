import cv2
import numpy as np
from camera_calibration import calibrate_camera

# 1. Load assets
target_img = cv2.imread('org_image.png') 
cap = cv2.VideoCapture('video.mp4')

# 2. GET CALIBRATED CAMERA PARAMETERS
try:
    # Try to load previously saved calibration
    calib_data = np.load('camera_calibration.npz')
    K = calib_data['camera_matrix']
    dist_coeffs = calib_data['dist_coeffs']
    print("Loaded calibration from camera_calibration.npz")
except FileNotFoundError:
    # If not found, run calibration
    print("Running camera calibration...")
    calib = calibrate_camera(verbose=False)
    K = calib['camera_matrix']
    dist_coeffs = calib['dist_coeffs']
    print("Camera calibration complete")

# 3. DEFINE 3D CUBE POINTS (World Coordinates)
# We use the target image size as the base. Z is height.
h_ref, w_ref = target_img.shape[:2]
# Cube sitting on the image. Top is at Z = -w_ref/2 (half the width high)
cube_height = w_ref // 2
obj_pts_cube = np.float32([[0,0,0], [w_ref,0,0], [w_ref,h_ref,0], [0,h_ref,0],
                           [0,0,-cube_height], [w_ref,0,-cube_height], 
                           [w_ref,h_ref,-cube_height], [0,h_ref,-cube_height]])

# 3D points of the 4 corners of the flat image for solvePnP
obj_pts_plane = np.float32([[0,0,0], [w_ref,0,0], [w_ref,h_ref,0], [0,h_ref,0]])

# 4. SIFT Setup
sift = cv2.SIFT_create(nfeatures=1000)
kp_target, des_target = sift.detectAndCompute(target_img, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))

# 2. Get video properties for the writer
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 3. Initialize VideoWriter 
# 'mp4v' is a common codec for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('part2_output.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    kp_frame, des_frame = sift.detectAndCompute(frame, None)

    if des_frame is not None and len(des_frame) > 0:
        matches = flann.knnMatch(des_target, des_frame, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > 30:
            # Get Homography just to find the 4 corners of the box in the video
            src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            if H is not None:
                # Find where the 4 corners are in the video frame
                ref_corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
                img_pts = cv2.perspectiveTransform(ref_corners, H)

                # --- PART 2 CORE: SolvePnP ---
                # This finds the Rotation (rvec) and Translation (tvec) of the camera
                success, rvec, tvec = cv2.solvePnP(obj_pts_plane, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    # Project the 8 3D cube points into 2D pixel coordinates
                    imgpts, _ = cv2.projectPoints(obj_pts_cube, rvec, tvec, K, dist_coeffs)
                    imgpts = np.int32(imgpts).reshape(-1, 2)

                    # Draw the Cube
                    # Bottom face (Green)
                    frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
                    # Vertical pillars (Blue)
                    for i, j in zip(range(4), range(4, 8)):
                        frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)
                    # Top face (Red)
                    frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)
                    # --- SAVE THE FRAME ---
    out.write(frame)
    cv2.imshow('Part 2: 3D Cube AR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()