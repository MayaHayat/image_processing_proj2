import cv2
import numpy as np

# 1. Load assets
target_img = cv2.imread('org_image.png')
panorama_img = cv2.imread('3602.jpg')       
cap = cv2.VideoCapture('video.mp4')

# Load camera calibration
calib_data = np.load('camera_calibration.npz')
camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

# 2. Get video properties for the writer
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 3. Initialize VideoWriter 
# 'mp4v' is a common codec for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('AR_output_circle.mp4', fourcc, fps, (frame_width, frame_height))

sift = cv2.SIFT_create()
kp_target, des_target = sift.detectAndCompute(target_img, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

# Calculate circle parameters for the target image (will be warped)
circle_center = (target_img.shape[1] // 2, target_img.shape[0] // 2)
circle_radius = min(target_img.shape[0], target_img.shape[1]) // 3

# Define 3D points of the target plane (assuming it's flat on z=0)
# These correspond to the corners of the target image
target_height, target_width = target_img.shape[:2]
object_points_3d = np.array([
    [0, 0, 0],
    [target_width, 0, 0],
    [target_width, target_height, 0],
    [0, target_height, 0]
], dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    kp_frame, des_frame = sift.detectAndCompute(frame, None)

    if des_frame is not None and len(des_frame) > 0:
        matches = flann.knnMatch(des_target, des_frame, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) > 15:
            src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Get corresponding 3D points for the matched keypoints
                object_points = np.array([[[kp_target[m.queryIdx].pt[0], 
                                           kp_target[m.queryIdx].pt[1], 
                                           0]] for m in good_matches], dtype=np.float32)
                image_points = dst_pts
                
                # Use solvePnP to get accurate camera pose
                success, rvec, tvec = cv2.solvePnP(
                    object_points.reshape(-1, 3),
                    image_points.reshape(-1, 2),
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Convert rotation vector to rotation matrix
                    rmat, _ = cv2.Rodrigues(rvec)
                    
                    # Extract Euler angles from rotation matrix
                    # Using the convention: R = Rz * Ry * Rx
                    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
                    
                    if sy > 1e-6:
                        yaw = np.arctan2(rmat[1, 0], rmat[0, 0]) * 180 / np.pi
                        pitch = np.arctan2(-rmat[2, 0], sy) * 180 / np.pi
                        roll = np.arctan2(rmat[2, 1], rmat[2, 2]) * 180 / np.pi
                    else:
                        yaw = np.arctan2(-rmat[1, 2], rmat[1, 1]) * 180 / np.pi
                        pitch = np.arctan2(-rmat[2, 0], sy) * 180 / np.pi
                        roll = 0
                    
                    # Map the angles to 360° image coordinates
                    # Panorama is typically equirectangular: width = 360°, height = 180°
                    pano_height, pano_width = panorama_img.shape[:2]
                    
                    # Convert angles to panorama coordinates
                    # yaw: -180 to 180 -> 0 to pano_width
                    # pitch: -90 to 90 -> 0 to pano_height
                    center_x_pano = int((yaw % 360) / 360 * pano_width)
                    center_y_pano = int((pitch + 90) / 180 * pano_height)
                    
                    # Get the center position of the target in the frame using homography
                    center_pt = np.array([[circle_center[0]], [circle_center[1]], [1.0]])
                    transformed_center = H @ center_pt
                    transformed_center = transformed_center / transformed_center[2]
                    center_x = int(transformed_center[0][0])
                    center_y = int(transformed_center[1][0])
                    
                    # Estimate scale from homography
                    scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
                    scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
                    avg_scale = (scale_x + scale_y) / 2
                    
                    # Calculate the scaled circle radius
                    scaled_radius = int(circle_radius * avg_scale)
                    circle_diameter = scaled_radius * 2
                    
                    # Extract a portion of the panorama based on camera angle
                    # The FOV (field of view) determines how much of the panorama to show
                    fov_scale = 0.3  # Adjust this to control how much the view changes
                    crop_width = int(pano_width * fov_scale)
                    crop_height = int(pano_height * fov_scale)
                    
                    # Calculate crop boundaries with wrapping for horizontal (yaw)
                    crop_x1 = (center_x_pano - crop_width // 2) % pano_width
                    crop_y1 = max(0, min(pano_height - crop_height, center_y_pano - crop_height // 2))
                    
                    # Handle horizontal wrapping for 360° panorama
                    if crop_x1 + crop_width <= pano_width:
                        # No wrapping needed
                        cropped_pano = panorama_img[crop_y1:crop_y1+crop_height, crop_x1:crop_x1+crop_width]
                    else:
                        # Need to wrap around
                        right_part = panorama_img[crop_y1:crop_y1+crop_height, crop_x1:]
                        left_part = panorama_img[crop_y1:crop_y1+crop_height, :(crop_x1+crop_width)%pano_width]
                        cropped_pano = np.hstack([right_part, left_part])
                    
                    # Resize cropped panorama to fit the circle
                    portal_img = cv2.resize(cropped_pano, (circle_diameter, circle_diameter))
                    
                    # Create a circular mask
                    portal_mask = np.zeros((circle_diameter, circle_diameter), dtype=np.uint8)
                    cv2.circle(portal_mask, (scaled_radius, scaled_radius), scaled_radius, 255, -1)
                    
                    # Apply mask to portal image
                    portal_masked = cv2.bitwise_and(portal_img, portal_img, mask=portal_mask)
                    
                    # Calculate the position to place the circle (top-left corner)
                    top_left_x = center_x - scaled_radius
                    top_left_y = center_y - scaled_radius
                    
                    # Create overlay on the frame
                    final_frame = frame.copy()
                    
                    # Check bounds and overlay the circular portal image
                    if (top_left_x >= 0 and top_left_y >= 0 and 
                        top_left_x + circle_diameter <= frame_width and 
                        top_left_y + circle_diameter <= frame_height):
                        
                        # Extract the region of interest from the frame
                        roi = final_frame[top_left_y:top_left_y+circle_diameter, 
                                         top_left_x:top_left_x+circle_diameter]
                        
                        # Blend using the mask
                        mask_inv = cv2.bitwise_not(portal_mask)
                        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                        portal_fg = cv2.bitwise_and(portal_masked, portal_masked, mask=portal_mask)
                        
                        combined = cv2.add(roi_bg, portal_fg)
                        final_frame[top_left_y:top_left_y+circle_diameter, 
                                  top_left_x:top_left_x+circle_diameter] = combined
                else:
                    final_frame = frame
            else:
                final_frame = frame
        else:
            final_frame = frame
    else:
        final_frame = frame

    # --- SAVE THE FRAME ---
    out.write(final_frame)
    
    cv2.imshow('AR Result with Circle', final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- IMPORTANT: Release everything ---
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved successfully as AR_output_circle.mp4")
