import cv2
import numpy as np

# 1. Load assets
target_img = cv2.imread('org_image.png')
panorama_img = cv2.imread('gemini_360.jpg')       
cap = cv2.VideoCapture('video.mp4')

# Load calibration
calib_data = np.load('camera_calibration.npz')
camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

# Video properties
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('smoothed_circular_portal.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

# SIFT Setup
sift = cv2.SIFT_create()
kp_target, des_target = sift.detectAndCompute(target_img, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

h_t, w_t = target_img.shape[:2]
object_pts = np.float32([[0,0,0], [w_t,0,0], [w_t,h_t,0], [0,h_t,0]])

# --- SMOOTHING VARIABLES ---
prev_center = None
prev_radius = None
prev_shift_x = 0
prev_shift_y = 0
# Alpha controls the smoothness: 0.1 is very smooth/slow, 0.8 is fast/jittery
alpha = 0.15 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    kp_frame, des_frame = sift.detectAndCompute(frame, None)
    if des_frame is not None:
        matches = flann.knnMatch(des_target, des_frame, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) > 25: # Increased threshold for better stability
            src_pts = np.float32([kp_target[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # 2. Get Camera Pose
                ref_corners = np.float32([[0,0], [w_t,0], [w_t,h_t], [0,h_t]]).reshape(-1,1,2)
                img_pts = cv2.perspectiveTransform(ref_corners, H)
                _, rvec, tvec = cv2.solvePnP(object_pts, img_pts, camera_matrix, dist_coeffs)

                # 3. Smoothed Parallax Math
                R_mat, _ = cv2.Rodrigues(rvec)
                cam_pos = -R_mat.T @ tvec 
                
                p_h, p_w = panorama_img.shape[:2]
                sensitivity = 1.8
                fov_scale = 0.25
                crop_w, crop_h = int(p_w * fov_scale), int(p_h * fov_scale)

                # Raw shift values
                curr_shift_x = (cam_pos[0] / cam_pos[2]) * p_w * sensitivity
                curr_shift_y = (cam_pos[1] / cam_pos[2]) * p_h * sensitivity
                
                # Apply Low-Pass Filter to shifts
                prev_shift_x = alpha * curr_shift_x + (1 - alpha) * prev_shift_x
                prev_shift_y = alpha * curr_shift_y + (1 - alpha) * prev_shift_y

                cx = (p_w // 2 + int(prev_shift_x)) % p_w
                cy = np.clip(p_h // 2 + int(prev_shift_y), crop_h // 2, p_h - crop_h // 2)
                x1, y1 = (cx - crop_w // 2) % p_w, cy - crop_h // 2

                # Panorama Crop with Wrapping
                if x1 + crop_w <= p_w:
                    view = panorama_img[y1:y1+crop_h, x1:x1+crop_w]
                else:
                    view = np.hstack([panorama_img[y1:y1+crop_h, x1:], panorama_img[y1:y1+crop_h, : (x1+crop_w) % p_w]])
                
                view = cv2.resize(view, (600, 600)) 

                # 4. SMOOTHED CIRCULAR GEOMETRY
                marker_center = np.array([[w_t//2], [h_t//2], [1.0]])
                center_f = H @ marker_center
                center_f = (center_f / center_f[2]).flatten()
                curr_center = np.array([center_f[0], center_f[1]])

                scale = np.sqrt(H[0,0]**2 + H[0,1]**2)
                curr_radius = (min(w_t, h_t) // 3) * scale

                # Initialize or smooth the geometry
                if prev_center is None:
                    prev_center = curr_center
                    prev_radius = curr_radius
                else:
                    prev_center = alpha * curr_center + (1 - alpha) * prev_center
                    prev_radius = alpha * curr_radius + (1 - alpha) * prev_radius

                center_px = (int(prev_center[0]), int(prev_center[1]))
                smooth_radius = int(prev_radius)

                # 5. Perspective Warp for Portal View
                side = (min(w_t, h_t) // 3) * 2
                p_pts_target = np.float32([
                    [w_t//2 - side//2, h_t//2 - side//2],
                    [w_t//2 + side//2, h_t//2 - side//2],
                    [w_t//2 + side//2, h_t//2 + side//2],
                    [w_t//2 - side//2, h_t//2 + side//2]
                ])
                p_pts_frame = cv2.perspectiveTransform(p_pts_target.reshape(-1,1,2), H)
                
                H_v, _ = cv2.findHomography(np.float32([[0,0], [600,0], [600,600], [0,600]]), p_pts_frame)
                warped_view = cv2.warpPerspective(view, H_v, (fw, fh))

                # 6. Masking and Blending
                mask = np.zeros((fh, fw), dtype=np.uint8)
                cv2.circle(mask, center_px, smooth_radius, 255, -1)
                
                frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
                portal_part = cv2.bitwise_and(warped_view, warped_view, mask=mask)
                frame = cv2.add(frame, portal_part)

                # 7. Visible Border
                cv2.circle(frame, center_px, smooth_radius, (255, 255, 255), 5)

    out.write(frame)
    cv2.imshow('Smoothed Circular Portal', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()