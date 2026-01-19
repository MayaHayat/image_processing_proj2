import cv2
import numpy as np

# --- 1. CONFIGURATION & ASSETS ---
target_files = ['org_image.png', 'mdco_gemini.png', 'flower_vib.png']
pano_files = ['forest_360.jpg', 'space_360.jpg', 'ocean_360.jpg']

calib_data = np.load('camera_calibration.npz')
camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

cap = cv2.VideoCapture('part5#3.mp4')
fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('final_multi_portal.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

sift = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

# --- 2. PRE-PROCESS TARGETS ---
markers = []
for i in range(3):
    t_img, p_img = cv2.imread(target_files[i]), cv2.imread(pano_files[i])
    
    if t_img is None or p_img is None:
        print(f"Error loading assets for marker {i}.")
        continue

    kp, des = sift.detectAndCompute(t_img, None)
    h, w = t_img.shape[:2]
    markers.append({
        'kp': kp, 'des': des, 'pano': p_img, 'w': w, 'h': h,
        'obj_pts': np.float32([[0,0,0], [w,0,0], [w,h,0], [0,h,0]]),
        'prev_center': None, 'prev_radius': None, 'prev_shift_x': 0.0, 'prev_shift_y': 0.0
    })

alpha = 0.15 

# --- 3. MAIN PROCESSING LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    output_frame = frame.copy()
    kp_frame, des_frame = sift.detectAndCompute(frame, None)
    
    if des_frame is not None:
        for m in markers:
            matches = flann.knnMatch(m['des'], des_frame, k=2)
            # Use a slightly stricter ratio (0.7) to prevent cross-marker jumping
            good = [mat for mat, n in matches if mat.distance < 0.7 * n.distance]

            if len(good) > 35:
                src_pts = np.float32([m['kp'][mat.queryIdx].pt for mat in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[mat.trainIdx].pt for mat in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

                # Only proceed if Homography is robust and enough points survived RANSAC
                if H is not None and np.sum(mask) > 25:
                    # A. Bounding Box & Pose
                    ref_corners = np.float32([[0,0], [m['w'],0], [m['w'],m['h']], [0,m['h']]]).reshape(-1,1,2)
                    img_pts = cv2.perspectiveTransform(ref_corners, H)
                    cv2.polylines(output_frame, [np.int32(img_pts)], True, (0, 255, 0), 2)
                    _, rvec, tvec = cv2.solvePnP(m['obj_pts'], img_pts, camera_matrix, dist_coeffs)

                    # B. Parallax Math
                    R_mat, _ = cv2.Rodrigues(rvec)
                    cam_pos = -R_mat.T @ tvec
                    ph, pw = m['pano'].shape[:2]
                    curr_sx = (cam_pos[0] / cam_pos[2]) * pw * 2.0
                    curr_sy = (cam_pos[1] / cam_pos[2]) * ph * 2.0
                    m['prev_shift_x'] = alpha * curr_sx + (1 - alpha) * m['prev_shift_x']
                    m['prev_shift_y'] = alpha * curr_sy + (1 - alpha) * m['prev_shift_y']

                    cw, ch = int(pw * 0.25), int(ph * 0.25)
                    cx = (pw // 2 + int(m['prev_shift_x'])) % pw
                    cy = np.clip(ph // 2 + int(m['prev_shift_y']), ch // 2, ph - ch // 2)
                    x1, y1 = (cx - cw // 2) % pw, cy - ch // 2
                    
                    # C. View Extraction with Wrap-around
                    if x1 + cw <= pw: view = m['pano'][y1:y1+ch, x1:x1+cw]
                    else: view = np.hstack([m['pano'][y1:y1+ch, x1:], m['pano'][y1:y1+ch, : (x1+cw)%pw]])
                    view = cv2.resize(view, (800, 800))

                    # D. Smoothed Geometry
                    m_center = H @ np.array([[m['w']//2], [m['h']//2], [1.0]])
                    center_f = (m_center / m_center[2]).flatten()
                    curr_r = (min(m['w'], m['h']) // 3) * (np.sqrt(H[0,0]**2 + H[0,1]**2))

                    if m['prev_center'] is None: m['prev_center'], m['prev_radius'] = center_f[:2], curr_r
                    else:
                        m['prev_center'] = alpha * center_f[:2] + (1 - alpha) * m['prev_center']
                        m['prev_radius'] = alpha * curr_r + (1 - alpha) * m['prev_radius']

                    # E. Rendering (With Exit Protection)
                    side = (min(m['w'], m['h']) // 3) * 2
                    p_pts_target = np.float32([[m['w']//2-side, m['h']//2-side], [m['w']//2+side, m['h']//2-side], [m['w']//2+side, m['h']//2+side], [m['w']//2-side, m['h']//2+side]])
                    p_pts_frame = cv2.perspectiveTransform(p_pts_target.reshape(-1,1,2), H)
                    Hv, _ = cv2.findHomography(np.float32([[0,0], [800,0], [800,800], [0,800]]), p_pts_frame)

                    # Only warp and blend if Hv is valid (Prevents crashes when marker exits)
                    if Hv is not None and Hv.shape == (3, 3):
                        warped = cv2.warpPerspective(view, Hv, (fw, fh))
                        mask = np.zeros((fh, fw), dtype=np.uint8)
                        center_px = (int(m['prev_center'][0]), int(m['prev_center'][1]))
                        cv2.circle(mask, center_px, int(m['prev_radius']), 255, -1)
                        
                        output_frame = cv2.bitwise_and(output_frame, output_frame, mask=cv2.bitwise_not(mask))
                        output_frame = cv2.add(output_frame, cv2.bitwise_and(warped, warped, mask=mask))
                        cv2.circle(output_frame, center_px, int(m['prev_radius']), (255, 255, 255), 5)

    out.write(output_frame)
    cv2.imshow('Final Multi-Portal', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()