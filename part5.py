import cv2
import numpy as np

# --- 1. CONFIGURATION & ASSETS ---
target_files = ['org_image.png', 'mdco_gemini.png', 'radio.png']
pano_files = ['forest_360.jpg', 'space_360.jpg', 'ocean_360.jpg']

# Load Calibration
try:
    calib_data = np.load('camera_calibration.npz')
    camera_matrix = calib_data['camera_matrix']
    dist_coeffs = calib_data['dist_coeffs']
except Exception as e:
    print(f"Calibration error: {e}")
    exit()

cap = cv2.VideoCapture('part5#5.mp4')
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('part5_output_#5_sift.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

sift = cv2.SIFT_create(nfeatures=2000) # Increased features for better stability
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=70))

# --- 2. PRE-PROCESS TARGETS ---
markers = []
for i in range(len(target_files)):
    t_img = cv2.imread(target_files[i])
    p_img = cv2.imread(pano_files[i])
    
    if t_img is None or p_img is None:
        print(f"Error loading assets for marker {i}.")
        continue

    kp, des = sift.detectAndCompute(t_img, None)
    h, w = t_img.shape[:2]
    markers.append({
        'kp': kp, 'des': des, 'pano': p_img, 'w': w, 'h': h,
        'obj_pts': np.float32([[0,0,0], [w,0,0], [w,h,0], [0,h,0]]),
        'prev_center': None, 'prev_radius': None, 
        'prev_shift_x': 0.0, 'prev_shift_y': 0.0
    })

# --- 3. TUNING PARAMETERS ---
alpha_geo = 0.4      # Smoothing for the portal position (0.1 - 0.3)
alpha_shift = 0.05    # Smoothing for internal movement (keep low for "less movement")
parallax_factor = 0.5 # Multiplier for camera motion (0.5 - 1.0 is stable)
match_ratio = 0.8    # Ratio test (Lower = stricter, prevents jumping)
min_good_matches = 70 # Minimum matches to trigger portal

# --- 4. MAIN PROCESSING LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    output_frame = frame.copy()
    kp_frame, des_frame = sift.detectAndCompute(frame, None)

    # # We use DRAW_RICH_KEYPOINTS to see the size and orientation of each point
    # output_frame = cv2.drawKeypoints(output_frame, kp_frame, None, 
    #                                  color=(0, 255, 255), 
    #                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    if des_frame is not None and len(des_frame) > 0:
        for m in markers:
            # IDENTITY LOCKING: Stricter matching
            matches = flann.knnMatch(m['des'], des_frame, k=2)
            good = [mat for mat, n in matches if mat.distance < match_ratio * n.distance]

            if len(good) > min_good_matches:
                src_pts = np.float32([m['kp'][mat.queryIdx].pt for mat in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[mat.trainIdx].pt for mat in good]).reshape(-1, 1, 2)
                
                # RANSAC check to filter out outliers from other markers
                H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

                if H is not None and np.sum(inlier_mask) > 45:
                    # A. Pose Estimation for Parallax
                    ref_corners = np.float32([[0,0], [m['w'],0], [m['w'],m['h']], [0,m['h']]]).reshape(-1,1,2)
                    img_pts = cv2.perspectiveTransform(ref_corners, H)
                    cv2.polylines(output_frame, [np.int32(img_pts)], True, (0, 255, 0), 2)

                    # 3. Calculate text position (using the smoothed center)
                    m_center = H @ np.array([[m['w']//2], [m['h']//2], [1.0]])
                    center_f = (m_center / m_center[2]).flatten()
                    text_pos = (int(center_f[0] + 100), int(center_f[1])) # Offset slightly to center text

                    # 4. PRINT len(good) matches
                    # We use a black background for the text to make it readable over the portal
                    label = f"Matches: {len(good)}"
                    cv2.putText(output_frame, label, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA) # Black outline
                    cv2.putText(output_frame, label, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA) # Green text
                    
                    # solvePnP is used to find camera position relative to target
                    _, rvec, tvec = cv2.solvePnP(m['obj_pts'], img_pts, camera_matrix, dist_coeffs)

                    # B. Dampened Parallax Calculation
                    R_mat, _ = cv2.Rodrigues(rvec)
                    cam_pos = -R_mat.T @ tvec
                    ph, pw = m['pano'].shape[:2]
                    
                    # Scale down camera position influence
                    curr_sx = (cam_pos[0] / (cam_pos[2] + 1e-6)) * pw * parallax_factor
                    curr_sy = (cam_pos[1] / (cam_pos[2] + 1e-6)) * ph * parallax_factor
                    
                    # Heavy Temporal Smoothing for interior movement
                    m['prev_shift_x'] = alpha_shift * curr_sx + (1 - alpha_shift) * m['prev_shift_x']
                    m['prev_shift_y'] = alpha_shift * curr_sy + (1 - alpha_shift) * m['prev_shift_y']

                    # C. Smooth View Extraction
                    cw, ch = int(pw * 0.5), int(ph * 0.5)
                    cx = (pw // 2 + int(m['prev_shift_x'])) % pw
                    cy = np.clip(ph // 2 + int(m['prev_shift_y']), ch // 2, ph - ch // 2)
                    
                    x1, y1 = (cx - cw // 2) % pw, cy - ch // 2
                    if x1 + cw <= pw: 
                        view = m['pano'][y1:y1+ch, x1:x1+cw]
                    else: 
                        view = np.hstack([m['pano'][y1:y1+ch, x1:], m['pano'][y1:y1+ch, : (x1+cw)%pw]])
                    view = cv2.resize(view, (800, 800))

                    # D. Portal Geometry Smoothing
                    m_center = H @ np.array([[m['w']//2], [m['h']//2], [1.0]])
                    center_f = (m_center / m_center[2]).flatten()
                    curr_r = (min(m['w'], m['h']) // 3) * (np.sqrt(H[0,0]**2 + H[0,1]**2))

                    if m['prev_center'] is None:
                        m['prev_center'], m['prev_radius'] = center_f[:2], curr_r
                    else:
                        m['prev_center'] = alpha_geo * center_f[:2] + (1 - alpha_geo) * m['prev_center']
                        m['prev_radius'] = alpha_geo * curr_r + (1 - alpha_geo) * m['prev_radius']

                    # E. Final Rendering
                    side = (min(m['w'], m['h']) // 3) * 2
                    p_pts_target = np.float32([[m['w']//2-side, m['h']//2-side], 
                                               [m['w']//2+side, m['h']//2-side], 
                                               [m['w']//2+side, m['h']//2+side], 
                                               [m['w']//2-side, m['h']//2+side]])
                    p_pts_frame = cv2.perspectiveTransform(p_pts_target.reshape(-1,1,2), H)
                    Hv, _ = cv2.findHomography(np.float32([[0,0], [800,0], [800,800], [0,800]]), p_pts_frame)

                    if Hv is not None:
                        warped = cv2.warpPerspective(view, Hv, (fw, fh))
                        mask = np.zeros((fh, fw), dtype=np.uint8)
                        center_px = (int(m['prev_center'][0]), int(m['prev_center'][1]))
                        cv2.circle(mask, center_px, int(m['prev_radius']), 255, -1)
                        
                        output_frame = cv2.bitwise_and(output_frame, output_frame, mask=cv2.bitwise_not(mask))
                        output_frame = cv2.add(output_frame, cv2.bitwise_and(warped, warped, mask=mask))
                        cv2.circle(output_frame, center_px, int(m['prev_radius']), (255, 255, 255), 4)

    out.write(output_frame)
    cv2.imshow('Stable Multi-Portal AR', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()