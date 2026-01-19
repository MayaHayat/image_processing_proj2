import cv2
import numpy as np

# --- CONFIGURATION ---
# List your 3 target images (markers)
target_files = ['radio.png', 'mdco_gemini.png', 'org_image.png']
# List 3 different panorama/360 images to show inside the portals
portal_views = ['forest.jpg', 'space.jpg', 'beach.jpg']

def get_parallax_view(panorama, rvec, tvec, K, portal_w, portal_h):
    """
    Calculates the parallax crop from a 360 image based on camera pose.
    """
    # 1. Calculate the camera position in the plane's coordinate system
    R_mat, _ = cv2.Rodrigues(rvec)
    # Camera position in world coords: C = -R^T * t
    cam_pos_in_plane = -R_mat.T @ tvec
    
    # 2. Use the camera's X and Y position to 'shift' the view
    # As the camera moves left, we see the right side of the 360 world
    shift_x = int(cam_pos_in_plane[0] * 0.5) 
    shift_y = int(cam_pos_in_plane[1] * 0.5)
    
    # 3. Crop a section of the panorama based on this shift
    h, w = panorama.shape[:2]
    center_x, center_y = w // 2 + shift_x, h // 2 + shift_y
    
    # Ensure crop stays within bounds
    x1 = np.clip(center_x - portal_w//2, 0, w - portal_w)
    y1 = np.clip(center_y - portal_h//2, 0, h - portal_h)
    
    view_crop = panorama[y1:y1+portal_h, x1:x1+portal_w]
    return cv2.resize(view_crop, (portal_w, portal_h))

def main():
    # 1. Load Assets
    targets = [cv2.imread(f) for f in target_files]
    views = [cv2.imread(v) for v in portal_views]
    cap = cv2.VideoCapture('part5_og_vid.mp4')
    
    # Load Calibration from Part 2
    calib = np.load('camera_calibration.npz')
    K, dist = calib['camera_matrix'], calib['dist_coeffs']
    
    # 2. Setup SIFT for 3 targets
    sift = cv2.SIFT_create()
    target_data = []
    for t in targets:
        kp, des = sift.detectAndCompute(t, None)
        h, w = t.shape[:2]
        # 3D points for solvePnP
        obj_pts = np.float32([[0,0,0], [w,0,0], [w,h,0], [0,h,0]])
        target_data.append({'kp': kp, 'des': des, 'obj_pts': obj_pts, 'size': (w, h)})

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        kp_frame, des_frame = sift.detectAndCompute(frame, None)
        if des_frame is None: continue

        # 3. Process each target marker
        for i, data in enumerate(target_data):
            matches = flann.knnMatch(data['des'], des_frame, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            
            if len(good) > 20:
                src_pts = np.float32([data['kp'][m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Solve PnP for camera pose relative to this marker
                    w, h = data['size']
                    ref_corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
                    img_pts = cv2.perspectiveTransform(ref_corners, H)
                    _, rvec, tvec = cv2.solvePnP(data['obj_pts'], img_pts, K, dist)

                    # --- RENDER PORTAL ---
                    # 1. Get the parallax view from the 360 image
                    portal_w, portal_h = w // 2, h // 2
                    view = get_parallax_view(views[i], rvec, tvec, K, portal_w, portal_h)
                    
                    # 2. Warp the portal view onto the frame
                    # Define portal corners in the center of the marker
                    p_x, p_y = w // 4, h // 4
                    portal_pts_src = np.float32([[0,0], [portal_w,0], [portal_w,portal_h], [0,portal_h]])
                    portal_pts_dst_on_marker = np.float32([[p_x, p_y], [p_x+portal_w, p_y], 
                                                         [p_x+portal_w, p_y+portal_h], [p_x, p_y+portal_h]])
                    
                    # Use Homography to map portal view to video frame
                    # We modify the portal coordinates by the marker's homography
                    warped_portal_pts = cv2.perspectiveTransform(portal_pts_dst_on_marker.reshape(-1,1,2), H)
                    
                    H_portal, _ = cv2.findHomography(portal_pts_src, warped_portal_pts)
                    warped_view = cv2.warpPerspective(view, H_portal, (frame.shape[1], frame.shape[0]))
                    
                    # 3. Blend and add a Border
                    mask_portal = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillConvexPoly(mask_portal, np.int32(warped_portal_pts), 255)
                    
                    frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_portal))
                    frame = cv2.add(frame, warped_view)
                    
                    # Draw Border/Frame
                    cv2.polylines(frame, [np.int32(warped_portal_pts)], True, (255, 255, 255), 4)

        cv2.imshow('Part 5: Multi-Portal AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()