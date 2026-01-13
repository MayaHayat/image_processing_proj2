import cv2
import numpy as np
import trimesh

# 1. Load Assets
target_img = cv2.imread('org_image.png') 
cap = cv2.VideoCapture('video.mp4')

# Load the 3D Model
mesh = trimesh.load('corgi.obj')

# Scale the mesh
scale_factor = (target_img.shape[1] * 0.5) / mesh.extents[0]
mesh.apply_scale(scale_factor)

vertices = np.array(mesh.vertices, dtype=np.float32)
faces = mesh.faces

# 2. Camera & Video Properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# --- NEW: VideoWriter Setup ---
# 'mp4v' is a standard codec for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output_corgi_ar.mp4', fourcc, fps, (width, height))
# ------------------------------

focal_length = width 
K = np.array([[focal_length, 0, width/2],
              [0, focal_length, height/2],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))

# 3. Reference Points for solvePnP
h_ref, w_ref = target_img.shape[:2]
obj_pts_plane = np.float32([[0,0,0], [w_ref,0,0], [w_ref,h_ref,0], [0,h_ref,0]])

# 4. SIFT Setup
sift = cv2.SIFT_create(nfeatures=1000)
kp_target, des_target = sift.detectAndCompute(target_img, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    kp_frame, des_frame = sift.detectAndCompute(frame, None)

    if des_frame is not None and len(des_frame) > 0:
        matches = flann.knnMatch(des_target, des_frame, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > 15:
            src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                ref_corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
                img_pts = cv2.perspectiveTransform(ref_corners, H)

                success, rvec, tvec = cv2.solvePnP(obj_pts_plane, img_pts, K, dist_coeffs)

                if success:
                    offset = np.array([w_ref/2, h_ref/2, 0], dtype=np.float32)
                    projected_pts, _ = cv2.projectPoints(vertices + offset, rvec, tvec, K, dist_coeffs)
                    projected_pts = np.int32(projected_pts).reshape(-1, 2)

                    for face in faces:
                        pts = projected_pts[face]
                        cv2.polylines(frame, [pts], True, (255, 100, 0), 1)
                        
    # --- NEW: Write the frame to the file ---
    out.write(frame)
    # ----------------------------------------

    cv2.imshow('Part 3: Corgi AR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Cleanup
cap.release()
out.release() # CRITICAL: If you don't release, the file may be corrupted
cv2.destroyAllWindows()
print("Video saved as output_corgi_ar.mp4")