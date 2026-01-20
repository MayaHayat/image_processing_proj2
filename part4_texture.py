import cv2
import numpy as np
import trimesh
from camera_calibration import calibrate_camera

# ========================================
# 1. 3D MESH RENDERING UTILITIES (TEXTURED)
# ========================================

def load_mesh_trimesh(filepath):
    try:
        mesh = trimesh.load(filepath, process=False)
        print(f"Loaded mesh: {filepath}")
        return mesh
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None

def project_3d_points(points_3d, rvec, tvec, K, dist_coeffs=None):
    # Convert rotation vector to rotation matrix
    R_mat, _ = cv2.Rodrigues(rvec)
    # Transform points to camera coordinates
    points_cam = (R_mat @ points_3d.T).T + tvec.reshape(1, 3)
    depths = points_cam[:, 2]
    
    # Project to normalized image coordinates
    points_normalized = points_cam[:, :2] / points_cam[:, 2:3]
    
    # Apply intrinsic matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    points_2d = np.zeros((len(points_normalized), 2))
    points_2d[:, 0] = points_normalized[:, 0] * fx + cx
    points_2d[:, 1] = points_normalized[:, 1] * fy + cy
    
    return points_2d, depths

def render_mesh_on_frame(frame, mesh, rvec, tvec, K, dist_coeffs=None, scale=1.0, offset=(0, 0, 0)):
    # 1. Prepare geometry
    vertices = (mesh.vertices * scale) + np.array(offset)
    points_2d, depths = project_3d_points(vertices, rvec, tvec, K, dist_coeffs)
    faces = mesh.faces
    
    # 2. Get Normals and Colors
    normals = mesh.face_normals 
    vertex_colors = mesh.visual.vertex_colors[:, :3] if hasattr(mesh.visual, 'vertex_colors') else np.full((len(vertices), 3), 150)

    # 3. Painter's Algorithm Sorting
    triangles_with_depth = []
    for i, face in enumerate(faces):
        if any(depths[face] <= 0): continue
        triangles_with_depth.append((np.mean(depths[face]), i, face))
    triangles_with_depth.sort(key=lambda x: x[0], reverse=True)

    # 4. LIGHTING CONFIGURATION
    light_dir = np.array([0.0, 0.0, -1.0]) # Light comes from camera
    light_dir /= np.linalg.norm(light_dir)
    ambient_intensity = 0.7 

    output = frame.copy()

    for _, face_idx, face in triangles_with_depth:
        pts_2d = points_2d[face].astype(np.int32)
        normal = normals[face_idx]
        
        # Diffuse
        diffuse_intensity = abs(np.dot(normal, -light_dir)) 
        
        # Specular
        view_dir = np.array([0, 0, -1]) 
        reflect_dir = light_dir - 2 * np.dot(light_dir, normal) * normal
        spec_dot = np.dot(reflect_dir, view_dir)
        specular_intensity = pow(max(0, spec_dot), 32) 

        # Base color
        base_color = np.mean(vertex_colors[face], axis=0)
        
        # Mixing
        lighting_factor = ambient_intensity + (0.3 * diffuse_intensity)
        final_color = (base_color * lighting_factor) + (np.array([255, 255, 255]) * specular_intensity * 0.8)
        final_color = np.clip(final_color, 0, 255).astype(np.uint8)
        
        # RGB to BGR
        bgr_color = (int(final_color[2]), int(final_color[1]), int(final_color[0]))

        cv2.fillConvexPoly(output, pts_2d, bgr_color)
        
        # Seams
        edge_color = (max(0, bgr_color[0]-25), max(0, bgr_color[1]-25), max(0, bgr_color[2]-25))
        cv2.polylines(output, [pts_2d], True, edge_color, 1)

    return output

# ========================================
# 2. MAIN PIPELINE
# ========================================

def main():
    # Assets
    target_img = cv2.imread('org_image.png')
    cap = cv2.VideoCapture('video_hand.mp4')
    mesh_trimesh = load_mesh_trimesh('lego_bouquet.ply')
    
    if target_img is None or not cap.isOpened() or mesh_trimesh is None:
        print("Error: Could not load assets.")
        return

    # Mesh Pre-processing (Match your previous script)
    mesh_center = mesh_trimesh.bounds.mean(axis=0)
    mesh_trimesh.vertices -= mesh_center
    mesh_trimesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    
    h_ref, w_ref = target_img.shape[:2]
    mesh_size = np.max(mesh_trimesh.bounds[1] - mesh_trimesh.bounds[0])
    mesh_scale = (w_ref / mesh_size) * 0.5
    mesh_offset = ((w_ref / 2) - 200, (h_ref / 2) - 500, -w_ref / 3)

    # Calibration
    try:
        calib_data = np.load('camera_calibration.npz')
        K = calib_data['camera_matrix']
        dist_coeffs = calib_data['dist_coeffs']
    except:
        print("Running calibration...")
        calib = calibrate_camera(verbose=False)
        K, dist_coeffs = calib['camera_matrix'], calib['dist_coeffs']

    # SIFT Setup
    sift = cv2.SIFT_create(nfeatures=1000)
    kp_target, des_target = sift.detectAndCompute(target_img, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))
    obj_pts_plane = np.float32([[0, 0, 0], [w_ref, 0, 0], [w_ref, h_ref, 0], [0, h_ref, 0]])

    # Video Writer
    out = cv2.VideoWriter('part4_textured_occlusion_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                         int(cap.get(cv2.CAP_PROP_FPS)), 
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    print("Rendering with Texture and Occlusion...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        kp_frame, des_frame = sift.detectAndCompute(frame, None)
        
        if des_frame is not None and len(des_frame) > 0:
            matches = flann.knnMatch(des_target, des_frame, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            if len(good_matches) > 30:
                src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                
                if H is not None:
                    ref_corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
                    img_pts = cv2.perspectiveTransform(ref_corners, H)
                    success, rvec, tvec = cv2.solvePnP(obj_pts_plane, img_pts, K, dist_coeffs)
                    
                    if success:
                        # --- OCCLUSION MASKING (SKIN DETECTION) ---
                        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
                        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
                        hand_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
                        
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        hand_mask = cv2.dilate(cv2.erode(hand_mask, kernel), kernel, iterations=2)
                        visible_mask = cv2.bitwise_not(hand_mask)

                        # --- RENDER TEXTURED MESH TO LAYER ---
                        mesh_layer = np.zeros_like(frame)
                        mesh_layer = render_mesh_on_frame(
                            mesh_layer, mesh_trimesh, rvec, tvec, K, dist_coeffs,
                            scale=mesh_scale, offset=mesh_offset
                        )
                        
                        # --- APPLY OCCLUSION ---
                        occluded_mesh = cv2.bitwise_and(mesh_layer, mesh_layer, mask=visible_mask)

                        # --- COMBINE WITH BACKGROUND ---
                        mesh_gray = cv2.cvtColor(occluded_mesh, cv2.COLOR_BGR2GRAY)
                        _, mesh_area = cv2.threshold(mesh_gray, 1, 255, cv2.THRESH_BINARY)
                        bg_hole = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mesh_area))
                        frame = cv2.add(bg_hole, occluded_mesh)

        out.write(frame)
        cv2.imshow('AR: Texture + Occlusion', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()