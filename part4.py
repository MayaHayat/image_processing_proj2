import cv2
import numpy as np
from camera_calibration import calibrate_camera
import trimesh
from scipy.spatial.transform import Rotation as R

# ========================================
# 3D MESH RENDERING UTILITIES
# ========================================

def load_mesh_trimesh(filepath):
    """Load a 3D mesh using trimesh"""
    try:
        mesh = trimesh.load(filepath, process=False)
        print(f"Loaded mesh: {filepath}")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        if hasattr(mesh.visual, 'vertex_colors'):
            print(f"  Has vertex colors: Yes")
        return mesh
    except Exception as e:
        print(f"Error loading mesh with trimesh: {e}")
        return None


def project_3d_points(points_3d, rvec, tvec, K, dist_coeffs=None):
    """
    Project 3D points to 2D image coordinates using camera parameters
    
    Args:
        points_3d: Nx3 array of 3D points in world coordinates
        rvec: Rotation vector (3x1)
        tvec: Translation vector (3x1)
        K: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients (optional)
    
    Returns:
        points_2d: Nx2 array of 2D image coordinates
        depths: N array of depth values (z-coordinates in camera frame)
    """
    # Convert rotation vector to rotation matrix
    R_mat, _ = cv2.Rodrigues(rvec)
    
    # Transform points to camera coordinates
    points_cam = (R_mat @ points_3d.T).T + tvec.reshape(1, 3)
    
    # Get depths (z-coordinates in camera frame)
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
    """
    Render a 3D mesh onto a 2D frame using painter's algorithm
    
    Args:
        frame: Input image frame
        mesh: trimesh.Trimesh object
        rvec: Rotation vector
        tvec: Translation vector
        K: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        scale: Scale factor for the mesh
        offset: (x, y, z) offset for mesh positioning
    
    Returns:
        frame: Frame with rendered mesh
    """
    # Get mesh vertices and faces
    vertices = mesh.vertices * scale
    vertices = vertices + np.array(offset)
    faces = mesh.faces
    
    # Project vertices to 2D
    points_2d, depths = project_3d_points(vertices, rvec, tvec, K, dist_coeffs)
    
    # Get vertex colors if available
    if hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = mesh.visual.vertex_colors[:, :3]  # RGB only
    else:
        # Default color if no vertex colors
        vertex_colors = np.ones((len(vertices), 3), dtype=np.uint8) * 128
    
    # Create list of triangles with their average depth
    triangles_with_depth = []
    for face_idx, face in enumerate(faces):
        # Get the three vertices of the triangle
        v0, v1, v2 = face
        
        # Skip if any vertex is behind the camera
        if depths[v0] <= 0 or depths[v1] <= 0 or depths[v2] <= 0:
            continue
        
        # Calculate average depth for sorting (painter's algorithm)
        avg_depth = (depths[v0] + depths[v1] + depths[v2]) / 3.0
        
        # Get 2D points
        pts_2d = np.array([points_2d[v0], points_2d[v1], points_2d[v2]], dtype=np.int32)
        
        # Get colors for the vertices
        colors = [vertex_colors[v0], vertex_colors[v1], vertex_colors[v2]]
        avg_color = np.mean(colors, axis=0).astype(np.uint8)
        
        triangles_with_depth.append((avg_depth, pts_2d, avg_color))
    
    # Sort triangles by depth (back to front for painter's algorithm)
    triangles_with_depth.sort(key=lambda x: x[0], reverse=True)
    
    # Create a copy of the frame to render on
    output = frame.copy()
    
    # Draw triangles
    for depth, pts_2d, color in triangles_with_depth:
        # Simple lighting based on depth (optional)
        light_factor = 0.7 + 0.3 * (1.0 / (1.0 + depth / 1000.0))
        lit_color = (color * light_factor).astype(np.uint8)
        
        # Convert color from RGB to BGR for OpenCV
        bgr_color = (int(lit_color[2]), int(lit_color[1]), int(lit_color[0]))
        
        # Draw filled triangle
        cv2.fillConvexPoly(output, pts_2d, bgr_color)
        
        # Optional: Draw wireframe edges
        # cv2.polylines(output, [pts_2d], True, (50, 50, 50), 1)
    
    return output

def render_mesh_wireframe(frame, mesh, rvec, tvec, K, dist_coeffs=None, scale=1.0, offset=(0, 0, 0), color=(0, 255, 0)):
    """
    Render mesh as wireframe (faster, simpler visualization)
    """
    vertices = mesh.vertices * scale
    vertices = vertices + np.array(offset)
    faces = mesh.faces
    
    # Project vertices to 2D
    points_2d, depths = project_3d_points(vertices, rvec, tvec, K, dist_coeffs)
    
    output = frame.copy()
    
    # Draw edges of each face
    for face in faces:
        # Skip if any vertex is behind camera
        if depths[face[0]] <= 0 or depths[face[1]] <= 0 or depths[face[2]] <= 0:
            continue
        
        pts = points_2d[face].astype(np.int32)
        cv2.polylines(output, [pts], True, color, 1)
    
    return output

# ========================================
# MAIN AR RENDERING PIPELINE
# ========================================

def main():
    # 1. Load Assets
    # Updated to use org_image.png as per your previous setup
    target_img = cv2.imread('org_image.png')
    if target_img is None:
        print("Error: Could not load org_image.png")
        return
    
    cap = cv2.VideoCapture('video_hand.mp4')
    if not cap.isOpened():
        print("Error: Could not open hand_vid.mp4")
        return
    
    # 2. Load and Prep Mesh
    print("\n=== Loading 3D Mesh ===")
    mesh_trimesh = load_mesh_trimesh('lego_bouquet.ply')
    if mesh_trimesh is None:
        return
    
    # Center and rotate mesh to match OpenCV's coordinate system
    mesh_center = mesh_trimesh.bounds.mean(axis=0)
    mesh_trimesh.vertices -= mesh_center
    mesh_trimesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    
    h_ref, w_ref = target_img.shape[:2]
    mesh_size = np.max(mesh_trimesh.bounds[1] - mesh_trimesh.bounds[0])
    
    # Matching your previous smaller scale and offset
    mesh_scale = (w_ref / mesh_size) * 0.5
    mesh_offset = ((w_ref / 2) - 200, (h_ref / 2) - 500, -w_ref / 3)

    # 3. GET CALIBRATED CAMERA PARAMETERS
    try:
        calib_data = np.load('camera_calibration.npz')
        K = calib_data['camera_matrix']
        dist_coeffs = calib_data['dist_coeffs']
    except FileNotFoundError:
        print("Calibration file not found. Please run calibration first.")
        return
    
    # 4. SIFT and Video Setup
    sift = cv2.SIFT_create(nfeatures=1000)
    kp_target, des_target = sift.detectAndCompute(target_img, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))
    obj_pts_plane = np.float32([[0, 0, 0], [w_ref, 0, 0], [w_ref, h_ref, 0], [0, h_ref, 0]])

    # Video Properties for Saving
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('part4_occlusion_output.mp4', fourcc, fps, (frame_width, frame_height))
    
    print(f"Processing and saving to part4_occlusion_output.mp4...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
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
                        # --- PART 4: FOREGROUND DETECTION (OCCLUSION) ---
                        # Convert to YCrCb for stable skin segmentation
                        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                        lower_skin = np.array([0, 139, 100], dtype=np.uint8)
                        upper_skin = np.array([255, 160, 127], dtype=np.uint8)
                        hand_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

                        # Morphological Operations to clean noise and fill holes
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        hand_mask = cv2.erode(hand_mask, kernel, iterations=1)
                        hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
                        
                        # Visible mask is where the hand is NOT
                        visible_mask = cv2.bitwise_not(hand_mask)

                        # Render Mesh to a separate layer
                        mesh_layer = np.zeros_like(frame)
                        mesh_layer = render_mesh_on_frame(
                            mesh_layer, mesh_trimesh, rvec, tvec, K, dist_coeffs,
                            scale=mesh_scale, offset=mesh_offset
                        )
                        
                        # Mask the mesh using the visible area (Occlusion)
                        occluded_mesh = cv2.bitwise_and(mesh_layer, mesh_layer, mask=visible_mask)

                        # Merge with original frame
                        mesh_gray = cv2.cvtColor(occluded_mesh, cv2.COLOR_BGR2GRAY)
                        _, mesh_area = cv2.threshold(mesh_gray, 1, 255, cv2.THRESH_BINARY)
                        
                        # Create a "hole" in the background for the mesh
                        bg_hole = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mesh_area))
                        frame = cv2.add(bg_hole, occluded_mesh)

                        # # Optional: Draw tracking box
                        # cv2.polylines(frame, [np.int32(img_pts)], True, (255, 0, 0), 2)

        # Visual indicator that the video is being saved
        cv2.putText(frame, "● REC", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Write the processed frame to the video file
        out.write(frame)
        
        cv2.imshow('Part 4: Occlusion Handling', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release() # This finalizes the video file
    cv2.destroyAllWindows()
    print("Video saved successfully as part4_occlusion_output.mp4")

if __name__ == "__main__":
    main()