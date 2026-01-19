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
    # 1. Load assets
    target_img = cv2.imread('org_image.png')
    if target_img is None:
        print("Error: Could not load org_image.png")
        return
    
    cap = cv2.VideoCapture('video.mp4')
    if not cap.isOpened():
        print("Error: Could not open video.mp4")
        return
    
    # 2. Load 3D mesh
    print("\n=== Loading 3D Mesh ===")
    mesh_trimesh = load_mesh_trimesh('lego_bouquet.ply')
    
    if mesh_trimesh is None:
        print("Error: Could not load mesh")
        return
    
    # Center and normalize the mesh
    mesh_center = mesh_trimesh.bounds.mean(axis=0)
    mesh_trimesh.vertices -= mesh_center
    mesh_size = np.max(mesh_trimesh.bounds[1] - mesh_trimesh.bounds[0])
    
    # Rotate mesh 180 degrees around X-axis to flip it right-side up
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    mesh_trimesh.apply_transform(rotation_matrix)
    
    print(f"  Mesh centered at: {mesh_center}")
    print(f"  Mesh size: {mesh_size}")
    print(f"  Mesh rotated 180° around X-axis")
    
    # 3. GET CALIBRATED CAMERA PARAMETERS
    print("\n=== Loading Camera Calibration ===")
    try:
        calib_data = np.load('camera_calibration.npz')
        K = calib_data['camera_matrix']
        dist_coeffs = calib_data['dist_coeffs']
        print("Loaded calibration from camera_calibration.npz")
        print(f"Camera matrix:\n{K}")
    except FileNotFoundError:
        print("Running camera calibration...")
        calib = calibrate_camera(verbose=False)
        K = calib['camera_matrix']
        dist_coeffs = calib['dist_coeffs']
        print("Camera calibration complete")
    
    # 4. Define 3D plane points for tracking
    h_ref, w_ref = target_img.shape[:2]
    obj_pts_plane = np.float32([[0, 0, 0], [w_ref, 0, 0], [w_ref, h_ref, 0], [0, h_ref, 0]])
    
    # 5. SIFT Setup for tracking
    print("\n=== Setting up Feature Matching ===")
    sift = cv2.SIFT_create(nfeatures=1000)
    kp_target, des_target = sift.detectAndCompute(target_img, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))
    print(f"Detected {len(kp_target)} keypoints in target image")
    
    # 6. Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('part3_output.mp4', fourcc, fps, (frame_width, frame_height))
    
    print(f"\n=== Processing Video ===")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Mesh rendering parameters
    mesh_scale = w_ref / mesh_size  # Scale mesh to fit on the tracked plane
    mesh_offset = (w_ref / 2, h_ref / 2, -w_ref / 3)  # Position above the plane
    
    frame_count = 0
    tracked_frames = 0
    
    # Choose rendering mode: 'solid' or 'wireframe'
    render_mode = 'solid'  # Change to 'wireframe' for faster rendering
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect features in current frame
        kp_frame, des_frame = sift.detectAndCompute(frame, None)
        
        if des_frame is not None and len(des_frame) > 0:
            # Match features
            matches = flann.knnMatch(des_target, des_frame, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            if len(good_matches) > 30:
                # Get homography to find target position
                src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                
                if H is not None:
                    # Find corners in video frame
                    ref_corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
                    img_pts = cv2.perspectiveTransform(ref_corners, H)
                    
                    # Solve PnP to get camera pose
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts_plane, img_pts, K, dist_coeffs, 
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if success:
                        tracked_frames += 1
                        
                        # Render 3D mesh on frame
                        if render_mode == 'solid':
                            frame = render_mesh_on_frame(
                                frame, mesh_trimesh, rvec, tvec, K, dist_coeffs,
                                scale=mesh_scale, offset=mesh_offset
                            )
                        else:  # wireframe
                            frame = render_mesh_wireframe(
                                frame, mesh_trimesh, rvec, tvec, K, dist_coeffs,
                                scale=mesh_scale, offset=mesh_offset, color=(0, 255, 0)
                            )
                        
                        # Optional: Draw bounding box of tracked region
                        # cv2.polylines(frame, [np.int32(img_pts)], True, (255, 0, 0), 2)
        
        # Write frame to output
        out.write(frame)
        
        # Display
        cv2.imshow('Part 3: 3D Mesh Rendering', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Progress indicator
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, tracked in {tracked_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== Complete ===")
    print(f"Total frames: {frame_count}")
    print(f"Tracked frames: {tracked_frames}")
    print(f"Output saved to: part3_output.mp4")

if __name__ == "__main__":
    main()
