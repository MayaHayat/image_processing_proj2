import cv2
import numpy as np

# 1. Load the 3 target images
target_img1 = cv2.imread('part5_image1.png')  # Will try as image first
target_img2 = cv2.imread('part5_image2.png')  # Will try as image first
target_img3 = cv2.imread('part5_image3.png')

# 2. Load the video
cap = cv2.VideoCapture('part5#2.mp4')

# 3. Get video properties for the writer
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 4. Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('part5_1_output.mp4', fourcc, fps, (frame_width, frame_height))

# 5. SIFT Setup - create feature detectors for each target image
sift = cv2.SIFT_create(nfeatures=1000)

# Detect features in all target images
targets = []
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red

for i, target_img in enumerate([target_img1, target_img2, target_img3]):
    if target_img is not None:
        kp_target, des_target = sift.detectAndCompute(target_img, None)
        h_ref, w_ref = target_img.shape[:2]
        targets.append({
            'img': target_img,
            'kp': kp_target,
            'des': des_target,
            'width': w_ref,
            'height': h_ref,
            'color': colors[i],
            'name': f'Image {i+1}'
        })
        print(f"Loaded target image {i+1}: {w_ref}x{h_ref}, {len(kp_target)} keypoints")

# 6. FLANN matcher
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))

# 7. Store previous positions for smoothing
previous_positions = {}  # Dictionary to store previous corners for each target
frames_since_detection = {}  # Track frames since last successful detection for each target
max_movement_threshold = 150  # Maximum pixel distance allowed between frames
max_frames_without_detection = 10  # Stop using previous position after this many frames

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame_count += 1
    
    # Detect features in the current frame
    kp_frame, des_frame = sift.detectAndCompute(frame, None)
    
    if des_frame is not None and len(des_frame) > 0:
        # Try to detect each target image
        for idx, target in enumerate(targets):
            try:
                matches = flann.knnMatch(target['des'], des_frame, k=2)
                
                # Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                # If enough good matches found, compute homography
                if len(good_matches) > 40:
                    src_pts = np.float32([target['kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        # Get the 4 corners of the target image
                        w_ref = target['width']
                        h_ref = target['height']
                        ref_corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
                        
                        # Transform corners to frame coordinates
                        img_pts = cv2.perspectiveTransform(ref_corners, H)
                        img_pts = np.int32(img_pts)
                        
                        # Check if the detected square is within frame boundaries
                        pts_flat = img_pts.reshape(-1, 2)
                        center_x = np.mean(pts_flat[:, 0])
                        center_y = np.mean(pts_flat[:, 1])
                        
                        # Check if at least the center is within frame (with some margin)
                        margin = 50  # Allow some margin outside frame
                        is_within_bounds = (
                            -margin < center_x < frame_width + margin and 
                            -margin < center_y < frame_height + margin
                        )
                        
                        if not is_within_bounds:
                            # Square is outside frame, don't use this detection
                            print(f"Frame {frame_count}: {target['name']} is outside frame bounds, ignoring")
                            # Increment frames since detection as this is not a valid detection
                            frames_since_detection[idx] = frames_since_detection.get(idx, 0) + 1
                            # Check if we should clear the previous position
                            if idx in previous_positions and frames_since_detection[idx] > max_frames_without_detection:
                                del previous_positions[idx]
                                del frames_since_detection[idx]
                            continue
                        
                        # Smoothing: Check if this position is too far from previous position
                        if idx in previous_positions:
                            prev_pts = previous_positions[idx]
                            # Calculate center of current and previous squares
                            current_center = np.mean(img_pts.reshape(-1, 2), axis=0)
                            prev_center = np.mean(prev_pts.reshape(-1, 2), axis=0)
                            distance = np.linalg.norm(current_center - prev_center)
                            
                            # If moved too far, use previous position
                            if distance > max_movement_threshold:
                                img_pts = prev_pts
                                print(f"Frame {frame_count}: {target['name']} moved too far ({distance:.1f}px), using previous position")
                            else:
                                # Update previous position
                                previous_positions[idx] = img_pts.copy()
                                frames_since_detection[idx] = 0  # Reset counter on successful detection
                        else:
                            # First detection, store it
                            previous_positions[idx] = img_pts.copy()
                            frames_since_detection[idx] = 0
                        
                        # Draw rectangle around detected image
                        cv2.polylines(frame, [img_pts], True, target['color'], 3, cv2.LINE_AA)
                        
                        # Add label
                        label_pos = tuple(img_pts[0])
                        cv2.putText(frame, target['name'], label_pos, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, target['color'], 2)
                        
            except Exception as e:
                # Skip if matching fails
                pass
    
    # Write frame to output video
    out.write(frame)
    
    # Display
    cv2.imshow('Part 5.1: Multi-Image Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nDone! Processed {frame_count} frames.")
print("Output saved to: part5_1_output.mp4")
