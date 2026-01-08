import cv2
import numpy as np

# 1. Load assets
target_img = cv2.imread('org_image.png') 
over_img = cv2.imread('mcdo.png')        
cap = cv2.VideoCapture('video.mp4')

# Resize overlay to match target dimensions
over_img = cv2.rotate(over_img, cv2.ROTATE_90_CLOCKWISE)
over_img = cv2.resize(over_img, (target_img.shape[1], target_img.shape[0]))

# 2. Get video properties for the writer
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 3. Initialize VideoWriter 
# 'mp4v' is a common codec for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('AR_output.mp4', fourcc, fps, (frame_width, frame_height))

sift = cv2.SIFT_create()
kp_target, des_target = sift.detectAndCompute(target_img, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

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
                warped_over = cv2.warpPerspective(over_img, H, (frame_width, frame_height))
                mask_base = np.ones((target_img.shape[0], target_img.shape[1]), dtype=np.uint8) * 255
                warped_mask = cv2.warpPerspective(mask_base, H, (frame_width, frame_height))
                mask_inv = cv2.bitwise_not(warped_mask)
                frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                final_frame = cv2.add(frame_bg, warped_over)
            else:
                final_frame = frame
        else:
            final_frame = frame
    else:
        final_frame = frame

    # --- SAVE THE FRAME ---
    out.write(final_frame)
    
    cv2.imshow('AR Result', final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- IMPORTANT: Release everything ---
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved successfully as AR_output.mp4")