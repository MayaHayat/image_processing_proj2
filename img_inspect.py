import cv2
import numpy as np

# --- CONFIGURATION ---
video_path = 'video_hand.mp4'
target_second = 16.2
# Set your desired display size here
display_width = 800
display_height = 600

def get_pixel_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_bgr = param[y, x]
        pixel_ycrcb = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        
        print("-" * 30)
        print(f"Clicked at: ({x}, {y})")
        print(f"BGR: {pixel_bgr} | YCrCb: {pixel_ycrcb} | HSV: {pixel_hsv}")

# 1. Load Video and Jump to Frame
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_second * fps))
ret, frame = cap.read()
cap.release()

if ret:
    # 2. Setup Resizable Window
    window_name = 'Color Inspector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Allows resizing
    cv2.resizeWindow(window_name, display_width, display_height) # Set smaller size
    
    cv2.setMouseCallback(window_name, get_pixel_color, param=frame)

    while True:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cv2.destroyAllWindows()