import cv2
import numpy as np
from ultralytics import YOLO
from CustomVideoCapture import CustomVideoCapture

def apply_gaussian_to_heatmap(heatmap, center, sigma, intensity):
    y, x = np.ogrid[:heatmap.shape[0], :heatmap.shape[1]]
    radius = min(width, height) // 2
    gaussian = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2 
    # gaussian = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma ** 2))
    # gaussian = gaussian / np.max(gaussian)  # Normalize to [0, 1]
    heatmap += gaussian * intensity

# load yolo
yolo = YOLO('yolov8s.pt') 

# video capture
cap = CustomVideoCapture(video_source="video.mp4", framerate=30, capped_fps=True)
window_width = 1280
window_height = 720

# create blank heatmap
heatmap = np.zeros((window_height, window_width), dtype=np.float32)
heatmap_alpha = 0.5 # level of transparency of colormap vs orig img


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue
    
    frame = cv2.resize(frame, (window_width, window_height))

    # detection - people only
    results = yolo(frame, classes=[0], verbose=False)

    for res in results:
        for detection in res.boxes:
            x_center, y_center, width, height = map(int, detection.xywh[0])
            
            sigma = min(width, height) / 2
            intensity = 0.1
            apply_gaussian_to_heatmap(heatmap, (x_center, y_center), sigma, intensity)


    # create a color heatmap
    heatmap_display = np.clip(heatmap / np.max(heatmap) * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_PARULA)

    # blend image with heatmap
    heatmap_overlay = cv2.addWeighted(frame, 1-heatmap_alpha, heatmap_color, heatmap_alpha, 0)

    # display frame
    cv2.imshow('Frame with Heatmap Overlay', heatmap_overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
