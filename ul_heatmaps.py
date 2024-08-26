import cv2
from CustomVideoCapture import CustomVideoCapture
from ultralytics import YOLO, solutions

model = YOLO("yolov8s.pt")

cap = CustomVideoCapture("video.mp4", 30, True, True)
window_width = 1280
window_height = 720

assert cap.isOpened(), "Error reading video file"

heatmap_obj = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    view_img=True,
    shape="circle",
    classes_names=model.names,
    decay_factor=1,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        continue

    im0 = cv2.resize(im0, (window_width, window_height))

    tracks = model.track(im0, classes=[0], verbose=False, persist=True, show=False)
    
    if tracks[0].boxes.id is not None:
        im0 = heatmap_obj.generate_heatmap(im0, tracks)

cap.release()
cv2.destroyAllWindows()