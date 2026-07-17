from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Video path
video_path = "assets/videos/test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video not found!")
    exit()

# Output folder
os.makedirs("output/videos", exist_ok=True)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "output/videos/tracked_result.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Tracking", annotated_frame)

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Tracked video saved in output/videos/tracked_result.mp4")