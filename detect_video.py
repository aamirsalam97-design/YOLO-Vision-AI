from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Video path
video_path = "assets/videos/test.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video not found!")
    exit()

# Create output folder
os.makedirs("output/videos", exist_ok=True)

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
out = cv2.VideoWriter(
    "output/videos/result.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show video
    cv2.imshow("YOLO Video Detection", annotated_frame)

    # Save frame
    out.write(annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video saved in output/videos/result.mp4")