from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Image path
image_path = "assets/images/test1.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("❌ Image not found!")
    exit()

# Detect objects
results = model(image)

# Draw detections
annotated_image = results[0].plot()

# Show result
cv2.imshow("YOLO Image Detection", annotated_image)

# Save output
cv2.imwrite("output/images/result.jpg", annotated_image)

print("✅ Output saved in output/images/result.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()