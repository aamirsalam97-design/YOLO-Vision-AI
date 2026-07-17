from ultralytics import YOLO
import cv2


class ObjectDetector:

    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model(frame)

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                conf = float(box.conf[0])

                if conf >= self.confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]

                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "box": (x1, y1, x2, y2)
                    })

        return detections