import cv2
import time

from config.config import (
    CAMERA_INDEX,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    WINDOW_NAME
)

from detector.object_detector import ObjectDetector
from detector.utils import (
    draw_detection,
    draw_fps,
    draw_object_count
)


def main():

    detector = ObjectDetector(
        MODEL_PATH,
        CONFIDENCE_THRESHOLD
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("❌ Camera not found!")
        return

    prev_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections = detector.detect(frame)

        for detection in detections:
            draw_detection(frame, detection)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        draw_fps(frame, fps)
        draw_object_count(frame, len(detections))

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()