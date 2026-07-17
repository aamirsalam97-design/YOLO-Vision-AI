import cv2


def draw_detection(frame, detection):
    x1, y1, x2, y2 = detection["box"]
    label = detection["label"]
    confidence = detection["confidence"]

    text = f"{label} {confidence:.2f}"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


def draw_object_count(frame, count):
    cv2.putText(
        frame,
        f"Objects: {count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )