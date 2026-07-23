import os
import cv2
import av
import tempfile
import pandas as pd
from datetime import datetime
from PIL import Image

import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="YOLO Vision AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------
# LOAD YOLO MODEL
# ---------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# WEBCAM PROCESSOR
class YOLOProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model.predict(
            img,
            conf=0.40
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

# ---------------------------------
# CREATE OUTPUT FOLDERS
# ---------------------------------

os.makedirs("output/images", exist_ok=True)
os.makedirs("output/videos", exist_ok=True)

# ---------------------------------
# CREATE HISTORY FILE
# ---------------------------------

if not os.path.exists("history.csv"):

    history = pd.DataFrame(
        columns=[
            "Time",
            "Mode",
            "Objects"
        ]
    )

    history.to_csv(
        "history.csv",
        index=False
    )

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("🤖 YOLO Vision AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📷 Image Detection",
        "🎥 Video Detection",
        "📹 Webcam Detection",
        "🚗 Object Tracking",
        "📊 Dashboard",
        "⚙ Settings",
        "ℹ About"
    ]
)

# ---------------------------------
# HOME
# ---------------------------------

if page == "🏠 Home":

    st.title("🤖 YOLO Vision AI")

    st.markdown("---")

    st.header("Professional Object Detection using YOLOv8")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "YOLOv8")

    with col2:
        st.metric("Framework", "PyTorch")

    with col3:
        st.metric("Status", "🟢 Running")

    st.markdown("---")

    st.write("""
Welcome to **YOLO Vision AI**.

This application can perform:

✅ Image Detection

✅ Video Detection

✅ Live Webcam Detection

✅ Object Tracking

✅ Detection Dashboard

✅ Download Results
""")

    st.image("https://docs.ultralytics.com/images/bus.jpg")
    # ---------------------------------
# IMAGE DETECTION
# ---------------------------------

elif page == "📷 Image Detection":

    st.title("📷 Image Detection")

    confidence = st.slider(
        "Confidence Threshold",
        0.10,
        1.00,
        0.40
    )

    uploaded = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:

        image = Image.open(uploaded)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            image.save(tmp.name)

            results = model.predict(
                tmp.name,
                conf=confidence
            )

        result = results[0].plot()

        st.markdown("---")
        st.subheader("Detection Result")

        st.image(
            result,
            use_container_width=True
        )

        # Save Result
        output_path = "output/images/result.jpg"
        Image.fromarray(result).save(output_path)

        # Detection Summary
        boxes = results[0].boxes
        class_names = model.names

        detected_objects = []

        for box in boxes:
            cls = int(box.cls[0])
            detected_objects.append(class_names[cls])

        st.markdown("---")
        st.subheader("📊 Detection Summary")

        if len(detected_objects) == 0:

            st.warning("No objects detected.")

        else:

            counts = {}

            for obj in detected_objects:
                counts[obj] = counts.get(obj, 0) + 1

            st.metric(
                "Total Objects",
                len(detected_objects)
            )

            for name, count in counts.items():
                st.write(f"✅ **{name}** : {count}")

        # Save History
        history = pd.read_csv("history.csv")

        history.loc[len(history)] = {
            "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "Mode": "Image Detection",
            "Objects": len(detected_objects)
        }

        history.to_csv(
            "history.csv",
            index=False
        )

        # Download Button
        with open(output_path, "rb") as file:

            st.download_button(
                "⬇ Download Result",
                file,
                file_name="result.jpg",
                mime="image/jpeg"
            )
            # ---------------------------------
# VIDEO DETECTION
# ---------------------------------

elif page == "🎥 Video Detection":

    st.title("🎥 Video Detection")

    confidence = st.slider(
        "Confidence Threshold",
        0.10,
        1.00,
        0.40,
        key="video_conf"
    )

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        st.info("⏳ Processing video... Please wait.")

        cap = cv2.VideoCapture(temp_video.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = "output/videos/result.mp4"

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        current = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                frame,
                conf=confidence
            )

            annotated = results[0].plot()

            writer.write(annotated)

            current += 1

            if total_frames > 0:
                progress.progress(current / total_frames)

        cap.release()
        writer.release()

        st.success("✅ Video Processed Successfully!")

        st.video(output_path)

        history = pd.read_csv("history.csv")

        history.loc[len(history)] = {
            "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "Mode": "Video Detection",
            "Objects": "-"
        }

        history.to_csv(
            "history.csv",
            index=False
        )

        with open(output_path, "rb") as file:

            st.download_button(
                "⬇ Download Processed Video",
                file,
                file_name="result.mp4",
                mime="video/mp4"
            )
            # ---------------------------------
# WEBCAM DETECTION
# ---------------------------------

elif page == "📹 Webcam Detection":

    st.title("📹 Live Webcam Detection")

    st.write("Click **START** to begin real-time object detection.")

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

# ---------------------------------
# OBJECT TRACKING
# ---------------------------------

elif page == "🚗 Object Tracking":

    st.title("🚗 Object Tracking")

    confidence = st.slider(
        "Confidence Threshold",
        0.10,
        1.00,
        0.40,
        key="track_conf"
    )

    uploaded_video = st.file_uploader(
        "Upload Video for Tracking",
        type=["mp4", "avi", "mov"],
        key="tracking_video"
    )

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        st.info("⏳ Tracking objects... Please wait.")

        cap = cv2.VideoCapture(temp_video.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = "output/videos/tracked_result.mp4"

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        current = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                conf=confidence
            )

            annotated = results[0].plot()

            writer.write(annotated)

            current += 1

            if total_frames > 0:
                progress.progress(current / total_frames)

        cap.release()
        writer.release()

        st.success("✅ Object Tracking Completed!")

        st.video(output_path)

        history = pd.read_csv("history.csv")

        history.loc[len(history)] = {
            "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "Mode": "Object Tracking",
            "Objects": "-"
        }

        history.to_csv(
            "history.csv",
            index=False
        )

        with open(output_path, "rb") as file:

            st.download_button(
                "⬇ Download Tracked Video",
                file,
                file_name="tracked_result.mp4",
                mime="video/mp4"
            )
            # ---------------------------------
# DASHBOARD
# ---------------------------------

elif page == "📊 Dashboard":

    st.title("📊 YOLO Vision Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "YOLOv8")

    with col2:
        st.metric("Framework", "PyTorch")

    with col3:
        st.metric("Status", "🟢 Running")

    st.markdown("---")

    st.subheader("Detection History")

    history = pd.read_csv("history.csv")

    if history.empty:

        st.info("No detection history available.")

    else:

        st.dataframe(
            history,
            use_container_width=True
        )

        with open("history.csv", "rb") as file:

            st.download_button(
                "⬇ Download History",
                file,
                file_name="history.csv",
                mime="text/csv"
            )

# ---------------------------------
# SETTINGS
# ---------------------------------

elif page == "⚙ Settings":

    st.title("⚙ Settings")

    st.subheader("Model Information")

    st.write("**Model:** YOLOv8 Nano")
    st.write("**Framework:** Ultralytics")
    st.write("**Backend:** PyTorch")

    st.markdown("---")

    confidence = st.slider(
        "Default Confidence",
        0.10,
        1.00,
        0.40,
        key="default_conf"
    )

    st.success(f"Current Confidence: {confidence:.2f}")

    st.markdown("---")

    st.info("YOLO Vision AI Version 1.0")

# ---------------------------------
# ABOUT
# ---------------------------------

elif page == "ℹ About":

    st.title("ℹ About")

    st.write("""
# YOLO Vision AI

A professional Object Detection project developed using:

- Python
- OpenCV
- Ultralytics YOLOv8
- Streamlit
- PyTorch

---

## Features

✅ Image Detection

✅ Video Detection

✅ Webcam Detection

✅ Object Tracking

✅ Detection History

✅ Dashboard

✅ Download Results

---

### Developed By

**Mohd Aamir Salam**

AI / ML Engineer
""")