import streamlit as st
import cv2
import time
from ultralytics import YOLO

st.set_page_config(
    page_title="Advanced Crowd Safety",
    layout="wide"
)

st.title("🚨 Advanced Crowd Behaviour Detection System")
st.caption("YOLO11 + ByteTrack | Real-Time Crowd Safety")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Sidebar
st.sidebar.header("System Controls")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1,
    0.9,
    0.30,
    0.05
)

run = st.sidebar.checkbox("Start Camera")

# Metrics
col1, col2, col3, col4 = st.columns(4)

people_metric = col1.empty()
density_metric = col2.empty()
risk_metric = col3.empty()
fps_metric = col4.empty()

video_placeholder = st.empty()

if run:

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("Could not open webcam.")
        st.stop()

    previous_time = time.time()

    while run:

        ret, frame = cap.read()

        if not ret:
            st.error("Could not read webcam.")
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=confidence,
            verbose=False
        )

        result = results[0]

        # Count people
        if result.boxes is not None:
            people = len(result.boxes)
        else:
            people = 0

        # Density
        if people <= 10:
            density = "LOW"
        elif people <= 25:
            density = "MEDIUM"
        elif people <= 50:
            density = "HIGH"
        else:
            density = "EXTREME"

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Risk
        if density == "EXTREME":
            risk = "CRITICAL"
        elif density == "HIGH":
            risk = "HIGH"
        elif density == "MEDIUM":
            risk = "MEDIUM"
        else:
            risk = "SAFE"

        # Draw YOLO results
        output = result.plot()

        # Dashboard information
        cv2.putText(
            output,
            f"People: {people}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Density: {density}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Convert BGR → RGB
        output_rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        # Update metrics
        people_metric.metric("People", people)
        density_metric.metric("Density", density)
        risk_metric.metric("Risk Level", risk)
        fps_metric.metric("FPS", f"{fps:.1f}")

        video_placeholder.image(
            output_rgb,
            channels="RGB",
            use_container_width=True
        )

    cap.release()

else:

    st.info(
        "Enable 'Start Camera' from the sidebar to begin monitoring."
    )