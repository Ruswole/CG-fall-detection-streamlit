import cv2
import requests
import streamlit as st
import tempfile
import numpy as np

# =========================
# CONFIGURATION
# =========================
API_KEY = "1FfK6B9ypa412xqLynYN"
PROJECT_ID = "fall-detection-mbldh-epfrq"
MODEL_VERSION = "2"
INFERENCE_URL = f"https://detect.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}?api_key={API_KEY}"

# =========================
# FUNCTION TO RUN INFERENCE
# =========================
def infer_frame(frame, conf_threshold):
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(
        INFERENCE_URL,
        files={"file": img_encoded.tobytes()},
        data={"name": "video_frame"}
    )
    if response.status_code == 200:
        preds = response.json()
        if "predictions" in preds:
            return [p for p in preds["predictions"] if p["confidence"] >= conf_threshold]
    return []

def annotate_frame(frame, predictions):
    for pred in predictions:
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        class_name = pred["class"].lower()
        conf = pred["confidence"]

        if class_name == "standing":
            label, color = "Standing", (0, 255, 0)
        elif class_name == "fall detected":
            label, color = "Fall Detected", (0, 0, 255)
        else:
            label, color = class_name, (255, 255, 0)

        cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 3)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x - w//2, y - h//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

# =========================
# STREAMLIT APP
# =========================
st.title("CareGuardian Fall Detection Demo (Roboflow + Streamlit)")

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.05)
option = st.radio("Choose input source:", ["Upload Video", "Webcam"])

# -------------------------
# UPLOAD VIDEO
# -------------------------
if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        stframe = st.empty()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            predictions = infer_frame(frame, conf_threshold)
            annotated_frame = annotate_frame(frame, predictions)
            out.write(annotated_frame)
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        out.release()

        with open(out_path, "rb") as f:
            st.download_button(
                "Download Annotated Video",
                f,
                file_name="annotated_output.mp4",
                mime="video/mp4"
            )

# -------------------------
# WEBCAM
# -------------------------
elif option == "Webcam":
    camera_input = st.camera_input("Take a picture or short video")
    if camera_input is not None:
        file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        predictions = infer_frame(frame, conf_threshold)
        annotated_frame = annotate_frame(frame, predictions)
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")