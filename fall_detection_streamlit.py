import cv2
import requests
import streamlit as st
import tempfile
import numpy as np
import os

# =========================
# CONFIGURATION
# =========================
API_KEY = "ai9yrSqRpYbc412OeDRh"
PROJECT_ID = "fall-detection-xkufe-uhjo8"
MODEL_VERSION = "7"
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
    else:
        st.error(f"API Error {response.status_code}: {response.text}")
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

option = st.radio("Choose input source:", ["Upload Video", "Upload Image", "Webcam"])

# -------------------------
# CASE 1: UPLOAD VIDEO
# -------------------------
if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        stframe = st.empty()
        st.info("Processing video... please wait.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            predictions = infer_frame(frame, conf_threshold)
            annotated_frame = annotate_frame(frame, predictions)
            out.write(annotated_frame)
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        out.release()
        st.success("Video processing complete!")

        with open(out_path, "rb") as f:
            st.download_button(
                "Download Annotated Video",
                f,
                file_name="annotated_output.mp4",
                mime="video/mp4"
            )

# -------------------------
# CASE 2: UPLOAD IMAGE
# -------------------------
elif option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_image is not None:
        # Decode uploaded image into a NumPy array for OpenCV
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.subheader("Original Image")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        with st.spinner("Running fall detection..."):
            predictions = infer_frame(frame, conf_threshold)
            annotated_frame = annotate_frame(frame.copy(), predictions)

        st.subheader("Detection Result")
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Show a summary of what was detected
        if predictions:
            fall_count = sum(1 for p in predictions if p["class"].lower() == "fall detected")
            stand_count = sum(1 for p in predictions if p["class"].lower() == "standing")

            if fall_count > 0:
                st.error(f"⚠️ Fall Detected! ({fall_count} instance(s) found)")
            if stand_count > 0:
                st.success(f"✅ Standing detected ({stand_count} person(s))")
        else:
            st.warning("No detections found. Try lowering the confidence threshold.")

        # Download annotated image
        _, img_encoded = cv2.imencode('.jpg', annotated_frame)
        st.download_button(
            "Download Annotated Image",
            data=img_encoded.tobytes(),
            file_name="annotated_output.jpg",
            mime="image/jpeg"
        )

# -------------------------
# CASE 3: WEBCAM
# -------------------------
elif option == "Webcam":
    st.write("Click 'Start' to capture from webcam")

    camera_input = st.camera_input("Take a picture")

    if camera_input is not None:
        file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        with st.spinner("Running fall detection..."):
            predictions = infer_frame(frame, conf_threshold)
            annotated_frame = annotate_frame(frame, predictions)

        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if predictions:
            fall_count = sum(1 for p in predictions if p["class"].lower() == "fall detected")
            stand_count = sum(1 for p in predictions if p["class"].lower() == "standing")

            if fall_count > 0:
                st.error(f"⚠️ Fall Detected! ({fall_count} instance(s) found)")
            if stand_count > 0:
                st.success(f"✅ Standing detected ({stand_count} person(s))")
        else:
            st.warning("No detections found. Try lowering the confidence threshold.")