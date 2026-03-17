import cv2
import requests
import streamlit as st
import tempfile
import numpy as np
import os

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
    """
    Encodes the frame as JPG, sends it to Roboflow API,
    and returns predictions above a confidence threshold.
    """
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
    """
    Draw bounding boxes and labels on the frame
    (with fall/stand label flipping as per your requirement).
    """
    for pred in predictions:
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        class_name = pred["class"].lower()
        conf = pred["confidence"]

         #Labeling 
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

# DIFFERENCE: In Streamlit, we use interactive widgets (like sliders)
# instead of hardcoding values in code.
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.05)

# DIFFERENCE: Instead of running automatically on one input,
# we let the user choose between uploading a video or using webcam.
option = st.radio("Choose input source:", ["Upload Video", "Webcam"])

# -------------------------
# CASE 1: UPLOAD VIDEO
# -------------------------
if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # DIFFERENCE: In Streamlit we must save the uploaded file
        # to a temporary file before OpenCV can read it.
        tfile = tempfile.NamedTemporaryFile(delete=False)  
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        # DIFFERENCE: We also prepare an output temp file
        # so that we can later provide it for downloading.
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        # DIFFERENCE: Instead of cv2.imshow (not allowed in Streamlit),
        # we use st.empty() which acts as a "placeholder" for updating frames.
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            predictions = infer_frame(frame, conf_threshold)
            annotated_frame = annotate_frame(frame, predictions)

            # Save frame to output video
            out.write(annotated_frame)

            # Show frame live inside Streamlit browser
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        out.release()

        # DIFFERENCE: In OpenCV, you would save the file and print the path.
        # In Streamlit, we can directly offer a "Download" button in the UI.
        with open(out_path, "rb") as f:
            st.download_button(
                "Download Annotated Video",
                f,
                file_name="annotated_output.mp4",
                mime="video/mp4"
            )

# -------------------------
# CASE 2: WEBCAM
# -------------------------
elif option == "Webcam":
    st.write("Click 'Start' to capture from webcam")

    # DIFFERENCE: In Streamlit, we cannot directly use cv2.VideoCapture(0)
    # because Streamlit runs in a web browser. Instead, we use st.camera_input,
    # which lets the user capture an image or short video clip from webcam.
    camera_input = st.camera_input("Take a picture or record short video")

    if camera_input is not None:
        # Convert the uploaded webcam capture into a NumPy array
        file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        predictions = infer_frame(frame, conf_threshold)
        annotated_frame = annotate_frame(frame, predictions)

        # Show annotated frame in browser
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")