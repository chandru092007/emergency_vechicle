import tensorflow as tf
import streamlit as st
import cv2
import numpy as np
import tempfile

st.title("🚑 Multi-lane Ambulance Detection System")

# Load trained CNN model
model = tf.keras.models.load_model("ambulance_cnn.keras")

# Preprocess frame for CNN
def preprocess_frame(frame):
    img = cv2.resize(frame, (128,128))   # same size as training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Detect ambulance in uploaded video
def detect_ambulance(uploaded_file):
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmp_path = tmpfile.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error(f"Could not open video: {uploaded_file.name}")
        return False

    detected = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pred = model.predict(preprocess_frame(frame))[0][0]
        if pred > 0.5:   # threshold
            detected = True
            break
    cap.release()
    return detected

# Upload multiple lane videos
uploaded_files = st.file_uploader("Upload 4 lane videos", type=["mp4","jpeg","jpg"], accept_multiple_files=True)

results = {}
if uploaded_files:
    for i, file in enumerate(uploaded_files, start=1):
        st.write(f"Processing Lane {i}...")
        if detect_ambulance(file):
            results[f"Lane {i}"] = True
            st.write(f"Lane {i}: 🚑 Ambulance detected")
        else:
            results[f"Lane {i}"] = False
            st.write(f"Lane {i}: 🚗 No ambulance")

