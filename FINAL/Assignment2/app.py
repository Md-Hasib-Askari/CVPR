import cv2
import pandas as pd
import streamlit as st
from mtcnn import MTCNN
import os

from config.settings import *
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.attendance_manager import AttendanceManager
from src.utils.image_utils import preprocess_face

# Streamlit App
st.set_page_config(
    page_title="Face Recognition Attendance System",
    layout="wide"
)
st.title("Face Recognition Attendance System")
st.markdown("Real-time attendance marking using face recognition.")

@st.cache_resource
def load_models():
    register = MTCNN()
    detector = FaceDetector(CASCADE_PATH)
    recognizer = FaceRecognizer()
    recognizer.load(MODEL_PATH, LABEL_MAP_PATH)
    recognizer.model.read(MODEL_PATH)
    attendance = AttendanceManager(ATTENDANCE_FILE)
    return detector, recognizer, attendance

detector, recognizer, attendance = load_models()

# Tabs
tab1, tab2 = st.tabs(["Real-time Attendance", "Face Registration"])

with tab1:
    st.header("Real-time Attendance Marking")
        
    # UI Controls
    col1, col2, col3 = st.columns([1,1,1])
    cap = None
    with col1:
        start_button = st.button("Start Camera")

    with col2:
        stop_button = st.button("Stop Camera")
        if stop_button and cap is not None:
            cap.release()
            cap = None

    with col3:
        clear_button = st.button("Clear Attendance")
        if clear_button:
            attendance.clear()
            st.success("Attendance cleared.")

    # Display Frame
    col_cam, col_att = st.columns([2,1])

    with col_cam:
        st.header("Camera Feed")
        frame_window = st.image([])

    with col_att:
        st.header("Attendance Records")
        table_placeholder = st.empty()

    # Camera Feed
    if start_button:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

        st.session_state['running'] = True

    if stop_button:
        st.session_state['running'] = False

    if 'running' not in st.session_state:
        st.session_state['running'] = False

    while st.session_state['running']:
        if cap is None:
            st.error("Camera not started.")
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect(gray)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = preprocess_face(face)

            label, confidence = recognizer.predict(face)

            if confidence < CONFIDENCE_THRESHOLD:
                name = recognizer.label_map[label]
                marked = attendance.mark(name)
                color = (0,255,0)
                if marked:
                    st.success(f"Attendance marked for {name}")
                    df = pd.DataFrame(
                        attendance.records,
                        columns=["Name", "Date", "Time"]
                    )
                    table_placeholder.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                name = "Unknown"
                color = (0,0,255)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(
                frame, name, (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)
    
with tab2:
    st.header("Face Registration")
    st.markdown("Register new faces to the dataset.")

    student_name = st.text_input("Enter Student Name / ID")

    col1, col2 = st.columns(2)
    start_capture = col1.button("Start Capture")

    if start_capture and student_name.strip() != "":
        import subprocess
        subprocess.run([
            "python", "scripts/collect_faces.py", student_name.strip()
        ])

# Footer
st.markdown("---")
st.caption("LBPH-based Face Recognition System | Real-time Attendance System")
st.markdown("Developed by Md Hasib Askari")