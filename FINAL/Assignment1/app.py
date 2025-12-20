import streamlit as st
import cv2
import numpy as np
import keras
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from av import VideoFrame

from preprocess_img import preprocess_image

st.set_page_config(page_title="Real-time Digit Recognition", layout="centered")
st.title("Real-time Digit Recognition")
st.write("Show your handwritten digits (0-9) inside the box.")

# Load the pre-trained model
try:
    @st.cache_resource(show_spinner="Loading model...")
    def load_model():
        return keras.models.load_model('digit_model.keras')
    model = load_model()
except:
    st.error("Failed to load the model. Please ensure 'digit_model.keras' is in the correct path.")
    st.stop()

# Video processor
class DigitRecognizer(VideoProcessorBase):
    def __init__(self):
        if isinstance(model, keras.Model): 
            self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            processed_img = preprocess_image(img)
            if processed_img is None:
                return VideoFrame.from_ndarray(img.astype(np.uint8), format="bgr24")

            processed_img, (x, y, w, h) = processed_img

            reshaped = processed_img.reshape(1, 28, 28, 1)

            # Predict the digit
            prediction = self.model.predict(x=reshaped)
            confidence = np.max(prediction)

            if confidence < 0.7:
                label = "Uncertain"
            else:
                label = str(np.argmax(prediction))
            
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # Overlay the prediction on the frame
            cv2.putText(
                img=img, 
                text=f'Predicted Digit: {label} ({confidence:.2f})',
                org=(10, 30), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, 
                color=(0, 255, 0), 
                thickness=2, 
                lineType=cv2.LINE_AA
            )

            return VideoFrame.from_ndarray(img.astype(np.uint8), format="bgr24")
        except Exception as e:
            print(f"Error processing frame: {e}")
            return VideoFrame.from_ndarray(img.astype(np.uint8), format="bgr24")
    
# Start the webcam stream
webrtc_streamer(
    key="digit-recognizer",
    video_processor_factory=DigitRecognizer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)