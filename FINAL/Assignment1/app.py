import streamlit as st
import cv2
import numpy as np
import keras
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from av import VideoFrame

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
    st.error("Failed to load the model. Please ensure 'digit_model.h5' is in the correct path.")
    st.stop()

# Video processor
class DigitRecognizer(VideoProcessorBase):
    def __init__(self):
        if isinstance(model, keras.Model): 
            self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # img = cv2.flip(img, 1)  # Mirror the image

        h, w, _ = img.shape

        # ROI box
        x1, y1 = int(w * 0.3), int(h * 0.3)
        x2, y2 = int(w * 0.7), int(h * 0.7)

        cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
        roi = img[y1:y2, x1:x2]

        # Preprocess the image
        gray = cv2.cvtColor(src=roi, code=cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
        _, thresh = cv2.threshold(
            src=blur, 
            thresh=100, 
            maxval=255, 
            type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        resized = cv2.resize(src=thresh, dsize=(28, 28))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 28, 28, 1)

        # Display the processed ROI for debugging (optional)
        st.image(resized, caption='Processed ROI', width=150)

        # Predict the digit
        prediction = self.model.predict(x=reshaped)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # Overlay the prediction on the frame
        cv2.putText(
            img=img, 
            text=f'Predicted Digit: {digit} ({confidence:.2f})',
            org=(x1, y1 - 10), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, 
            color=(0, 255, 0), 
            thickness=2, 
            lineType=cv2.LINE_AA
        )

        return VideoFrame.from_ndarray(img.astype(np.uint8), format="bgr24")
    
# Start the webcam stream
webrtc_streamer(
    key="digit-recognizer",
    video_processor_factory=DigitRecognizer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)