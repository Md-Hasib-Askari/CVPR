import cv2

from src.utils.image_utils import preprocess_face

class FaceDetector:
    def __init__(self, cascade_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, gray_frame):
        return self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5
        )
    
    def _extract_face(self, img_gray):
        faces = self.detect(img_gray)
        
        if len(faces) == 0:
            return None
        
        # Take the largest detected face
        x, y, w, h = sorted(
            faces, key=lambda f: f[2] * f[3], reverse=True
        )[0]

        face = img_gray[y:y+h, x:x+w]
        face = preprocess_face(face)
        return face
