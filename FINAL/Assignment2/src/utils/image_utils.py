import cv2

def preprocess_face(face_img, size=(256, 256)):
    face_img = cv2.resize(face_img, size)
    return face_img