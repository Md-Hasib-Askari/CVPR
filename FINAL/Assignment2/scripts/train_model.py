from config.settings import DATASET_PATH, MODEL_PATH, LABEL_MAP_PATH, CASCADE_PATH
from src.face_recognizer import FaceRecognizer
from src.face_detector import FaceDetector

recognizer = FaceRecognizer()
detector = FaceDetector(CASCADE_PATH)
recognizer.train(detector, DATASET_PATH, MODEL_PATH, LABEL_MAP_PATH)

print("Model trained successfully")
