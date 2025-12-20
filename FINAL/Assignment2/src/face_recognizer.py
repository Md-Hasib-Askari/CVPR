import cv2
import os
import numpy as np
import json
from src.utils.image_utils import preprocess_face

class FaceRecognizer:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer.create()
        self.label_map = {}

    def load(self, model_path, label_path):
        self.model.read(model_path)

        with open(label_path, "r") as f:
            self.label_map = json.load(f)

        # JSON loads keys as strings → convert back to int
        self.label_map = {int(k): v for k, v in self.label_map.items()}


    def train(self, detector, dataset_path, model_path, label_path):
        faces, labels = [], []
        label_id = 0
        print(os.listdir(dataset_path))
        for person in os.listdir(dataset_path):
            self.label_map[label_id] = person
            
            for img in os.listdir(f"{dataset_path}/{person}"):
                img_path = f"{dataset_path}/{person}/{img}"
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # face = detector._extract_face(image)
                # if face is None:
                    # continue

                face = preprocess_face(image)
                faces.append(face)
                labels.append(label_id)
            label_id += 1

        print(len(faces), len(labels))
        self.model.train(faces, np.array(labels))
        self.model.save(model_path)

        with open(label_path, "w") as f:
            json.dump(self.label_map, f)


    def predict(self, face):
        return self.model.predict(face)
