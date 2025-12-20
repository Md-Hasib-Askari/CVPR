import cv2
import os
from mtcnn import MTCNN
import sys

# Initialize face detector
detector = MTCNN()

# Ask student name
if len(sys.argv) > 1:
    student_name = sys.argv[1].strip()
else:
    student_name = input("Enter student name/ID: ").strip()

# Dataset path
dataset_path = "dataset"
student_path = os.path.join(dataset_path, student_name)

os.makedirs(student_path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

count = 0
MAX_IMAGES = 20   # collect 20 face images

print("[INFO] Press 'q' to quit early")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)
    if not faces:
        cv2.imshow("Collecting Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for face in faces:
        if not face:
            continue
        
        x, y, w, h = face["box"] # type: ignore
        x, y = max(0, x), max(0, y)

        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        face_img = cv2.resize(face_img, (160, 160))
        count += 1

        file_name = f"{student_name}_{count}.jpg"
        cv2.imwrite(os.path.join(student_path, file_name), face_img)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"Saved: {count}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print(f"[DONE] Collected {count} images for {student_name}")