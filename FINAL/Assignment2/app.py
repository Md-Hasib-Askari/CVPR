import cv2
from config.settings import *
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.attendance_manager import AttendanceManager
from src.utils.image_utils import preprocess_face

detector = FaceDetector(CASCADE_PATH)
recognizer = FaceRecognizer()
recognizer.load(MODEL_PATH, LABEL_MAP_PATH)
recognizer.model.read(MODEL_PATH)
attendance = AttendanceManager(ATTENDANCE_FILE)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detect(gray)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = preprocess_face(face)
        label, confidence = recognizer.predict(face)

        print("="*20)
        print(recognizer.label_map)
        print(f"Label: {label}, Confidence: {confidence}")
        if confidence < CONFIDENCE_THRESHOLD:
            name = recognizer.label_map[label]
            attendance.mark(name)
            color = (0,255,0)
        else:
            name = "Unknown"
            color = (0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
