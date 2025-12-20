# Face Recognition Attendance System

A real-time face recognition system that automatically marks attendance using OpenCV and LBPH algorithm.

## Quick Start

# CVPR Assignment 2 - Face Recognition Attendance System
# Quick Setup and Usage Guide

## Prerequisites
- Python 3.8 or higher installed
- Webcam connected and working
- Command prompt/terminal access

## Step-by-Step Setup

### 1. Install Dependencies
Open command prompt/terminal in this directory and run:
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python -c "import cv2, streamlit; print('All packages installed successfully!')"
```

### 3. Register Faces
Collect face images for each person:
```bash
python scripts/collect_faces.py "John Doe"
```
- Look at the camera
- Wait for 20 images to be collected
- Repeat for each person

### 4. Train Model
After collecting faces for all people:
```bash
python scripts/train_model.py
```

### 5. Run Application
Start the web interface:
```bash
streamlit run app.py
```

## Usage Summary

### Register New Person
1. Open app -> "Face Registration" tab
2. Enter name -> Click "Start Capture"
3. Click "Train Model" after collection

### Mark Attendance
1. Open app -> "Real-time Attendance" tab
2. Click "Start Camera"
3. Face the camera
4. Attendance marked automatically

### View Records
1. Open app -> "View Records" tab
2. See statistics and download CSV

## Quick Commands Reference

```bash
# Collect faces
python scripts/collect_faces.py "Student Name"

# Train model
python scripts/train_model.py

# Run app
streamlit run app.py

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

## Troubleshooting

### Camera not working?
- Change CAMERA_INDEX in config/settings.py from 0 to 1
- Close other apps using camera

### Poor accuracy?
- Collect more images (20-30 per person)
- Ensure good lighting
- Lower CONFIDENCE_THRESHOLD in config/settings.py

### Import errors?
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

```
Assignment2/
├── app.py                          # Main application
├── config/settings.py
├── dataset/                        # Dataset directory
├── data/
│   ├── attendance/attendance.csv   # Attendance records
│   └── models/                     # Trained model files
├── scripts/
│   ├── collect_faces.py            # Collect face samples
│   └── train_model.py              # Train recognition model
└── src/
    ├── attendance_manager.py       # Attendance tracking
    ├── face_detector.py            # Face detection
    ├── face_recognizer.py          # Face recognition
    └── utils/image_utils.py        # Image preprocessing
```

## Configuration

Edit [config/settings.py](config/settings.py) to adjust:
- `CONFIDENCE_THRESHOLD`: Recognition confidence (default: 70)
- `CAMERA_INDEX`: Camera device (default: 0)
- File paths for dataset, model, and attendance log

## How It Works

1. **Detection**: Haar Cascade detects faces in video frames
2. **Recognition**: LBPH algorithm identifies known individuals
3. **Attendance**: Automatically logs recognized persons with timestamp

## Troubleshooting

- **Camera not opening**: Try changing `CAMERA_INDEX` to 1 or 2
- **Poor accuracy**: Add more training images, ensure good lighting
- **No face detected**: Face camera directly with adequate lighting

## Support
- Review troubleshooting section
- Verify all prerequisites are met

---
**Course**: Computer Vision and Pattern Recognition (CVPR)
**Author**: Md Hasib Askari