# Face Recognition Attendance System

A real-time face recognition system that automatically marks attendance using OpenCV and LBPH algorithm.

## Quick Start

### Installation

```bash
pip install opencv-python opencv-contrib-python numpy
```

### Usage

1. **Prepare Training Data**
   - Create folders in `data/dataset/` for each person (e.g., `person_1`, `person_2`)
   - Add 10-30 clear face images per person

2. **Train Model**
   ```bash
   python scripts/train_model.py
   ```

3. **Run Attendance System**
   ```bash
   python app.py
   ```
   Press ESC to exit.

## Project Structure

```
Assignment2/
├── app.py                          # Main application
├── config/settings.py              # Configuration
├── data/
│   ├── attendance/attendance.csv   # Attendance records
│   ├── dataset/                    # Training images
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
