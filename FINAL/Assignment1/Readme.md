# Real-time Digit Recognition

A Streamlit web application that performs real-time handwritten digit recognition (0-9) using a webcam feed and a trained Keras CNN model.

## Features

- **Real-time Recognition**: Processes webcam video stream in real-time
- **Automatic Digit Detection**: Identifies and isolates handwritten digits from the video frame
- **Confidence Scoring**: Displays prediction confidence for each recognition
- **Visual Feedback**: Shows bounding box around detected digits and overlay predictions

## Project Structure

```
Assignment1/
├── app.py                 # Main Streamlit application
├── preprocess_img.py      # Image preprocessing pipeline
├── digit_model.keras      # Trained Keras model (required)
└── Readme.md             # Project documentation
```

## Requirements

- Python 3.7+
- streamlit
- opencv-python (cv2)
- numpy
- keras
- streamlit-webrtc
- av

## Installation

```bash
pip install streamlit opencv-python numpy keras streamlit-webrtc av
```

## Usage

1. Ensure `digit_model.keras` is in the project directory
2. Run the application:
    ```bash
    streamlit run app.py
    ```
3. Allow camera access when prompted
4. Show handwritten digits (0-9) to the camera
5. The app will display predictions with confidence scores

## How It Works

### Image Preprocessing (`preprocess_img.py`)

1. **Grayscale Conversion**: Converts BGR frame to grayscale
2. **CLAHE Enhancement**: Applies Contrast Limited Adaptive Histogram Equalization
3. **Noise Reduction**: Uses Gaussian blur to reduce noise
4. **Adaptive Thresholding**: Creates binary image with adaptive thresholding
5. **Morphological Operations**: Opens and closes to clean up noise
6. **Contour Detection**: Finds and validates digit contours
7. **ROI Extraction**: Isolates the largest valid digit region
8. **Normalization**: Resizes to 28x28 pixels with proper centering

### Real-time Recognition (`app.py`)

- Uses `streamlit-webrtc` for webcam streaming
- `DigitRecognizer` class processes each video frame
- Applies preprocessing pipeline to extract digit
- Feeds processed image to Keras model for prediction
- Displays prediction with confidence threshold (0.7)
- Renders bounding box and label overlay on video feed

## Model Requirements

The application expects a Keras model (`digit_model.keras`) trained on 28x28 grayscale digit images with the following specifications:

- Input shape: (1, 28, 28, 1)
- Output: 10 classes (digits 0-9)
- Format: Keras SavedModel or H5 format

## Configuration

### Confidence Threshold
Default: 0.7 (can be modified in `app.py`)

### Preprocessing Parameters
- CLAHE clip limit: 2.0
- Gaussian blur kernel: (5,5)
- Adaptive threshold block size: 11
- Morphological kernel: (3,3)
- Minimum contour area: 300 pixels
- Target digit size: 20x20 (on 28x28 canvas)

## Limitations

- Requires good lighting conditions
- Works best with clear, isolated digits
- May struggle with overlapping or touching digits
- Border-touching digits are rejected
- Very large contours (>90% of frame) are ignored

## Troubleshooting

**Model not loading**: Ensure `digit_model.keras` is in the correct path

**No prediction**: Check that digit is fully visible and not touching frame borders

**Low confidence**: Improve lighting, write more clearly, or adjust confidence threshold

**Camera not working**: Grant browser camera permissions for localhost