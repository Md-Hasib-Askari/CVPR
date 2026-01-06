import cv2
from matplotlib import pyplot as plt
import numpy as np

def preprocess_roi(thresh):
    h_img, w_img = thresh.shape

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # reject contours touching border
        if x == 0 or y == 0 or x+w == w_img or y+h == h_img:
            continue

        # reject huge contours (likely paper/background)
        if area > 0.9 * h_img * w_img:
            continue

        valid_contours.append(c)

    if not valid_contours:
        return None

    c = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    digit = thresh[y:y+h, x:x+w]
    return digit, (x, y, w, h)

def preprocess_image(img):
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    plt.imshow(thresh, cmap='gray')

    roi = preprocess_roi(thresh)
    if roi is None:
        return None
    digit_bin, (x, y, w, h) = roi

    ys, xs = np.where(digit_bin > 0)

    digit_crop = digit_bin[
        ys.min():ys.max()+1,
        xs.min():xs.max()+1
    ]

    kernel = np.ones((2,2), np.uint8)
    digit_crop = cv2.dilate(digit_crop, kernel, iterations=1)

    h, w = digit_crop.shape
    scale = 20.0 / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    digit_resized = cv2.resize(
        digit_crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    _, digit_resized = cv2.threshold(
        digit_resized, 127, 255, cv2.THRESH_BINARY
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    y_off = (28 - digit_resized.shape[0]) // 2
    x_off = (28 - digit_resized.shape[1]) // 2

    canvas[
        y_off:y_off+digit_resized.shape[0],
        x_off:x_off+digit_resized.shape[1]
    ] = digit_resized

    return canvas, (x, y, w, h)