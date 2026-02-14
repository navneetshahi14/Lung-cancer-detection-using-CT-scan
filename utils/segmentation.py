import os
import cv2
import numpy as np

def segment_lung(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Failed to read: {image_path}")
        return None

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    mask = np.zeros_like(img, dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(mask, [cnt], -1, 255, -1)

    segmented = cv2.bitwise_and(img, mask)

    # 🔥 Ensure correct dtype
    segmented = segmented.astype("uint8")

    return segmented.astype("uint8")



