import cv2
import numpy as np

def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    img = cv2.resize(img, (200, 200))

    # --- Color Features ---
    mean_rgb = np.mean(img, axis=(0, 1))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv, axis=(0, 1))

    # --- Texture Features ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_gray = np.mean(gray)
    std_gray = np.std(gray)

    # Combine all features
    features = np.hstack([
        mean_rgb,
        mean_hsv,
        mean_gray,
        std_gray
    ])

    return features
