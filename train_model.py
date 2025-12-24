import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from features import extract_features

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_PATH = "dataset"
CLASSES = ["healthy", "leaf_spot", "blight"]

# -----------------------------
# LOAD DATA
# -----------------------------
X = []
y = []

print("Loading dataset and extracting features...")

for label, class_name in enumerate(CLASSES):
    class_folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(class_folder):
        raise FileNotFoundError(f"Folder not found: {class_folder}")

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)

        try:
            features = extract_features(img_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Skipping {img_name}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}")

# -----------------------------
# FEATURE SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN RANDOM FOREST MODEL
# -----------------------------
print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_scaled, y)

# -----------------------------
# SAVE MODEL & SCALER
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained successfully!")
print("Saved files: model.pkl, scaler.pkl")
