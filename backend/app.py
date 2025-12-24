import sys
import os

# Allow backend to access root folder (for features.py)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

from features import extract_features

# --------------------------------
# APP SETUP
# --------------------------------
app = Flask(__name__)
CORS(app)  # Enable frontend-backend connection

# --------------------------------
# LOAD MODEL & SCALER
# --------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------
# CLASS LABELS
# --------------------------------
classes = ["Healthy", "Leaf Spot", "Blight"]

# --------------------------------
# DISEASE INFO
# --------------------------------
info = {
    "Healthy": {
        "description": "The leaf appears healthy with no visible disease symptoms.",
        "precaution": "No action required. Maintain regular watering and nutrient care."
    },
    "Leaf Spot": {
        "description": "Leaf spot is a plant disease characterized by dark circular spots on leaves.",
        "precaution": "Remove infected leaves, avoid overhead watering, and apply fungicide if needed."
    },
    "Blight": {
        "description": "Blight is a serious disease causing rapid browning and drying of leaves.",
        "precaution": "Use disease-free seeds, improve air circulation, and apply recommended fungicides."
    }
}

# --------------------------------
# PREDICTION API
# --------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if image is sent
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        img_path = "temp.jpg"
        file.save(img_path)

        # Extract features
        features = extract_features(img_path)
        features = scaler.transform([features])

        # Predict
        probs = model.predict_proba(features)[0]
        idx = int(np.argmax(probs))

        confidence = float(probs[idx] * 100)
        disease = classes[idx]

        # --------------------------------
        # CONFIDENCE THRESHOLD LOGIC
        # --------------------------------
        if confidence < 60:
            disease = "Healthy"
            description = "The leaf appears healthy or the model is uncertain."
            precaution = "No immediate action required. Monitor plant health regularly."
        else:
            description = info[disease]["description"]
            precaution = info[disease]["precaution"]

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "description": description,
            "precaution": precaution
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------
# RUN SERVER
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True)
