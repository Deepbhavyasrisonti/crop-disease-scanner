import sys
import os
import joblib
import numpy as np
import uuid

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

sys.path.append(ROOT_DIR)
from features import extract_features

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)
CORS(app)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

classes = ["Healthy", "Leaf Spot", "Blight"]

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

@app.route("/health")
def health():
    return jsonify({"status": "Backend is running"})

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        img_path = os.path.join(BASE_DIR, f"{uuid.uuid4()}.jpg")
        file.save(img_path)

        features = extract_features(img_path)
        features = scaler.transform([features])

        probs = model.predict_proba(features)[0]
        idx = int(np.argmax(probs))

        confidence = float(probs[idx] * 100)
        disease = classes[idx]

        if confidence < 60:
            disease = "Healthy"
            description = "The leaf appears healthy or the model is uncertain."
            precaution = "No immediate action required."
        else:
            description = info[disease]["description"]
            precaution = info[disease]["precaution"]

        os.remove(img_path)

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "description": description,
            "precaution": precaution
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
