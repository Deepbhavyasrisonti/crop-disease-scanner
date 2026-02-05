🌿 Crop Disease Scanner

A web-based Crop Disease Detection System developed using Python 3.11, Flask, and Classical Machine Learning.
The application allows users to upload leaf images and predicts whether the crop is Healthy, affected by Leaf Spot, or Blight, along with confidence, disease description, and precautionary recommendations.

🔍 Project Overview

Crop diseases significantly affect agricultural productivity. This project aims to provide an accessible and lightweight solution for early disease detection using image-based feature extraction and machine learning, without relying on heavy deep learning frameworks.

The system uses color and texture features extracted from leaf images and classifies them using a Random Forest model. The trained model is deployed using a Flask backend and accessed via a simple HTML/CSS frontend.

✨ Features

Upload leaf images through a web interface

Detects:

Healthy

Leaf Spot

Blight

Displays:

Disease name

Confidence score

Disease description

Precautionary measures

Confidence threshold to avoid false disease alerts

Lightweight and Python 3.11 compatible

🧠 Machine Learning Details

Model: Random Forest Classifier

Why Random Forest?

Handles non-linear relationships well

Performs better on small datasets

Robust to noise and feature overlap

Features Extracted:

Mean RGB values

Mean HSV values

Grayscale mean

Grayscale standard deviation (texture)

🛠️ Technologies Used

Python 3.11

Flask

Flask-CORS

Scikit-learn

OpenCV

NumPy

HTML & CSS

Git & GitHub

📂 Project Structure
crop-disease-scanner/
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   └── style.css
│
├── dataset/           # (ignored in GitHub)
│   ├── healthy/
│   ├── leaf_spot/
│   └── blight/
│
├── features.py
├── train_model.py
├── README.md
└── .gitignore

🚀 How to Run the Project
1️⃣ Install dependencies
pip install flask flask-cors scikit-learn opencv-python numpy

2️⃣ Train the model
python train_model.py


This generates:

model.pkl

scaler.pkl

3️⃣ Start the backend
cd backend
python app.py


Backend runs at:

http://127.0.0.1:5000

4️⃣ Open frontend

Open the file below in a browser:

frontend/index.html


Upload a leaf image and click Predict.

📊 Output Example

Disease: Leaf Spot

Confidence: 78.45%

Description: Leaf spot causes dark lesions on leaves.

Precaution: Remove infected leaves and apply fungicide.

Low-confidence predictions are treated as Healthy to prevent false alarms.

⚠️ Limitations

Works best with clear leaf images

Accuracy depends on dataset size and diversity

Classical ML may struggle with very subtle disease patterns

🔮 Future Enhancements

Increase dataset size

Add more disease classes

Use advanced texture features (GLCM)

Integrate CNN-based models

Deploy as a mobile application

📌 Note

Dataset and trained model files are excluded from GitHub using .gitignore

This project is intended for academic and educational purposes

👩‍💻 Author

Bhavya (Deepbhavyasrisonti)
GitHub: https://github.com/Deepbhavyasrisonti

📜 License

This project is licensed under the MIT License.
