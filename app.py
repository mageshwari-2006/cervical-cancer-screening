from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Model path
MODEL_PATH = os.path.join(
    "model",
    "FINAL_Cervical_Cancer_ResNet50_CLAHE.keras"
)

# Load model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("MODEL LOADED SUCCESSFULLY")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # Resize
    image = cv2.resize(image, (224, 224))

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # RGB → LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # CLAHE
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    # Merge LAB channels
    lab = cv2.merge((l, a, b))

    # LAB → RGB
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Normalize
    image = image.astype(np.float32) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Prediction
    output = model.predict(image, verbose=0)

    normal_probability = float(output[0][0])
    abnormal_probability = 1.0 - normal_probability

    # Classification
    if normal_probability >= 0.50:
        prediction = "Normal"
    else:
        prediction = "Abnormal"

    return jsonify({
        "prediction": prediction,
        "normal_probability": normal_probability,
        "abnormal_probability": abnormal_probability
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )