from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the model and configurations
model = load_model("C:/Gender_detection/gender_detection.model")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
GENDER_CLASSES = ["Male", "Female"]
input_shape = model.input_shape[1:3]
input_channels = model.input_shape[-1]

def predict_gender(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_roi = image_np[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, input_shape)

        if input_channels == 1:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            face_roi = np.expand_dims(face_roi, axis=-1)

        face_roi = face_roi.astype("float32") / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)

        prediction = model.predict(face_roi)
        gender_output = prediction[0]
        gender_index = np.argmax(gender_output)
        gender = GENDER_CLASSES[gender_index]
        confidence = round(gender_output[gender_index] * 100, 2)

        return {"gender": gender, "confidence": confidence}

    return {"gender": "No face detected", "confidence": 0}

@app.route('/detect', methods=['POST'])
def detect_from_webcam():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame file provided"}), 400

    file = request.files['frame']
    image = Image.open(file.stream).convert("RGB")
    image_np = np.array(image)
    result = predict_gender(image_np)
    return jsonify(result)

@app.route('/detect-image', methods=['POST'])
def detect_from_upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    image_np = np.array(image)
    result = predict_gender(image_np)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
