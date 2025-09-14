from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
import joblib
import tempfile
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

app = FastAPI()

# === Load model and CNN once ===
MODEL_PATH = r"C:\Users\HP\Desktop\BorderAI\CCTV-ANOMALY-DETECTION\svm_model.pkl"
svm_model = joblib.load(MODEL_PATH)
cnn_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

# === Utility functions ===
def extract_frames(video_path, max_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def extract_cnn_features(frames):
    features = []
    for frame in frames:
        img = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)
        feature_vector = cnn_model.predict(img_array, verbose=0)
        features.append(feature_vector.flatten())
    return np.mean(features, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Process video
        frames = extract_frames(temp_path)
        if not frames:
            return JSONResponse(content={"error": "No frames extracted from video"}, status_code=400)

        features = extract_cnn_features(frames)
        prediction = svm_model.predict([features])[0]
        label = "Threat" if prediction == 1 else "Normal"

        # Cleanup
        os.remove(temp_path)

        return {"prediction": label}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
