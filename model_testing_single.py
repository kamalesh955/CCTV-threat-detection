import os
import cv2
import numpy as np
import joblib
import warnings
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress TensorFlow logs (only show errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# === Paths ===
video_path = r"C:\Users\HP\Desktop\BorderAI\fight-detection-surv-dataset-master\data\Normal_Videos032_x264.mp4"
model_path = r"C:\Users\HP\Desktop\BorderAI\CCTV-ANOMALY-DETECTION\svm_model.pkl"

# === Frame extraction ===
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

# === Feature extraction ===
def extract_cnn_features(frames):
    model = ResNet50(weights="imagenet", include_top=False, pooling='avg')
    features = []
    for frame in frames:
        img = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)
        feature_vector = model.predict(img_array, verbose=0)
        features.append(feature_vector.flatten())
    return np.mean(features, axis=0)

# === Load model and predict ===
try:
    svm_model = joblib.load(model_path)
except:
    print("Error: Could not load SVM model.")
    exit()

frames = extract_frames(video_path)
if not frames:
    print("Error: No frames extracted.")
else:
    features = extract_cnn_features(frames)
    prediction = svm_model.predict([features])[0]
    print("Threat" if prediction == 1 else "Normal")
