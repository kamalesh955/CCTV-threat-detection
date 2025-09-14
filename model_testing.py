import os
import cv2
import numpy as np
import joblib
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Suppress warnings and logs
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# === Paths ===
fight_folder = r"C:\Users\HP\Desktop\BorderAI\fight-detection-surv-dataset-master\data\fight"
nofight_folder = r"C:\Users\HP\Desktop\BorderAI\fight-detection-surv-dataset-master\data\no_fight"
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
def extract_cnn_features(frames, model):
    features = []
    for frame in frames:
        img = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)
        feature_vector = model.predict(img_array, verbose=0)
        features.append(feature_vector.flatten())
    return np.mean(features, axis=0)

# === Load model ===
try:
    svm_model = joblib.load(model_path)
except:
    print("Error: Could not load SVM model.")
    exit()

# === Load CNN only once ===
cnn_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

y_true = []
y_pred = []

# === Process Fight videos (Threat = 1) ===
for file_name in os.listdir(fight_folder):
    if file_name.lower().endswith(".mp4"):
        video_path = os.path.join(fight_folder, file_name)
        frames = extract_frames(video_path)
        if not frames:
            print(f"Skipping {file_name} (no frames extracted).")
            continue
        features = extract_cnn_features(frames, cnn_model)
        prediction = svm_model.predict([features])[0]
        y_true.append(1)  # Fight = Threat = 1
        y_pred.append(prediction)

# === Process No-Fight videos (Normal = 0) ===
for file_name in os.listdir(nofight_folder):
    if file_name.lower().endswith(".mp4"):
        video_path = os.path.join(nofight_folder, file_name)
        frames = extract_frames(video_path)
        if not frames:
            print(f"Skipping {file_name} (no frames extracted).")
            continue
        features = extract_cnn_features(frames, cnn_model)
        prediction = svm_model.predict([features])[0]
        y_true.append(0)  # No Fight = Normal = 0
        y_pred.append(prediction)

# === Evaluate results ===
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
accuracy = accuracy_score(y_true, y_pred)

print("\nConfusion Matrix (Rows=True, Cols=Pred):")
print("      Pred:Normal  Pred:Threat")
print(f"True:Normal    {cm[0][0]:3d}          {cm[0][1]:3d}")
print(f"True:Threat    {cm[1][0]:3d}          {cm[1][1]:3d}")
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
