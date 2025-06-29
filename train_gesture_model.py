# train_gesture_model.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === CONFIGURATION ===
DATA_DIR = "data"
MODEL_PATH = "models/gesture_classifier.pkl"

# === LOAD AND COMBINE DATA ===
data_frames = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
        data_frames.append(df)

all_data = pd.concat(data_frames, ignore_index=True)
X = all_data.iloc[:, :-1].values  # landmark features
y = all_data.iloc[:, -1].values   # labels

# === ENCODE LABELS ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === NORMALIZE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === TRAIN MLP MODEL ===
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# === EVALUATE ===
y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === SAVE MODEL AND ENCODER ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump({
    "model": model,
    "scaler": scaler,
    "label_encoder": label_encoder
}, MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")
