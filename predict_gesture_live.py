# predict_gesture_live.py

import cv2
import numpy as np
import mediapipe as mp
import joblib
from gesture_actions import trigger_action
import time


# === LOAD MODEL AND TOOLS ===
model_bundle = joblib.load("models/gesture_classifier.pkl")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
label_encoder = model_bundle["label_encoder"]

# === MEDIAPIPE SETUP ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")


gesture_cooldown = 1.0  # seconds
last_time = time.time()
last_gesture = None
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    annotated_image = image.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            X = scaler.transform([landmarks])
            y_pred = model.predict(X)
            gesture = label_encoder.inverse_transform(y_pred)[0]

            cv2.putText(annotated_image, f"Gesture: {gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            current_time = time.time()
            if gesture == last_gesture:
                if current_time - last_time >= gesture_cooldown:
                    trigger_action(gesture)
                    last_time = current_time
            else:
                trigger_action(gesture)
                last_gesture = gesture
                last_time = current_time       

    cv2.imshow('Real-Time Gesture Prediction', annotated_image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
