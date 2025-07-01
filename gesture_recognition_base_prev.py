import cv2
import os
import csv
import numpy as np
import mediapipe as mp
from datetime import datetime

# CONFIGURATION
GESTURE_LABEL = "Fist"  # Change this when collecting each gesture
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, f"{GESTURE_LABEL}.csv")

# INITIALIZATION
os.makedirs(DATA_DIR, exist_ok=True)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Press 's' to save a sample, 'q' to quit.")

with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)

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

                # Extract and flatten landmark data
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Save data on key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    writer.writerow(landmarks + [GESTURE_LABEL])
                    print(f"Saved sample for label: {GESTURE_LABEL} at {datetime.now().strftime('%H:%M:%S')}")

        cv2.imshow('Hand Gesture Collection', annotated_image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
