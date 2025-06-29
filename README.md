# Gesture-Controlled Media and Presentation Controller 🎯🤘

This project uses computer vision and machine learning to recognise hand gestures via webcam and perform actions in **Windows Media Player**, **VLC**, or **Microsoft PowerPoint**. It's built with OpenCV, MediaPipe, and a custom-trained gesture classifier, with a custom dataset.

> ✨ Designed for real-time control of media and presentations using simple hand gestures without the use of a keyboard.

---

## 📸 Gestures & Mapped Actions

| Gesture     | PowerPoint        | Media Player / VLC      |
|-------------|-------------------|--------------------------|
| Thumbs Up   | Start slideshow   | Increase volume          |
| Thumbs Down | Exit slideshow    | Decrease volume          |
| Open Palm   | Black screen      | Play / Pause             |
| Peace       | Next slide        | Skip forward (30 sec)    |
| Fist        | Previous slide    | Rewind (30 sec)          |

---

## 🧠 Model Performance

The model was trained on 5 gestures, with over 100 - 150 samples each, achieving:

- **100% Accuracy** on test set (caution: data distribution is imbalanced)
- **Confusion Matrix** showed **no misclassifications**
- **F1-scores** all at **1.0**

Model saved as: `models/gesture_classifier.pkl`

---

## 🛠️ Project Structure

├── data/ # Collected gesture data (.csv)
├── trained model/ # Pre-trained classifier (.pkl)
├── requirements.txt # Dependencies
├── gesture_recognition_base.py # Script to collect custom gesture samples (Change the name of the file in line 11 for each new custom gesture dataset created)
├── train_gesture_model.py # Train & save gesture classification model on the custom dataset
├── gesture_actions.py # Maps gestures to app-specific actions
├── predict_gesture_live.py # Main script: runs live gesture detection + triggers actions
├── README.md

## ▶️ How to Run

1. **Clone the repo** and navigate into it:
   ```bash
   git clone https://github.com/roshantariq/Gesture-Control.git
   cd Gesture-Control
2. Create a virtual environment
   ```bash
   conda create -n gesture-env python=3.10
   conda activate gesture-env
3. Install dependencies:
   ```bash
   pip install -r requirements/requirements.txt
4. Run the main script:
   ```bash
   python predict_gesture_live.py

To collect new gesture samples and train your own classifier:
   python gesture_recognition_base.py  # Collect samples
   python train_gesture_model.py       # Train model & save to models/

## 💡 Motivation

This project combines my passion for real-world machine learning applications with computer vision. It's designed to show how intuitive human-computer interaction can be — and it works fully offline, in real-time.

Built as part of my portfolio to demonstrate applied ML skills in both classification and deployment.

## 🛠️ Tech Stack
Python 3.10
OpenCV
MediaPipe
scikit-learn
PyAutoGUI
pygetwindow
