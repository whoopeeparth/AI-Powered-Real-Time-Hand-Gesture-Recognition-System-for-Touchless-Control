# 🎮 Real-Time Hand Gesture Controlled Game

This project implements a **real-time hand gesture recognition system** using deep learning and computer vision to control a game. The system detects hand gestures through a webcam and maps them to keyboard actions for interactive gameplay.

---

## 🚀 Features

* Real-time hand gesture detection using **MediaPipe**
* Deep learning model based on **MobileNetV2**
* Smooth and optimized gesture prediction
* Gesture-based control for games (accelerate / reverse)
* Lightweight and fast inference for real-time performance

---

## 🧠 Model Details

* **Architecture:** MobileNetV2 (Transfer Learning)
* **Input Size:** 160 × 160 RGB images
* **Classes:**

  * Fist
  * Palm
  * Index
  * OK
  * Thumb

---

## 📊 Model Performance

* **Validation Accuracy:** **92%**
* Evaluated using:

  * Accuracy & Loss curves
  * Confusion Matrix
  * Classification Report (Precision, Recall, F1-score)

---

## 📂 Project Structure

```
gesture/
│── dataset/                     # Training images (5 gesture classes)
│── gesture_model_final.keras   # Trained model
│── train_model.ipynb           # Model training notebook
│── run_game2.py                # Real-time gesture control script
│── requirements.txt            # Dependencies
│── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Python

Make sure you are using:

```
Python 3.10.11
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the Application

```bash
python run_game2.py
```

---

## 🎮 How to Use (Game Control)

1. After running the script, **your camera will turn ON**
2. Sit in **proper lighting conditions** for better detection
3. Open your game (e.g., Hill Climb Racing)
4. **Click on the game window to focus it**
5. Keep the **camera window visible on the side** so you can see your hand
6. Now focus on the game and control it using gestures:

* ✋ **Palm → Accelerate**
* ✊ **Fist → Brake / Reverse**
* ❌ **No gesture → Stop**

👉 Keep your hand inside the green box for best results.

---

## 🖥️ Requirements

* Webcam
* Python 3.10.11
* Minimum 8GB RAM recommended

---

## ⚠️ Notes

* Ensure proper lighting for accurate detection
* Keep hand inside the detection region (green box)
* Performance may vary based on system hardware

---

## 🔮 Future Improvements

* Cross-platform compatibility (Windows + macOS)
* Real-time performance metrics (FPS & latency)
* Support for more gestures
* Integration with multiple games

---

## 👨‍💻 Author

Developed as a real-time AI-based gesture control system using deep learning and computer vision.

---
