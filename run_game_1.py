import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time
import ctypes

# ===== Windows Native Key Input (SAFE) =====
KEY_LEFT = 0x25
KEY_RIGHT = 0x27

def press_key(key):
    ctypes.windll.user32.keybd_event(key, 0, 0, 0)

def release_key(key):
    ctypes.windll.user32.keybd_event(key, 0, 2, 0)

# Load model
model = tf.keras.models.load_model("gesture_model_final.keras")
classes = ['fist','index','ok','palm','thumb']

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
IMG_SIZE = 160

# ===== Smoothing =====
pred_buffer = deque(maxlen=4)
smooth_pred = None
alpha = 0.6

# ===== Stability =====
last_gesture = None
gesture_start_time = 0
GESTURE_HOLD_TIME = 0.12

# ===== Key State =====
current_key = None

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    h, w, _ = frame.shape

    # ROI
    size = 300
    cx, cy = w//2, h//2
    xmin = cx - size//2
    xmax = cx + size//2
    ymin = cy - size//2
    ymax = cy + size//2

    roi = frame[ymin:ymax, xmin:xmax]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    text = "Waiting..."

    if results.multi_hand_landmarks and roi.size != 0:

        img = cv2.resize(roi,(IMG_SIZE,IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img,axis=0)

        pred = model.predict(img, verbose=0)[0]

        # Bias tweak (your original)
        pred[1] *= 1.25
        pred[3] *= 1.25

        # Buffer smoothing
        pred_buffer.append(pred)
        avg_pred = np.mean(pred_buffer, axis=0)

        # EMA smoothing
        if smooth_pred is None:
            smooth_pred = avg_pred
        else:
            smooth_pred = alpha * avg_pred + (1 - alpha) * smooth_pred

        # Instant override
        instant_id = np.argmax(pred)
        instant_conf = pred[instant_id]

        if instant_conf > 0.85:
            class_id = instant_id
            confidence = instant_conf
        else:
            class_id = np.argmax(smooth_pred)
            confidence = smooth_pred[class_id]

        # Stable decision
        if confidence > 0.6:
            gesture = classes[class_id]
            current_time = time.time()

            if gesture != last_gesture:
                gesture_start_time = current_time
                last_gesture = gesture

            if current_time - gesture_start_time > GESTURE_HOLD_TIME:

                text = f"{gesture} {confidence:.2f}"

                # ===== CONTROL =====
                if gesture == "palm":
                    if current_key != "right":
                        release_key(KEY_LEFT)
                        release_key(KEY_RIGHT)
                        press_key(KEY_RIGHT)
                        current_key = "right"

                elif gesture == "fist":
                    if current_key != "left":
                        release_key(KEY_RIGHT)
                        release_key(KEY_LEFT)
                        press_key(KEY_LEFT)
                        current_key = "left"

                else:
                    if current_key is not None:
                        release_key(KEY_LEFT)
                        release_key(KEY_RIGHT)
                        current_key = None

    else:
        pred_buffer.clear()
        smooth_pred = None
        last_gesture = None

        if current_key is not None:
            release_key(KEY_LEFT)
            release_key(KEY_RIGHT)
            current_key = None

    # Draw ROI
    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)

    cv2.putText(frame,
                text,
                (xmin, ymin-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Gesture Recognition",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
release_key(KEY_LEFT)
release_key(KEY_RIGHT)
cap.release()
cv2.destroyAllWindows()