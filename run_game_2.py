import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time
import ctypes

# ===== KEY CONTROL =====
KEY_LEFT = 0x25
KEY_RIGHT = 0x27

def press_key(k):
    ctypes.windll.user32.keybd_event(k, 0, 0, 0)

def release_key(k):
    ctypes.windll.user32.keybd_event(k, 0, 2, 0)

# ===== MODEL =====
model = tf.keras.models.load_model("gesture_model_final.keras")
classes = ['fist','index','ok','palm','thumb']

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

IMG_SIZE = 160

# ===== PERFORMANCE CONTROL =====
FRAME_SKIP = 3          # 🔥 process every 3rd frame
frame_count = 0
FPS_LIMIT = 15
frame_time = 1 / FPS_LIMIT

# ===== SMOOTHING =====
pred_buffer = deque(maxlen=3)
smooth_pred = None
alpha = 0.7

# ===== STATE =====
current_key = None
last_gesture = None

while True:

    start_time = time.time()

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

    gesture = last_gesture  # keep previous gesture

    # ===== PROCESS ONLY SOME FRAMES =====
    if frame_count % FRAME_SKIP == 0:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and roi.size != 0:

            img = cv2.resize(roi,(IMG_SIZE,IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img,axis=0)

            pred = model.predict(img, verbose=0)[0]

            pred[1] *= 1.25
            pred[3] *= 1.25

            pred_buffer.append(pred)
            avg_pred = np.mean(pred_buffer, axis=0)

            if smooth_pred is None:
                smooth_pred = avg_pred
            else:
                smooth_pred = alpha * avg_pred + (1 - alpha) * smooth_pred

            class_id = np.argmax(smooth_pred)
            confidence = smooth_pred[class_id]

            if confidence > 0.6:
                gesture = classes[class_id]
                last_gesture = gesture

    # ===== CONTROL (VERY LIGHT) =====
    if gesture == "palm":
        if current_key != "right":
            release_key(KEY_LEFT)
            press_key(KEY_RIGHT)
            current_key = "right"

    elif gesture == "fist":
        if current_key != "left":
            release_key(KEY_RIGHT)
            press_key(KEY_LEFT)
            current_key = "left"

    else:
        if current_key is not None:
            release_key(KEY_LEFT)
            release_key(KEY_RIGHT)
            current_key = None

    # ===== UI =====
    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)

    if gesture:
        cv2.putText(frame, gesture, (xmin, ymin-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Gesture Recognition",frame)

    # ===== FPS LIMIT =====
    elapsed = time.time() - start_time
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
release_key(KEY_LEFT)
release_key(KEY_RIGHT)
cap.release()
cv2.destroyAllWindows()