import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time


# LOAD MODEL

model = tf.keras.models.load_model("gesture_model_final.keras")

# Class labels
classes = ['fist', 'index', 'ok', 'palm', 'thumb']


# SETTINGS

IMG_SIZE = 160
CONF_THRESHOLD = 0.70
SWITCH_DELAY = 0.25
SMOOTH_ALPHA = 0.55
GUIDE_BOX = 320


# MEDIAPIPE

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)


# CAMERA

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# SMOOTHING VARIABLES

ema_pred = None
gesture_history = deque(maxlen=8)

stable_label = "Waiting..."
last_switch_time = time.time()


# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape


    # CENTER GUIDE BOX

    cx, cy = w // 2, h // 2

    gx1 = cx - GUIDE_BOX // 2
    gy1 = cy - GUIDE_BOX // 2
    gx2 = cx + GUIDE_BOX // 2
    gy2 = cy + GUIDE_BOX // 2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_text = stable_label
    box_color = (0, 255, 0)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]


        # Hand Bounding Box

        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        xmin = int(min(x_coords) * w) - 40
        xmax = int(max(x_coords) * w) + 40
        ymin = int(min(y_coords) * h) - 40
        ymax = int(max(y_coords) * h) + 40

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)


        # Check if hand inside center box

        hand_center_x = (xmin + xmax) // 2
        hand_center_y = (ymin + ymax) // 2

        inside_box = (
            gx1 < hand_center_x < gx2 and
            gy1 < hand_center_y < gy2
        )

        if inside_box:


            # Square ROI crop

            box_w = xmax - xmin
            box_h = ymax - ymin
            size = max(box_w, box_h)

            cx_box = (xmin + xmax) // 2
            cy_box = (ymin + ymax) // 2

            xmin2 = max(0, cx_box - size // 2)
            xmax2 = min(w, cx_box + size // 2)
            ymin2 = max(0, cy_box - size // 2)
            ymax2 = min(h, cy_box + size // 2)

            roi = frame[ymin2:ymax2, xmin2:xmax2]

            if roi.size != 0:

                # Preprocess

                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)

                pred = model.predict(img, verbose=0)[0]


                # Bias correction

                pred[1] *= 1.30   # Boost INDEX
                pred[4] *= 0.88   # Reduce THUMB dominance

                pred = pred / np.sum(pred)


                # EMA smoothing

                if ema_pred is None:
                    ema_pred = pred
                else:
                    ema_pred = (
                        SMOOTH_ALPHA * pred +
                        (1 - SMOOTH_ALPHA) * ema_pred
                    )

                class_id = np.argmax(ema_pred)
                confidence = ema_pred[class_id]

                if confidence > CONF_THRESHOLD:
                    label = classes[class_id]
                    gesture_history.append(label)


                    # Majority voting

                    most_common = max(
                        set(gesture_history),
                        key=gesture_history.count
                    )

                    count = gesture_history.count(most_common)

                    # Require 4/8 frames
                    if count >= 4:
                        now = time.time()

                        if (
                            most_common != stable_label and
                            now - last_switch_time > SWITCH_DELAY
                        ):
                            stable_label = most_common
                            last_switch_time = now

                display_text = f"{stable_label} ({confidence:.2f})"
                box_color = (0, 255, 0)

        else:
            display_text = "Place hand inside box"
            box_color = (0, 0, 255)
            gesture_history.clear()
            ema_pred = None


        # Blue hand tracking rectangle

        cv2.rectangle(
            frame,
            (xmin, ymin),
            (xmax, ymax),
            (255, 0, 0),
            2
        )

    else:
        display_text = "Show Hand"
        gesture_history.clear()
        ema_pred = None
        stable_label = "Waiting..."
        box_color = (0, 255, 255)


    # Draw Center Guide Box

    cv2.rectangle(
        frame,
        (gx1, gy1),
        (gx2, gy2),
        box_color,
        3
    )


    # Display Gesture Text

    cv2.putText(
        frame,
        display_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        box_color,
        2
    )

    cv2.imshow("Gesture Recognition", frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# CLEANUP

cap.release()
cv2.destroyAllWindows()