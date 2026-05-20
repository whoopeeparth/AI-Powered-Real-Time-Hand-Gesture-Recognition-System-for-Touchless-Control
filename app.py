import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

model = tf.keras.models.load_model("gesture_model_final.keras")

classes = ['fist','index','ok','palm','thumb']

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

IMG_SIZE = 160

pred_buffer = deque(maxlen=12)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    h, w, _ = frame.shape

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

        pred = model.predict(img,verbose=0)[0]

        pred[1] *= 1.25   
        pred[3] *= 1.25   

        pred_buffer.append(pred)

        avg_pred = np.mean(pred_buffer, axis=0)

        class_id = np.argmax(avg_pred)
        confidence = avg_pred[class_id]

        if confidence > 0.55:
            text = f"{classes[class_id]} {confidence:.2f}"

    else:
        pred_buffer.clear()

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

cap.release()
cv2.destroyAllWindows()