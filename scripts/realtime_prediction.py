import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# Load trained model
model = tf.keras.models.load_model("models/sign_model.keras", compile=False)

# Labels
labels = [
"A","B","C","D","E","F","G","H","I","J",
"K","L","M","N","O","P","Q","R","S","T",
"U","V","W","X","Y","Z","del","nothing","space"
]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

sequence = deque(maxlen=30)
prediction_buffer = deque(maxlen=10)

prev_position = None
prev_velocity = None

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted_letter = ""
    confidence = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            position = []

            for lm in hand_landmarks.landmark:
                position.extend([lm.x, lm.y, lm.z])

            position = np.array(position)

            # velocity
            if prev_position is None:
                velocity = np.zeros_like(position)
            else:
                velocity = position - prev_position

            # acceleration
            if prev_velocity is None:
                acceleration = np.zeros_like(position)
            else:
                acceleration = velocity - prev_velocity

            feature_vector = np.concatenate(
                (position, velocity, acceleration)
            )

            sequence.append(feature_vector)

            prev_position = position
            prev_velocity = velocity

            if len(sequence) == 30:

                input_data = np.array(sequence).reshape(1,30,189)

                prediction = model.predict(input_data, verbose=0)

                predicted_index = np.argmax(prediction)

                predicted_letter = labels[predicted_index]
                confidence = prediction[0][predicted_index]

                prediction_buffer.append(predicted_letter)

                predicted_letter = max(set(prediction_buffer), key=prediction_buffer.count)

    cv2.rectangle(frame,(10,10),(300,80),(0,0,0),-1)

    cv2.putText(
        frame,
        f"Letter: {predicted_letter}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence*100:.2f}%",
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()