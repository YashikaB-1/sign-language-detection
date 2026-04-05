from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque

app = Flask(__name__)

# -----------------------------
# CUSTOM ATTENTION LAYER
# -----------------------------

class Attention(layers.Layer):

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1],1),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):

        score = tf.nn.tanh(tf.matmul(inputs,self.W))
        weights = tf.nn.softmax(score,axis=1)

        context = weights * inputs
        context = tf.reduce_sum(context,axis=1)

        return context


# -----------------------------
# LOAD MODEL
# -----------------------------

model = tf.keras.models.load_model(
    "models/sign_model.keras",
    custom_objects={"Attention": Attention},
    compile=False
)

# -----------------------------
# LABELS
# -----------------------------

labels = [
"A","B","C","D","E","F","G","H","I","J",
"K","L","M","N","O","P","Q","R","S","T",
"U","V","W","X","Y","Z","del","nothing","space"
]

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

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


# -----------------------------
# VIDEO STREAM
# -----------------------------

def generate_frames():

    global prev_position, prev_velocity

    while True:

        success, frame = cap.read()

        if not success:
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
                    position.extend([lm.x,lm.y,lm.z])

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
                    (position,velocity,acceleration)
                )

                sequence.append(feature_vector)

                prev_position = position
                prev_velocity = velocity

                if len(sequence) == 30:

                    input_data = np.array(sequence).reshape(1,30,189)

                    prediction = model.predict(input_data,verbose=0)

                    predicted_index = np.argmax(prediction)

                    predicted_letter = labels[predicted_index]

                    confidence = prediction[0][predicted_index]

                    prediction_buffer.append(predicted_letter)

                    predicted_letter = max(
                        set(prediction_buffer),
                        key=prediction_buffer.count
                    )

                    cv2.putText(
                        frame,
                        f"{predicted_letter} : {confidence*100:.1f}%",
                        (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0,255,0),
                        3
                    )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -----------------------------
# FLASK ROUTES
# -----------------------------

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)