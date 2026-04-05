import cv2
import mediapipe as mp
import numpy as np


# MEDIAPIPE SETUP
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils


# WEBCAM
cap = cv2.VideoCapture(0)

prev_landmarks = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    valid_frame = False

    if results.multi_hand_landmarks and results.multi_handedness:

        confidence = results.multi_handedness[0].classification[0].score

        # -----------------------------
        # LOW CONFIDENCE FILTER
        # -----------------------------
        if confidence < 0.6:
            cv2.putText(
                frame,
                "Low Confidence Frame",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

        else:

            for hand_landmarks in results.multi_hand_landmarks:

                current = []

                for lm in hand_landmarks.landmark:
                    current.extend([lm.x, lm.y, lm.z])

                current = np.array(current)

               
                # MOTION FILTER
            
                if prev_landmarks is not None:

                    motion = np.linalg.norm(current - prev_landmarks)

                    if motion < 0.01:

                        cv2.putText(
                            frame,
                            "Low Motion Frame",
                            (10,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2
                        )

                    else:

                        valid_frame = True

                prev_landmarks = current

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

   
    # VALID FRAME
   
    if valid_frame:

        cv2.putText(
            frame,
            "Valid Frame",
            (10,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.imshow("Frame Filtering", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()