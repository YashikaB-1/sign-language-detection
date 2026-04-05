import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

dataset_path = "dataset/asl_alphabet_train"

X = []
y = []

labels = sorted(os.listdir(dataset_path))

for label_id, label in enumerate(labels):

    folder = os.path.join(dataset_path, label)

    print("Processing:", label)

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                X.append(landmarks)
                y.append(label_id)

print("Total samples extracted:", len(X))

np.save("dataset/X_landmarks.npy", np.array(X))
np.save("dataset/y_labels.npy", np.array(y))