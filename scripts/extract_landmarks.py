import cv2
import mediapipe as mp
import numpy as np
import os


# MEDIAPIPE HAND SETUP

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


# DATASET PATH

dataset_path = "dataset/asl_alphabet_train"

X = []
y = []

labels = sorted(os.listdir(dataset_path))

print("Total classes:", len(labels))


# PROCESS DATASET

for label_id, label in enumerate(labels):

    folder = os.path.join(dataset_path, label)

    if not os.path.isdir(folder):
        continue

    print("Processing:", label)

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # Convert to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

      
        # EXTRACT LANDMARKS
       
        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks = []

            for lm in hand_landmarks.landmark:

                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            X.append(landmarks)
            y.append(label_id)


# CONVERT TO NUMPY

X = np.array(X)
y = np.array(y)

print("Total samples extracted:", len(X))
print("Feature shape:", X.shape)


# SAVE DATASET

os.makedirs("dataset", exist_ok=True)

np.save("dataset/X_landmarks.npy", X)
np.save("dataset/y_labels.npy", y)

print("Landmarks dataset saved!")