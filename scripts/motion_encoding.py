import numpy as np

# Load landmark dataset
X = np.load("dataset/X_landmarks.npy")

features = []

prev_position = None
prev_velocity = None

for position in X:

    # Compute velocity
    if prev_position is None:
        velocity = np.zeros_like(position)
    else:
        velocity = position - prev_position

    # Compute acceleration
    if prev_velocity is None:
        acceleration = np.zeros_like(position)
    else:
        acceleration = velocity - prev_velocity

    # Combine features
    combined = np.concatenate((position, velocity, acceleration))

    features.append(combined)

    prev_position = position
    prev_velocity = velocity

features = np.array(features)

print("New feature shape:", features.shape)

np.save("dataset/X_features.npy", features)