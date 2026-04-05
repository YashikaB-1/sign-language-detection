import numpy as np


# LOAD FEATURES

X = np.load("dataset/X_features.npy")
y = np.load("dataset/y_labels.npy")

sequence_length = 30

sequences = []
labels = []


# CREATE TEMPORAL SEQUENCES
for i in range(len(X)):

    base_frame = X[i]

    sequence = []

    for j in range(sequence_length):

        # add small noise to simulate motion
        noise = np.random.normal(0,0.002,base_frame.shape)

        new_frame = base_frame + noise

        sequence.append(new_frame)

    sequences.append(sequence)
    labels.append(y[i])

X_seq = np.array(sequences)
y_seq = np.array(labels)

print("Sequence dataset shape:", X_seq.shape)

np.save("dataset/X_sequences.npy",X_seq)
np.save("dataset/y_sequences.npy",y_seq)