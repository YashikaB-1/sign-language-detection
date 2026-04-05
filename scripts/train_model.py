import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# LOAD DATA
X = np.load("dataset/X_sequences.npy")
y = np.load("dataset/y_sequences.npy")


# SHUFFLE DATASET
indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]

# TRAIN TEST SPLIT

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Train shape:",X_train.shape)
print("Test shape:",X_test.shape)


# ATTENTION LAYER

class Attention(layers.Layer):

    def build(self,input_shape):

        self.W = self.add_weight(
            shape=(input_shape[-1],1),
            initializer="random_normal",
            trainable=True
        )

    def call(self,inputs):

        score = tf.nn.tanh(tf.matmul(inputs,self.W))
        weights = tf.nn.softmax(score,axis=1)

        context = weights * inputs
        context = tf.reduce_sum(context,axis=1)

        return context


# MODEL

model = keras.Sequential([

    layers.LSTM(128,return_sequences=True,input_shape=(30,189)),

    layers.Dropout(0.3),

    layers.LSTM(64,return_sequences=True),

    layers.Dropout(0.3),

    Attention(),

    layers.Dense(64,activation="relu"),

    layers.BatchNormalization(),

    layers.Dense(len(np.unique(y)),activation="softmax")
])


# COMPILE

model.compile(

    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# CALLBACKS

early_stop = keras.callbacks.EarlyStopping(

    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# TRAIN
history = model.fit(

    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test,y_test),
    callbacks=[early_stop]
)


# SAVE MODEL

model.save("models/sign_model.keras")

print("Training complete.")