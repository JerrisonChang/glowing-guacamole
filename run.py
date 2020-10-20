import tensorflow as tf
import config
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Build the RNN model
def build_model(units, input_dim, output_size, LSTM=True):
    # Builds model using LSTM RNN
    if LSTM:
        # The LSTM layer with default options
        layer = keras.layers.LSTM(units, input_shape=(None, input_dim))

    #Builds model using simple RNN
    else:
        # The RNN layer with default options
        layer = keras.layers.RNN(keras.layers.RNNCell(units), input_shape=(None, input_dim))

    model = keras.models.Sequential(
        [
            layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )
    return model

x, y = config.load_dataset(open('All_data.csv'))

batch_size = x.shape[0]
# Each batch is a tensor of shape (students, questions, skill).
# Each input sequence will be of size (question, skill).
input_dim = x.shape[2]

units = batch_size
output_size = 2  # labels are from 0 to 1

model = build_model(units, input_dim, output_size, LSTM=True)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)

model.fit(x, y, validation_split=0.2, batch_size=batch_size, epochs=1)
