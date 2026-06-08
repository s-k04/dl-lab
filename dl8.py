import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout
)

# AlexNet-style model
model = Sequential([
    Conv2D(
        96,
        (11, 11),
        strides=4,
        activation='relu',
        input_shape=(227, 227, 3)
    ),

    BatchNormalization(),
    MaxPooling2D((3, 3), strides=2),

    Conv2D(
        256,
        (5, 5),
        padding='same',
        activation='relu'
    ),

    BatchNormalization(),
    MaxPooling2D((3, 3), strides=2),

    Conv2D(
        384,
        (3, 3),
        padding='same',
        activation='relu'
    ),

    Conv2D(
        384,
        (3, 3),
        padding='same',
        activation='relu'
    ),

    Conv2D(
        256,
        (3, 3),
        padding='same',
        activation='relu'
    ),

    MaxPooling2D((3, 3), strides=2),

    Flatten(),

    Dense(4096, activation='relu'),
    Dropout(0.5),

    Dense(4096, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Dummy training data
x = np.random.rand(2, 227, 227, 3).astype('float32')

y = tf.keras.utils.to_categorical(
    [0, 1],
    num_classes=10
)

# Quick training run
model.fit(
    x,
    y,
    epochs=1,
    verbose=1
)
