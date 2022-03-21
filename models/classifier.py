"""
References for implementation:
    https://github.com/aamini/introtodeeplearning/blob/master/lab2/Part2_Debiasing.ipynb
"""

import tensorflow as tf
import functools
from tensorflow.keras import layers

n_filters = 12  # base number of convolutional filters


def construct_classifier(n_outputs=1):
    """
    Construct a face classifier network.
    """
    Conv2D = functools.partial(layers.Conv2D, padding='same', activation='relu')
    BatchNormalization = layers.BatchNormalization
    Flatten = layers.Flatten
    Dense = functools.partial(layers.Dense, activation='relu')

    model = tf.keras.Sequential([
        Conv2D(filters=1*n_filters, kernel_size=5,  strides=2),
        BatchNormalization(),

        Conv2D(filters=2*n_filters, kernel_size=5,  strides=2),
        BatchNormalization(),

        Conv2D(filters=4*n_filters, kernel_size=3,  strides=2),
        BatchNormalization(),

        Conv2D(filters=6*n_filters, kernel_size=3,  strides=2),
        BatchNormalization(),

        Flatten(),
        Dense(512),
        Dense(n_outputs, activation=None),
    ])

    return model
