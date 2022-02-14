"""
Custom callbacks.
"""

import os
import tensorflow as tf

def save_weights(checkpoint_path):
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_total_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    return cp_callback
