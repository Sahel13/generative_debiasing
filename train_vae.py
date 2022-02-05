import os
import math

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from models.vae import VAE
from utils.callbacks import save_weights

# Hyper-parameters
batch_size = 128
learning_rate = 5e-4
num_epochs = 50

# Load data.
data_folder = os.path.join('datasets', 'celeba')

num_images = 202599
steps_per_epoch = math.ceil(num_images/batch_size)

data_gen = ImageDataGenerator(rescale=1./255)
data_flow = data_gen.flow_from_directory(
    data_folder,
    target_size=(128, 128),
    batch_size=batch_size,
    shuffle=True,
    class_mode='input'
)

# Load and compile the model.
vae = VAE()
optimizer = tf.keras.optimizers.Adam(learning_rate)
vae.compile(optimizer)

# Specify path to save model weights.
checkpoint_path = os.path.join('results', 'vae', 'celeba', 'cp.ckpt')
cp_callback = save_weights(checkpoint_path)

# Train the model.
vae.fit(
    data_flow,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=[cp_callback]
)
