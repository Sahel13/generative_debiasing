import os
import argparse

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from models.classifier import construct_classifier
from utils.callbacks import save_weights

# Use only a specific GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir',
                    required=True,
                    help="The name of the output directory.")
args = parser.parse_args()
output_dir = args.output_dir

# Hyper-parameters
batch_size = 128
learning_rate = 5e-4
num_epochs = 2
input_dim = (128, 128, 3)

# Load data.
data_folder = os.path.join('datasets', 'combined_dataset')

def get_data(source_dir):
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_data = data_gen.flow_from_directory(
        source_dir,
        target_size=input_dim[:2],
        batch_size=batch_size,
        shuffle=True,
        classes=['imagenet', 'celeba'],
        class_mode='binary',
        follow_links=True,
        subset='training'
    )
    val_data = data_gen.flow_from_directory(
        source_dir,
        target_size=input_dim[:2],
        batch_size=batch_size,
        shuffle=True,
        classes=['imagenet', 'celeba'],
        class_mode='binary',
        subset='validation',
        follow_links=True
    )
    return train_data, val_data

train_data, val_data = get_data(data_folder)

# Load and compile the model.
model = construct_classifier()
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

# Specify path to save model weights.
checkpoint_path = os.path.join('results', 'classifier', output_dir, 'cp.ckpt')

# Callback to save weights after each epoch.
cp_callback = save_weights(checkpoint_path, monitor='val_loss')

# Callback for early stopping.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3)

# Train the model.
model.fit(
    x=train_data,
    epochs=num_epochs,
    validation_data=val_data,
    callbacks=[early_stop, cp_callback]
)
