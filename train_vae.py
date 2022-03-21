import os
import argparse
import tensorflow as tf

from models.vae import VAE
from utils.callbacks import save_weights
from utils.dataloader import load_vae_data

# Use only a specific GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--kl_weight',
                    default=1e-5,
                    type=float,
                    help="The kl_weight value to use.")
parser.add_argument('-o', '--output_dir',
                    required=True,
                    help="The name of the output directory.")
args = parser.parse_args()
kl_weight = args.kl_weight
output_dir = args.output_dir

# Hyper-parameters
batch_size = 128
learning_rate = 5e-4
num_epochs = 100
input_dim = (128, 128, 3)

# Load data.
data_folder = os.path.join('datasets', 'celeba')
train_data, val_data = load_vae_data(data_folder, input_dim, batch_size)

# Load and compile the model.
vae = VAE(kl_weight=kl_weight)
optimizer = tf.keras.optimizers.Adam(learning_rate)
vae.compile(optimizer)

# Specify path to save model weights.
checkpoint_path = os.path.join('results', 'vae', output_dir, 'cp.ckpt')

# Callback to save weights after each epoch.
cp_callback = save_weights(checkpoint_path, 'val_total_loss')

# Callback for early stopping.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_total_loss', patience=3)

# Train the model.
vae.fit(
    x=train_data,
    epochs=num_epochs,
    validation_data=val_data,
    callbacks=[early_stop, cp_callback]
)
