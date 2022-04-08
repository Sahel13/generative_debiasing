import os
import shutil
import argparse
import numpy as np
import pandas as pd
from models.vae import VAE
from utils.minority_dataset import load_celeba, MinorityDataset

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--wdir',
                    required=True,
                    help="The checkpoint directory for weights.")
args = parser.parse_args()

# Hyper-parameters
input_shape = (128, 128, 3)
batch_size = 256
latent_dim = 200

# Specify paths for data, pre-trained weights and results.
data_folder = os.path.join('datasets', 'celeba')
image_folder = os.path.join(data_folder, 'img_align_celeba')

checkpoint_path = os.path.join('results', 'vae', args.wdir, 'cp.ckpt')

# Load the model.
vae = VAE()
vae.load_weights(checkpoint_path)

# Load the data.
df = pd.read_csv(os.path.join(data_folder, 'list_attr_celeba.csv'))
data_flow = load_celeba(df, image_folder, input_shape[:2], batch_size)

# Create the minority dataset.
minority_data = MinorityDataset(data_flow, vae, latent_dim)

mean_array_path = os.path.join(os.path.dirname(checkpoint_path),
                               'mean_array.npy')

if not os.path.exists(mean_array_path):
    mean = minority_data.get_latent_mean()
    np.save(mean_array_path, mean)
else:
    mean = np.load(mean_array_path)

# Choose what percentage of images to use as minority.
minority_list = minority_data.get_sub_dataset(mean, bins=100, extremes=18)
percentage = minority_list.sum()/len(minority_list) * 100

output_folder = os.path.join(
    os.path.dirname(checkpoint_path), f'minority_{percentage:02.0f}')

dest_dir = os.path.join(output_folder, f'minority_{percentage:02.0f}_images')

os.mkdir(output_folder)
os.mkdir(dest_dir)
output_file = os.path.join(output_folder, 'minority_dataset.csv')

# Create and save the minority dataset to a new file.
minority_data.create_new_df(minority_list, df, output_file)

min_df = pd.read_csv(output_file)
image_names = min_df.image_id

for _, img_name in image_names.items():
    img_path = os.path.join(image_folder, img_name)
    shutil.copy(img_path, dest_dir)
