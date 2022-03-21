import os
import pickle
import random
import shutil
from pathlib import Path

# The number of images of a particular type to select.
num_imgs_of_type = 400

source_dir = os.path.join(Path.home(), 'Code', 'Imagenet', 'ILSVRC', 'Data',
                          'DET', 'train', 'ILSVRC2013_train')

img_dirs = os.listdir(source_dir)

image_dict = {}

# For each subdirectory (object type)
for img_dir in img_dirs:
    img_dir_path = os.path.join(source_dir, img_dir)
    imgs_in_dir = os.listdir(img_dir_path)

    if num_imgs_of_type > len(imgs_in_dir):
        imgs = imgs_in_dir
    else:
        imgs = random.sample(imgs_in_dir, num_imgs_of_type)

    image_dict[img_dir] = imgs

# Save the image names to a file.
with open('imagenet_images.pkl', 'wb') as file:
    pickle.dump(image_dict, file)

# Load the image names from the pickled file. 
with open('imagenet_images.pkl', 'rb') as file:
    image_dict = pickle.load(file)

# Specify (and create if neeed) the destination directory.
dest_dir = 'imagenet'
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

# Copy all files to the destination directory.
for folder in image_dict:
    for img_name in image_dict[folder]:
        img_path = os.path.join(source_dir, folder, img_name)
        shutil.copy(img_path, dest_dir)
