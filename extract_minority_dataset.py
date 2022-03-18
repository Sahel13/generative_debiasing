import os
import shutil
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--attr_file_path',
                    required=True,
                    help="The path to the attribute file of the minority \
                    datasest.")
parser.add_argument('-s', '--src',
                    default='datasets/celeba/img_align_celeba',
                    help="The source directory.")
parser.add_argument('-d', '--dest',
                    required=True,
                    help="The destination directory.")
args = parser.parse_args()

# Load list of minority images.
min_df_path = args.attr_file_path
min_df = pd.read_csv(min_df_path)
image_names = min_df.image_id

# Copy the minority images from source to destination.
src_dir = args.src
dest = args.dest
os.mkdir(dest)

for _, img_name in image_names.items():
    img_path = os.path.join(src_dir, img_name)
    shutil.copy(img_path, dest)
