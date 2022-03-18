import os
import math
from tqdm import tqdm
from PIL import Image

# Code to remove images that are smaller than 128 pixels along either axis.

max_allowed_size = 512

max_width = 0
min_width = math.inf

num_files_deleted = 0

for img_name in tqdm(os.listdir('imagenet')):
    filepath = os.path.join('imagenet', img_name)
    with Image.open(filepath) as img:
        width, height = img.size

    if (width < 128 or height < 128 or
            width > max_allowed_size or height > max_allowed_size):
        os.remove(filepath)
        num_files_deleted += 1
        continue

    max_width = max(width, max_width)
    min_width = min(width, min_width)

print(f"Min-width = {min_width}")
print(f"Max-width = {max_width}")
print(f"Deleted {num_files_deleted} files.\n")
