# Dataset Debiasing Experiments with Generative Models

This repository contains code to conduct dataset debiasing experiments using generative models. There are many reasons as to why a machine learning model may be biased (for a comprehensive survey see link), but one of the most common ones is due to a biased dataset.

This is a work in progress, and the objectives and details of the experiments are continuously evolving. As of now, I'm working on the following two aspects:
Given a biased dataset,
1. use variational auto-encoders (VAEs) to find what is missing (more specifically, the under-represented features).
2. use generative adversarial networks (GANs) to generate what is missing (duplicate images with the under-represented features).

## Setup

### Using Pip
Create a python virtual environment and install the latest TensorFlow version. Then run
```
$ pip install -r requirements.txt
$ pip install -e .
```

### Using Docker
Ensure [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) is installed for GPU support.

Change the USER\_NAME, USER\_ID and GROUP\_ID build time variables inside the Dockerfile corresponding to your user (use `id -u` and `id -g` to find the USER\_ID and GROUP\_ID respectively).

## Experiments

### Step 1
Train the variational autoencoder.
```
$ python train_vae.py -o output_dir
```

Now run the notebook `notebooks/02_create_minority_dataset.ipynb` to create a file containing the names and attributes of images that constitute the minority dataset.
