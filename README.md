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
[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) is needed for GPU support.

Change the USER\_NAME, USER\_ID and GROUP\_ID build time variables inside the Dockerfile corresponding to your user (use `id -u` and `id -g` to find your USER\_ID and GROUP\_ID respectively).

## The debiasing process

### Step 1
Train the variational autoencoder. The network architecture is given in `models/vae.py`.
```
$ python train_vae.py -o output_dir
```
Run the notebook `notebooks/01_visualize_vae_output.ipynb` to visualize training results.

### Step 2
Use the VAE to extract a minority dataset.
```
$ python extract_minority_dataset.py -w checkpoint_dir -d destination_dir
```
The size of this minority dataset (as a percentage of the original) can be adjusted within the code. Run the notebook `notebooks/02_inspect_minority_dataset.ipynb` to understand the different properties of the minority dataset thus created.

### Step 3
Train a generative adversarial network on the minority dataset that we just created. I am using [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) made available by [Nvidia Research Projects](https://github.com/NVlabs).