import numpy as np
from subprocess import call

kl_weight_values = np.arange(start=1e-5, stop=1e-3, step=2e-5)

for i in range(len(kl_weight_values)):
    call('python train_vae.py' + f' -k {kl_weight_values[i]}'
         + f' -o celeb_{i:02d}', shell=True)
