import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import math
import torch
import os

def display_capsule_grid(capsule_out):
    capsule_out = capsule_out.cpu().detach().numpy()
    num_caps = capsule_out.shape[1]
    caps_dimension = capsule_out.shape[2]
    capsule_out = capsule_out.squeeze(0)

    for capsule_index in range(1):    
        for img in capsule_out[capsule_index][:1]:
            fig, axs = plt.subplots()
            print(img.shape)
            axs.imshow(img)
            axs.axis('off')

            plt.tight_layout()
            plt.show()
            
