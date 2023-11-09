#Load tensorflow model
import wandb
import numpy as np

import tensorflow as tf

from pilot_utils import conv_model, show_density_map, load_image

best_model = wandb.restore('model.h5', run_path="fergus-k-steel/pilot-model/yvqacsdb")

model = conv_model()

# use the "name" attribute of the returned object if your framework expects a filename, e.g. as in Keras
model.load_weights(best_model.name)

a = model.predict(load_image('../data/train/x/0.png'))
print(a.shape)


