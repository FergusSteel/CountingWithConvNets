# This script needs these libraries to be installed: 
#   tensorflow, numpy

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import random
import numpy as np
import tensorflow as tf

import os

from pilot_utils import load_image, show_density_map
from pilot_utils import load_batch, conv_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Start a run, tracking hyperparameters
run = wandb.init(
    # set the wandb project where this run will be logged
    project="pilot-model",
)




# get the data
dat = load_batch(1000)
x_train = dat[0][:800]
x_test = dat[0][800:]
y_train = dat[1][:800]
y_test = dat[1][800:]


# build a model
model = conv_model()

# compile the model
model.compile(optimizer="adam",
              loss="mean_squared_error",
              metrics=["accuracy",]
              )

# artifact = run.use_artifact('fergus-k-steel/pilot-model/run_bvtqj9vm_model:v2', type='model')
# artifact_dir = artifact.download()

# model.load_weights
# https://openaccess.thecvf.com/content_ECCV_2018/papers/Haroon_Idrees_Composition_Loss_for_ECCV_2018_paper.pdf training hyperparams from here
history = model.fit(x=x_train, y=y_train,
                    epochs=5,
                    batch_size=20,
                    validation_data=(x_test, y_test),
                    callbacks=[
                      WandbMetricsLogger(log_freq="epoch"),
                      WandbModelCheckpoint("models")
                    ])

model.save(os.path.join(wandb.run.dir, "model.h5"))
wandb.save('model.h5')
wandb.save('../logs/*ckpt*')
wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

a = model.predict(load_image('../data/train/x/0.png'))
show_density_map(0, a)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()