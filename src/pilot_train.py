# This script needs these libraries to be installed: 
#   tensorflow, numpy

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import random
import numpy as np
import tensorflow as tf

from pilot_utils import load_batch, conv_model



# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="pilot-model",

    # track hyperparameters and run metadata with wandb.config
)

# [optional] use wandb.config as your config
config = wandb.config

# get the data
dat = load_batch(10000)
x_train = dat[0][:8000]
x_test = dat[0][8000:]
y_train = dat[1][:8000]
y_test = dat[1][8000:]


# build a model
model = conv_model()

# compile the model
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy",]
              )

# https://openaccess.thecvf.com/content_ECCV_2018/papers/Haroon_Idrees_Composition_Loss_for_ECCV_2018_paper.pdf training hyperparams from here
history = model.fit(x=x_train, y=y_train,
                    epochs=70,
                    batch_size=16,
                    validation_data=(x_test, y_test),
                    callbacks=[
                      WandbMetricsLogger(log_freq="epoch"),
                      WandbModelCheckpoint("models")
                    ])

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()