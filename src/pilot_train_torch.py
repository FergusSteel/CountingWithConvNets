import wandb
import numpy as np
import os
import torch
import copy
from pilot_utils import load_image, show_density_map
from pilot_utils import load_batch, conv_model, TorchUNetModel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TorchUNetModel(10).to(device)

# build a model
def train_model(model, optimiser, scheduler, epochs):     
  # get the data
  dat = load_batch(5)
  x_train = dat[0][:4]
  x_test = dat[0][4:]
  y_train = dat[1][:4]
  y_test = dat[1][4:]
  best_loss = 1e10
  best_model_wts = copy.deepcopy(model.state_dict())

  for epoch in range(epochs):
    print(f"Epoch ({epoch}/{epochs})")
    print()

    for phase in ["train", "validation"]:
      if phase == "train":
        scheduler.step()
        for param_group in optimiser.param_groups:
          print("LR", param_group['lr'])
        
        model.train()
      else:
        model.eval()

      history = {}
      epoch_samples = 0

      if phase == "train":
        inputs = torch.from_numpy(x_train).float().to(device)
        ground_truth = torch.from_numpy(y_train).float().to(device)

        optimiser.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
          outputs = model(inputs)
          loss = torch.nn.functional.mse_loss(outputs, ground_truth)
          loss.backward()
          optimiser.step()

          epoch_samples += inputs.size(0)

          history["loss"] = loss.item()

          print(history, epoch_samples, phase)
          epoch_loss = history["loss"] / epoch_samples
      elif phase == "val":
        inputs = torch.from_numpy(x_test).float().to(device)
        ground_truth = torch.from_numpy(y_test).float().to(device)

        optimiser.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
          outputs = model(inputs)
          loss = torch.nn.functional.mse_loss(outputs, ground_truth)

          epoch_samples += inputs.size(0)

          history["loss"] = loss.item()

          print(history, epoch_samples, phase)
          epoch_loss = history["loss"] / epoch_samples
      
      
      if phase == "val" and epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "best_model.pth")

  print(f"Best validation loss: {best_loss}")
  model.load_state_dict(best_model_wts)
  return model


# compile the model
optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
lr_schedule = lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
model = train_model(model, optimiser, lr_schedule, 5)
model.eval()


dat = load_batch(5)
inputs = torch.from_numpy(dat[0]).float().to(device)
outputs = model(inputs)

for i in range(5):
  show_density_map(inputs[i], outputs[i])
  print(sum(sum(sum(outputs[i]))))


# model.load_weights
# https://openaccess.thecvf.com/content_ECCV_2018/papers/Haroon_Idrees_Composition_Loss_for_ECCV_2018_paper.pdf training hyperparams from here
# a = model.predict(load_image('../data/train/x/0.png'))
# show_density_map(0, a)

# [optional] finish the wandb run, necessary in notebooks
#wandb.finish()