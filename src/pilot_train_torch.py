import numpy as np
import os
import torch
import copy
from pilot_utils import load_image, show_density_map
from pilot_utils import load_batch, conv_model, TorchUNetModel
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# class ScatterMNISTDataset(Dataset),
#   def __init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TorchUNetModel(10).to(device)

# build a model
def train_model(model, optimiser, scheduler, batch_size, epochs):     
  # get the data
  n= 4500
  n_test = 1500
  dat = load_batch(12000)
  x_train = dat[0][:9000]
  x_test = dat[0][9000:]
  y_train = dat[1][:9000]
  y_test = dat[1][9000:]
  best_loss = 1e10
  best_model_wts = copy.deepcopy(model.state_dict())

  for epoch in range(epochs):
    print(f"Epoch ({epoch+1}/{epochs})")
    print()

    for phase in (["train", "val"]):
      print(phase)
      if phase == "train":
        for param_group in optimiser.param_groups:
          print("LR", param_group['lr'])
        
        model.train()
      else:
        model.eval()

      history = {}
      epoch_samples = 0

      if phase == "train":
        for batch in range((n//batch_size)-1):
          inputs = torch.from_numpy(x_train[batch*batch_size:batch_size*(batch+1)]).float().to(device)
          ground_truth = torch.from_numpy(y_train[batch*batch_size:batch_size*(batch+1)]).float().to(device)

          optimiser.zero_grad()

          with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            #print(outputs.shape, ground_truth.shape)
            loss = torch.nn.functional.mse_loss(outputs, ground_truth)
            
            loss.backward()
            optimiser.step()
            

          epoch_samples += batch_size

          history["loss"] = loss.item()
        scheduler.step()
        print(loss)
        print(history, epoch_samples, phase)
        epoch_loss = history["loss"] / epoch_samples
      if phase == "val":
        for test_batch in range((n_test//batch_size)-1):
          inputs = torch.from_numpy(x_test[test_batch*batch_size:batch_size*(test_batch+1)]).float().to(device)
          ground_truth = torch.from_numpy(y_test[test_batch*batch_size:batch_size*(test_batch+1)]).float().to(device)

          optimiser.zero_grad()

          with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, ground_truth)

          epoch_samples += n_test//batch_size

          history["loss"] = loss.item()

        print(history, epoch_samples, phase)
        epoch_loss = history["loss"] / epoch_samples
      
      
      if phase == "val" and epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "best_model.pth")

  torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'best_model_wts': best_model_wts
            }, "model.pt")

  print(f"Best validation loss: {best_loss}")
  model.load_state_dict(best_model_wts)
  return model


# compile the model
optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
lr_schedule = lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
# model = train_model(model, optimiser, lr_schedule, 1, 10)
# model.eval()

# checkpoint = torch.load("model.pt")
# model.load_state_dict(checkpoint['model_state_dict'])

model.load_state_dict(torch.load("best_model.pth"))

dat = load_batch(10)

inputs = torch.from_numpy(dat[0]).float().to(device)
outputs = model(inputs)

def show_n_example(n):
  for i in range(n):
    print(f"Image Number {i+1}:")
    print("-"*10)
    print(f"True image count: {round(sum(sum(sum(dat[1][i]))) / 100)}")
    print(f"Predicted image count: {round(sum(sum(sum(outputs.cpu().detach().numpy()[i]))) / 100)}")
    for j in range(10):
      print("-"*10)
      print(f"Count for Digit {j}: True {round(sum(sum(dat[1][i][j])) / 100)}, Predicted: {(sum(sum(outputs.cpu().detach().numpy()[i][j])) / 100):.2f}")
      print(f"Confidence Error (True Count - Predicted Count): {(sum(sum(dat[1][i][j])) / 100 - sum(sum(outputs.cpu().detach().numpy()[i][j])) / 100):.2f}")
      if i == 1 and j == 9:
        show_density_map(inputs[i], outputs[i][j])
    
    show_density_map(inputs[i], sum(outputs[i]))
