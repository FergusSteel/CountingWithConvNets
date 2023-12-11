# From https://github.com/ImMrMa/SegCaps.pytorch/tree/master
import os
import time
import sys

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import grad
import pytorch_msssim

import wandb
wandb.init(project='SegCaps-PyTorch')

## Custom Loss Function uses Dice-BinaryCrossEntropy Loss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        BCE = 0.0
        dice_loss = 0.0
        print(inputs.shape, targets.shape)
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        #flatten label and prediction tensors
        l = F.sigmoid(inputs)
        r = targets
            
        intersection = (l * r).sum()                            
        dice_loss += 1 - (2.*intersection + smooth)/(l.sum() + r.sum() + smooth)  
        BCE += F.binary_cross_entropy(l, r, reduction='mean')
        
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class MSSLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSSLoss, self).__init__()
        self.lf = pytorch_msssim.msssim

    def forward(self, y_pred, y_true, smooth=1e-5):
        # loss = torch.tensor([])
        loss = torch.tensor(0.0).cuda()
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        for cat in range(10):
            loss += self.lf(y_pred[cat].unsqueeze(0).unsqueeze(1), y_true[cat].unsqueeze(0).unsqueeze(1))
            print(loss)
        
        return 1 - torch.mean(loss)

#lf = CategoricalMSELoss()
#lf = MSSLoss()
lf = pytorch_msssim.msssim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_loss(output, target):
    class_loss = (target * F.relu(0.9 - output)+ 0.5 * (1 - target) * F.relu(output - 0.1)).mean()
    return class_loss

def compute_acc(predict,target):
    predict[predict>=0.7]=1
    predict[predict<=0.3]=0
    predict = predict != target
    acc = torch.sum(predict).float() / torch.numel(target.data)
    return acc

def train_epoch(model, loader,optimizer, epoch, n_epochs, ):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    model.train()
    end = time.time()
    for batch_index, data in enumerate(loader):
        
        inputs = data["image"].float().to(device)
        inputs.unsqueeze_(1)
        target = data["dmap"].float().to(device)
        

        output = model(inputs)
        #loss = compute_loss(output, target)
        #loss = torch.nn.functional.mse_loss(output[0], target.squeeze())
        loss = F.mse_loss(output[0], target.squeeze()) + (1 - lf(output, target, normalize="relu"))

        batch_size = target.size(0)
        losses.update(loss.data, batch_size)

        optimizer.zero_grad()
        loss.backward()
    
        optimizer.step()
        acc=compute_acc(output.detach(),target)
        accs.update(acc)
        batch_time.update(time.time() - end)
        end = time.time()
        res = '\t'.join([
            'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
            'Batch: [%d/%d]' % (batch_index, len(loader)),
            'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
            'Loss %.4f (%.4f)' % (losses.val, losses.avg),
            'Error %.4f (%.4f)' % (accs.val, accs.avg),
        ])
        print(res)
    return batch_time.avg, losses.avg  , accs.avg



def test_epoch(model,loader,epoch,n_epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # Model on eval mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch_index, data in enumerate(loader):
            inputs = data["image"].float().to(device)
            inputs.unsqueeze_(1)
            target = data["dmap"].float().to(device)
            output = model(inputs)

            #loss = compute_loss(output, target)
            #loss = torch.nn.functional.mse_loss(output, target)
            #loss = lf(output[0], target.squeeze())
            loss = F.mse_loss(output[0], target.squeeze()) + (1 - lf(output, target, normalize="relu"))

            batch_size = target.size(0)
            losses.update(loss.data, batch_size)
            acc = compute_acc(output, target)
            accs.update(acc)
            batch_time.update(time.time() - end)
            end = time.time()
            res = '\t'.join([
                'Test',
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Batch: [%d/%d]' % (batch_index, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (accs.val, accs.avg),
            ])
            print(res)
    wandb.log({'Test Loss': losses.avg, 'Test Accuracy': 1 - accs.avg})
    return batch_time.avg, losses.avg, accs.avg



def train(args, model,train_loader, test_loader, decreasing_lr, wd=0.0001, momentum=0.9):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    best_train_loss = 10
    for epoch in range(args.nepoch):
        scheduler.step()
        
        _, train_loss,train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=args.nepoch,
        )
        _, test_loss,test_acc = test_epoch(
            loader=test_loader,
            model=model,
            epoch=epoch,
            n_epochs=args.nepoch,
        )
        if best_train_loss>train_loss:
            best_train_loss=train_loss
            print('best_loss'+str(best_train_loss))
            torch.save(model.state_dict(),args.params_name)
        print(train_loss)
    
        wandb.log({'Epoch': epoch,'Train Loss': train_loss, 'Train Accuracy': train_acc})
