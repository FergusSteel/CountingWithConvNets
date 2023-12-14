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

    def forward(self, y_pred, y_true, threshold = 0.5, smooth=1e-5):
        #hard_dice = torch.FloatTensor()
        y_pred = (y_pred > threshold).type(torch.FloatTensor)
        y_true =(y_true > threshold).type(torch.FloatTensor)
        inse = torch.sum(torch.multiply(y_pred, y_true))
        l = torch.sum(y_pred)
        r = torch.sum(y_true)

        hard_dice = (2. * inse + smooth) / (l + r + smooth)
    
        return torch.mean(hard_dice).cuda()

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
    
class UNET3Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(UNET3Loss, self).__init__()
        
        # focal loss adapted from https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
        def focal_loss(y_true, y_pred):
            y_pred = y_pred.clamp(1e-7, 1 - 1e-7)
            pt = torch.where(y_true == 1, y_pred, 1 - y_pred)
            loss = -0.25 * (1 - pt) ** 2 * torch.log(pt)

            return torch.mean(loss)
        self.focal_loss = focal_loss
    
        def jaccard_loss(y_true, y_pred):
            y_true_flattened = torch.flatten(y_true)
            y_pred_flattened = torch.flatten(y_pred)
            intersection = torch.sum(y_true * y_pred)
            union = (torch.sum(y_true + y_pred)) - intersection
            loss = (intersection + 1e-7) / (union + 1e-7)

            return 1 - loss
        self.jaccard_loss = jaccard_loss
        
        # self simm from https://github.com/jorge-pessoa/pytorch-msssim
        self.ssim = pytorch_msssim.msssim


    def forward(self, y_pred, y_true, smooth=1e-5):
        # loss = torch.tensor([])
        loss = torch.tensor(0.0).cuda()
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        for cat in range(10):
            loss += 1 - self.ssim(y_pred[cat].unsqueeze(0).unsqueeze(1), y_true[cat].unsqueeze(0).unsqueeze(1), normalize="relu")
            loss += self.focal_loss(y_true[cat].unsqueeze(0).unsqueeze(1), y_pred[cat].unsqueeze(0).unsqueeze(1))
            loss += self.jaccard_loss(y_true[cat].unsqueeze(0).unsqueeze(1), y_pred[cat].unsqueeze(0).unsqueeze(1))
        
        return loss

#lf = CategoricalMSELoss()
#lf = MSSLoss()
#lf = pytorch_msssim.msssim
lf = UNET3Loss()


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
    predict = predict[0].cpu().detach()
    target = target[0].cpu().detach()
    for i in range(10):
        predict[i][predict[i]>=0.7]=1
        predict[i][predict[i]<=0.3]=0
        predict[i] = predict[i] != target[i]
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
        #print(inputs.shape)
        inputs.unsqueeze_(1)
        target = data["dmap"].float().to(device)
        output = model(inputs)
        #loss = compute_loss(output, target)
        #loss = torch.nn.functional.mse_loss(output[0], target.squeeze())
        loss = F.mse_loss(output[0], target.squeeze())
        # for i in range(10):
        #     t_pred_map = output[0][i][None, None, :]
        #     t_true_map = target[0][i][None, None, :]
        #     loss += (1 - lf(t_pred_map, t_true_map, normalize="relu"))
        #loss += lf(output[0], target.squeeze())

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
            loss = F.mse_loss(output[0], target.squeeze())
            # for i in range(10):
            #     t_pred_map = output[0][i][None, None, :]
            #     t_true_map = target[0][i][None, None, :]
            #     loss += (1 - lf(t_pred_map, t_true_map, normalize="relu"))
            #loss += lf(output[0], target.squeeze())

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
