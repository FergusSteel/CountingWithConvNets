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
import math

class MSELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSELoss, self).__init__()
        self.lf = nn.MSELoss()

    def forward(self, y_pred, y_true, smooth=1e-5):
        loss = torch.tensor(0.0).cuda()
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        for cat in range(10):
            loss += (self.lf(y_pred[cat], y_true[cat])) 
        return loss
lf = MSELoss()


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


def compute_acc(predict,target, digit_class):
    y_pred = predict.squeeze()
    y_true = target.squeeze()
    true_count = sum(sum(y_true[digit_class]))
    pred_count = sum(sum(y_pred[digit_class]))
    err = abs(true_count - pred_count)
    return err

def train_epoch(capsmodel, model, loader,optimizer, epoch, n_epochs, n_classes):
    # Evaluation Metrics
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    model.train()
    capsmodel.to(device)
    end = time.time()
    for batch_index, data in enumerate(loader):
        
        inputs = data["image"].float().to(device)
        #print(inputs.shape)
        inputs.unsqueeze_(1)
        target = data["dmap"].float().to(device)
        
        with torch.no_grad():
            _, CapsOut = capsmodel(inputs)

        print("caps", CapsOut.shape)
        recon = model(CapsOut)

        # Reconstruction loss
        loss = lf(recon, inputs.squeeze())
        #loss += 0.0005 * lf(output[1].squeeze(0), inputs.squeeze(0))


        batch_size = target.size(0)
        losses.update(loss.data, batch_size)

        optimizer.zero_grad()
        loss.backward()
    
        optimizer.step()
        # acc=compute_acc(output[1].detach(),target, 0)
        # accs.update(acc)

        batch_time.update(time.time() - end)
        end = time.time()
        res = '\t'.join([
            'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
            'Batch: [%d/%d]' % (batch_index, len(loader)),
            'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
            'Loss %.4f (%.4f)' % (losses.val, losses.avg),
        ])
        print(res)
    return batch_time.avg, losses.avg, accs.avg



def test_epoch(capsmodel,model,loader,epoch,n_epochs,n_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    
    capsmodel.to(device)

    # Model on eval mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch_index, data in enumerate(loader):
            inputs = data["image"].float().to(device)
            #print(inputs.shape)
            inputs.unsqueeze_(1)
            target = data["dmap"].float().to(device)
            with torch.no_grad():
                _, CapsOut = capsmodel(inputs)
            recon = model(CapsOut)

            # Reconstruction loss
            loss = lf(recon, inputs.squeeze())
            #loss += 0.0005 * lf(output[1].squeeze(0), inputs.squeeze(0))


            batch_size = target.size(0)
            losses.update(loss.data, batch_size)
            # acc = compute_acc(output[1], target,0)
            # accs.update(acc)
            batch_time.update(time.time() - end)
            end = time.time()
            res = '\t'.join([
                'Test',
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Batch: [%d/%d]' % (batch_index, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
            ])
            print(res)
    return batch_time.avg, losses.avg, accs.avg



def recontrain(args, capsmodel, model,train_loader, test_loader, decreasing_lr, n_classes):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    best_train_loss = np.inf
    for epoch in range(args.nepoch):
        scheduler.step()
        
        _, train_loss,train_acc = train_epoch(
            capsmodel=capsmodel,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=args.nepoch,
            n_classes=n_classes
        )
        _, test_loss,test_acc = test_epoch(
            capsmodel=capsmodel,
            loader=test_loader,
            model=model,
            epoch=epoch,
            n_epochs=args.nepoch,
            n_classes=n_classes
        )
        if train_loss < best_train_loss:
            best_train_loss=train_loss
            print('best_loss'+str(best_train_loss))
            torch.save(model.state_dict(),args.params_name)
        print(train_loss)
    