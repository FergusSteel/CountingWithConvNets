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

class MSELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSELoss, self).__init__()
        self.lf = nn.MSELoss()

    def forward(self, y_pred, y_true, smooth=1e-5):
        # loss = torch.tensor([])
        loss = torch.tensor(0.0).cuda()
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        for cat in range(10):
            loss += (self.lf(y_pred[cat], y_true[cat])) 
            loss += (abs(sum(sum(y_true[cat])) - sum(sum(y_pred[cat]))) / 100_000)
            # Count loss is breaking 
            #loss *= abs(sum(sum(y_true[cat])) - sum(sum(y_pred[cat]))) * 0.001
            # disjoint_loss = torch.tensor(0.0).cuda()
            # for disjoint_cats in range(10):
            #     if disjoint_cats != cat:
            #         disjoint_loss += (self.lf(y_pred[cat], y_true[disjoint_cats]))
            # loss += (1/(disjoint_loss)) * 0.005
        return loss
    
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
        ssimloss = torch.tensor(0.0).cuda()
        focalloss = torch.tensor(0.0).cuda()
        jaccardloss = torch.tensor(0.0).cuda()
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        for cat in range(10):
            ssimloss += 1 - self.ssim(y_pred[cat].unsqueeze(0).unsqueeze(1), y_true[cat].unsqueeze(0).unsqueeze(1), normalize="relu")
            focalloss += self.focal_loss(y_true[cat].unsqueeze(0).unsqueeze(1), y_pred[cat].unsqueeze(0).unsqueeze(1))
            jaccardloss += self.jaccard_loss(y_true[cat].unsqueeze(0).unsqueeze(1), y_pred[cat].unsqueeze(0).unsqueeze(1))
        
        return (ssimloss / 10) + (focalloss / 10) + (jaccardloss / 10)

#lf = CategoricalMSELoss()
lf = MSELoss()
#lf = pytorch_msssim.msssim
# lf = UNET3Loss()
#lf = DiceBCELoss()


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
    true_count = sum(sum(y_true[digit_class])) / 100
    pred_count = sum(sum(y_pred[digit_class])) / 100
    err = abs(true_count - pred_count)
    return err

def train_epoch(model, loader,optimizer, epoch, n_epochs, n_classes):
    # Evaluation Metrics
    batch_time = AverageMeter()
    losses = AverageMeter()
    per_class_count_err = [AverageMeter() for i in range(n_classes)]
    accs = AverageMeter()

    model.train()
    end = time.time()
    for batch_index, data in enumerate(loader):
        
        inputs = data["image"].float().to(device)
        #print(inputs.shape)
        inputs.unsqueeze_(1)
        target = data["dmap"].float().to(device)
        output = model(inputs)
        print(output[1].shape)
        print(sum(target).shape)

        # Reconstruction loss
        loss = lf(output[0], target.squeeze())
        #loss += 0.0005 * lf(output[1].squeeze(0), inputs.squeeze(0))

        print(output[1].shape, inputs.shape)

        batch_size = target.size(0)
        losses.update(loss.data, batch_size)

        optimizer.zero_grad()
        loss.backward()
    
        optimizer.step()
        # acc=compute_acc(output[1].detach(),target, 0)
        # accs.update(acc)

        for digit_class in range(n_classes):
            per_class_count_err[digit_class].update(compute_acc(output[0].detach().cpu().numpy(), target.detach().cpu().numpy(), digit_class), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()
        res = '\t'.join([
            'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
            'Batch: [%d/%d]' % (batch_index, len(loader)),
            'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
            'Loss %.4f (%.4f)' % (losses.val, losses.avg),
            f'P.C. Accs {[(i, val) for i, val in enumerate([round(acc.avg,2) for acc in per_class_count_err])]}',
        ])
        print(res)
    return batch_time.avg, losses.avg, accs.avg



def test_epoch(model,loader,epoch,n_epochs,n_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    per_class_count_err = [AverageMeter() for i in range(n_classes)]

    # Model on eval mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch_index, data in enumerate(loader):
            inputs = data["image"].float().to(device)
            inputs.unsqueeze_(1)
            target = data["dmap"].float().to(device)
            output = model(inputs)

            # Reconstruction loss
            #loss = 0.005 * F.mse_loss(output[1], inputs.squeeze())
                # for i in range(10):
                #     t_pred_map = output[0][i][None, None, :]
                #     t_true_map = target[0][i][None, None, :]
                #     loss += (1 - lf(t_pred_map, t_true_map, normalize="relu"))
            loss = lf(output[0], target.squeeze())
            #loss += 0.0005 * lf(output[1].squeeze(0), inputs.squeeze(0))


            batch_size = target.size(0)
            losses.update(loss.data, batch_size)
            # acc = compute_acc(output[1], target,0)
            # accs.update(acc)
            for digit_class in range(n_classes):
                per_class_count_err[digit_class].update(compute_acc(output[0].detach().cpu().numpy(), target.detach().cpu().numpy(), digit_class), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()
            res = '\t'.join([
                'Test',
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Batch: [%d/%d]' % (batch_index, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                f'P.C. MAE {[(i, val) for i, val in enumerate([round(acc.avg,2) for acc in per_class_count_err])]}',
            ])
            print(res)
    wandb.log({'Test Loss': losses.avg, 'Test Accuracy': 1 - accs.avg})
    return batch_time.avg, losses.avg, accs.avg



def train(args, model,train_loader, test_loader, decreasing_lr, n_classes):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    best_train_loss = np.inf
    for epoch in range(args.nepoch):
        scheduler.step()
        
        _, train_loss,train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=args.nepoch,
            n_classes=n_classes
        )
        _, test_loss,test_acc = test_epoch(
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
    
        wandb.log({'Epoch': epoch,'Train Loss': train_loss, 'Train Accuracy': train_acc})
