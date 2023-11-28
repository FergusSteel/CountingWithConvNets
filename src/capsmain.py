# FROM https://github.com/ImMrMa/SegCaps.pytorch/tree/master

import os
import torch
from capsnet import SegCaps
from torch.utils.data import DataLoader
from torchdataset import *
import argparse
from capstrain import train
from convert_to_npz import *
from pilot_utils import show_density_map, load_batch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init(args):
    
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")
    os.environ['CUDA_VISIBLE_DEVICES'] =args.gpu
    print('{}:{}'.format('cuda',torch.cuda.is_available()))
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader = DataLoader(SpreadMNISTDataset(1000), batch_size=1, shuffle=True)
    model=SegCaps()
    model.cuda()
    model.to(device)
    
    decreasing_lr = list(map(int,args.dlr.split(',')))
    return  train_loader,model,decreasing_lr

def show_n_example(model, n):
    dat = load_batch(n)
    inputs = torch.from_numpy(dat[0]).float().to(device)
    outputs = model(inputs)
    show_density_map(inputs[0][0].cpu().detach().numpy(), outputs.cpu().detach().numpy()[0])


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Little Ma's train")
    parser.add_argument('--batch_size_train', type=int, default=2, help='input batch size for training (default: 160)')
    parser.add_argument('--batch_size_test', type=int, default=2, help='input batch size for testing (default: 80)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
    parser.add_argument('--gpu', default='0', help='index of gpus to use')
    parser.add_argument('--dlr', default='10,25', help='decreasing strategy')
    parser.add_argument('--model', default='segcaps-train', help='which model (default: xception)')
    parser.add_argument('--data_root', default='./', help='data_root (default: ./)')
    parser.add_argument('--nepoch', type=int,default=10, help='epochs (default: 200)')
    parser.add_argument('--seed', type=int,default='10', help='seed (default: 1)')
    parser.add_argument('--pretrain', type=int, default='0', help='pretrain (default: 1)')
    parser.add_argument('--data_name', default='train', help='data_name (default: train)')
    parser.add_argument('--params_name', default='segcaps.pkl', help='params_name (default: segcaps.pkl)')
    parser.add_argument('--load_params_name', default='segcapsv0.pkl', help='params_name (default: segcaps.pkl)')
    args = parser.parse_args()

    if args.pretrain == 1:
        model.load_state_dict(torch.load(args.load_params_name))
    else:
        print("Training")
        train_loader,model,decreasing_lr=init(args)
        train(args,model,train_loader,
                decreasing_lr,wd=0.0001, momentum=0.9)

    show_n_example(model, 5)
    print('hhh')
    print('Done!')

