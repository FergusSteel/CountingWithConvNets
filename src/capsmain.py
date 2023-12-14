# FROM https://github.com/ImMrMa/SegCaps.pytorch/tree/master

import os
import torch
from capsnet import SegCaps
from torch.utils.data import DataLoader
from torchdataset import *
from torchdataset_masks import *
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
    # if args.pretrain == 1:
    #     train_loader = DataLoader(MaskedSpreadMNISTDataset(1), batch_size=args.batch_size_train, shuffle=True)
    #     test_loader = DataLoader(MaskedSpreadMNISTDataset(1, train=False), batch_size=args.batch_size_train, shuffle=True)
    # else:
    #     train_loader = DataLoader(MaskedSpreadMNISTDataset(args.n_images), batch_size=args.batch_size_train, shuffle=True)
    #     test_loader = DataLoader(MaskedSpreadMNISTDataset(int(args.n_images*0.25), train=False), batch_size=args.batch_size_train, shuffle=True)
    if args.pretrain == 1:
        train_loader = DataLoader(SpreadMNISTDataset(1), batch_size=args.batch_size_train, shuffle=True)
        test_loader = DataLoader(SpreadMNISTDataset(1, train=False), batch_size=args.batch_size_train, shuffle=True)
    else:
        train_loader = DataLoader(SpreadMNISTDataset(args.n_images), batch_size=args.batch_size_train, shuffle=True)
        test_loader = DataLoader(SpreadMNISTDataset(int(args.n_images*0.25), train=False), batch_size=args.batch_size_train, shuffle=True)
    model=SegCaps()
    model.cuda()
    model.to(device)
    
    decreasing_lr = list(map(int,args.dlr.split(',')))
    return  train_loader, test_loader, model, decreasing_lr

def show_n_example(model, n):
    dat = load_batch(n)
    inputs = torch.from_numpy(dat[0]).float().to(device)
    
    # for i in range(n):
    #     print(f"Image Number {i+1}:")
    #     print("-"*10)
    #     print(f"True image count: {round(sum(sum(sum(dat[1][i]))) / 100)}")
    #     print(f"Predicted image count: {round(sum(sum(sum(outputs.cpu().detach().numpy()[i]))) / 100)}")
    #     for j in range(10):
    #         print("-"*10)
    #         print(f"Count for Digit {j}: True {round(sum(sum(dat[1][i][j])) / 100)}, Predicted: {(sum(sum(outputs.cpu().detach().numpy()[i][j])) / 100):.2f}")
    #         print(f"Confidence Error (True Count - Predicted Count): {(sum(sum(dat[1][i][j])) / 100 - sum(sum(outputs.cpu().detach().numpy()[i][j])) / 100):.2f}")
    for i in range(n):
        outputs = model(inputs[i].unsqueeze(0)).float().to(device)
        # print(inputs.shape)
        # print(outputs.shape)
        for j in range(10):
             show_density_map(inputs[i][0].cpu().detach().numpy(), outputs[0][j].cpu().detach().numpy())



        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Little Ma's train")
    parser.add_argument('--batch_size_train', type=int, default=1, help='input batch size for training (default: 160)')
    parser.add_argument('--batch_size_test', type=int, default=1, help='input batch size for testing (default: 80)')
    parser.add_argument("--n_images", type=int, default=1000, help="Number of images to train on.")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--gpu', default='0', help='index of gpus to use')
    parser.add_argument('--dlr', default='10,25', help='decreasing strategy')
    parser.add_argument('--model', default='segcaps-train', help='which model (default: xception)')
    parser.add_argument('--data_root', default='./', help='data_root (default: ./)')
    parser.add_argument('--nepoch', type=int,default=1, help='epochs (default: 200)')
    parser.add_argument('--seed', type=int,default='10', help='seed (default: 1)')
    parser.add_argument('--pretrain', type=int, default='0', help='pretrain (default: 1)')
    parser.add_argument('--data_name', default='train', help='data_name (default: train)')
    parser.add_argument('--params_name', default='segcaps.pkl', help='params_name (default: segcaps.pkl)')
    parser.add_argument('--load_params_name', default='segcaps_good.pkl', help='params_name (default: segcaps.pkl)')
    args = parser.parse_args()

    if args.pretrain == 1:
        train_loader,test_loader,model,decreasing_lr=init(args)
        model.load_state_dict(torch.load(args.load_params_name))
    else:
        print("Training")
        train_loader,test_loader,model,decreasing_lr=init(args)
        train(args,model,train_loader,test_loader,
                decreasing_lr,wd=0.0001, momentum=0.9)

    show_n_example(model, 10)
    print('hhh')
    print('Done!')

