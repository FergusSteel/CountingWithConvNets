# FROM https://github.com/ImMrMa/SegCaps.pytorch/tree/master

import os
import torch
from capsnet import SegCaps, CapsNetBasic, CCCaps, SegCapsOld
from torch.utils.data import DataLoader
from torchdataset import *
from torchdataset_masks import *
import argparse
from capstrain import train
from convert_to_npz import *
from pilot_utils import show_density_map, load_batch
from introspection_utils import display_capsule_grid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init(args):
    os.environ['CUDA_VISIBLE_DEVICES'] =args.gpu
    print('{}:{}'.format('cuda',torch.cuda.is_available()))
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader = DataLoader(SpreadMNISTDataset(args.n_images), batch_size=args.batch_size_train, shuffle=False)
    test_loader = DataLoader(SpreadMNISTDataset(int(args.n_images*0.2), train=False), batch_size=args.batch_size_train, shuffle=False)
    if args.model == "Seg":
        model=CapsNetBasic(10)
    elif args.model == "SegCaps":
        model = SegCapsOld()
    elif args.model == "CCC":
        model = CCCaps()

    model.cuda()
    model.to(device)
    
    decreasing_lr = list(map(int,args.dlr.split(',')))
    return  train_loader, test_loader, model, decreasing_lr

def show_n_example(model, n):
    dat = load_batch(n)
    inputs = torch.from_numpy(dat[0]).float().to(device)
    # for i in range(3):
    #     show_density_map(inputs[i][0].cpu().detach().numpy(), sum(dat[1][i]))
    for i in range(n):
        outputs = model(inputs[i].unsqueeze(0))[0].float().to(device)
        reconstructed = model(inputs[i].unsqueeze(0))[1][0][0].float().to(device)
        print(outputs.shape)
        print(f"Image Number {i+1}:")
        show_density_map(np.zeros((256,256)), reconstructed.cpu().detach().numpy())
        print("-"*10)
        print("True count = ", round(sum(sum(sum(dat[1][i]))) / 1000))
        print("Total count = ", round(sum(sum(sum(outputs.cpu().detach().numpy()[0]))) / 1000))
        for j in range(10):
             print(f"{j}'s... True Count = {round(sum(sum(dat[1][i][j])) / 1000)}, Predicted Count = {sum(sum(outputs.cpu().detach().numpy()[0][j]))}")
             show_density_map(inputs[i][0].cpu().detach().numpy(), outputs[0][j].cpu().detach().numpy())
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Counting With SegCaps")
    parser.add_argument('--batch_size_train', type=int, default=1, help='input batch size for training (default: 160)')
    parser.add_argument('--model', default="Seg", help='Which model to use (default: Seg)')
    parser.add_argument("--n_images", type=int, default=1000, help="Number of images to train on.")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--dlr', default='10,25', help='decreasing strategy')
    parser.add_argument('--nepoch', type=int,default=1, help='epochs (default: 200)')
    parser.add_argument('--seed', type=int,default='10', help='seed (default: 1)')
    parser.add_argument('--pretrain', type=int, default='0', help='pretrain (default: 1)')
    parser.add_argument('--gpu', default='0', help='index of gpus to use')
    parser.add_argument('--data_name', default='train', help='data_name (default: train)')
    parser.add_argument('--params_name', default='segcaps_longsigma3.pkl', help='params_name (default: segcaps.pkl)')
    parser.add_argument('--load_params_name', default='segcaps_best_so_far.pkl', help='params_name (default: segcaps.pkl)')
    args = parser.parse_args()

    if args.pretrain == 1:
        train_loader,test_loader,model,decreasing_lr=init(args)
        model.load_state_dict(torch.load(args.load_params_name))
        if args.nepoch > 1:
            print("Continuing Training")
            train_loader,test_loader,model,decreasing_lr=init(args)
            train(args,model,train_loader,test_loader,
                    decreasing_lr)
    else:
        print("Training")
        train_loader,test_loader,model,decreasing_lr=init(args)
        train(args,model,train_loader,test_loader,
                decreasing_lr)

    show_n_example(model, 10)

