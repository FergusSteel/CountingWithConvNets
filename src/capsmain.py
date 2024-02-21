# FROM https://github.com/ImMrMa/SegCaps.pytorch/tree/master

import os
import torch
from capsnet import SegCaps, CapsNetBasic, CCCaps, SegCapsOld, ReconstructionNet
from torch.utils.data import DataLoader
from torchdataset import *
from torchdataset_masks import *
import argparse
from capstrain import train
from caps_recontrain import recontrain
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
        model=CapsNetBasic(args.n_classes)
    elif args.model == "SegCaps":
        model = SegCapsOld()
    elif args.model == "CCC":
        model = CCCaps()

    
    global n_classes
    n_classes = args.n_classes

    model.cuda()
    model.to(device)
    
    decreasing_lr = list(map(int,args.dlr.split(',')))
    return  train_loader, test_loader, model, decreasing_lr

def show_n_example(model, n):
    test_loader = DataLoader(SpreadMNISTDataset(n, train=False))
    iterater = iter(test_loader)
    for i in range(n):
        input_dict = next(iterater)
        img = input_dict['image'].to(device)
        ground_truth = input_dict["dmap"]
        print(ground_truth.shape)
        outputs = model(img.unsqueeze(0).float().to(device))[0]
        print(outputs.shape)
        print(f"Image Number {i+1}:")
        show_density_map(img.cpu().detach().numpy()[0], sum(sum(ground_truth.cpu().detach().numpy())))
        print("-"*10)
        print("True count = ", round(sum(sum(sum(ground_truth.cpu().detach().numpy()[0])))))
        print("Total count = ", round(sum(sum(sum(outputs.cpu().detach().numpy()[0])))))
        for j in range(n_classes):
             print(f"{j}'s... True Count = {round(sum(sum(ground_truth.cpu().detach().numpy()[0][j])))}, Predicted Count = {sum(sum(outputs.cpu().detach().numpy()[0][j]))}")
             show_density_map(img.cpu().detach().numpy()[0], outputs[0][j].cpu().detach().numpy())

def introspect_example(capsmodel, reconmodel, n):
    test_loader = DataLoader(SpreadMNISTDataset(n, train=False))
    iterater = iter(test_loader)
    input_dict = next(iterater)
    img = input_dict['image'].to(device)
    ground_truth = input_dict["dmap"]
    
    output = model(capsmodel(img.unsqueeze(0).float().to(device))[1])
    print(output.shape)
    print(f"Reconstruction")
    show_density_map(img.cpu().detach().numpy()[0], output.cpu().detach().numpy().squeeze(0))
    show_density_map(np.zeros((256,256)), output.cpu().detach().numpy().squeeze(0))
    
        

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
    parser.add_argument('--params_name', default='segcaps_newest.pkl', help='params_name (default: segcaps.pkl)')
    parser.add_argument('--load_params_name', default='segcaps_longlongsigma2.pkl', help='params_name (default: segcaps.pkl)')
    parser.add_argument("--n_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument('--recon', type=int, default=0, help='run evaluation model')
    args = parser.parse_args()
    
    # Training the reconstructionnetwork guy
    if args.recon == 1:
        capsmodel = CapsNetBasic(args.n_classes)
        capsmodel.load_state_dict(torch.load(args.load_params_name))
        
        train_loader,test_loader,model,decreasing_lr=init(args)
        model = ReconstructionNet().cuda().to(device)
        recontrain(args,capsmodel,model,train_loader,test_loader,
                decreasing_lr, n_classes)
        
        introspect_example(capsmodel, model, 1)

    elif args.pretrain == 1:
        train_loader,test_loader,model,decreasing_lr=init(args)
        model.load_state_dict(torch.load(args.load_params_name))
        if args.nepoch > 1:
            print("Continuing Training")
            train_loader,test_loader,model,decreasing_lr=init(args)
            train(args,model,train_loader,test_loader,
                    decreasing_lr, n_classes)
        else:
            print("Evaluation Only")
            show_n_example(model, 1)
    else:
        print("Training")
        train_loader,test_loader,model,decreasing_lr=init(args)
        train(args,model,train_loader,test_loader,
                decreasing_lr, n_classes)

        show_n_example(model, 10)

