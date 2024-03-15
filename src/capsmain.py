# FROM https://github.com/ImMrMa/SegCaps.pytorch/tree/master

import os
import torch
from capsnet import SegCaps, CapsNetBasic, CCCaps, SegCapsOld, ReconstructionNet, CapsNetComplex
from pilot_utils import TorchUNetModel, TorchUNetModel2, ReconstructionUNetModel
from torch.utils.data import DataLoader
from torchdataset import *
from torchdataset_masks import *
import argparse
from capstrain import train, AverageMeter
from caps_recontrain import recontrain
from convert_to_npz import *
from pilot_utils import show_density_map, load_batch
from introspection_utils import display_capsule_grid
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython import display
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init(args):
    os.environ['CUDA_VISIBLE_DEVICES'] =args.gpu
    print('{}:{}'.format('cuda',torch.cuda.is_available()))
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader = DataLoader(SpreadMNISTDataset(args.n_images, path="noisy"), batch_size=args.batch_size_train, shuffle=False)
    test_loader = DataLoader(SpreadMNISTDataset(int(args.n_images*0.2), train=False), batch_size=args.batch_size_train, shuffle=False)
    if args.unet == 1:
        model = TorchUNetModel(10)
    else:   
        if args.model == "Seg":
            model=CapsNetBasic(args.n_classes)
        elif args.model == "Seg2":
            model = CapsNetComplex(args.n_classes)
        elif args.model == "CCC":
            model = CCCaps()

    
    global n_classes
    n_classes = args.n_classes

    model.cuda()
    model.to(device)
    
    decreasing_lr = list(map(int,args.dlr.split(',')))
    return  train_loader, test_loader, model, decreasing_lr

def show_n_example(model, n, save=False):
    test_loader = DataLoader(SpreadMNISTDataset(n, train=False, path="noisy"), shuffle=True)
    iterater = iter(test_loader)
    for i in range(n):
        input_dict = next(iterater)
        img = input_dict['image'].to(device)
        ground_truth = input_dict["dmap"]
        print(ground_truth.shape)
        outputs = model(img.unsqueeze(0).float().to(device))[0]
        print(outputs.shape)
        print(f"Image Number {i+1}:")
        print("-"*10)
        print("True count = ", round(sum(sum(sum(ground_truth.cpu().detach().numpy()[0])))/100))
        print("Total count = ", round(sum(sum(sum(outputs.cpu().detach().numpy()[0])))/100))

        show_density_map(img.cpu().detach().numpy()[0], sum(ground_truth.cpu().detach().numpy()[0]))
        show_density_map(img.cpu().detach().numpy()[0], sum(outputs[0].cpu().detach().numpy()))
        

        for j in range(n_classes):
             print(f"{j}'s... True Count = {round(sum(sum(ground_truth.cpu().detach().numpy()[0][j]))/100):.2f}, Predicted Count = {sum(sum(outputs.cpu().detach().numpy()[0][j]))/100:.2f}")
             show_density_map(img.cpu().detach().numpy()[0], outputs[0][j].cpu().detach().numpy())
             if save:
                save_img = img + outputs[0][j]
                print(save_img.shape)
                plt.imsave(f"outputs/pred_img{i}class{j}.png",save_img)

def introspect_example(capsmodel, reconmodel, n):
    test_loader = DataLoader(SpreadMNISTDataset(n, train=True))
    iterater = iter(test_loader)
    for j in range(n):
        input_dict = next(iterater)
        img = input_dict['image'].to(device)
        ground_truth = input_dict["dmap"]
        os.makedirs(f"reconexp\img{j}", exist_ok=True)
        capsout = capsmodel(img.unsqueeze(0).float().to(device))[1]
        print(capsout.shape)

        reconstructed_base = model(capsout)
        print(f"Base Reconstruction")
        show_density_map(img.cpu().detach().numpy()[0], img.cpu().detach().numpy()[0])
        show_density_map(reconstructed_base.cpu().detach().numpy().squeeze(0), reconstructed_base.cpu().detach().numpy().squeeze(0))

        for dimension_to_vary in range(16):
            variance = np.arange(-10, 12, 2)
            for i, var in enumerate(variance):
                capscopy = capsout.clone()
                var = round(var,2)
                print("Varying Dimension [", dimension_to_vary, "] by scalar amount: ", round(var,2))
                capscopy[dimension_to_vary] *= var
                reconstructed = model(capscopy)

                # show_density_map(img.cpu().detach().numpy()[0], reconstructed.cpu().detach().numpy().squeeze(0))
                plt.imsave(f"reconexp\img{j}\dim{dimension_to_vary}var{i}.png", reconstructed.cpu().detach().numpy().squeeze(0), cmap="gray")


    
        

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
    parser.add_argument('--load_params_name', default='segcaps_100Sigma2Good.pkl', help='params_name (default: segcaps.pkl)')
    parser.add_argument("--n_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument('--recon', type=int, default=0, help='run evaluation model')
    parser.add_argument("--unet", type=int, default=0, help="Use UNet model")
    args = parser.parse_args()
    
    # reconstruciton training
    if args.recon == 1:
        capsmodel = CapsNetComplex(args.n_classes).cuda().to(device)
        capsmodel.load_state_dict(torch.load("segcaps_5by5routing.pkl"))
        
        train_loader,test_loader,model,decreasing_lr=init(args)
        model = ReconstructionUNetModel().cuda().to(device)

        if args.pretrain == 1:
            model.load_state_dict(torch.load("ReconWeigths.pkl"))
        else:
            recontrain(args,capsmodel,model,train_loader,test_loader,
                decreasing_lr, n_classes)
        
        introspect_example(capsmodel, model, 3)
    # using pretrained model, or kepe training or whatever
    elif args.pretrain == 1:
        train_loader,test_loader,model,decreasing_lr=init(args)
        model.load_state_dict(torch.load(args.load_params_name))
        if args.nepoch > 1:
            print("Continuing Training")
            train_loader,test_loader,model,decreasing_lr=init(args)
            train(args,model,train_loader,test_loader,
                    decreasing_lr, n_classes)
            show_n_example(model, 1)
        else:
            print("Evaluation Only")
            test_loader = iter(DataLoader(SpreadMNISTDataset(int(10000), train=False, path="noisy"), batch_size=args.batch_size_train, shuffle=False))
            count_error = 0
            pc_count_error = [0]*10
            for i, input_dict in enumerate(test_loader):
                img = input_dict['image'].to(device)
                ground_truth = input_dict["dmap"]
                outputs = model(img.unsqueeze(0).float().to(device))[0]
                print(f"Image Number {i+1}:")
                print("-"*10)
                print("True count = ", round(sum(sum(sum(ground_truth.cpu().detach().numpy()[0])))) / 100)
                print("Predicted count = ", round(sum(sum(sum(outputs.cpu().detach().numpy()[0]))))/ 100)
                error = abs(sum(sum(sum(ground_truth.cpu().detach().numpy()[0]))) - sum(sum(sum(outputs.cpu().detach().numpy()[0]))))
                count_error += error
                for cat in range(10):
                    pc_count_error[cat] += abs(sum(sum(ground_truth.cpu().detach().numpy()[0][cat])) - sum(sum(outputs.cpu().detach().numpy()[0][cat])))
                    print(f"Digit {cat} G.T. count = ", round(sum(sum(ground_truth.cpu().detach().numpy()[0][cat])) / 100))
                    print(f"Predicted {cat} count = ", round(sum(sum(outputs.cpu().detach().numpy()[0][cat])) / 100))
                    print(f"Digit {cat} error = ", abs(sum(sum(ground_truth.cpu().detach().numpy()[0][cat])) - sum(sum(outputs.cpu().detach().numpy()[0][cat]))) / 100) 
            
            print("Mean Count Error Across Test Data Set = ", count_error/len(test_loader))
            print("Per Class Counting Error:")
            for cat in range(10):
                print(f"Category {cat}: {pc_count_error[cat]/len(test_loader) / 100}")


            show_n_example(model, 1, False)
            count = sum(p.numel() for p in model.parameters())

            print(count)
    # Normal training
    else:
        # print("Training")
        train_loader,test_loader,model,decreasing_lr=init(args)
        train(args,model,train_loader,test_loader,
                decreasing_lr, n_classes)

        show_n_example(model, 10)
        count = sum(p.numel() for p in model.parameters())

        print(count)

    
