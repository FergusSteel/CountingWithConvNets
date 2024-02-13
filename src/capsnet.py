# FROM https://github.com/ImMrMa/SegCaps.pytorch/tree/master
import torch
import torch.nn as nn
from capslayer import CapsuleLayer
import nn_
from introspection_utils import display_capsule_grid


class SegCaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, padding=2,bias=False),

        )
        self.step_1 = nn.Sequential(  # 1/2
            CapsuleLayer(1, 32, "conv", kernel_size=5, stride=2, num_output_capsules=2, output_capsules_dimension=32, routing=1),
            CapsuleLayer(2, 32, "conv", kernel_size=5, stride=1, num_output_capsules=4, output_capsules_dimension=32, routing=3),
        )
        self.step_2 = nn.Sequential(  # 1/4
            CapsuleLayer(4, 32, "conv", kernel_size=5, stride=2, num_output_capsules=4, output_capsules_dimension=64, routing=3),
            CapsuleLayer(4, 64, "conv", kernel_size=5, stride=1, num_output_capsules=8, output_capsules_dimension=64, routing=3)
        )
        self.step_3 = nn.Sequential(  # 1/8
            CapsuleLayer(8, 64, "conv", kernel_size=5, stride=2, num_output_capsules=8, output_capsules_dimension=128, routing=3),
            CapsuleLayer(8, 128, "conv", kernel_size=5, stride=1, num_output_capsules=8, output_capsules_dimension=64, routing=3)
        )
        self.step_4 = CapsuleLayer(8, 64, "deconv", kernel_size=5, stride=2, num_output_capsules=8, output_capsules_dimension=64, routing=3)

        self.step_5 = CapsuleLayer(16, 64, "conv", kernel_size=5, stride=1, num_output_capsules=4, output_capsules_dimension=64, routing=3)

        self.step_6 = CapsuleLayer(4, 64, "deconv", kernel_size=5, stride=2, num_output_capsules=4, output_capsules_dimension=32, routing=5)
        self.step_7 = CapsuleLayer(8, 32, "conv", kernel_size=5, stride=1, num_output_capsules=4, output_capsules_dimension=32, routing=5)
        self.step_8 = CapsuleLayer(4, 32, "deconv", kernel_size=5, stride=2, num_output_capsules=2, output_capsules_dimension=32, routing=5)
        self.step_10 = CapsuleLayer(3, 32, "conv", kernel_size=1, stride=1, num_output_capsules=1, output_capsules_dimension=32, routing=5) # TEST THIS
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 10, 5, 1, padding=2),
        )

        self.recon_1 = nn.Sequential(
            nn.Conv2d(10, 64, 1, 1, padding=0),
            nn.ReLU(),
        )

        self.recon_2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, padding=0),
            nn.ReLU(),
        )

        self.out_recon = nn.Sequential(
            nn.Conv2d(128, 1, 1, 1, padding=0),
        )


    def forward(self, x):
        x = self.conv_1(x)
        x.unsqueeze_(1)
        skip_1 = x  
        x = self.step_1(x)

        skip_2 = x  # [N,4,16,H/2,W/2]
        x = self.step_2(x)

        
        skip_3 = x  # [N,8,32,H/4,W/4]

        x = self.step_3(x)


        x = self.step_4(x)  # [N,8,32,H/4,W/4
       
        x = torch.cat((x, skip_3), 1)  # [N,16,32,H/4,W/4]

        x = self.step_5(x)  # [N,4,32,H/4,W/4]

        x = self.step_6(x)  # [N,4,16,H/2,W/2]

        x = torch.cat((x, skip_2), 1)   # [N,8,16,H/2,W/2]
        x = self.step_7(x)  # [N,4,16,H/2,W/2]
    
        x = self.step_8(x)  # [N,2,16,H,W]

        x=torch.cat((x,skip_1),1)
        #print("step cat", x.shape)
        x=self.step_10(x)
        output_caps = x
        x.squeeze_(1)
        x=self.conv_2(x)
        # #print("conv2", x.shape)
        x = x.squeeze_(1)
        #print("A", x.shape)
        v_lens = torch.norm(x, p=2, dim=0)
        # print("pnorm", x.shape)
        # # #print("B", v_lens.shape)
        # #print("v_lens", v_lens.shape)
        v_lens = v_lens.unsqueeze(0)

        reconstructed = self.recon_1(x)
        reconstructed = self.recon_2(reconstructed)
        reconstructed = self.out_recon(reconstructed)
        print("recon", reconstructed.shape)

        return x, reconstructed
    
class CCCaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, padding=1),
            nn.Conv2d(64, 64, 3, 1, padding=1),
        )

        self.caps = nn.Sequential(
            CapsuleLayer(1, 64, "conv", kernel_size=5, stride=1, num_output_capsules=2, output_capsules_dimension=32, routing=1),
            CapsuleLayer(2, 32, "conv", kernel_size=5, stride=1, num_output_capsules=2, output_capsules_dimension=64, routing=3),
            CapsuleLayer(2, 64, "conv", kernel_size=5, stride=1, num_output_capsules=2, output_capsules_dimension=128, routing=1),
            CapsuleLayer(2, 128, "conv", kernel_size=1, stride=1, num_output_capsules=1, output_capsules_dimension=256, routing=3)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(256, 256, 5, 1, padding=4, dilation=2),
            nn.Conv2d(256, 256, 5, 1, padding=4, dilation=2),
            nn.Conv2d(256, 128, 5, 1, padding=4, dilation=2),
            nn.Conv2d(128, 64, 5, 1, padding=4, dilation=2),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 10, 1, 1, padding=0),
        )

    def forward(self, x):
        x = self.conv_1(x)
        print("conv1", x.shape)
        x = torch.reshape(x, (x.shape[0], 1, 64, x.shape[2], x.shape[3]))
        print("reshape1", x.shape)
        x = self.caps(x)
        print("caps", x.shape)
        x = torch.reshape(x, (x.shape[0], 256, x.shape[3], x.shape[4]))
        print("reshape2", x.shape)
        x = self.conv_2(x)
        print("conv2", x.shape)
        x = self.conv_3(x)
        print("conv3", x.shape)

        return x, torch.zeros(1, 1, 256, 256)


class SegCapsOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, padding=2,bias=False),

        )
        self.step_1 = nn.Sequential(  # 1/2
            CapsuleLayer(1, 16, "conv", kernel_size=5, stride=2, num_output_capsules=2, output_capsules_dimension=16, routing=1),
            CapsuleLayer(2, 16, "conv", kernel_size=5, stride=1, num_output_capsules=4, output_capsules_dimension=16, routing=3),
        )
        self.step_2 = nn.Sequential(  # 1/4
            CapsuleLayer(4, 16, "conv", kernel_size=5, stride=2, num_output_capsules=4, output_capsules_dimension=32, routing=3),
            CapsuleLayer(4, 32, "conv", kernel_size=5, stride=1, num_output_capsules=8, output_capsules_dimension=32, routing=3)
        )
        self.step_3 = nn.Sequential(  # 1/8
            CapsuleLayer(8, 32, "conv", kernel_size=5, stride=2, num_output_capsules=8, output_capsules_dimension=64, routing=3),
            CapsuleLayer(8, 64, "conv", kernel_size=5, stride=1, num_output_capsules=8, output_capsules_dimension=32, routing=3)
        )
        self.step_4 = CapsuleLayer(8, 32, "deconv", kernel_size=5, stride=2, num_output_capsules=8, output_capsules_dimension=32, routing=3)

        self.step_5 = CapsuleLayer(16, 32, "conv", kernel_size=5, stride=1, num_output_capsules=4, output_capsules_dimension=32, routing=3)

        self.step_6 = CapsuleLayer(4, 32, "deconv", kernel_size=5, stride=2, num_output_capsules=4, output_capsules_dimension=16, routing=3)
        self.step_7 = CapsuleLayer(8, 16, "conv", kernel_size=5, stride=1, num_output_capsules=4, output_capsules_dimension=16, routing=3)
        self.step_8 = CapsuleLayer(4, 16, "deconv", kernel_size=5, stride=2, num_output_capsules=2, output_capsules_dimension=16, routing=3)
        self.step_10 = CapsuleLayer(3, 16, "conv", kernel_size=5, stride=1, num_output_capsules=1, output_capsules_dimension=16, routing=3) # TEST THIS
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 10, 5, 1, padding=2),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x.unsqueeze_(1)

        skip_1 = x  # [N,1,16,H,W]

        x = self.step_1(x)

        skip_2 = x  # [N,4,16,H/2,W/2]
        x = self.step_2(x)

        skip_3 = x  # [N,8,32,H/4,W/4]

        x = self.step_3(x)  # [N,8,32,H/8,W/8]


        x = self.step_4(x)  # [N,8,32,H/4,W/4]
        x = torch.cat((x, skip_3), 1)  # [N,16,32,H/4,W/4]

        x = self.step_5(x)  # [N,4,32,H/4,W/4]

        x = self.step_6(x)  # [N,4,16,H/2,W/2]

        x = torch.cat((x, skip_2), 1)   # [N,8,16,H/2,W/2]
        x = self.step_7(x)  # [N,4,16,H/2,W/2]
        x = self.step_8(x)  # [N,2,16,H,W]
        #print("step 8", x.shape)

        x=torch.cat((x,skip_1),1)
        #print("step cat", x.shape)
        x=self.step_10(x)
        #print("step 10", x.shape)
        x.squeeze_(1)
        print(x.shape)
        x=self.conv_2(x)
        #print("conv2", x.shape)
        x = x.squeeze_(1)
        #print("A", x.shape)
        v_lens = torch.norm(x, p=2, dim=0)
        # print("pnorm", x.shape)
        # # #print("B", v_lens.shape)
        # #print("v_lens", v_lens.shape)
        v_lens = v_lens.unsqueeze(0)
        return v_lens, torch.zeros(1, 1, 256, 256)

class CapsNetBasic(nn.Module):
    def __init__(self, n_classes=10):
        self.n_classes=n_classes
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 128, 5, 1, padding="same"),
        )
        #[N,CAPS,C,H,W]
        self.primary_caps = nn.Sequential(  # 1/2
            CapsuleLayer(1, 128, "conv", kernel_size=5, stride=1, num_output_capsules=32, output_capsules_dimension=8, routing=1),
        )

        self.seg_caps= nn.Sequential(  # 1/2s
            CapsuleLayer(32, 8, "conv", kernel_size=1, stride=1, num_output_capsules=1, output_capsules_dimension=16, routing=5),
        ) 

        # self.conv_2 = nn.Sequential(
        #     nn.Conv3d(10, n_classes, 5, 1, padding=2),
        # )

        # self.reconstruction_conv1 = nn.Sequential(
        #     nn.Conv2d(16, 128, 1, 1, padding=0),
        #     nn.ReLU()
            
        
        self.reconstruction_conv2 = nn.Sequential(
            nn.Conv2d(16, 10, 5, 1, padding=2),
        )


    def forward(self, x):
        x = self.conv_1(x)
        x.unsqueeze_(1)
        #print(x.shape)

        x = self.primary_caps(x)
        #print(x.shape)

        x = self.seg_caps(x)
        #display_capsule_grid(x)
        #print(x.shape)
        x = x.squeeze_(1)
        #x=self.conv_2(x)
        #print("conv2", x.shape)
        x = x.squeeze_(0)
        #x = self.reconstruction_conv1(x)
        x = self.reconstruction_conv2(x)
        # x  = torch.tensor([self.reconstruction_conv2(x[caps]) for caps in range(self.n_classes)])
        #print("2", x.shape)
        
        # Fix the shape following the reconstruction convs broski
        x = x.squeeze_(1)
        x = x.unsqueeze(0)
        
        
        v_lens = torch.norm(x, p=2, dim=0)
        # print("pnorm", x.shape)
        #print("A", v_lens.shape)
        # # #print("B", v_lens.shape)
        # #print("v_lens", v_lens.shape)
        v_lens = v_lens.unsqueeze(0)
        return v_lens, x
