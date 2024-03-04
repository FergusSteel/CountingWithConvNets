# From https://github.com/ImMrMa/SegCaps.pytorch/tree/master
import torch
import torch.nn as nn
import torch.nn.functional as F
import nn_ as nn_
import torch.optim as optim

class CapsuleLayer(nn.Module):
    def __init__(self, num_input_capsules,input_capsule_dimension, op, kernel_size, stride, num_output_capsules, output_capsules_dimension, routing):
        super().__init__()
        self.num_output_capsules = num_output_capsules
        self.output_capsules_dimension = output_capsules_dimension
        self.op = op
        self.kernel_size = kernel_size
        self.stride = stride
        self.routing = routing
        self.convs = nn.ModuleList()
        self.num_input_capsules=num_input_capsules
        for _ in range(num_input_capsules):
            if self.op=='conv':
                if self.kernel_size == 1:
                    self.convs.append(nn.Conv2d(input_capsule_dimension, self.num_output_capsules*self.output_capsules_dimension, self.kernel_size, self.stride,padding=0,bias=False))
                else:
                    self.convs.append(nn.Conv2d(input_capsule_dimension, self.num_output_capsules*self.output_capsules_dimension, self.kernel_size, self.stride,padding=2,bias=False))
            else:
                self.convs.append(nn.ConvTranspose2d(input_capsule_dimension, self.num_output_capsules * self.output_capsules_dimension, self.kernel_size, self.stride,padding=2,output_padding=1))

    def forward(self, u): 
        if u.shape[1]!=self.num_input_capsules:
            raise ValueError("Wrong type of operation for capsule")
        op = self.op
        kernel_size = self.kernel_size
        stride = self.stride
        num_output_capsules = self.num_output_capsules
        output_capsules_dimension = self.output_capsules_dimension
        routing = self.routing
        N = u.shape[0] # This is always 1
        H_1=u.shape[3]
        W_1=u.shape[4]
        num_input_capsules = self.num_input_capsules

        u_t_list = [u_t.squeeze(1) for u_t in u.split(1, 1)]  # The received capsule outputs from previous layer

        u_hat_t_list = []

        for i, u_t in zip(range(self.num_input_capsules), u_t_list): # For each input capsules index and its output
            if op == "conv":
                u_hat_t = self.convs[i](u_t) # Run the convolutional operation over that capsules output
            elif op == "deconv":
                u_hat_t = self.convs[i](u_t)
            else:
                raise ValueError("Wrong type of operation for capsule")
            H_1 = u_hat_t.shape[2]
            W_1 = u_hat_t.shape[3]
            u_hat_t = u_hat_t.reshape(N, num_output_capsules,output_capsules_dimension,H_1, W_1).transpose_(1,3).transpose_(2,4) # Rearranges capsule output so spatial dimensions are first for the funky routing process
            u_hat_t_list.append(u_hat_t)
        v=self.update_routing(u_hat_t_list,kernel_size,N,H_1,W_1,num_input_capsules,num_output_capsules,routing)
        return v
    
    def update_routing(self,u_hat_t_list, kernel_size, N, H_1, W_1, num_input_capsules, num_output_capsules, routing):
        local_kernel = torch.ones(1, num_output_capsules, kernel_size, kernel_size).cuda()  
        logits_b = torch.zeros(N, H_1, W_1, num_input_capsules, num_output_capsules).cuda() # Initisalises the output logits
        b_t_list = [b_t.squeeze(3) for b_t in logits_b.split(1, 3)] # removes the input dimensions so there are num_input many logit lists
        u_hat_t_list_sg = []
        for u_hat_t in u_hat_t_list: # This part removes the outputs ofrom the GPU
            u_hat_t_sg=u_hat_t.detach()
            u_hat_t_list_sg.append(u_hat_t_sg)

        # Routing loop
        for d in range(routing):
            if d < routing - 1:
                u_hat_t_list_ = u_hat_t_list_sg
            else:
                u_hat_t_list_ = u_hat_t_list

            r_t_mul_u_hat_t_list = []
            # For each capsule type.... do routing :)
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # b_t.transpose_(1, 3).transpose_(2, 3)  
                # c_t = torch.nn.functional.softmax(b_t, dim=3)

                # sum_c_t = nn_.conv2d_same(c_t, one_kernel, stride=(1, 1))
                # routing softmax (N,H_1,W_1,t_1)
                b_t.transpose_(1, 3).transpose_(2, 3) # reorganises the logits to be [dimensions, height, width] I do not understand why he does all this rearranging, but it doesn't change algorithm
                # This then calculates the coupling coefficients by using softmax over the logits, I had to alter thsi code to allow for 0 padding in my network
                try:
                    b_t_max = torch.nn.functional.max_pool2d(b_t,kernel_size,1,padding=2)
                    b_t_max = b_t_max.max(1, True)[0]
                    c_t = torch.exp(b_t - b_t_max) # this is such a bizarre way to compute this???? But it still normalises it and computes the same as the softmax, its just wacky
                except:
                    c_t = torch.nn.functional.softmax(b_t, dim=3)   
                    
                sum_c_t = nn_.conv2d_same(c_t, local_kernel, stride=(1, 1)) # Runs the local kernel over the coupling coefficents
                r_t = c_t / sum_c_t  
                r_t = r_t.transpose(1, 3).transpose(1, 2)
                r_t = r_t.unsqueeze(4)
                r_t_mul_u_hat_t_list.append(r_t * u_hat_t) # Calculates the P in the paper, (before summing, basically its the outputs * the coupling coefficients)
            p = sum(r_t_mul_u_hat_t_list)
            v = squash(p)
            if d < routing - 1:
                b_t_list_ = [] # resets to the new logits if not the last one (the prior probabilities shoudl always be the previous capsule outputs as stated in papers)
                for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                    b_t.transpose_(1,3).transpose_(2,1)
                    b_t_list_.append(b_t + (u_hat_t * v).sum(4))
        v.transpose_(1, 3).transpose_(2, 4)
        return v

    def update_rout123ing(self, u_hat_list, kernel_size, batch_size, height, width, num_input_capsules, num_output_capsules, num_iterations):
        kernel_ones = torch.ones(1, num_output_capsules, kernel_size, kernel_size).cuda()
        logits_b = torch.zeros(batch_size, height, width, num_input_capsules, num_output_capsules).cuda()
        logits_b_list = [logit.squeeze(3) for logit in logits_b.split(1, 3)]
        u_hat_detached_list = [u_hat.detach() for u_hat in u_hat_list]

        for iteration in range(num_iterations):
            current_u_hat_list = u_hat_detached_list if iteration < num_iterations - 1 else u_hat_list
            scaled_u_hat_list = []

            for logit_b, u_hat in zip(logits_b_list, current_u_hat_list):
                logit_b.transpose_(1, 3).transpose_(2, 3)
                print(logit_b.shape)
                coupling_c = torch.nn.functional.softmax(logit_b, dim=1)
                sum_c = nn_.conv2d_same(coupling_c, kernel_ones, stride=(1, 1))
                coupling_coefficients = coupling_c / sum_c
                coupling_coefficients = coupling_coefficients.transpose(1, 3).transpose(1, 2)
                coupling_coefficients = coupling_coefficients.unsqueeze(4)
                scaled_u_hat_list.append(coupling_coefficients * u_hat)

            pre_squashed_v = sum(scaled_u_hat_list)
            output_v = squash(pre_squashed_v)

            if iteration < num_iterations - 1:
                logits_b_list_updated = []
                for logit_b, u_hat in zip(logits_b_list, current_u_hat_list):
                    logit_b = logit_b.transpose(1, 3).transpose(2, 3)
                    print(logit_b.shape)
                    print()
                    logits_b_list_updated.append(logit_b + (u_hat * output_v).sum(4))

        output_v = output_v.transpose(1, 3).transpose(2, 4)

        return output_v

    
def squash(p):
    p_norm_sq = (p * p).sum(-1, True)
    p_norm = (p_norm_sq + 1e-9).sqrt()
    v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
    return v
