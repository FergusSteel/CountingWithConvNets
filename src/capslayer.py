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
                self.convs.append(nn.Conv2d(input_capsule_dimension, self.num_output_capsules*self.output_capsules_dimension, self.kernel_size, self.stride,padding=2,bias=False))
            else:
                self.convs.append(nn.ConvTranspose2d(input_capsule_dimension, self.num_output_capsules * self.output_capsules_dimension, self.kernel_size, self.stride,padding=2,output_padding=1))

    def forward(self, u):  # input [N,CAPS,C,H,W]
        if u.shape[1]!=self.num_input_capsules:
            raise ValueError("Wrong type of operation for capsule")
        op = self.op
        kernel_size = self.kernel_size
        stride = self.stride
        num_output_capsules = self.num_output_capsules
        output_capsules_dimension = self.output_capsules_dimension
        routing = self.routing
        N = u.shape[0]
        H_1=u.shape[3]
        W_1=u.shape[4]
        num_input_capsules = self.num_input_capsules

        u_t_list = [u_t.squeeze(1) for u_t in u.split(1, 1)]  # 将cap分别取出来

        u_hat_t_list = []

        for i, u_t in zip(range(self.num_input_capsules), u_t_list):  # u_t: [N,C,H,W]
            if op == "conv":
                u_hat_t = self.convs[i](u_t)  # 卷积方式
            elif op == "deconv":
                u_hat_t = self.convs[i](u_t) #u_hat_t: [N,num_output_capsules*output_capsules_dimension,H,W]
            else:
                raise ValueError("Wrong type of operation for capsule")
            H_1 = u_hat_t.shape[2]
            W_1 = u_hat_t.shape[3]
            u_hat_t = u_hat_t.reshape(N, num_output_capsules,output_capsules_dimension,H_1, W_1).transpose_(1,3).transpose_(2,4)
            u_hat_t_list.append(u_hat_t)    #[N,H_1,W_1,num_output_capsules,output_capsules_dimension]
        v=self.update_routing(u_hat_t_list,kernel_size,N,H_1,W_1,num_input_capsules,num_output_capsules,routing)
        return v
    
    def update_routing(self,u_hat_t_list, kernel_size, N, H_1, W_1, num_input_capsules, num_output_capsules, routing):
        one_kernel = torch.ones(1, num_output_capsules, kernel_size, kernel_size).cuda()  # 不需要学习
        b = torch.zeros(N, H_1, W_1, num_input_capsules, num_output_capsules).cuda()  # 不需要学习
        b_t_list = [b_t.squeeze(3) for b_t in b.split(1, 3)]
        u_hat_t_list_sg = []
        for u_hat_t in u_hat_t_list:
            u_hat_t_sg=u_hat_t.detach()
            u_hat_t_list_sg.append(u_hat_t_sg)

        for d in range(routing):
            if d < routing - 1:
                u_hat_t_list_ = u_hat_t_list_sg
            else:
                u_hat_t_list_ = u_hat_t_list

            r_t_mul_u_hat_t_list = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # routing softmax (N,H_1,W_1,num_output_capsules)
                b_t.transpose_(1, 3).transpose_(2, 3)  #[N,num_output_capsules,H_1, W_1]
                b_t_max = torch.nn.functional.max_pool2d(b_t,kernel_size,1,padding=2)
                b_t_max = b_t_max.max(1, True)[0]
                c_t = torch.exp(b_t - b_t_max)
                sum_c_t = nn_.conv2d_same(c_t, one_kernel, stride=(1, 1))  # [... , 1]
                r_t = c_t / sum_c_t  # [N,num_output_capsules, H_1, W_1]
                r_t = r_t.transpose(1, 3).transpose(1, 2)  # [N, H_1, W_1,num_output_capsules]
                r_t = r_t.unsqueeze(4)  # [N, H_1, W_1,num_output_capsules, 1]
                r_t_mul_u_hat_t_list.append(r_t * u_hat_t)  # [N, H_1, W_1, num_output_capsules, output_capsules_dimension]
            p = sum(r_t_mul_u_hat_t_list)  # [N, H_1, W_1, num_output_capsules, output_capsules_dimension]
            v = squash(p)
            if d < routing - 1:
                b_t_list_ = []
                for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                    # b_t     : [N, num_output_capsules,H_1, W_1]
                    # u_hat_t : [N, H_1, W_1, num_output_capsules, output_capsules_dimension]
                    # v       : [N, H_1, W_1, num_output_capsules, output_capsules_dimension]
                    # [N,H_1,W_1,num_output_capsules]
                    b_t.transpose_(1,3).transpose_(2,1)
                    b_t_list_.append(b_t + (u_hat_t * v).sum(4))
        v.transpose_(1, 3).transpose_(2, 4)
        # print(v.grad)
        return v
    
def squash(p):
    p_norm_sq = (p * p).sum(-1, True)
    p_norm = (p_norm_sq + 1e-9).sqrt()
    v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
    return v
