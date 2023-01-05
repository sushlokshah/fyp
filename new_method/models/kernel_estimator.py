import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch.utils.data



class Kernel_estimation(nn.Module):
    def __init__(self,kernel_size) -> None:
        super(Kernel_estimation, self).__init__()
        self.kernel_size = kernel_size
        self.pad = torch.nn.ReplicationPad2d([(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2])
        # self.batchnorm = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, input, kernel):
        N, C, H, W = input.shape
        print("kernel_estimation_input",input.shape)
        # print(input.shape, kernel.shape)
        input = self.pad(input)
        unfolded_input= input.unfold(2,self.kernel_size,1).unfold(3,self.kernel_size,1)
        
        N, C_kernel, H, W = kernel.shape
        print("kernel:",kernel.shape)
        kernel = kernel.reshape(N,H,W,-1)
        kernel = kernel.reshape(N,H,W,self.kernel_size,self.kernel_size, -1)
        kernel = kernel.view(N,-1,H,W,self.kernel_size,self.kernel_size)
        # print(kernel.shape, unfolded_input.shape)
        print("kernel_estimation_output",kernel.shape)
        output = kernel*unfolded_input
        output = output.sum(4).sum(4)
        output = self.relu(output)
        
        return output