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


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation=[1, 1], bias=True):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                      dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                      dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.stem(x) + x
        return out
