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

from models.resnet import ResnetBlock


# feature extractor model
class conv_encoder(nn.Module):
    def __init__(self, nin, nout, maxpool=True):
        super(conv_encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        if self.maxpool:
            return self.maxpool(self.conv_block(input))
        else:
            return self.conv_block(input)


class encoder(nn.Module):
    def __init__(self, output_channels, input_channels, resblocks=False) -> None:
        super(encoder, self).__init__()
        out_ch = 16
        self.output_channels = output_channels
        self.conv1 = conv_encoder(input_channels, out_ch)  # 256*256*16
        self.resblock11 = ResnetBlock(out_ch, kernel_size=3)
        # self.resblock12 = ResnetBlock(out_ch, kernel_size=3)
        # self.resblock13 = ResnetBlock(out_ch, kernel_size=3)

        self.conv2 = conv_encoder(out_ch, out_ch*2)  # 128*128*32
        self.resblock21 = ResnetBlock(out_ch*2, kernel_size=3)
        # self.resblock22 = ResnetBlock(out_ch*2, kernel_size=3)
        # self.resblock23 = ResnetBlock(out_ch*2, kernel_size=3)

        self.conv3 = conv_encoder(out_ch*2, out_ch*4)  # 64*64*64
        self.resblock31 = ResnetBlock(out_ch*4, kernel_size=3)
        # self.resblock32 = ResnetBlock(out_ch*4, kernel_size=3)
        # self.resblock33 = ResnetBlock(out_ch*4, kernel_size=3)

        self.conv4 = conv_encoder(out_ch*4, out_ch*4)  # 32*32*64
        self.resblock41 = ResnetBlock(out_ch*4, kernel_size=3)
        # self.resblock42 = ResnetBlock(output_channels, kernel_size=3)
        # self.resblock43 = ResnetBlock(output_channels, kernel_size=3)

        self.conv5 = conv_encoder(out_ch*4, out_ch*8)  # 16*16*128
        self.resblock51 = ResnetBlock(out_ch*8, kernel_size=3)

        self.conv6 = conv_encoder(
            out_ch*8, output_channels)  # 8*8*output_channels
        self.resblock61 = ResnetBlock(output_channels, kernel_size=3)

        self.conv7 = conv_encoder(
            output_channels, output_channels)  # 4*4*output_channels
        self.resblock71 = ResnetBlock(output_channels, kernel_size=3)

        # 1*1*output_channels
        self.conv8 = nn.Conv2d(output_channels, output_channels, 4, 1, 0)

        self.resblocks = resblocks

    def forward(self, sharp_image):
        if self.resblocks:
            h1 = self.resblock11(self.conv1(sharp_image))
            h2 = self.resblock21(self.conv2(h1))
            h3 = self.resblock31(self.conv3(h2))
            h4 = self.resblock41(self.conv4(h3))
            h5 = self.resblock51(self.conv5(h4))
            h6 = self.resblock61(self.conv6(h5))
            # h7 = self.resblock71(self.conv7(h6))
            encoding = self.conv8(h6)

        else:
            #print(sharp_image.shape)
            h1 = self.conv1(sharp_image)
            #print(h1.shape)
            h2 = self.conv2(h1)
            #print(h2.shape)
            h3 = self.conv3(h2)
            #print(h3.shape)
            h4 = self.conv4(h3)
            #print(h4.shape)
            h5 = self.conv5(h4)
            #print(h5.shape)
            h6 = self.conv6(h5)
            #print(h6.shape)
            # h7 = self.conv7(h6)
            # #print(h7.shape)
            encoding = self.conv8(h6)
            #print(encoding.shape)
        return encoding.view(-1, self.output_channels), [h1, h2, h3, h4, h5, h6]
