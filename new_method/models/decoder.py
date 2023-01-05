import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch.utils.data

from models.resnet import ResnetBlock


class conv_decoder(nn.Module):
    def __init__(self, nin, nout) -> None:
        super(conv_decoder, self).__init__()
        self.upconv_block = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        return self.upsample(self.upconv_block(input))


class decoder(nn.Module):
    def __init__(self, input_channels, output_channels, resblocks=False) -> None:
        super(decoder, self).__init__()
        out_ch = 16
        # 1*1*input_channels to 4*4*output_channels
        self.output_channels = output_channels
        self.convT1 = nn.ConvTranspose2d(
            input_channels, input_channels, 4, 1, 0)  # /4*4*128

        self.deconv1 = conv_decoder(2*input_channels, input_channels)
        self.resblock11 = ResnetBlock(
            input_channels, kernel_size=3)  # 8*8*128
        # self.resblock12 = ResnetBlock(out_ch*4, kernel_size=3)
        # self.resblock13 = ResnetBlock(out_ch*4, kernel_size=3)

        self.deconv2 = conv_decoder(2*input_channels, out_ch*4)
        self.resblock21 = ResnetBlock(out_ch*4, kernel_size=3)  # 16*16*64
        # self.resblock22 = ResnetBlock(out_ch*2, kernel_size=3)
        # self.resblock23 = ResnetBlock(out_ch*2, kernel_size=3)

        self.deconv3 = conv_decoder(2*out_ch*4, out_ch*4)
        self.resblock31 = ResnetBlock(out_ch*4, kernel_size=3)  # 32*32*64
        # self.resblock32 = ResnetBlock(out_ch, kernel_size=3)
        # self.resblock33 = ResnetBlock(out_ch, kernel_size=3)

        self.deconv4 = conv_decoder(2*out_ch*4, out_ch*2)
        self.resblock41 = ResnetBlock(out_ch*2, kernel_size=3)  # 64*64*32
        # self.resblock42 = ResnetBlock(output_channels, kernel_size=3)
        # self.resblock43 = ResnetBlock(output_channels, kernel_size=3)

        self.deconv5 = conv_decoder(2*out_ch*2, out_ch)  # 128*128*16
        self.resblock51 = ResnetBlock(out_ch, kernel_size=3)  # 128*128*16

        self.deconv6 = conv_decoder(2*out_ch, output_channels)
        self.resblock61 = ResnetBlock(
            output_channels, kernel_size=3)  # 256*256*3

        # self.refinement = refinement_module(output_channels)
        # self.convT2 = nn.ConvTranspose2d(
        #     2*out_ch, output_channels, 3, 1, 1)  # 256*256*3

        self.resblocks = resblocks

    def forward(self, encoding, catche):
        if self.resblocks:
            encoding = encoding.unsqueeze(2).unsqueeze(3)
            d0 = self.convT1(encoding)
            d1 = self.resblock11(self.deconv1(
                torch.cat((d0, catche[6]), dim=1)))
            d2 = self.resblock21(self.deconv2(
                torch.cat((d1, catche[5]), dim=1)))
            d3 = self.resblock31(self.deconv3(
                torch.cat((d2, catche[4]), dim=1)))
            d4 = self.resblock41(self.deconv4(
                torch.cat((d3, catche[3]), dim=1)))
            d5 = self.resblock51(self.deconv5(
                torch.cat((d4, catche[2]), dim=1)))
            d6 = self.resblock61(self.deconv6(
                torch.cat((d5, catche[1]), dim=1)))
            d7 = self.convT2(torch.cat((d6, catche[0]), dim=1))

        else:
            encoding = encoding.unsqueeze(2).unsqueeze(3)
            d0 = self.convT1(encoding)  # 4*4*128
            d1 = self.deconv1(torch.cat((d0, catche[5]), dim=1))  # 8*8*128
            d2 = self.deconv2(torch.cat((d1, catche[4]), dim=1))  # 16*16*64
            d3 = self.deconv3(torch.cat((d2, catche[3]), dim=1))  # 32*32*64
            d4 = self.deconv4(torch.cat((d3, catche[2]), dim=1))  # 64*64*32
            d5 = self.deconv5(torch.cat((d4, catche[1]), dim=1))  # 128*128*16
            d6 = self.deconv6(torch.cat((d5, catche[0]), dim=1))

        return d6


class Refinement_Decoder(nn.Module):
    def __init__(self, output_channels, input_channels):
        super(Refinement_Decoder, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels

        self.norm1 = nn.BatchNorm2d(self.input_channels//2)
        self.norm2 = nn.BatchNorm2d(self.input_channels//4)
        self.norm3 = nn.BatchNorm2d(self.output_channels)

        self.dconv1 = nn.ConvTranspose2d(
            self.input_channels, self.input_channels//2, kernel_size=4, stride=2, padding=1)
        self.deconv_level2 = nn.ConvTranspose2d(
            self.input_channels//2, self.input_channels//2, kernel_size=3, stride=1, padding=1)
        self.dconv2 = nn.ConvTranspose2d(
            self.input_channels//2, self.input_channels//4, kernel_size=4, stride=2, padding=1)
        self.deconv_level1 = nn.ConvTranspose2d(
            self.input_channels//4, self.input_channels//4, kernel_size=3, stride=1, padding=1)
        self.dconv3 = nn.ConvTranspose2d(
            self.input_channels//4, self.output_channels, kernel_size=8, stride=2, padding=3)
        self.deconv_level0 = nn.ConvTranspose2d(
            self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x, cache):
        f1 = F.relu(self.norm1(self.dconv1(x))) #h/4*w/4*outputput_channels//2
        # print(f1)
        f11 = F.relu(self.norm1(self.deconv_level2((f1 + cache[2])/2)))
        
        f2 = F.relu(self.norm2(self.dconv2(f11))) #h/2*w/2*outputput_channels//4
        
        f22 = F.relu(self.norm2(self.deconv_level1((f1 + cache[1])/2)))
        # print(f2)
        f3 = F.relu(self.norm3(self.dconv3(f22))) #h*w*outputput_channels
        
        f33 = F.sigmoid(self.norm3(self.deconv_level0((f2 + cache[0])/2)))
        # print(f3)
        return f33


class refinement_module(nn.Module):
    def __init__(self, output_channels):
        super(refinement_module, self).__init__()
        self.output_channels = output_channels
        self.block1_conv1 = nn.Conv2d(
            self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        self.block1_conv2 = nn.Conv2d(
            self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        self.block1_conv3 = nn.Conv2d(
            self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)

        self.norm1 = nn.BatchNorm2d(self.output_channels)

        self.block2_conv1 = nn.Conv2d(
            self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        self.block2_conv2 = nn.Conv2d(
            self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        self.block2_conv3 = nn.Conv2d(
            self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.BatchNorm2d(self.output_channels)

    def forward(self, x, warped_image=None):

        x1 = F.relu(self.block1_conv1(x))
        x1 = torch.sigmoid(self.block1_conv2(x1))
        # x1 = F.relu(self.block1_conv3(x1))
        if warped_image is not None:
            x2 = F.relu(self.block2_conv1(warped_image))
            x2 = torch.sigmoid(self.block2_conv2(x2))
        # x2 = F.relu(self.block2_conv3(x2))
            return 0.5*x1 + 0.5*x2
        else:
            return x1


if __name__ == '__main__':
    model = Refinement_Decoder(3, 128)
    x = torch.randn(8, 128, 16, 16)
    cache = [torch.randn(8, 32, 64, 64), torch.randn(8, 64, 32, 32), x]
    y = model(x, cache)
    print(y.shape)
