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

class conv_decoder(nn.Module):
    def __init__(self,nin, nout) -> None:
        super(conv_decoder, self).__init__()
        self.upconv_block = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        return self.upsample(self.upconv_block(input))      
        

class decoder(nn.Module):
    def __init__(self, input_channels, output_channels, resblocks = True) -> None:
        super(decoder, self).__init__()
        out_ch = 16
        # 1*1*input_channels to 4*4*output_channels
        self.output_channels = output_channels
        self.convT1 = nn.ConvTranspose2d(input_channels, input_channels, 4, 1, 0) #/4*4*input_channels
        
        self.deconv1 = conv_decoder(2*input_channels, input_channels) 
        self.resblock11 = ResnetBlock(input_channels, kernel_size=3) # 8*8*input_channels
        # self.resblock12 = ResnetBlock(out_ch*4, kernel_size=3)
        # self.resblock13 = ResnetBlock(out_ch*4, kernel_size=3)
        
        
        self.deconv2 = conv_decoder(2*input_channels, out_ch*8) 
        self.resblock21 = ResnetBlock(out_ch*8, kernel_size=3) # 16*16*128
        # self.resblock22 = ResnetBlock(out_ch*2, kernel_size=3)
        # self.resblock23 = ResnetBlock(out_ch*2, kernel_size=3)
        
        self.deconv3 = conv_decoder(2*out_ch*8, out_ch*4)
        self.resblock31 = ResnetBlock(out_ch*4, kernel_size=3) # 32*32*64
        # self.resblock32 = ResnetBlock(out_ch, kernel_size=3)
        # self.resblock33 = ResnetBlock(out_ch, kernel_size=3)
        
        self.deconv4 = conv_decoder(2*out_ch*4, out_ch*4) 
        self.resblock41 = ResnetBlock(out_ch*4, kernel_size=3) # 64*64*64
        # self.resblock42 = ResnetBlock(output_channels, kernel_size=3)
        # self.resblock43 = ResnetBlock(output_channels, kernel_size=3)
        
        self.deconv5 = conv_decoder(2*out_ch*4, out_ch*2) # 128*128*32
        self.resblock51 = ResnetBlock(out_ch*2, kernel_size=3) # 128*128*32
        
        self.deconv6 = conv_decoder(2*out_ch*2, out_ch)
        self.resblock61 = ResnetBlock(out_ch, kernel_size=3) # 256*256*16
        
        self.convT2 = nn.ConvTranspose2d(2*out_ch, output_channels, 3, 1, 1) # 256*256*3
        
    
    def forward(self, encoding, catche):
        if self.resblocks:
            d0 = self.convT1(encoding)
            d1 = self.resblock11(self.deconv1(torch.cat((d0, catche[6]), dim=1)))
            d2 = self.resblock21(self.deconv2(torch.cat((d1, catche[5]), dim=1)))
            d3 = self.resblock31(self.deconv3(torch.cat((d2, catche[4]), dim=1)))
            d4 = self.resblock41(self.deconv4(torch.cat((d3, catche[3]), dim=1)))
            d5 = self.resblock51(self.deconv5(torch.cat((d4, catche[2]), dim=1)))
            d6 = self.resblock61(self.deconv6(torch.cat((d5, catche[1]), dim=1)))
            d7 = self.convT2(torch.cat((d6, catche[0]), dim=1))
            
        
        else:
            d0 = self.convT1(encoding)
            d1 = self.deconv1(torch.cat((d0, catche[6]), dim=1))
            d2 = self.deconv2(torch.cat((d1, catche[5]), dim=1))
            d3 = self.deconv3(torch.cat((d2, catche[4]), dim=1))
            d4 = self.deconv4(torch.cat((d3, catche[3]), dim=1))
            d5 = self.deconv5(torch.cat((d4, catche[2]), dim=1))
            d6 = self.deconv6(torch.cat((d5, catche[1]), dim=1))
            d7 = self.convT2(torch.cat((d6, catche[0]), dim=1))
            
        return d7