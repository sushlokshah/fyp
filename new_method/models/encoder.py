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
from torch.autograd import Variable


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


# basic pyramidal feature extractor model
class Pyramidal_feature_encoder(nn.Module):
    def __init__(self, output_channels, input_channels, dropout = 0):
        super(Pyramidal_feature_encoder, self).__init__()
    
        self.output_channels = output_channels
        
        self.norm1 = nn.BatchNorm2d(self.output_channels//4)
        self.norm2 = nn.BatchNorm2d(self.output_channels//2)
        self.norm3 = nn.BatchNorm2d(self.output_channels)

        self.conv1 = nn.Conv2d(input_channels, self.output_channels//4, kernel_size=7, stride=2, padding=3) #H/2,W/2
        self.conv2 = nn.Conv2d(self.output_channels//4, self.output_channels//2, kernel_size=3, stride=2, padding=1) #H/4,W/4
        self.conv3 = nn.Conv2d(self.output_channels//2,self.output_channels, kernel_size=3, stride=2, padding=1) # H/8 W/8

        # # output convolution; this can solve mixed memory warning, not know why
        # self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
    def forward(self, x):
        
        # print(x.shape)
        f1 = F.relu(self.norm1(self.conv1(x)), inplace=True)
        f2 = F.relu(self.norm2(self.conv2(f1)), inplace=True)
        f3 = F.relu(self.norm3(self.conv3(f2)), inplace=True)
        
        if self.dropout is not None:
            output = self.dropout(f3)
        else:
            output = f3
        return output, [f1, f2, f3] 
    
    
#multiheaded self attention module over the feature maps

class Feature_extractor(nn.Module):
    def __init__(self, output_channels, input_channels, nheads, dropout = 0):
        super(Feature_extractor, self).__init__()
        
        self.feature_encoder = Pyramidal_feature_encoder(output_channels, input_channels, dropout)
        self.self_attention = nn.MultiheadAttention(output_channels, nheads, dropout=dropout, kdim=output_channels, vdim=output_channels)
    
    def forward(self, x):
        encoded_features, feature_scale = self.feature_encoder(x)
        # print(encoded_features.shape)
        input_features = encoded_features.reshape(encoded_features.shape[0], encoded_features.shape[1], -1)
        input_features = input_features.permute(2,0,1)
        # print(encoded_features.shape)
        attn_features, attn_map = self.self_attention(input_features, input_features, input_features)
        attn_features = attn_features.permute(1,2,0)
        attn_features = attn_features.reshape(encoded_features.shape)
        attn_map = attn_map.reshape(encoded_features.shape[0], encoded_features.shape[2],encoded_features.shape[3], encoded_features.shape[2], encoded_features.shape[3])
        return attn_features, attn_map, encoded_features, feature_scale
    

class Feature_forcaster(nn.Module):
    def __init__(self, history_in_channels,current_in_channels, out_channels, nheads, dropout = 0):
        super(Feature_forcaster, self).__init__()
        
        self.feature_projector1 = nn.Sequential(
            nn.Linear(current_in_channels, current_in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(current_in_channels, current_in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(current_in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        self.feature_projector2 = nn.Sequential(
            nn.Linear(history_in_channels, history_in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(history_in_channels, history_in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(history_in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cross_correlation = nn.MultiheadAttention(out_channels, nheads, dropout=dropout, kdim=out_channels, vdim=out_channels)
        
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        
        
    def forward(self, sharp_features, history_encoding, current_encoding):
        # sharp encoding : [batch, in_channel, H, W]
        # history encoding : [batch, in_channel, H, W]
        # current encoding : [batch, out_channel, H, W]
        N, fC, fH, fW = current_encoding.shape
        # project each features from [batch, in_channel, H, W] to [batch, out_channel, H, W]
        projected_sharp_features = sharp_features.permute(1,0,2,3)
        projected_sharp_features = projected_sharp_features.reshape(projected_sharp_features.shape[0], -1)
        projected_sharp_features = projected_sharp_features.permute(1,0)
        projected_sharp_features = self.feature_projector1(projected_sharp_features)
        projected_sharp_features = projected_sharp_features.permute(1,0)
        projected_sharp_features = projected_sharp_features.reshape(projected_sharp_features.shape[0],sharp_features.shape[0], sharp_features.shape[2], sharp_features.shape[3])
        projected_sharp_features = projected_sharp_features.permute(1,0,2,3)
        
        projected_blurred_features = history_encoding.permute(1,0,2,3)
        projected_blurred_features = projected_blurred_features.reshape(projected_blurred_features.shape[0], -1)
        projected_blurred_features = projected_blurred_features.permute(1,0)
        projected_blurred_features = self.feature_projector2(projected_blurred_features)
        projected_blurred_features = projected_blurred_features.permute(1,0)
        projected_blurred_features = projected_blurred_features.reshape(projected_blurred_features.shape[0],history_encoding.shape[0], history_encoding.shape[2], history_encoding.shape[3])
        projected_blurred_features = projected_blurred_features.permute(1,0,2,3)
        
         
        q = projected_sharp_features.reshape(projected_sharp_features.shape[0], projected_sharp_features.shape[1], -1)
        q = q.permute(2,0,1)
        
        k = projected_blurred_features.reshape(projected_blurred_features.shape[0], projected_blurred_features.shape[1], -1)
        k = k.permute(2,0,1)
        
        v = current_encoding.reshape(current_encoding.shape[0], current_encoding.shape[1], -1)
        v = v.permute(2,0,1)
        
        # print(encoded_features.shape)
        attn_features, attn_map = self.cross_correlation(q, k, v)
        attn_features = attn_features.permute(1,2,0)
        attn_features = attn_features.reshape(current_encoding.shape)
        # print(attn_features.shape, attn_map.shape)
        correlation_map = attn_map.reshape(current_encoding.shape[0], current_encoding.shape[2],current_encoding.shape[3], current_encoding.shape[2], current_encoding.shape[3])
        match12, match_idx12 = attn_map.max(dim=2) # (N, fH*fW)
        match21, match_idx21 = attn_map.max(dim=1)

        for b_idx in range(N):
            match21_b = match21[b_idx,:]
            match_idx12_b = match_idx12[b_idx,:]
            match21[b_idx,:] = match21_b[match_idx12_b]

        matched = (match12 - match21) == 0  # (N, fH*fW)
        coords_index = torch.arange(fH*fW).unsqueeze(0).repeat(N,1).to(current_encoding.device)
        coords_index[matched] = match_idx12[matched]

        # matched coords
        coords_index = coords_index.reshape(N, fH, fW)
        coords_x = coords_index % fW
        coords_y = coords_index // fW

        coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
        
        return attn_features, correlation_map, coords_xy
        
def warp(features, flow):
    B, C, H, W = features.size()
    # mesh grid
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float().to(features.device)
    vgrid = Variable(grid) + flow

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    print(features.device, vgrid.device)
    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    flow = flow.permute(0 ,2 ,3 ,1)
    output = F.grid_sample(features, vgrid.to(features.device),align_corners=True)
    mask = torch.autograd.Variable(torch.ones(features.size())).to(features.device)
    mask = F.grid_sample(mask, vgrid, align_corners=True)
    mask[mask <0.9999] = 0
    mask[mask >0] = 1
    
    return output*mask        



if __name__ == "__main__":
    model = Feature_extractor(128, 3, 4)
    x = torch.randn(8, 3, 128, 128)
    y, attn_map, encoded_features, feature_scale = model(x)
    # print(attn_map.shape)
    print(feature_scale[0].shape)
    print(feature_scale[1].shape)
    print(feature_scale[2].shape)
    # model2 = Feature_forcaster(128,128,8)
    # y2, corr_map, co_ord = model2(y, encoded_features, encoded_features)
    # print(y2.shape, corr_map.shape, co_ord)
    # y3 = warp(y2,co_ord)
    # print(y3.shape)