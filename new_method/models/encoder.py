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
from models.kernel_estimator import Kernel_estimation

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
            # # # print(sharp_image.shape)
            h1 = self.conv1(sharp_image)
            # # # print(h1.shape)
            h2 = self.conv2(h1)
            # # # print(h2.shape)
            h3 = self.conv3(h2)
            # # # print(h3.shape)
            h4 = self.conv4(h3)
            # # # print(h4.shape)
            h5 = self.conv5(h4)
            # # # print(h5.shape)
            h6 = self.conv6(h5)
            # # # print(h6.shape)
            # h7 = self.conv7(h6)
            # ## # print(h7.shape)
            encoding = self.conv8(h6)
            # # # print(encoding.shape)
        return encoding.view(-1, self.output_channels), [h1, h2, h3, h4, h5, h6]


# basic pyramidal feature extractor model
class Pyramidal_feature_encoder(nn.Module):
    def __init__(self, output_channels, input_channels, dropout=0):
        super(Pyramidal_feature_encoder, self).__init__()

        self.output_channels = output_channels

        # self.norm1 = nn.BatchNorm2d(self.output_channels//4)
        # self.norm2 = nn.BatchNorm2d(self.output_channels//2)
        # self.norm3 = nn.BatchNorm2d(self.output_channels)

        self.conv_level1 = nn.Conv2d(
            input_channels, output_channels//8, kernel_size=3, stride=1, padding=1)
        # self.resblock1 = ResnetBlock(output_channels//8, kernel_size=3)
        self.resblock2 = ResnetBlock(output_channels//8, kernel_size=3)

        self.conv1 = nn.Conv2d(output_channels//8, self.output_channels //
                               4, kernel_size=5, stride=2, padding=2)  # H/2,W/2
        # self.resblock3 = ResnetBlock(output_channels//4, kernel_size=3)
        self.resblock4 = ResnetBlock(output_channels//4, kernel_size=3)

        self.conv2 = nn.Conv2d(self.output_channels//4, self.output_channels //
                               2, kernel_size=5, stride=2, padding=2)  # H/4,W/4
        # self.resblock5 = ResnetBlock(output_channels//2, kernel_size=3)
        self.resblock6 = ResnetBlock(output_channels//2, kernel_size=3)

        self.conv3 = nn.Conv2d(self.output_channels//2, self.output_channels,
                               kernel_size=3, stride=2, padding=1)  # H/8 W/8
        # self.resblock7 = ResnetBlock(output_channels, kernel_size=3)
        self.resblock8 = ResnetBlock(output_channels, kernel_size=3)
        # # output convolution; this can solve mixed memory warning, not know why
        # self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):

        # # # print(x.shape)
        f1 = F.relu(self.conv_level1(x), inplace=True)
        # f1 = self.resblock1(f1)
        f1 = self.resblock2(f1)  # H*W*output_channels//8
        # # print("cache:0 ", f1.shape)
        f11 = F.relu(self.conv1(f1), inplace=True)
        # f2 = self.resblock3(f11)
        f2 = self.resblock4(f11)  # H/2*W/2*output_channels//4
        # # print("cache:1 ", f2.shape)
        f22 = F.relu(self.conv2(f2), inplace=True)
        # f3 = self.resblock5(f22)
        f3 = self.resblock6(f22)  # H/4*W/4*output_channels//2
        # # print("cache:2 ", f3.shape)
        f4 = F.relu(self.conv3(f3), inplace=True)
        # f4 = self.resblock7(f4)
        f4 = self.resblock8(f4)  # H/8*W/8*output_channels
        # # print("output", f4.shape)
        if self.dropout is not None:
            output = self.dropout(f4)
        else:
            output = f4
        return output, [f1, f2, f3]

# multiheaded self attention module over the feature maps


class Feature_extractor(nn.Module):
    def __init__(self, output_channels, input_channels, nheads, dropout=0):
        super(Feature_extractor, self).__init__()

        self.feature_encoder = Pyramidal_feature_encoder(
            output_channels, input_channels, dropout)
        self.self_attention = nn.MultiheadAttention(
            output_channels, nheads, dropout=dropout, kdim=output_channels, vdim=output_channels)

    def forward(self, x):
        features, feature_scale = self.feature_encoder(x)
        # # # print(encoded_features.shape)
        encoded_features = posemb_sincos_2d(features) + features

        input_features = encoded_features.reshape(
            encoded_features.shape[0], encoded_features.shape[1], -1)
        input_features = input_features.permute(2, 0, 1)
        # # # print(encoded_features.shape)
        attn_features, attn_map = self.self_attention(
            input_features, input_features, input_features)
        attn_features = attn_features.permute(1, 2, 0)
        attn_features = attn_features.reshape(encoded_features.shape)
        attn_map = attn_map.reshape(encoded_features.shape[0], encoded_features.shape[2],
                                    encoded_features.shape[3], encoded_features.shape[2], encoded_features.shape[3])
        return attn_features, attn_map, features, feature_scale


# define deblurring network
class Deblurring_net_encoder(nn.Module):
    def __init__(self, output_channels, input_channels, kernel_size, dropout=0):
        super(Deblurring_net_encoder, self).__init__()
        self.combined_feature_encoder = Pyramidal_feature_encoder(
            output_channels*kernel_size*kernel_size, 2*input_channels, dropout)
        self.past_feature_encoder = Pyramidal_feature_encoder(
            output_channels, input_channels, dropout)
        self.current_feature_encoder = Pyramidal_feature_encoder(
            output_channels, input_channels, dropout)
        self.past_kernel = Kernel_estimation(kernel_size)
        self.current_kernel = Kernel_estimation(kernel_size)

    def forward(self, last_blur, current_blur):
        # # # print("last_blur.shape", last_blur.shape)
        # # print("past blur")
        last_blur_features, last_blur_feature_scale = self.past_feature_encoder(
            last_blur)
        # # print("current_blur")
        current_blur_features, current_blur_feature_scale = self.current_feature_encoder(
            current_blur)
        # # print("combined")
        combined_features, combined_feature_scale = self.combined_feature_encoder(
            torch.cat((last_blur, current_blur), dim=1))

        # # # print("last_blur_features.shape", last_blur_features.shape)
        # # # print("current_blur_features.shape", current_blur_features.shape)
        # # # print("combined_features.shape", combined_features.shape)
        # # # print("combined_feature_scale[0].shape", combined_feature_scale[0].shape)
        # # # print("combined_feature_scale[1].shape", combined_feature_scale[1].shape)
        # # # print("combined_feature_scale[2].shape", combined_feature_scale[2].shape)

        # # # print("last_blur_feature_scale[0].shape", last_blur_feature_scale[0].shape)
        # # # print("last_blur_feature_scale[1].shape", last_blur_feature_scale[1].shape)
        # # # print("last_blur_feature_scale[2].shape", last_blur_feature_scale[2].shape)

        # # # print("current_blur_feature_scale[0].shape", current_blur_feature_scale[0].shape)
        # # # print("current_blur_feature_scale[1].shape", current_blur_feature_scale[1].shape)
        # # # print("current_blur_feature_scale[2].shape", current_blur_feature_scale[2].shape)

        # past_features_0 = self.past_kernel(
        #     last_blur_feature_scale[0], combined_feature_scale[0])
        # current_features_0 = self.current_kernel(
        #     current_blur_feature_scale[0], combined_feature_scale[0]) # H*w

        # # # print("past_features_0.shape", past_features_0.shape)

        past_features_1 = self.past_kernel(
            last_blur_feature_scale[1], combined_feature_scale[1])
        current_features_1 = self.current_kernel(
            current_blur_feature_scale[1], combined_feature_scale[1])  # H/2 *W/2
        # # print("level 1 done")
        past_features_2 = self.past_kernel(
            last_blur_feature_scale[2], combined_feature_scale[2])
        current_features_2 = self.current_kernel(
            current_blur_feature_scale[2], combined_feature_scale[2])  # H/4 * W/4
        # # print("level 2 done")
        past_features_3 = self.past_kernel(
            last_blur_features, combined_features)
        current_features_3 = self.current_kernel(
            current_blur_features, combined_features)  # H/4 * W/4
        # # print("level 3 done")
        last_scale_feature = 0.5*past_features_3 + 0.5*current_features_3

        features = [current_blur_feature_scale[0], 0.5*past_features_1 +
                    0.5*current_features_1, 0.5*past_features_2 + 0.5*current_features_2]
        return last_scale_feature, features, current_blur_features, current_blur_feature_scale


class Feature_forcaster(nn.Module):
    def __init__(self, history_in_channels, current_in_channels, out_channels, nheads, dropout=0):
        super(Feature_forcaster, self).__init__()

        self.feature_projector1 = nn.Sequential(
            nn.Linear(current_in_channels, current_in_channels),
            nn.ReLU(inplace=True),
            # nn.Linear(current_in_channels, current_in_channels),
            # nn.ReLU(inplace=True),
            nn.Linear(current_in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        self.feature_projector2 = nn.Sequential(
            nn.Linear(history_in_channels, history_in_channels),
            nn.ReLU(inplace=True),
            # nn.Linear(history_in_channels, history_in_channels),
            # nn.ReLU(inplace=True),
            nn.Linear(history_in_channels, out_channels),
            nn.ReLU(inplace=True)
        )

        self.cross_correlation = nn.MultiheadAttention(
            out_channels, nheads, dropout=dropout, kdim=out_channels, vdim=out_channels)

        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self, sharp_features_in, history_encoding_in, current_encoding_in):
        # sharp encoding : [batch, in_channel, H, W]
        # history encoding : [batch, in_channel, H, W]
        # current encoding : [batch, out_channel, H, W]
        sharp_features = posemb_sincos_2d(
            sharp_features_in) + sharp_features_in
        history_encoding = posemb_sincos_2d(
            history_encoding_in) + history_encoding_in
        current_encoding = posemb_sincos_2d(
            current_encoding_in) + current_encoding_in
        N, fC, fH, fW = current_encoding.shape
        # project each features from [batch, in_channel, H, W] to [batch, out_channel, H, W]
        projected_sharp_features = sharp_features.permute(1, 0, 2, 3)
        projected_sharp_features = projected_sharp_features.reshape(
            projected_sharp_features.shape[0], -1)
        projected_sharp_features = projected_sharp_features.permute(1, 0)
        projected_sharp_features = self.feature_projector1(
            projected_sharp_features)
        projected_sharp_features = projected_sharp_features.permute(1, 0)
        projected_sharp_features = projected_sharp_features.reshape(
            projected_sharp_features.shape[0], sharp_features.shape[0], sharp_features.shape[2], sharp_features.shape[3])
        projected_sharp_features = projected_sharp_features.permute(1, 0, 2, 3)

        projected_blurred_features = history_encoding.permute(1, 0, 2, 3)
        projected_blurred_features = projected_blurred_features.reshape(
            projected_blurred_features.shape[0], -1)
        projected_blurred_features = projected_blurred_features.permute(1, 0)
        projected_blurred_features = self.feature_projector2(
            projected_blurred_features)
        projected_blurred_features = projected_blurred_features.permute(1, 0)
        projected_blurred_features = projected_blurred_features.reshape(
            projected_blurred_features.shape[0], history_encoding.shape[0], history_encoding.shape[2], history_encoding.shape[3])
        projected_blurred_features = projected_blurred_features.permute(
            1, 0, 2, 3)

        q = projected_sharp_features.reshape(
            projected_sharp_features.shape[0], projected_sharp_features.shape[1], -1)
        q = q.permute(2, 0, 1)

        k = projected_blurred_features.reshape(
            projected_blurred_features.shape[0], projected_blurred_features.shape[1], -1)
        k = k.permute(2, 0, 1)

        v = current_encoding.reshape(
            current_encoding.shape[0], current_encoding.shape[1], -1)
        v = v.permute(2, 0, 1)

        # # # print(encoded_features.shape)
        attn_features, attn_map = self.cross_correlation(q, k, v)
        attn_features = attn_features.permute(1, 2, 0)
        attn_features = attn_features.reshape(current_encoding.shape)
        # # # print(attn_map.shape)
        # attn_map = F.softmax(attn_map, dim=2) * F.softmax(attn_map, dim=1)

        # attn_map = self.softmax1(attn_map)
        # # # print(attn_features.shape, attn_map.shape)
        correlation_map = attn_map.reshape(
            current_encoding.shape[0], current_encoding.shape[2], current_encoding.shape[3], current_encoding.shape[2], current_encoding.shape[3])
        correlation_map = F.softmax(
            correlation_map, dim=2) * F.softmax(correlation_map, dim=1)
        # # # print(correlation_map.shape)
        match12, match_idx12 = attn_map.max(dim=2)  # (N, fH*fW)
        match21, match_idx21 = attn_map.max(dim=1)

        for b_idx in range(N):
            match21_b = match21[b_idx, :]
            match_idx12_b = match_idx12[b_idx, :]
            match21[b_idx, :] = match21_b[match_idx12_b]

        matched = (match12 - match21) == 0  # (N, fH*fW)
        coords_index = torch.arange(
            fH*fW).unsqueeze(0).repeat(N, 1).to(current_encoding.device)
        coords_index[matched] = match_idx12[matched]

        # matched coords
        coords_index = coords_index.reshape(N, fH, fW)
        coords_x = coords_index % fW
        coords_y = coords_index // fW

        coords_xy = torch.stack([coords_x, coords_y], dim=1).float()

        return attn_features, correlation_map, coords_xy


class Feature_predictor(nn.Module):
    def __init__(self, sharp_feature_channels, blurred_feature_channels, generation_time_encoding_channels, nheads, dropout=0):
        super(Feature_predictor, self).__init__()
        input_channels = sharp_feature_channels + \
            blurred_feature_channels + generation_time_encoding_channels + 1
        output_channels = sharp_feature_channels
        #################################################################
        # for sharing blur information using attention
        #################################################################
        self.blur_feature_projector = nn.Sequential(
            nn.Linear(output_channels, 2*output_channels),
            nn.ReLU(inplace=True),
            # nn.Linear(current_in_channels, current_in_channels),
            # nn.ReLU(inplace=True),
            nn.Linear(2*output_channels, output_channels),
            nn.ReLU(inplace=True)
        )
        self.blur_self_attention = nn.MultiheadAttention(
            output_channels, nheads, dropout=dropout, kdim=output_channels, vdim=output_channels)

        #################################################################
        # for predicting next frame from sharp image given history and time stamp
        #################################################################
        self.sharp_feature_projector = nn.Sequential(
            nn.Linear(input_channels, 2*output_channels),
            nn.ReLU(inplace=True),
            # nn.Linear(current_in_channels, current_in_channels),
            # nn.ReLU(inplace=True),
            nn.Linear(2*output_channels, output_channels),
            nn.ReLU(inplace=True)
        )

        self.sharp_self_attention = nn.MultiheadAttention(
            output_channels, nheads, dropout=dropout, kdim=output_channels, vdim=output_channels)

        #################################################################
        # cross attention for sampling next image from history
        #################################################################
        self.sampler_feature_projector = nn.Sequential(
            nn.Linear(output_channels +
                      generation_time_encoding_channels + 1, 2*output_channels),
            nn.ReLU(inplace=True),
            # nn.Linear(current_in_channels, current_in_channels),
            # nn.ReLU(inplace=True),
            nn.Linear(2*output_channels, output_channels),
            nn.ReLU(inplace=True)
        )

        self.cross_attention = nn.MultiheadAttention(
            output_channels, nheads, dropout=dropout, kdim=output_channels, vdim=output_channels)

        self.flow_conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.flow_conv2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.flow_conv3 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, blur_feature, blur_feature_scale, sharp_feature, sharp_feature_scale, time_encoding):
        N, fC, fH, fW = blur_feature.shape
        # print("blur_feature", blur_feature.shape)
        #############################################################
        # blur attention encoding
        #############################################################
        features = posemb_sincos_2d(blur_feature) + blur_feature
        # print("features", features.shape)

        projected_blur_feature = features.permute(1, 0, 2, 3)
        # print("projected_blur_feature", projected_blur_feature.shape)
        projected_blur_feature = projected_blur_feature.reshape(
            projected_blur_feature.shape[0], -1)
        # print("projected_blur_feature", projected_blur_feature.shape)
        projected_blur_feature = projected_blur_feature.permute(1, 0)
        # print("projected_blur_feature", projected_blur_feature.shape)
        projected_blur_feature = self.blur_feature_projector(
            projected_blur_feature)
        # print("projected_blur_feature", projected_blur_feature.shape)
        projected_blur_feature = projected_blur_feature.permute(1, 0)
        # print("projected_blur_feature", projected_blur_feature.shape)
        projected_blur_feature = projected_blur_feature.reshape(
            projected_blur_feature.shape[0], blur_feature.shape[0], blur_feature.shape[2], blur_feature.shape[3])
        # print("projected_blur_feature", projected_blur_feature.shape)
        projected_blur_feature = projected_blur_feature.permute(1, 0, 2, 3)
        # print("projected_blur_feature", projected_blur_feature.shape)

        input_features1 = projected_blur_feature.reshape(
            projected_blur_feature.shape[0], projected_blur_feature.shape[1], -1)
        # print("input_features1", input_features1.shape)
        input_features1 = input_features1.permute(2, 0, 1)
        # print("input_features1", input_features1.shape)
        blur_attn_features, blur_attn_map = self.blur_self_attention(
            input_features1, input_features1, input_features1)

        blur_attn_features = blur_attn_features.permute(1, 2, 0)
        blur_attn_features = blur_attn_features.reshape(
            projected_blur_feature.shape)
        # print("blur_attn_features", blur_attn_features.shape)
        blur_attn_map = blur_attn_map.reshape(projected_blur_feature.shape[0], projected_blur_feature.shape[2],
                                              projected_blur_feature.shape[3], projected_blur_feature.shape[2], projected_blur_feature.shape[3])
        # print("blur_attn_features", blur_attn_features.shape)
        y = blur_attn_map.permute(0, 3, 4, 1, 2).reshape(
            blur_attn_map.shape[0], blur_attn_map.shape[3], blur_attn_map.shape[4], -1).max(dim=3)[0]

        range_variation = (1/y)*(1/(2*3.1415))*(1/blur_attn_map.shape[3])
        range_variation = range_variation.unsqueeze(1)
        # print("range_variation", range_variation.shape)
        # _____________________________________________________________
        # output: blur_attn_features, blur_attn_map, range_variation
        # _____________________________________________________________

        #############################################################
        # next frame prediction using sharp image
        #############################################################

        latent_features = posemb_sincos_2d(sharp_feature) + sharp_feature
        # print("latent_features", latent_features.shape)
        latent_features = torch.cat(
            [latent_features, blur_attn_features, time_encoding, range_variation], dim=1)
        # print("latent_features", latent_features.shape)
        projected_latent_features = latent_features.permute(1, 0, 2, 3)
        # print("projected_latent_features", projected_latent_features.shape)
        projected_latent_features = projected_latent_features.reshape(
            projected_latent_features.shape[0], -1)
        # print("projected_latent_features", projected_latent_features.shape)
        projected_latent_features = projected_latent_features.permute(1, 0)
        # print("projected_latent_features", projected_latent_features.shape)
        projected_latent_features = self.sharp_feature_projector(
            projected_latent_features)
        # print("projected_latent_features", projected_latent_features.shape)
        projected_latent_features = projected_latent_features.permute(1, 0)
        # print("projected_latent_features", projected_latent_features.shape)
        projected_latent_features = projected_latent_features.reshape(
            projected_latent_features.shape[0], latent_features.shape[0], latent_features.shape[2], latent_features.shape[3])
        # print("projected_latent_features", projected_latent_features.shape)
        projected_latent_features = projected_latent_features.permute(
            1, 0, 2, 3)  # n,D,h,w
        # print("projected_latent_features", projected_latent_features.shape)
        input_features2 = projected_latent_features.reshape(
            projected_latent_features.shape[0], projected_latent_features.shape[1], -1)
        # print("input_features2", input_features2.shape)
        input_features2 = input_features2.permute(2, 0, 1)  # h*w, n, D
        # print("input_features2", input_features2.shape)
        v1 = sharp_feature.reshape(
            sharp_feature.shape[0], sharp_feature.shape[1], -1)
        v1 = v1.permute(2, 0, 1)
        # print("v1", v1.shape)
        new_features_predicted, predictor_map = self.sharp_self_attention(
            input_features2, input_features2, v1)
        # print("new_features_predicted", new_features_predicted.shape)
        new_features_predicted = new_features_predicted.permute(
            1, 2, 0)  # n, D, h*w
        # print("new_features_predicted", new_features_predicted.shape)
        new_features_predicted = new_features_predicted.reshape(
            projected_blur_feature.shape)
        # print("new_features_predicted", new_features_predicted.shape)
        predictor_map_4d = predictor_map.reshape(latent_features.shape[0], latent_features.shape[2],
                                                 latent_features.shape[3], latent_features.shape[2], latent_features.shape[3])
        # print("predictor_map_4d", predictor_map_4d.shape)
        # _____________________________________________________________
        # output:  new_features, predictor_map
        # _____________________________________________________________

        ##############################################################
        # next frame sampler
        ##############################################################
        key = posemb_sincos_2d(sharp_feature) + sharp_feature
        key = key.reshape(
            key.shape[0], key.shape[1], -1)
        key = key.permute(2, 0, 1)
        # print("key", key.shape)
        sampled_feature = torch.cat([posemb_sincos_2d(
            blur_attn_features) + blur_attn_features, time_encoding, range_variation], dim=1)
        # print("sampled_feature", sampled_feature.shape)
        projected_sampled_image_feature = sampled_feature.permute(1, 0, 2, 3)
        projected_sampled_image_feature = projected_sampled_image_feature.reshape(
            projected_sampled_image_feature.shape[0], -1)
        projected_sampled_image_feature = projected_sampled_image_feature.permute(
            1, 0)
        # print("projected_sampled_image_feature",
        #   projected_sampled_image_feature.shape)
        projected_sampled_image_feature = self.sampler_feature_projector(
            projected_sampled_image_feature)
        projected_sampled_image_feature = projected_sampled_image_feature.permute(
            1, 0)
        projected_sampled_image_feature = projected_sampled_image_feature.reshape(
            projected_sampled_image_feature.shape[0], sampled_feature.shape[0], sampled_feature.shape[2], sampled_feature.shape[3])
        projected_sampled_image_feature = projected_sampled_image_feature.permute(
            1, 0, 2, 3)  # n,D,h,w
        # print("projected_sampled_image_feature",
        #   projected_sampled_image_feature.shape)
        input_features3 = projected_sampled_image_feature.reshape(
            projected_sampled_image_feature.shape[0], projected_sampled_image_feature.shape[1], -1)

        input_features3 = input_features3.permute(2, 0, 1)  # h*w, n, D
        # print("input_features3", input_features3.shape)
        new_sampled_features, sampled_map = self.cross_attention(
            input_features3, key, v1)
        # print("new_sampled_features", new_sampled_features.shape)
        new_sampled_features = new_sampled_features.permute(
            1, 2, 0)  # n, D, h*w
        new_sampled_features = new_sampled_features.reshape(
            projected_blur_feature.shape)
        # print("new_sampled_features", new_sampled_features.shape)
        sampled_map_4d = sampled_map.reshape(sampled_feature.shape[0], sampled_feature.shape[2],
                                             sampled_feature.shape[3], sampled_feature.shape[2], sampled_feature.shape[3])

        ###############################################################
        # belief propagation
        ###############################################################
        final_transformation_map = predictor_map*sampled_map
        # print("final_transformation_map", final_transformation_map.shape)
        ###############################################################
        # flow computation
        ###############################################################

        match12, match_idx12 = final_transformation_map.max(
            dim=2)  # (N, fH*fW)
        match21, match_idx21 = final_transformation_map.max(dim=1)

        for b_idx in range(N):
            match21_b = match21[b_idx, :]
            match_idx12_b = match_idx12[b_idx, :]
            match21[b_idx, :] = match21_b[match_idx12_b]

        matched = (match12 - match21) == 0  # (N, fH*fW)
        coords_index = torch.arange(
            fH*fW).unsqueeze(0).repeat(N, 1).to(sharp_feature.device)
        coords_index[matched] = match_idx12[matched]

        # matched coords
        coords_index = coords_index.reshape(N, fH, fW)
        coords_x = coords_index % fW
        coords_y = coords_index // fW

        coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
        new_size = (2 * coords_xy.shape[2], 2 * coords_xy.shape[3])
        coords_xy_3 = 2 * \
            F.interpolate(coords_xy, size=new_size,
                          mode='bilinear', align_corners=True)
        coords_xy_3 = self.flow_conv1(coords_xy_3)
        # print("coords_xy_3", coords_xy_3.shape)
        ###############################################################
        # multi-scale feature warping
        ###############################################################
        # print("sharp_feature_scale2", sharp_feature_scale[2].shape)
        sharp_feature_scale[2] = warp(sharp_feature_scale[2], -1*coords_xy_3)
        # print("sharp_feature_scale2", sharp_feature_scale[2].shape)

        new_size = (2 * coords_xy_3.shape[2], 2 * coords_xy_3.shape[3])
        coords_xy_2 = 2 * \
            F.interpolate(coords_xy_3, size=new_size,
                          mode='bilinear', align_corners=True)
        # # # print(sharp_init_feature_scale[1].max())
        coords_xy_2 = self.flow_conv2(coords_xy_2)
        # print("coords_xy_2", coords_xy_2.shape)
        # print("sharp_feature_scale1", sharp_feature_scale[1].shape)
        sharp_feature_scale[1] = warp(sharp_feature_scale[1], -1*coords_xy_2)
        # print("sharp_feature_scale1", sharp_feature_scale[1].shape)
        new_size = (2 * coords_xy_2.shape[2], 2 * coords_xy_2.shape[3])
        coords_xy_3 = 2 * \
            F.interpolate(coords_xy_2, size=new_size,
                          mode='bilinear', align_corners=True)
        # # # print(sharp_init_feature_scale[1].max())
        coords_xy_3 = self.flow_conv3(coords_xy_3)
        sharp_feature_scale[0] = warp(sharp_feature_scale[0], -1*coords_xy_3)

        final_features = new_sampled_features*0.4 + \
            sharp_feature*0.2 + new_features_predicted*0.4
        # print(final_features.device, sharp_feature_scale[0].device,
        #   blur_attn_features.device, final_transformation_map.device)
        return blur_attn_features, final_transformation_map, final_features, sharp_feature_scale


# class Feature_sampler(nn.Module):

def warp(features, flow):
    B, C, H, W = features.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(features.device)
    vgrid = Variable(grid) + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    # # # print(features.device, vgrid.device)
    vgrid = vgrid.permute(0, 2, 3, 1)
    flow = flow.permute(0, 2, 3, 1)
    output = F.grid_sample(features, vgrid.to(
        features.device), align_corners=True)
    mask = torch.autograd.Variable(
        torch.ones(features.size())).to(features.device)
    mask = F.grid_sample(mask, vgrid, align_corners=True)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output*mask


def posemb_sincos_2d(patches, temperature=1000, dtype=torch.float32):
    _, dim, h, w, device, dtype = *patches.shape, patches.device, patches.dtype
    # # # print(patches.shape, h, w, dim, device, dtype)
    y, x = torch.meshgrid(torch.arange(h, device=device),
                          torch.arange(w, device=device))
    # # # print(y.shape, x.shape)
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    pe = pe.reshape(1, h, w, dim).permute(0, 3, 1, 2)
    return pe.type(dtype)


if __name__ == "__main__":
    # model = Feature_extractor(128, 3, 8).cuda()
    x = torch.randn(8, 128, 32, 32).to('cuda')
    # # y, y_cache = model(x)
    # # # # print(y.shape)
    # # # # print(y_cache[0].shape, y_cache[1].shape, y_cache[2].shape)
    # attn_features, attn_map, features, feature_scale = model(x)
    # # # print(attn_features.shape, attn_map.shape, features.shape)
    # model2 = Feature_forcaster(128,128,128,8).cuda()
    # y2, corr_map, co_ord = model2(x,x,x)
    # # # print(y2.shape, corr_map.shape, co_ord)
    # y3 = warp(y2,co_ord)
    # # # print(y3.shape)
