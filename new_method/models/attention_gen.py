import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import os
from models.encoder import Pyramidal_feature_encoder, Feature_extractor, Feature_forcaster, warp
from models.decoder import Refinement_Decoder
from models.positional_encoding import Positional_encoding
from utils.loss import KLCriterion, PSNR, SSIM, SmoothMSE


class Attention_Gen(nn.Module):
    """parameters for the model

    Args:
        sharp_encoder:
            output_channels
            input_channels
            nheads
        
        blur_encoder:
            output_channels
            input_channels
            nheads
        
        positional:
            output_channels
        
        feature_forcasting:
            output_channels
            input_channels
            nheads
        
            
            
    """
    def __init__(self, args, batch_size=2, prob_for_frame_drop=0, lr=0.001,dropout = 0):
        super(Attention_Gen, self).__init__()
        self.args = args
        if args.train or args.evaluate:
            self.batch_size = args.training_parameters["batch_size"]
            self.dropout = args.training_parameters["dropout"]
        elif args.test:
            self.batch_size = args.testing_parameters["batch_size"]
            self.dropout = 0
        else:
            self.batch_size = batch_size
            self.dropout = dropout
            
        if args.test != True:
            if args.train or args.evaluate:
                self.prob_for_frame_drop = args.training_parameters["prob_for_frame_drop"]
                self.lr = args.training_parameters["lr"]

            else:
                self.prob_for_frame_drop = prob_for_frame_drop
                self.lr = lr
        else:
            self.prob_for_frame_drop = 0
            
        self.sharp_encoder = Feature_extractor(self.args.attention_gen['sharp_encoder']['output_channels'], self.args.attention_gen['sharp_encoder']['input_channels'], self.args.attention_gen['sharp_encoder']['nheads'],self.dropout)
        self.blur_encoder = Feature_extractor(self.args.attention_gen['blur_encoder']['output_channels'], self.args.attention_gen['blur_encoder']['input_channels'], self.args.attention_gen['blur_encoder']['nheads'],self.dropout)
        
        # positional encoding
        self.pos_encoder = Positional_encoding(
            self.args.attention_gen["positional"]['output_channels'])
        
        self.feature_forcasting = Feature_forcaster(self.args.attention_gen['feature_forcasting']['output_channels'], self.args.attention_gen['feature_forcasting']['input_channels'], self.args.attention_gen['feature_forcasting']['nheads'],self.dropout)
        
        self.decoder = Refinement_Decoder(self.args.attention_gen['decoder']['output_channels'], self.args.attention_gen['decoder']['input_channels'])

    def sequence_training(self, sharp_images, motion_blur_image):
        # blur_encoder
        attn_blur_features, blur_attn_map, encoded_blur_features, blur_feature_scale = self.blur_encoder(motion_blur_image)
        
        frame_use = np.random.uniform(
                0, 1, len(sharp_images)) >= self.prob_for_frame_drop
        
        generated_sequence = {}
        gt_sequence = {}
        
        last_time_stamp = 0
        init_flow = torch.zeros((self.batch_size, 2, sharp_images.shape[2]//4,sharp_images.shape[3]//4)).to(sharp_images.device)
        for i in range(1, len(sharp_images)):
                if frame_use[i]:
                    gt_sequence[i] = sharp_images[i].detach().cpu()
                    # inital sharp encoder
                    attn_sharp_init_features, init_sharp_attn_map, encoded_sharp_init_features, sharp_init_feature_scale = self.sharp_encoder(sharp_images[last_time_stamp])
                    
                    # inital positional encoding
                    init_time_info = self.pos_encoder(
                        last_time_stamp, last_time_stamp, len(sharp_images), self.batch_size).to(encoded_sharp_init_features.device)
                    # stack inital time info with each feature from the encoder
                    init_time_info = init_time_info.repeat(1, 1, encoded_sharp_init_features[0], encoded_sharp_init_features[1])
                    
                    # inital feature info for feature forcasting
                    init_feature_info = torch.cat( (attn_sharp_init_features, init_time_info,init_flow), dim=1)
                    
                    
                    
                    # genration positional encoding
                    gen_time_info = self.pos_encoder(
                        last_time_stamp, i, len(sharp_images), self.batch_size).to(encoded_sharp_init_features.device)
                    gen_time_info = gen_time_info.repeat(1, 1, encoded_sharp_init_features[0], encoded_sharp_init_features[1])
                    
                    # distribution for the feature forcasting corresponding to the current time stamp
                    blur_feature_info = torch.cat((attn_blur_features, gen_time_info), dim=1)
                    
                    
                    # feature forcasting
                    attn_features_i, correlation_map_i, coords_xy_i = self.feature_forcasting(init_feature_info, blur_feature_info, attn_sharp_init_features)
                    
                    # warping sharp_image_features based on the flow
                    # scale  = 1/4
                    sharp_init_feature_scale[2] = warp(sharp_init_feature_scale[2], coords_xy_i)
                    # scale = 1/2
                    # upsample the flow to the size of the feature map
                    new_size = (2* coords_xy_i.shape[2], 2* coords_xy_i.shape[3])
                    coords_xy_i_2 = 2 * F.interpolate(coords_xy_i, size=new_size, mode='bilinear', align_corners=True)
                    sharp_init_feature_scale[1] = warp(sharp_init_feature_scale[1], coords_xy_i_2)
                    # scale = 1
                    # upsample the flow to the size of the feature map
                    new_size = (2* coords_xy_i_2.shape[2], 2* coords_xy_i_2.shape[3])
                    coords_xy_i_4 = 2 * F.interpolate(coords_xy_i_2, size=new_size, mode='bilinear', align_corners=True)
                    sharp_init_feature_scale[0] = warp(sharp_init_feature_scale[0], coords_xy_i_4)
                    #refinement decoder
                    gen_sharp_image = self.decoder(attn_features_i, sharp_init_feature_scale) 
                    generated_sequence[i] = gen_sharp_image.detach().cpu()
                    
                    init_flow = coords_xy_i -  init_flow
                    # normalized flow
                    init_flow[:,0,:,:] = init_flow[:,0,:,:] / (sharp_images.shape[3]//4)
                    init_flow[:,1,:,:] = init_flow[:,1,:,:] / (sharp_images.shape[2]//4)
                    last_time_stamp = i
                    
                else:
                    continue

        return generated_sequence, gt_sequence
        
    def single_image_training(self, sharp_images, motion_blur_image):
        return NotImplementedError
    
    def sequence_testing(self, sharp_images, motion_blur_image):
        return NotImplementedError
    
    def single_image_testing(self, sharp_images, motion_blur_image):
        return NotImplementedError   

    def forward(self, sharp_images, motion_blur_image, mode, single_image_prediction=False):
        if mode == "train":
            if single_image_prediction:
                gen_seq, losses = self.single_image_training(sharp_images, motion_blur_image)
            else:
                gen_seq, losses = self.sequence_training(sharp_images, motion_blur_image)
        else:
            if single_image_prediction:
                gen_seq, losses = self.single_image_testing(sharp_images, motion_blur_image)
            else:
                gen_seq, losses = self.sequence_testing(sharp_images, motion_blur_image)
         
        return gen_seq, losses