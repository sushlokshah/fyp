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
        
        decoder:
            output_channels
            input_channels
            
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
            
            if args.optimizer["optimizer_name"] == "AdamW":
                self.optimizer = optim.AdamW
        else:
            self.prob_for_frame_drop = 0
            
        self.sharp_encoder = Feature_extractor(self.args.attention_gen['sharp_encoder']['output_channels'], self.args.attention_gen['sharp_encoder']['input_channels'], self.args.attention_gen['sharp_encoder']['nheads'],self.dropout)
        self.blur_encoder = Feature_extractor(self.args.attention_gen['blur_encoder']['output_channels'], self.args.attention_gen['blur_encoder']['input_channels'], self.args.attention_gen['blur_encoder']['nheads'],self.dropout)
        
        # positional encoding
        self.pos_encoder = Positional_encoding(
            self.args.attention_gen["positional"]['output_channels'])
        
        history_in_channels = self.args.attention_gen['blur_encoder']['output_channels'] + self.args.attention_gen['positional']['output_channels']
        current_in_channels = self.args.attention_gen['sharp_encoder']['output_channels'] + self.args.attention_gen['positional']['output_channels'] + 2
        #  history_in_channels,current_in_channels, out_channels, nheads, dropout = 0
        self.feature_forcasting = Feature_forcaster(history_in_channels, current_in_channels, self.args.attention_gen['sharp_encoder']['output_channels'], self.args.attention_gen['feature_forcasting']['nheads'], self.dropout)
        self.decoder = Refinement_Decoder(self.args.attention_gen['decoder']['output_channels'], self.args.attention_gen['decoder']['input_channels'])
        
        self.flow_conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.flow_conv2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.mse_criterion = nn.L1Loss()
        self.ssim_criterion = SSIM()
        self.psnr_criterion = PSNR()
        
        if args.test != True:
            self.init_optimizer()
            
    def init_optimizer(self):
        self.sharp_encoder_optimizer = self.optimizer(self.sharp_encoder.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.blur_encoder_optimizer = self.optimizer(self.blur_encoder.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.feature_forcasting_optimizer = self.optimizer(self.feature_forcasting.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.decoder_optimizer = self.optimizer(self.decoder.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.flow_conv1_optimizer = self.optimizer(self.flow_conv1.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.flow_conv2_optimizer = self.optimizer(self.flow_conv2.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))      
    
    def sequence_training(self, sharp_images, motion_blur_image):
        # blur_encoder
        attn_blur_features, blur_attn_map, encoded_blur_features, blur_feature_scale = self.blur_encoder(motion_blur_image)
        
        frame_use = np.random.uniform(
                0, 1, len(sharp_images)) >= self.prob_for_frame_drop
        
        generated_sequence = {}
        gt_sequence = {}
        self.reconstruction_loss_post = torch.tensor(0)
        self.psnr_post = torch.tensor(0)
        self.ssim_post = torch.tensor(0)
        last_time_stamp = 0
        #print(sharp_images.shape)
        import sys
        init_flow = torch.zeros((self.batch_size, 2, sharp_images[0].shape[2]//8,sharp_images[0].shape[3]//8)).to(sharp_images.device)
        rows = torch.arange(0, sharp_images[0].shape[2]//8).view(1, -1).repeat(sharp_images[0].shape[3]//8, 1)
        coloumns = torch.arange(0, sharp_images[0].shape[3]//8).view(-1, 1).repeat(1, sharp_images[0].shape[2]//8)
        init_corrdinates = torch.stack((rows, coloumns), dim=2).permute(2,0,1).unsqueeze(0).repeat(self.batch_size, 1, 1, 1).to(sharp_images.device)
        # print(init_corrdinates.shape)
        # print(init_flow.indices())
        # sys.exit(0)
        for i in range(0, len(sharp_images)):
            if frame_use[i]:
                gt_sequence[i] = sharp_images[i].detach().cpu()
                # inital sharp encoder
                attn_sharp_init_features, init_sharp_attn_map, encoded_sharp_init_features, sharp_init_feature_scale = self.sharp_encoder(sharp_images[last_time_stamp])
                
                # inital positional encoding
                init_time_info = self.pos_encoder(
                    last_time_stamp, last_time_stamp, len(sharp_images), self.batch_size).to(encoded_sharp_init_features.device)
                # stack inital time info with each feature from the encoder
                #print("init_time_info", init_time_info.shape)
                init_time_info = init_time_info.repeat(encoded_sharp_init_features.shape[2], encoded_sharp_init_features.shape[3],1,1).permute(2,3,0,1)
                #print("init_time_info", init_time_info.shape)	
                #print("attention_sharp_init_features", attn_sharp_init_features.shape)
                #print("init_flow", init_flow.shape)	
                # inital feature info for feature forcasting
                init_feature_info = torch.cat( (attn_sharp_init_features, init_time_info,init_flow), dim=1)
                
                
                
                # genration positional encoding
                gen_time_info = self.pos_encoder(
                    last_time_stamp, i, len(sharp_images), self.batch_size).to(encoded_sharp_init_features.device)
                gen_time_info = gen_time_info.repeat(encoded_sharp_init_features.shape[2], encoded_sharp_init_features.shape[3],1,1).permute(2,3,0,1)
                
                # distribution for the feature forcasting corresponding to the current time stamp
                blur_feature_info = torch.cat((attn_blur_features, gen_time_info), dim=1)
                
                
                # feature forcasting
                attn_features_i, correlation_map_i, coords_xy_i = self.feature_forcasting(init_feature_info, blur_feature_info, attn_sharp_init_features)
                
                # #print("attn_features_i", attn_features_i)
                current_flow = self.flow_conv1(init_corrdinates - coords_xy_i)
                # warping sharp_image_features based on the flow
                # scale  = 1/4
                sharp_init_feature_scale[2] = warp(sharp_init_feature_scale[2], current_flow)
                # scale = 1/2
                # #print(sharp_init_feature_scale[2])
                # upsample the flow to the size of the feature map
                new_size = (2* coords_xy_i.shape[2], 2* coords_xy_i.shape[3])
                coords_xy_i_2 = 2 * F.interpolate(current_flow, size=new_size, mode='bilinear', align_corners=True)
                coords_xy_i_2 = self.flow_conv2(coords_xy_i_2)
                sharp_init_feature_scale[1] = warp(sharp_init_feature_scale[1], coords_xy_i_2)
                
                # #print("sharp_init_feature_scale[1]", sharp_init_feature_scale[1])
                # scale = 1
                # upsample the flow to the size of the feature map
                new_size = (2* coords_xy_i_2.shape[2], 2* coords_xy_i_2.shape[3])
                coords_xy_i_4 = 2 * F.interpolate(coords_xy_i_2, size=new_size, mode='bilinear', align_corners=True)
                # coords_xy_i_4 = self.flow_conv(coords_xy_i_4)
                sharp_init_feature_scale[0] = warp(sharp_init_feature_scale[0], coords_xy_i_4)
                # #print("sharp_init_feature_scale[0]", sharp_init_feature_scale[0])
                #refinement decoder
                
                # fine level feature
                coords_xy_i_8 = 2 * F.interpolate(coords_xy_i_4, size=(2* coords_xy_i_4.shape[2], 2* coords_xy_i_4.shape[3]), mode='bilinear', align_corners=True)
                # coords_xy_i_8 = self.flow_conv(coords_xy_i_8)
                sharp_image_features = warp(sharp_images[last_time_stamp], coords_xy_i_8)
                
                gen_sharp_image = self.decoder(attn_features_i, sharp_init_feature_scale,sharp_image_features) 
                # import sys
                # #print(gen_sharp_image)
                # sys.exit(0)
                generated_sequence[i] = gen_sharp_image.detach().cpu()
                
                self.reconstruction_loss_post = self.reconstruction_loss_post + self.mse_criterion(gen_sharp_image, sharp_images[i])
                self.ssim_post = self.ssim_post + self.ssim_criterion(gen_sharp_image, sharp_images[i])
                self.psnr_post = self.psnr_post + self.psnr_criterion(gen_sharp_image, sharp_images[i])
                
                #print("reconstruction_loss_post", self.reconstruction_loss_post.item())
                #print("ssim_post", self.ssim_post.item())
                #print("psnr_post", self.psnr_post.item())
                init_flow = current_flow
                # normalized flow
                init_flow[:,0,:,:] = init_flow[:,0,:,:] / (sharp_images[0].shape[3]//8)
                init_flow[:,1,:,:] = init_flow[:,1,:,:] / (sharp_images[0].shape[2]//8)
                
                init_corrdinates = coords_xy_i
                #print(init_flow)
                last_time_stamp = i
                # sys.exit(0)
                # #print("generated image", gen_sharp_image)
                
            else:
                continue
        
        self.reconstruction_loss_post = self.reconstruction_loss_post / (len(generated_sequence) - 1)
        self.psnr_post = self.psnr_post / (len(generated_sequence) - 1)
        self.ssim = self.ssim_post / (len(generated_sequence) - 1)
        
        return [gt_sequence, generated_sequence], [self.reconstruction_loss_post.item()],[self.psnr_post.item(), self.ssim_post.item()]
        
    def single_image_training(self, sharp_images, motion_blur_image):
        # blur_encoder
        attn_blur_features, blur_attn_map, encoded_blur_features, blur_feature_scale = self.blur_encoder(motion_blur_image)
        
        frame_use = np.random.uniform(
                0, 1, len(sharp_images)) >= self.prob_for_frame_drop
        
        generated_sequence = {}
        gt_sequence = {}
        self.reconstruction_loss_post = torch.tensor(0)
        self.psnr_post = torch.tensor(0)
        self.ssim_post = torch.tensor(0)
        last_time_stamp = 0
        init_flow = torch.zeros((self.batch_size, 2, sharp_images[0].shape[2]//8,sharp_images[0].shape[3]//8)).to(sharp_images.device)
        rows = torch.arange(0, sharp_images[0].shape[2]//8).view(1, -1).repeat(sharp_images[0].shape[3]//8, 1)
        coloumns = torch.arange(0, sharp_images[0].shape[3]//8).view(-1, 1).repeat(1, sharp_images[0].shape[2]//8)
        init_corrdinates = torch.stack((rows, coloumns), dim=2).permute(2,0,1).unsqueeze(0).repeat(self.batch_size, 1, 1, 1).to(sharp_images.device)
        import sys
        # print(init_corrdinates)
        # print(init_corrdinates.shape)
        # print(init_corrdinates.shape)
        initial_frame = sharp_images[last_time_stamp]
        for i in range(0, len(sharp_images)):
            if frame_use[i]:
                gt_sequence[i] = sharp_images[i].detach().cpu()
                # print("sharp_images[i]", sharp_images[i].shape)
                # print(sharp_images[i].max())
                # inital sharp encoder
                attn_sharp_init_features, init_sharp_attn_map, encoded_sharp_init_features, sharp_init_feature_scale = self.sharp_encoder(initial_frame)
                # print("attn_sharp_init_features", attn_sharp_init_features)
                # sys.exit(0)
                # inital positional encoding
                init_time_info = self.pos_encoder(
                    last_time_stamp, i, len(sharp_images), self.batch_size).to(encoded_sharp_init_features.device)
                # stack inital time info with each feature from the encoder
                # print("init_time_info", init_time_info.shape)	
                # print("attention_sharp_init_features", attn_sharp_init_features.shape)
                # print("init_flow", init_flow.shape)	
                init_time_info = init_time_info.repeat(encoded_sharp_init_features.shape[2], encoded_sharp_init_features.shape[3],1,1).permute(2,3,0,1)
                # print("init_time_info", init_time_info.shape)
                # inital feature info for feature forcasting
                init_feature_info = torch.cat( (attn_sharp_init_features, init_time_info,init_flow), dim=1)
                
                # generation positional encoding
                gen_time_info = self.pos_encoder(
                    last_time_stamp, i, len(sharp_images), self.batch_size).to(encoded_sharp_init_features.device)
                gen_time_info = gen_time_info.repeat(encoded_sharp_init_features.shape[2], encoded_sharp_init_features.shape[3],1,1).permute(2,3,0,1)
                
                # distribution for the feature forcasting corresponding to the current time stamp
                blur_feature_info = torch.cat((attn_blur_features, gen_time_info), dim=1)
                
                
                # feature forcasting
                attn_features_i, correlation_map_i, coords_xy_i = self.feature_forcasting(init_feature_info, blur_feature_info, attn_sharp_init_features)
                
                # print(coords_xy_i)
                # sys.exit(0)
                
                
                current_flow =  self.flow_conv1(init_corrdinates - coords_xy_i)
                # print(current_flow)
                # print(current_flow.shape)
                # warping sharp_image_features based on the flow
                # scale  = 1/4
                # print(attn_sharp_init_features.max())
                # print(sharp_init_feature_scale[2].max())
                sharp_init_feature_scale[2] = warp(sharp_init_feature_scale[2], current_flow)
                # print(sharp_init_feature_scale[2].max())
                # sys.exit(0)
                # scale = 1/2
                # upsample the flow to the size of the feature map
                new_size = (2* coords_xy_i.shape[2], 2* coords_xy_i.shape[3])
                coords_xy_i_2 = 2 * F.interpolate(current_flow, size=new_size, mode='bilinear', align_corners=True)
                # print(sharp_init_feature_scale[1].max())
                coords_xy_i_2 = self.flow_conv2(coords_xy_i_2)
                sharp_init_feature_scale[1] = warp(sharp_init_feature_scale[1], coords_xy_i_2)
                # print(sharp_init_feature_scale[1].max())
                # sys.exit(0)
                # scale = 1
                # upsample the flow to the size of the feature map
                new_size = (2* coords_xy_i_2.shape[2], 2* coords_xy_i_2.shape[3])
                coords_xy_i_4 = 2 * F.interpolate(coords_xy_i_2, size=new_size, mode='bilinear', align_corners=True)
                # print(sharp_init_feature_scale[0].max())
                sharp_init_feature_scale[0] = warp(sharp_init_feature_scale[0], coords_xy_i_4)
                # print(sharp_init_feature_scale[0].max())
                # sys.exit(0)
                #refinement decoder
                # fine level features
                coords_xy_i_8 = 2 * F.interpolate(coords_xy_i_4, size=(2* coords_xy_i_4.shape[2], 2* coords_xy_i_4.shape[3]), mode='bilinear', align_corners=True)
                warped_sharp_image = warp(initial_frame, coords_xy_i_8)
                
                gen_sharp_image = self.decoder(attn_features_i, sharp_init_feature_scale, warped_sharp_image) 
                generated_sequence[i] = gen_sharp_image.detach().cpu()
                # print("gen_sharp_image", gen_sharp_image.shape)
                # print(gen_sharp_image.max())
                # sys.exit(0)
                self.reconstruction_loss_post = self.reconstruction_loss_post + self.mse_criterion(gen_sharp_image, sharp_images[i])
                self.ssim_post = self.ssim_post + self.ssim_criterion(gen_sharp_image, sharp_images[i])
                self.psnr_post = self.psnr_post + self.psnr_criterion(gen_sharp_image, sharp_images[i])
                
                init_flow = current_flow
                # print("init_flow", init_flow.shape)
                # print("init_flow", init_flow.max())
                # sys.exit(0)
                # init_corrdinates = coords_xy_i
                # print(init_corrdinates.max())
                # normalized flow
                init_flow[:,0,:,:] = init_flow[:,0,:,:] / (sharp_images[0].shape[3]//8)
                init_flow[:,1,:,:] = init_flow[:,1,:,:] / (sharp_images[0].shape[2]//8)
                # last_time_stamp = i
                # initial_frame = gen_sharp_image
                # print(init_flow.max())
                # sys.exit(0)                
            else:
                continue
        
        self.reconstruction_loss_post = self.reconstruction_loss_post / (len(generated_sequence) - 1)
        self.psnr_post = self.psnr_post / (len(generated_sequence) - 1)
        self.ssim = self.ssim_post / (len(generated_sequence) - 1)
        
        return [gt_sequence, generated_sequence], [self.reconstruction_loss_post.item()],[self.psnr_post.item(), self.ssim_post.item()]


    def forward(self, sharp_images, motion_blur_image, mode, single_image_prediction=False):
        if mode == "train":
            if single_image_prediction:
                gen_seq, losses, metric = self.single_image_training(sharp_images, motion_blur_image)
            else:
                gen_seq, losses, metric = self.sequence_training(sharp_images, motion_blur_image)
        else:
            if single_image_prediction:
                gen_seq, losses, metric = self.single_image_training(sharp_images, motion_blur_image)
            else:
                gen_seq, losses, metric = self.sequence_training(sharp_images, motion_blur_image)
         
        return gen_seq, losses, metric
    
    def update_model(self):
        # set all optimizers to zero grad
        self.sharp_encoder_optimizer.zero_grad()
        self.blur_encoder_optimizer.zero_grad()
        self.feature_forcasting_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # self.flow_conv1_optimizer.zero_grad()
        # self.flow_conv2_optimizer.zero_grad()
        loss = self.reconstruction_loss_post + torch.exp(-0.05*self.psnr_post) + 0.7*torch.abs(1-self.ssim_post)
        #print(loss.item())
        loss.backward(retain_graph=True)
        
        self.sharp_encoder_optimizer.step()
        self.blur_encoder_optimizer.step()
        self.feature_forcasting_optimizer.step()
        self.decoder_optimizer.step()
        # self.flow_conv1_optimizer.step()
        # self.flow_conv2_optimizer.step()
        return [loss.item()]
    
    def save(self, fname):
        states = {
            'sharp_encoder': self.sharp_encoder.state_dict(),
            'blur_encoder': self.blur_encoder.state_dict(),
            'feature_forcasting': self.feature_forcasting.state_dict(),
            'decoder': self.decoder.state_dict(),
            'flow_conv1': self.flow_conv1.state_dict(),
            'flow_conv2': self.flow_conv2.state_dict(),
            'sharp_encoder_optimizer': self.sharp_encoder_optimizer.state_dict(),
            'blur_encoder_optimizer': self.blur_encoder_optimizer.state_dict(),
            'feature_forcasting_optimizer': self.feature_forcasting_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
        }   
        torch.save(states, fname)
        
    def load(self, fname):
        states = torch.load(fname)
        self.sharp_encoder.load_state_dict(states['sharp_encoder'])
        self.blur_encoder.load_state_dict(states['blur_encoder'])
        self.feature_forcasting.load_state_dict(states['feature_forcasting'])
        self.decoder.load_state_dict(states['decoder'])
        self.flow_conv1.load_state_dict(states['flow_conv1'])
        self.flow_conv2.load_state_dict(states['flow_conv2'])
        # self.flow_conv.load_state_dict(states['flow_conv'])
        # self.sharp_encoder_optimizer.load_state_dict(states['sharp_encoder_optimizer'])
        # self.blur_encoder_optimizer.load_state_dict(states['blur_encoder_optimizer'])
        # self.feature_forcasting_optimizer.load_state_dict(states['feature_forcasting_optimizer'])
        # self.decoder_optimizer.load_state_dict(states['decoder_optimizer'])
    