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
from models.encoder import Deblurring_net_encoder, Feature_forcaster, Feature_extractor
from models.decoder import Refinement_Decoder
from models.positional_encoding import Positional_encoding
from utils.loss import KLCriterion, PSNR, SSIM, SmoothMSE


class Blur_decoder(nn.Module):
    """parameters for the model

    Args:
        sharp_encoder:
            output_channels
            input_channels
            kernel_size
        
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
        super(Blur_decoder, self).__init__()
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
        
        self.sharp_encoder = Deblurring_net_encoder(self.args.blur_decoder["sharp_encoder"]["output_channels"],self.args.blur_decoder["sharp_encoder"]["input_channels"],self.args.blur_decoder["sharp_encoder"]["kernel_size"],dropout=self.dropout)
        # self.blur_encoder = Feature_extractor(self.args.blur_decoder['blur_encoder']['output_channels'], self.args.blur_decoder['blur_encoder']['input_channels'], self.args.blur_decoder['blur_encoder']['nheads'],self.dropout)
        
        # positional encoding
        # self.pos_encoder = Positional_encoding(
            # self.args.blur_decoder["positional"]['output_channels'])
        
        # history_in_channels = self.args.blur_decoder['blur_encoder']['output_channels'] + self.args.blur_decoder['positional']['output_channels']
        # current_in_channels = self.args.blur_decoder['sharp_encoder']['output_channels'] + self.args.blur_decoder['positional']['output_channels'] + 2
        # #  history_in_channels,current_in_channels, out_channels, nheads, dropout = 0
        # self.feature_forcasting = Feature_forcaster(history_in_channels, current_in_channels, self.args.blur_decoder['sharp_encoder']['output_channels'], self.args.blur_decoder['feature_forcasting']['nheads'], self.dropout)
        self.decoder = Refinement_Decoder(self.args.blur_decoder['decoder']['output_channels'], self.args.blur_decoder['decoder']['input_channels'])
        
        self.mse_criterion = nn.L1Loss()
        self.ssim_criterion = SSIM()
        self.psnr_criterion = PSNR()
        
        if args.test != True:
            self.init_optimizer()
        
        
    def init_optimizer(self):
        self.sharp_encoder_optimizer = self.optimizer(self.sharp_encoder.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        # self.blur_encoder_optimizer = self.optimizer(self.blur_encoder.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        # self.feature_forcasting_optimizer = self.optimizer(self.feature_forcasting.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.decoder_optimizer = self.optimizer(self.decoder.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
    
    def train_image_deblurring(self, past_blur_image, current_blur_image, sharp_image):
        
        # encoder
        sharp_feature, sharp_feature_scale, current_blur_features, current_blur_feature_scale = self.sharp_encoder(past_blur_image,current_blur_image)
        # print(sharp_feature.shape)
        # decoder
        current_sharp_image = self.decoder(sharp_feature, sharp_feature_scale)
        
        self.deblurring_reconstruction_loss = self.mse_criterion(current_sharp_image, sharp_image)
        self.deblurring_ssim = self.ssim_criterion(current_sharp_image, sharp_image)
        self.deblurring_psnr = self.psnr_criterion(current_sharp_image, sharp_image)
        
        return [{0:current_sharp_image}, {0:sharp_image}], self.deblurring_reconstruction_loss.item(), [self.deblurring_psnr.item(), self.deblurring_ssim.item()]
    
    def update_deblurring(self):
        self.sharp_encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        loss = 0.4*self.deblurring_reconstruction_loss + 0.4*torch.exp(-0.05*self.deblurring_psnr) + 0.2*torch.abs(1-self.deblurring_ssim)
        loss.backward(retain_graph=True)
        
        self.sharp_encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        return loss.item()
    
    def update_forcaster(self):
        return NotImplementedError
    
    def update_model(self):
        return NotImplementedError
    
    def train_forcaster_sequence(self, past_blur_image, current_blur_image, sharp_images):
        return NotImplementedError
    
    def train_sequence(self, past_blur_image, current_blur_image, sharp_images):
        return NotImplementedError
    
    def train_forcaster_image(self, past_blur_image, current_blur_image, sharp_image):
        return NotImplementedError
    
    def train_image_pred(self, past_blur_image, current_blur_image, sharp_image):
        return NotImplementedError
    
    def forward(self, sharp_images, past_blur_image, current_blur_image, mode):
        if mode == "train_image_deblurring":
            generation, loss, metric = self.train_image_deblurring(past_blur_image, current_blur_image, sharp_images)
        elif mode == "train_forcaster_sequence":
            generation, loss, metric = self.train_forcaster_sequence(past_blur_image, current_blur_image, sharp_images)
        elif mode == "train_sequence":
            generation, loss, metric = self.train_sequence(past_blur_image, current_blur_image, sharp_images)
        elif mode == "train_forcaster_image":
            generation, loss, metric = self.train_forcaster_image(past_blur_image, current_blur_image, sharp_images)
        elif mode == "train_image_pred":
            generation, loss, metric = self.train_image_pred(past_blur_image, current_blur_image, sharp_images)
        else:
            raise NotImplementedError
        
        return generation, loss, metric
    
    def save(self, fname):
        states = {
            'sharp_encoder': self.sharp_encoder.state_dict(),
            # 'blur_encoder': self.blur_encoder.state_dict(),
            # 'feature_forcasting': self.feature_forcasting.state_dict(),
            'decoder': self.decoder.state_dict(),
            
            'sharp_encoder_optimizer': self.sharp_encoder_optimizer.state_dict(),
            # 'blur_encoder_optimizer': self.blur_encoder_optimizer.state_dict(),
            # 'feature_forcasting_optimizer': self.feature_forcasting_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
        }   
        torch.save(states, fname)
        
    def load(self, fname):
        states = torch.load(fname)
        self.sharp_encoder.load_state_dict(states['sharp_encoder'])
        # self.blur_encoder.load_state_dict(states['blur_encoder'])
        # self.feature_forcasting.load_state_dict(states['feature_forcasting'])
        self.decoder.load_state_dict(states['decoder'])
        
    