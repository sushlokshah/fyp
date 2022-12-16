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
from new_method.models.encoder import encoder
from new_method.models.decoder import decoder
from new_method.models.motion_encoder import Corr_Encoder
from new_method.models.latent_modeling import lstm, gaussian_lstm
from new_method.models.positional_encoding import Positional_encoding
from new_method.utils.loss import KLCriterion, PSNR, SSIM, SmoothMSE

class variational_gen(nn.Module):
    def __init__(self, args,batch_size=2):
        super(variational_gen, self).__init__()
        self.args = args
        
        self.batch_size = batch_size
        # sharp image encoder for both prior and posterior
        self.encoder = encoder(self.args.encoder['output_channels'], 3,resblocks=True)
        self.decoder = decoder(self.args.encoder['output_channels'], 3, resblocks=True)
        
        # motion encoder for posterior
        self.motion_encoder = Corr_Encoder()
        
        # latent modeling
        self.prior_lstm = gaussian_lstm(2*self.args.encoder['output_channels'] + self.args.positional['output_channel'], self.args.latent['output_size'],self.args.latent['hidden_size'], self.args.latent['num_layers'], self.batch_size)
        self.posterior_lstm = gaussian_lstm(2*self.args.encoder['output_channels'] + self.args.positional['output_channel'], self.args.latent['output_size'],self.args.latent['hidden_size'], self.args.latent['num_layers'], self.batch_size)
        
        self.decoder_lstm = lstm(self.args.latent['output_size'], self.args.encoder['output_channels'], self.args.latent['hidden_size'], self.args.latent['num_layers'], self.batch_size)
        
        # positional encoding
        self.pos_encoder = Positional_encoding(self.args.positional['output_channel'])
        
        if args.test != True:
            self.prob_for_frame_drop = args.prob_for_frame_drop
        else:
            self.prob_for_frame_drop = 0
        
        self.mse_criterion = nn.MSELoss() 
        self.latent_mse = nn.MSELoss()# recon and cpc
        self.kl_criterion = KLCriterion()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.align_criterion = KLCriterion()
            
    def init_hidden(self):
        self.decoder_lstm.hidden = self.decoder_lstm.init_hidden(batch_size=self.batch_size)
        self.posterior_lstm.hidden       = self.posterior_lstm.init_hidden(batch_size=self.batch_size)
        self.prior_lstm.hidden           = self.prior_lstm.init_hidden(batch_size=self.batch_size)

    def forward(self, sharp_images,motion_blur_image):
        self.init_hidden()
        
        #motion encoding
        blur_features = self.encoder(motion_blur_image)
        blur_features = self.motion_encoder(blur_features[1][5]) #8*8*8*8
        blur_features = blur_features.view(self.batch_size, -1)
        
        
        reconstruction_loss_post = 0
        reconstruction_loss_prior = 0
        alignment_loss = 0
        kl_loss_prior = 0
        latent_loss = 0
        last_frame_gen_loss = 0
        generated_sequence = {}
        #sequence generation
        frame_use = np.random.uniform(0,1,len(sharp_images)) >= self.prob_for_frame_drop
        last_time_stamp = 0
        for i in range(1,len(sharp_images)):
            #encoder
            if frame_use[i]:
                last_time_stamp = i
                
                # posterior
                sharp_features_encoding, feature_cache = self.encoder(sharp_images[last_time_stamp])
                time_info = self.pos_encoder(last_time_stamp,i,len(sharp_images),self.batch_size)
                posterior_input = torch.cat((sharp_features_encoding,blur_features,time_info),1)
                
                #prior
                if self.args.test != True:
                    target_encoding, target_cache = self.encoder(sharp_images[i])
                    time_info = self.pos_encoder(i,i,len(sharp_images),self.batch_size)
                    prior_input = torch.cat((target_encoding,blur_features,time_info),1)
                
                z_i_post, mu_i_post, logvar_i_post = self.posterior_lstm(posterior_input)
                
                if self.args.test != True:
                    z_i_prior, mu_i_prior, logvar_i_prior = self.prior_lstm(prior_input)
                
                #decoder
                z_decoder = self.decoder_lstm(z_i_post)
                x_i = self.decoder(z_decoder,feature_cache)
                
                generated_sequence[i] = x_i
                
                if self.args.test != True:
                    z_p = self.decoder_lstm(z_i_prior)
                    target_i = self.decoder(z_p,target_cache)
                
                reconstruction_loss_post = reconstruction_loss_post + self.mse_criterion(x_i, sharp_images[i]) + self.psnr(x_i, sharp_images[i]) + self.ssim(x_i, sharp_images[i])
                if self.args.test != True:
                    reconstruction_loss_prior = reconstruction_loss_prior + self.mse_criterion(target_i, sharp_images[i]) + self.psnr(target_i, sharp_images[i]) + self.ssim(target_i, sharp_images[i])
                    alignment_loss = alignment_loss + self.align_criterion(mu_i_post, logvar_i_post, mu_i_prior, logvar_i_prior)
                    kl_loss_prior = kl_loss_prior + self.kl_criterion(mu_i_prior, logvar_i_prior,0,0)
                    latent_loss = latent_loss + self.latent_mse(z_decoder,target_encoding)
                
                if i == len(sharp_images)-1:
                    if self.args.test != True:
                        last_frame_gen_loss = self.mse_criterion(target_i, sharp_images[i]) + self.psnr(target_i, sharp_images[i]) + self.ssim(target_i, sharp_images[i])    
                
            else:
                continue
        
        # loss = reconstruction_loss_post + alignment_loss + latent_loss
        # loss.backward(retain_graph=True)
        # self.update_model_without_prior()
        
        # self.prior_lstm.zero_grad()
        # prior_loss = kl_loss_prior + reconstruction_loss_prior + last_frame_gen_loss
        # prior_loss.backward()
        # self.update_prior()
        if self.args.test != True:
            return generated_sequence, reconstruction_loss_post, alignment_loss, latent_loss, kl_loss_prior, reconstruction_loss_prior, last_frame_gen_loss
        else:
            return generated_sequence, reconstruction_loss_post
    # def update_model_without_prior(self):
    #     for param in self.prior_lstm.parameters():
    #         param.requires_grad = False
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #     for param in self.prior_lstm.parameters():
    #         param.requires_grad = True
    
    # def update_prior(self):
        
        