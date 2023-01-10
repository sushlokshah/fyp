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
from models.encoder import Deblurring_net_encoder, Feature_forcaster, Feature_extractor, Feature_predictor
from models.decoder import Refinement_Decoder
from models.positional_encoding import Positional_encoding
from utils.loss import KLCriterion, PSNR, SSIM, SmoothMSE, image_gradient, image_laplacian


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

    def __init__(self, args, batch_size=2, prob_for_frame_drop=0, lr=0.001, dropout=0):
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

        #################################################################
        # Sharp image feature generator along with blur feature encoding
        #################################################################
        self.sharp_encoder = Deblurring_net_encoder(self.args.blur_decoder["sharp_encoder"]["output_channels"],
                                                    self.args.blur_decoder["sharp_encoder"]["input_channels"],
                                                    self.args.blur_decoder["sharp_encoder"]["kernel_size"],
                                                    dropout=self.dropout)
        #################################################################
        # positional encoding
        #################################################################
        self.pos_encoder = Positional_encoding(
            self.args.blur_decoder["positional"]['output_channels'])

        #################################################################
        # feature forecastor from sharp image and pixel relation co-variance
        #################################################################
        self.feature_predictor = Feature_predictor(self.args.blur_decoder["sharp_encoder"]["output_channels"],
                                                   self.args.blur_decoder["sharp_encoder"]["output_channels"],
                                                   self.args.blur_decoder["positional"]['output_channels'],
                                                   self.args.blur_decoder['feature_predictor']["nheads"],
                                                   dropout=self.dropout)

        #################################################################
        # decoder and final refinement
        #################################################################
        self.decoder = Refinement_Decoder(self.args.blur_decoder['decoder']['output_channels'],
                                          self.args.blur_decoder['decoder']['input_channels'])
        ##################################################################

        ##################################################################
        # losses and metric
        ##################################################################
        self.mse_criterion = nn.MSELoss()
        self.grad_x_mse_criterion = nn.BCELoss()
        self.grad_y_mse_criterion = nn.BCELoss()
        self.laplacian_mse_criterion = nn.BCELoss()
        self.ssim_criterion = SSIM()
        self.psnr_criterion = PSNR()

        if args.test != True:
            self.init_optimizer()

    def init_optimizer(self):
        self.sharp_encoder_optimizer = self.optimizer(self.sharp_encoder.parameters(
        ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        # self.refinement_max_scale_optimizer = self.optimizer(self.refinement_max_scale.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        # self.blur_encoder_optimizer = self.optimizer(self.blur_encoder.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        # self.feature_forcasting_optimizer = self.optimizer(self.feature_forcasting.parameters(), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.decoder_optimizer = self.optimizer(self.decoder.parameters(
        ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.feature_predictor_optimizer = self.optimizer(self.feature_predictor.parameters(
        ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))

    def train_image_deblurring(self, past_blur_image, current_blur_image, sharp_image):

        #################################################################
        # Sharp image feature generator along with blur feature encoding
        #################################################################
        sharp_feature, sharp_feature_scale, current_blur_features, current_blur_feature_scale = self.sharp_encoder(
            past_blur_image, current_blur_image)

        #################################################################
        # decoder with skip connections
        #################################################################
        current_sharp_image = self.decoder(sharp_feature, sharp_feature_scale)
        # current_sharp_image = current_sharp_image + \
        #     image_laplacian(current_blur_image)
        # print(current_sharp_image.max(), current_sharp_image.min())
        ##################################################################
        # losses and metric
        ##################################################################
        grad_x, grad_y = image_gradient(current_sharp_image)
        sharp_image2 = torchvision.transforms.functional.equalize(sharp_image)
        sharp_image2 = sharp_image2 + image_laplacian(sharp_image2)
        gt_grad_x, gt_grad_y = image_gradient(sharp_image2)

        shape_out = grad_x.shape
        threshold = 0.1
        grad_x = grad_x.squeeze(1).view(current_sharp_image.size(0), -1)
        grad_y = grad_y.squeeze(1).view(current_sharp_image.size(0), -1)

        grad_x = torch.where(grad_x > threshold, 0.99*torch.ones_like(
            grad_x), grad_x + 0.0001*torch.ones_like(grad_x))
        grad_y = torch.where(grad_y > threshold, 0.99*torch.ones_like(
            grad_y), grad_y + 0.0001*torch.ones_like(grad_y))

        gt_grad_x = gt_grad_x.squeeze(1).view(current_sharp_image.size(0), -1)
        gt_grad_y = gt_grad_y.squeeze(1).view(current_sharp_image.size(0), -1)

        # apply thresholding over gradients gt
        gt_grad_x = torch.where(gt_grad_x > threshold, torch.ones_like(
            gt_grad_x), torch.zeros_like(gt_grad_x))
        gt_grad_y = torch.where(gt_grad_y > threshold, torch.ones_like(
            gt_grad_y), torch.zeros_like(gt_grad_y))
        self.grad_x_loss = self.grad_x_mse_criterion(grad_x, gt_grad_x)
        self.grad_y_loss = self.grad_y_mse_criterion(grad_y, gt_grad_y)

        laplacian = image_laplacian(current_sharp_image)
        gt_laplacian = image_laplacian(sharp_image2)

        laplacian = laplacian.squeeze(1).view(current_sharp_image.size(0), -1)

        laplacian = torch.where(laplacian > threshold, 0.99*torch.ones_like(
            laplacian), laplacian + 0.0001*torch.ones_like(laplacian))

        gt_laplacian = gt_laplacian.squeeze(
            1).view(current_sharp_image.size(0), -1)

        gt_laplacian = torch.where(gt_laplacian > threshold, torch.ones_like(
            gt_laplacian), torch.zeros_like(gt_laplacian))

        edge_map = grad_x + grad_y + laplacian
        edge_map_gt = gt_grad_x + gt_grad_y + gt_laplacian
        # print(edge_map_gt.shape)
        edge_map = edge_map.unsqueeze(1).reshape(shape_out)
        edge_map_gt = edge_map_gt.unsqueeze(1).reshape(shape_out)

        self.laplacian_loss = self.laplacian_mse_criterion(
            laplacian, gt_laplacian)
        self.deblurring_reconstruction_loss = self.mse_criterion(
            current_sharp_image, sharp_image)
        self.deblurring_ssim = self.ssim_criterion(
            current_sharp_image, sharp_image)
        self.deblurring_psnr = self.psnr_criterion(
            current_sharp_image, sharp_image)

        # make gray scale edge map to 3 channel
        edge_map = edge_map.repeat(1, 3, 1, 1)
        edge_map_gt = edge_map_gt.repeat(1, 3, 1, 1)
        return [{0: current_sharp_image.detach().cpu()}, {0: sharp_image.detach().cpu()}, {0: (edge_map.detach().cpu()/1.5 - 1)}, {0: (edge_map_gt.detach().cpu()/1.5 - 1)}, {0: current_blur_image.detach().cpu()}], self.deblurring_reconstruction_loss.item(), [self.deblurring_psnr.item(), self.deblurring_ssim.item()]

    def train_sequence(self, past_blur_image, current_blur_image, sharp_images):
        #################################################################
        # Sharp image feature generator along with blur feature encoding
        #################################################################
        sharp_feature, sharp_feature_scale, current_blur_features, current_blur_feature_scale = self.sharp_encoder(
            past_blur_image, current_blur_image)
        # print(sharp_feature.device, sharp_feature_scale[0].device,
        #   current_blur_features.device, current_blur_feature_scale[0].device)
        self.reconstruction_loss = 0
        self.psnr_metric = 0
        self.ssim_metric = 0
        self.grad_x_loss = 0
        self.grad_y_loss = 0
        generated_sequence = {}
        gt_sequence = {}
        gen_length = len(sharp_images)
        for i in range(len(sharp_images)):
            gt_sequence[i] = sharp_images[i].detach().cpu()
            ######################################################################
            # positional encoding
            ######################################################################
            time_info = self.pos_encoder(
                0, i, gen_length, self.batch_size).to(sharp_feature.device)
            time_info = time_info.repeat(
                sharp_feature.shape[2], sharp_feature.shape[3], 1, 1).permute(2, 3, 0, 1)
            # print(time_info.device)
            ######################################################################
            # feature forcasting
            ######################################################################
            blur_attn_features, final_transformation_map, final_features, sharp_feature_scale = self.feature_predictor(
                current_blur_features, current_blur_feature_scale, sharp_feature, sharp_feature_scale, time_info)

            ######################################################################
            # decode new image
            ######################################################################
            current_sharp_image = self.decoder(
                final_features, sharp_feature_scale)
            # print(current_sharp_image.device)
            ######################################################################
            # losses and metric
            ######################################################################
            grad_x, grad_y = image_gradient(current_sharp_image)
            gt_grad_x, gt_grad_y = image_gradient(sharp_images[i])
            self.grad_x_loss += self.grad_x_mse_criterion(grad_x, gt_grad_x)
            self.grad_y_loss += self.grad_y_mse_criterion(grad_y, gt_grad_y)
            self.reconstruction_loss += self.mse_criterion(
                current_sharp_image, sharp_images[i])
            self.ssim_metric += self.ssim_criterion(
                current_sharp_image, sharp_images[i])
            self.psnr_metric += self.psnr_criterion(
                current_sharp_image, sharp_images[i])
            generated_sequence[i] = current_sharp_image.detach().cpu()

        self.grad_x_loss = self.grad_x_loss/gen_length
        self.grad_y_loss = self.grad_y_loss/gen_length
        self.reconstruction_loss = self.reconstruction_loss/gen_length
        self.psnr_metric = self.psnr_metric/gen_length
        self.ssim_metric = self.ssim_metric/gen_length

        return [generated_sequence, gt_sequence], self.reconstruction_loss.item(), [self.psnr_metric.item(), self.ssim_metric.item()]

    def train_image_pred(self, past_blur_image, current_blur_image, sharp_images, gen_index, gen_length):
        #################################################################
        # Sharp image feature generator along with blur feature encoding
        #################################################################
        sharp_feature, sharp_feature_scale, current_blur_features, current_blur_feature_scale = self.sharp_encoder(
            past_blur_image, current_blur_image)
        self.reconstruction_loss = 0
        self.psnr_metric = 0
        self.ssim_metric = 0
        self.grad_x_loss = 0
        self.grad_y_loss = 0
        generated_sequence = {}
        gt_sequence = {}
        gt_sequence[gen_index] = sharp_images[gen_index].detach().cpu()
        ######################################################################
        # positional encoding
        ######################################################################
        time_info = self.pos_encoder(
            0, gen_index, gen_length, self.batch_size).to(sharp_feature.device)

        ######################################################################
        # feature forcasting
        ######################################################################
        blur_attn_features, final_transformation_map, final_features, sharp_feature_scale = self.feature_predictor(
            current_blur_features, current_blur_feature_scale, sharp_feature, sharp_feature_scale, time_info)

        ######################################################################
        # decode new image
        ######################################################################
        current_sharp_image = self.decoder(final_features, sharp_feature_scale)

        ######################################################################
        # losses and metric
        ######################################################################
        grad_x, grad_y = image_gradient(current_sharp_image)
        gt_grad_x, gt_grad_y = image_gradient(sharp_images[gen_index])
        self.grad_x_loss += self.grad_x_mse_criterion(grad_x, gt_grad_x)
        self.grad_y_loss += self.grad_y_mse_criterion(grad_y, gt_grad_y)
        self.reconstruction_loss += self.mse_criterion(
            current_sharp_image, sharp_images[gen_index])
        self.ssim_metric += self.ssim_criterion(
            current_sharp_image, sharp_images[gen_index])
        self.psnr_metric += self.psnr_criterion(
            current_sharp_image, sharp_images[gen_index])
        generated_sequence[gen_index] = current_sharp_image.detach().cpu()

        return [generated_sequence, gt_sequence], self.reconstruction_loss.item(), [self.psnr_metric.item(), self.ssim_metric.item()]

    def update_deblurring(self):
        self.sharp_encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.feature_predictor_optimizer.zero_grad()
        # self.refinement_max_scale_optimizer.zero_grad()

        # loss = self.deblurring_reconstruction_loss + 1 * \
        #     torch.exp(-0.05*self.deblurring_psnr) + 1 * \
        #     torch.abs(1-self.deblurring_ssim) + \
        loss = self.deblurring_reconstruction_loss + \
            0.2*self.laplacian_loss + \
            torch.abs(1-self.deblurring_ssim) + \
            0.2*self.grad_x_loss + 0.2*self.grad_y_loss

        loss.backward(retain_graph=True)

        self.sharp_encoder_optimizer.step()
        self.decoder_optimizer.step()
        # self.refinement_max_scale_optimizer.step()

        return loss.item()

    def update_forcaster(self):
        self.sharp_encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.feature_predictor_optimizer.zero_grad()
        # self.refinement_max_scale_optimizer.zero_grad()

        loss = 0.4*self.reconstruction_loss + 0.4 * \
            torch.exp(-0.05*self.psnr_metric) + 0.2 * \
            torch.abs(1-self.ssim_metric) + self.grad_x_loss + self.grad_y_loss
        loss.backward(retain_graph=True)

        self.feature_predictor_optimizer.step()

    def update_model(self):
        self.sharp_encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.feature_predictor_optimizer.zero_grad()
        # self.refinement_max_scale_optimizer.zero_grad()

        loss = 0.4*self.reconstruction_loss + 0.4 * \
            torch.exp(-0.05*self.psnr_metric) + 0.2 * \
            torch.abs(1-self.ssim_metric) + self.grad_x_loss + self.grad_y_loss
        loss.backward(retain_graph=True)

        self.sharp_encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.feature_predictor_optimizer.step()

    def forward(self, sharp_images, past_blur_image, current_blur_image, mode, gen_index=0, gen_length=1):
        if mode == "train_image_deblurring":
            generation, loss, metric = self.train_image_deblurring(
                past_blur_image, current_blur_image, sharp_images)
        elif mode == "train_sequence" or mode == "train_forcaster_sequence":
            generation, loss, metric = self.train_sequence(
                past_blur_image, current_blur_image, sharp_images)
        elif mode == "train_image_pred" or mode == "train_forcaster_image":
            generation, loss, metric = self.train_image_pred(
                past_blur_image, current_blur_image, sharp_images, gen_index, gen_length)
        else:
            raise NotImplementedError

        return generation, loss, metric

    def save(self, fname):
        states = {
            'sharp_encoder': self.sharp_encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'feature_predictor': self.feature_predictor.state_dict(),

            'sharp_encoder_optimizer': self.sharp_encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'feature_predictor_optimizer': self.feature_predictor_optimizer.state_dict(),
        }
        torch.save(states, fname)

    def load(self, fname):
        states = torch.load(fname)
        self.sharp_encoder.load_state_dict(states['sharp_encoder'])
        self.decoder.load_state_dict(states['decoder'])
        if 'feature_predictor' in states:
            self.feature_predictor.load_state_dict(states['feature_predictor'])
