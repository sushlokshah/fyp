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
from models.encoder import encoder
from models.decoder import decoder, refinement_module
from models.motion_encoder import Corr_Encoder
from models.latent_modeling import lstm, gaussian_lstm
from models.positional_encoding import Positional_encoding
from utils.loss import KLCriterion, PSNR, SSIM, SmoothMSE


class Variational_Gen(nn.Module):
    # added
    def __init__(self, args, batch_size=2, prob_for_frame_drop=0, lr=0.001):
        super(Variational_Gen, self).__init__()
        self.args = args
        if args.train or args.evaluate:
            self.batch_size = args.training_parameters["batch_size"]
        elif args.test:
            self.batch_size = args.testing_parameters["batch_size"]
        else:
            self.batch_size = batch_size
        # sharp image encoder for both prior and posterior
        self.encoder = encoder(
            self.args.variational_gen["encoder"]['output_channels'], 3, resblocks=False)
        self.decoder = decoder(
            self.args.variational_gen["encoder"]['output_channels'], 3, resblocks=False)

        # motion encoder for posterior
        self.motion_encoder = Corr_Encoder()

        # latent modeling
        self.prior_lstm = gaussian_lstm(self.args.variational_gen["encoder"]['output_channels'] + self.args.variational_gen["positional"]['output_channels'] + 4*4*4*4,
                                        self.args.variational_gen["latent"]['output_channels'], self.args.variational_gen["latent"]['hidden_size'], self.args.variational_gen["latent"]['num_layers'], self.batch_size)
        self.posterior_lstm = gaussian_lstm(self.args.variational_gen["encoder"]['output_channels'] + self.args.variational_gen["positional"]['output_channels'] + 4*4*4*4,
                                            self.args.variational_gen["latent"]['output_channels'], self.args.variational_gen["latent"]['hidden_size'], self.args.variational_gen["latent"]['num_layers'], self.batch_size)

        self.decoder_lstm = lstm(self.args.variational_gen["latent"]['output_channels'], self.args.variational_gen["encoder"]['output_channels'],
                                 self.args.variational_gen["latent"]['hidden_size'], self.args.variational_gen["latent"]['num_layers'], self.batch_size)

        self.refinement = refinement_module(3)
        # positional encoding
        self.pos_encoder = Positional_encoding(
            self.args.variational_gen["positional"]['output_channels'])

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

        self.mse_criterion = nn.MSELoss()
        self.latent_mse = nn.MSELoss()  # recon and cpc
        self.kl_criterion = KLCriterion()
        self.align_criterion = KLCriterion()
        self.ssim_criterion = SSIM()
        self.psnr_criterion = PSNR()

        if args.test != True:
            self.init_optimizer()

    def init_optimizer(self):
        self.decoder_lstm_optimizer = self.optimizer(self.decoder_lstm.parameters(
        ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.posterior_optimizer = self.optimizer(self.posterior_lstm.parameters(
        ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.prior_optimizer = self.optimizer(self.prior_lstm.parameters(
        ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.encoder_optimizer = self.optimizer(self.encoder.parameters(
        ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.decoder_optimizer = self.optimizer(self.decoder.parameters(
        ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        self.refinement_optimizer = self.optimizer(self.refinement.parameters(
        ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        # self.motion_encoder_optimizer = self.optimizer(self.motion_encoder.parameters(
        # ), lr=self.lr, weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))
        # self.positional_encoder_optimizer = self.optimizer(self.pos_encoder.parameters(
        # ), lr=self.lr,  weight_decay=self.args.optimizer["weight_decay"], eps=float(self.args.optimizer["eps"]))

    def init_hidden(self):
        self.decoder_lstm.hidden = self.decoder_lstm.init_hidden(
            batch_size=self.batch_size)
        self.posterior_lstm.hidden = self.posterior_lstm.init_hidden(
            batch_size=self.batch_size)
        self.prior_lstm.hidden = self.prior_lstm.init_hidden(
            batch_size=self.batch_size)

    def sequence_training(self, sharp_images, motion_blur_image):
        self.init_hidden()

        # motion encoding
        blur_features = self.encoder(motion_blur_image)
        blur_features = self.motion_encoder(blur_features[1][5])  # 8*8*8*8
        blur_features = blur_features.view(self.batch_size, -1)

        self.reconstruction_loss_post = 0
        self.reconstruction_loss_prior = 0
        self.alignment_loss = 0
        self.kl_loss_prior = 0
        self.latent_loss = 0
        self.last_frame_gen_loss = 0
        generated_sequence_posterior = {}
        generated_sequence_prior = {}
        gt_sequence = {}
        self.psnr_post = 0
        self.ssim_post = 0
        self.psnr_prior = 0
        self.ssim_prior = 0
        # sequence generation
        frame_use = np.random.uniform(
            0, 1, len(sharp_images)) >= self.prob_for_frame_drop
        last_time_stamp = 0

        for i in range(1, len(sharp_images)):
            # encoder
            if frame_use[i]:
                gt_sequence[i] = sharp_images[i].detach().cpu()

                sharp_features_encoding, feature_cache = self.encoder(
                    sharp_images[last_time_stamp])
                time_info = self.pos_encoder(
                    last_time_stamp, i, len(sharp_images), self.batch_size).to(sharp_features_encoding.device)

                posterior_input = torch.cat(
                    (sharp_features_encoding, blur_features, time_info), 1)

                # prior
                target_encoding, target_cache = self.encoder(
                    sharp_images[i])
                time_info = self.pos_encoder(
                    i, i, len(sharp_images), self.batch_size).to(target_encoding.device)
                prior_input = torch.cat(
                    (target_encoding, blur_features, time_info), 1)

                z_i_post, mu_i_post, logvar_i_post = self.posterior_lstm(
                    posterior_input)

                z_i_prior, mu_i_prior, logvar_i_prior = self.prior_lstm(
                    prior_input)

                # decoder
                z_decoder = self.decoder_lstm(z_i_post)
                # print(z_decoder.shape)
                x_i = self.decoder(z_decoder, feature_cache)

                generated_sequence_posterior[i] = x_i.detach().cpu()

                z_p = self.decoder_lstm(z_i_prior)
                target_i = self.decoder(z_p, target_cache)
                generated_sequence_prior[i] = target_i.detach().cpu()

                self.reconstruction_loss_post = self.reconstruction_loss_post + self.mse_criterion(
                    x_i, sharp_images[i])

                self.psnr_post = self.psnr_post + \
                    self.psnr_criterion(x_i, sharp_images[i])
                self.ssim_post = self.ssim_post + \
                    self.ssim_criterion(x_i, sharp_images[i])

                self.reconstruction_loss_prior = self.reconstruction_loss_prior + self.mse_criterion(
                    target_i, sharp_images[i])
                self.alignment_loss = self.alignment_loss + \
                    self.align_criterion(
                        mu_i_post, logvar_i_post, mu_i_prior, logvar_i_prior)
                self.kl_loss_prior = self.kl_loss_prior + \
                    self.kl_criterion(
                        mu_i_prior, logvar_i_prior, torch.tensor(0), torch.tensor(0))
                self.latent_loss = self.latent_loss + \
                    self.latent_mse(z_decoder, target_encoding)

                self.psnr_prior = self.psnr_prior + \
                    self.psnr_criterion(target_i, sharp_images[i])
                self.ssim_prior = self.ssim_prior + \
                    self.ssim_criterion(target_i, sharp_images[i])

                if i == len(sharp_images)-1:
                    self.last_frame_gen_loss = self.mse_criterion(
                        target_i, sharp_images[i])
                last_time_stamp = i
            else:
                continue

        # average all losses over the sequence
        self.reconstruction_loss_post = self.reconstruction_loss_post / \
            (len(generated_sequence_posterior) - 1)
        self.reconstruction_loss_prior = self.reconstruction_loss_prior / \
            (len(generated_sequence_posterior) - 1)
        self.alignment_loss = self.alignment_loss / \
            (len(generated_sequence_posterior) - 1)
        self.kl_loss_prior = self.kl_loss_prior / \
            (len(generated_sequence_posterior) - 1)
        self.latent_loss = self.latent_loss / \
            (len(generated_sequence_posterior) - 1)
        self.psnr_post = self.psnr_post / \
            (len(generated_sequence_posterior) - 1)
        self.ssim_post = self.ssim_post / \
            (len(generated_sequence_posterior) - 1)
        self.psnr_prior = self.psnr_prior / \
            (len(generated_sequence_posterior) - 1)
        self.ssim_prior = self.ssim_prior / \
            (len(generated_sequence_posterior) - 1)

        return [gt_sequence, generated_sequence_posterior, generated_sequence_prior], [self.reconstruction_loss_post.item(), self.alignment_loss.item(), self.latent_loss.item(), self.kl_loss_prior.item(), self.reconstruction_loss_prior.item(), self.last_frame_gen_loss.item()], [self.psnr_post.item(), self.ssim_post.item(), self.psnr_prior.item(), self.ssim_prior.item()]

    def single_image_training(self, sharp_images, motion_blur_image):
        self.init_hidden()
        seq_len = len(sharp_images)

        # motion encoding
        blur_features = self.encoder(motion_blur_image)
        blur_features = self.motion_encoder(blur_features[1][5])  # 8*8*8*8
        blur_features = blur_features.view(self.batch_size, -1)

        self.reconstruction_loss_post = torch.tensor(0)
        self.reconstruction_loss_prior = torch.tensor(0)
        self.alignment_loss = torch.tensor(0)
        self.kl_loss_prior = torch.tensor(0)
        self.latent_loss = torch.tensor(0)
        self.last_frame_gen_loss = torch.tensor(0)
        generated_sequence_posterior = {}
        generated_sequence_prior = {}
        gt_sequence = {}
        self.psnr_post = torch.tensor(0)
        self.ssim_post = torch.tensor(0)
        self.psnr_prior = torch.tensor(0)
        self.ssim_prior = torch.tensor(0)
        # sequence generation
        frame_use = np.random.uniform(
            0, 1, len(sharp_images)) >= self.prob_for_frame_drop
        last_time_stamp = 0
        print(frame_use)
        initial_frame = sharp_images[last_time_stamp]
        for i in range(1, seq_len):
            if frame_use[i]:
                gt_sequence[i] = sharp_images[i].detach().cpu()

                sharp_features_encoding, feature_cache = self.encoder(
                    initial_frame)

                time_info = self.pos_encoder(
                    last_time_stamp, i, len(sharp_images), self.batch_size).to(sharp_features_encoding.device)

                posterior_input = torch.cat(
                    (sharp_features_encoding, blur_features, time_info), 1)

                # prior
                target_encoding, target_cache = self.encoder(
                    sharp_images[i])
                time_info = self.pos_encoder(
                    i, i, len(sharp_images), self.batch_size).to(target_encoding.device)
                prior_input = torch.cat(
                    (target_encoding, blur_features, time_info), 1)

                z_i_post, mu_i_post, logvar_i_post = self.posterior_lstm(
                    posterior_input)

                z_i_prior, mu_i_prior, logvar_i_prior = self.prior_lstm(
                    prior_input)

                # decoder
                z_decoder = self.decoder_lstm(z_i_post)
                # print(z_decoder.shape)
                x_i = self.decoder(z_decoder, feature_cache)

                generated_sequence_posterior[i] = x_i.detach().cpu()

                z_p = self.decoder_lstm(z_i_prior)
                target_i = self.decoder(z_p, target_cache)
                generated_sequence_prior[i] = target_i.detach().cpu()

                self.reconstruction_loss_post = self.reconstruction_loss_post + self.mse_criterion(
                    x_i, sharp_images[i])

                self.psnr_post = self.psnr_post + \
                    self.psnr_criterion(x_i, sharp_images[i])
                self.ssim_post = self.ssim_post + \
                    self.ssim_criterion(x_i, sharp_images[i])

                self.reconstruction_loss_prior = self.reconstruction_loss_prior + self.mse_criterion(
                    target_i, sharp_images[i])
                self.alignment_loss = self.alignment_loss + \
                    self.align_criterion(
                        mu_i_post, logvar_i_post, mu_i_prior, logvar_i_prior)
                self.kl_loss_prior = self.kl_loss_prior + \
                    self.kl_criterion(
                        mu_i_prior, logvar_i_prior, torch.tensor(0), torch.tensor(0))
                self.latent_loss = self.latent_loss + \
                    self.latent_mse(z_decoder, target_encoding)

                self.psnr_prior = self.psnr_prior + \
                    self.psnr_criterion(target_i, sharp_images[i])
                self.ssim_prior = self.ssim_prior + \
                    self.ssim_criterion(target_i, sharp_images[i])

                if i == len(sharp_images)-1:
                    self.last_frame_gen_loss = self.mse_criterion(
                        target_i, sharp_images[i])
                last_time_stamp = i
                initial_frame = x_i
            else:
                continue

        # average all losses over the sequence
        self.reconstruction_loss_post = self.reconstruction_loss_post / \
            (len(generated_sequence_posterior) - 1)
        self.reconstruction_loss_prior = self.reconstruction_loss_prior / \
            (len(generated_sequence_posterior) - 1)
        self.alignment_loss = self.alignment_loss / \
            (len(generated_sequence_posterior) - 1)
        self.kl_loss_prior = self.kl_loss_prior / \
            (len(generated_sequence_posterior) - 1)
        self.latent_loss = self.latent_loss / \
            (len(generated_sequence_posterior) - 1)
        self.psnr_post = self.psnr_post / \
            (len(generated_sequence_posterior) - 1)
        self.ssim_post = self.ssim_post / \
            (len(generated_sequence_posterior) - 1)
        self.psnr_prior = self.psnr_prior / \
            (len(generated_sequence_posterior) - 1)
        self.ssim_prior = self.ssim_prior / \
            (len(generated_sequence_posterior) - 1)

        return [gt_sequence, generated_sequence_posterior, generated_sequence_prior], [self.reconstruction_loss_post.item(), self.alignment_loss.item(), self.latent_loss.item(), self.kl_loss_prior.item(), self.reconstruction_loss_prior.item(), self.last_frame_gen_loss.item()], [self.psnr_post.item(), self.ssim_post.item(), self.psnr_prior.item(), self.ssim_prior.item()]

    def single_image_testing(self, sharp_images, motion_blur_image):
        self.init_hidden()
        seq_len = len(sharp_images)

        # motion encoding
        blur_features = self.encoder(motion_blur_image)
        blur_features = self.motion_encoder(blur_features[1][5])  # 8*8*8*8
        blur_features = blur_features.view(self.batch_size, -1)

        # losses
        self.reconstruction_loss_post = 0
        self.psnr_post = 0
        self.ssim_post = 0
        generated_sequence_posterior = {}
        gt_sequence = {}

        last_image = sharp_images[0]
        print("last image shape", last_image.shape)
        print("seq_len", seq_len)
        for i in range(1, seq_len):
            last_time_stamp = i
            current_image = last_image

            # updating ground truth
            gt_sequence[i] = sharp_images[i].detach().cpu()

            # encoder
            sharp_features_encoding, feature_cache = self.encoder(
                current_image)

            time_info = self.pos_encoder(
                last_time_stamp, i, len(sharp_images), self.batch_size).to(sharp_features_encoding.device)

            posterior_input = torch.cat(
                (sharp_features_encoding, blur_features, time_info), 1)

            # posterior
            z_i_post, mu_i_post, logvar_i_post = self.posterior_lstm(
                posterior_input)

            # decoder_lstm
            z_decoder = self.decoder_lstm(z_i_post)

            # decoder
            x_i = self.decoder(z_decoder, feature_cache)

            generated_sequence_posterior[i] = x_i.detach().cpu()

            self.reconstruction_loss_post = self.reconstruction_loss_post + self.mse_criterion(
                x_i, sharp_images[i])

            self.psnr_post = self.psnr_post + \
                self.psnr_criterion(x_i, sharp_images[i])
            self.ssim_post = self.ssim_post + \
                self.ssim_criterion(x_i, sharp_images[i])

            last_image = x_i

        # average all losses over the sequence
        self.reconstruction_loss_post = self.reconstruction_loss_post / \
            (len(generated_sequence_posterior) - 1)

        self.ssim_post = self.ssim_post / \
            (len(generated_sequence_posterior) - 1)

        self.psnr_post = self.psnr_post / \
            (len(generated_sequence_posterior) - 1)

        return [gt_sequence, generated_sequence_posterior], [self.reconstruction_loss_post.item()], [self.psnr_post.item(), self.ssim_post.item()]

    def sequence_testing(self, sharp_images, motion_blur_image):
        self.init_hidden()

        # motion encoding
        blur_features = self.encoder(motion_blur_image)
        blur_features = self.motion_encoder(blur_features[1][5])  # 8*8*8*8
        blur_features = blur_features.view(self.batch_size, -1)

        self.reconstruction_loss_post = 0
        generated_sequence_posterior = {}
        gt_sequence = {}
        self.psnr_post = 0
        self.ssim_post = 0
        # sequence generation
        frame_use = np.random.uniform(
            0, 1, len(sharp_images)) >= self.prob_for_frame_drop
        last_time_stamp = 0

        for i in range(1, len(sharp_images)):
            # encoder
            if frame_use[i]:
                gt_sequence[i] = sharp_images[i].detach().cpu()

                sharp_features_encoding, feature_cache = self.encoder(
                    sharp_images[last_time_stamp])
                time_info = self.pos_encoder(
                    last_time_stamp, i, len(sharp_images), self.batch_size).to(sharp_features_encoding.device)

                posterior_input = torch.cat(
                    (sharp_features_encoding, blur_features, time_info), 1)

                z_i_post, mu_i_post, logvar_i_post = self.posterior_lstm(
                    posterior_input)

                # decoder
                z_decoder = self.decoder_lstm(z_i_post)
                # print(z_decoder.shape)
                x_i = self.decoder(z_decoder, feature_cache)

                generated_sequence_posterior[i] = x_i.detach().cpu()

                self.reconstruction_loss_post = self.reconstruction_loss_post + self.mse_criterion(
                    x_i, sharp_images[i])

                self.psnr_post = self.psnr_post + \
                    self.psnr_criterion(x_i, sharp_images[i])
                self.ssim_post = self.ssim_post + \
                    self.ssim_criterion(x_i, sharp_images[i])

                last_time_stamp = i
            else:
                continue

        # average all losses over the sequence
        self.reconstruction_loss_post = self.reconstruction_loss_post / \
            (len(generated_sequence_posterior) - 1)
        self.psnr_post = self.psnr_post / \
            (len(generated_sequence_posterior) - 1)
        self.ssim_post = self.ssim_post / \
            (len(generated_sequence_posterior) - 1)

        return [gt_sequence, generated_sequence_posterior], [self.reconstruction_loss_post.item()], [self.psnr_post.item(), self.ssim_post.item()]

    def forward(self, sharp_images, motion_blur_image, mode, single_image_prediction=False):
        # print(mode)
        if mode == "train":
            if single_image_prediction:
                gen_seq, losses, metric = self.single_image_training(
                    sharp_images, motion_blur_image)
            else:
                gen_seq, losses, metric = self.sequence_training(
                    sharp_images, motion_blur_image)
        elif mode == "test":
            if single_image_prediction:
                gen_seq, losses, metric = self.single_image_testing(
                    sharp_images, motion_blur_image)
            else:
                gen_seq, losses, metric = self.sequence_testing(
                    sharp_images, motion_blur_image)

        return gen_seq, losses, metric

    def update_model(self):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.posterior_optimizer.zero_grad()
        self.prior_optimizer.zero_grad()
        self.decoder_lstm_optimizer.zero_grad()

        # convert psnr to loss

        loss = self.reconstruction_loss_post + self.alignment_loss + \
            self.latent_loss + 1.5 * \
            torch.exp(-1*self.psnr_post) + 1.5*(1-self.ssim_post)
        loss.backward(retain_graph=True)

        self.prior_lstm.zero_grad()
        prior_loss = self.kl_loss_prior + \
            self.reconstruction_loss_prior + 1.5*self.last_frame_gen_loss + \
            2*torch.exp(-1*self.psnr_prior) + 2*(1-self.ssim_prior)
        prior_loss.backward(retain_graph=True)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.posterior_optimizer.step()
        self.prior_optimizer.step()
        self.decoder_lstm_optimizer.step()

        return loss.item(), prior_loss.item()

    def save(self, fname):
        # save weights of each module
        states = {
            'encoder': self.encoder.state_dict(),
            'motion_encoder': self.motion_encoder.state_dict(),
            'pos_encoder': self.pos_encoder.state_dict(),
            'posterior_lstm': self.posterior_lstm.state_dict(),
            'prior_lstm': self.prior_lstm.state_dict(),
            'decoder_lstm': self.decoder_lstm.state_dict(),
            'decoder': self.decoder.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'refinement': self.refinement.state_dict(),
            # 'motion_encoder_optimizer': self.motion_encoder_optimizer.state_dict(),
            # 'pos_encoder_optimizer': self.pos_encoder_optimizer.state_dict(),
            'posterior_optimizer': self.posterior_optimizer.state_dict(),
            'prior_optimizer': self.prior_optimizer.state_dict(),
            'decoder_lstm_optimizer': self.decoder_lstm_optimizer.state_dict(),
            'refinement_optimizer': self.refinement_optimizer.state_dict(),
        }
        torch.save(states, fname)

    def load(self, fname):
        # load weights of each module
        states = torch.load(fname)
        self.encoder.load_state_dict(states['encoder'])
        self.motion_encoder.load_state_dict(states['motion_encoder'])
        self.pos_encoder.load_state_dict(states['pos_encoder'])
        self.posterior_lstm.load_state_dict(states['posterior_lstm'])
        self.prior_lstm.load_state_dict(states['prior_lstm'])
        self.decoder_lstm.load_state_dict(states['decoder_lstm'])
        self.decoder.load_state_dict(states['decoder'])
        # self.refinement.load_state_dict(states['refinement'])
        # if self.args.test != True:
        #     self.decoder_optimizer.load_state_dict(states['decoder_optimizer'])
        #     # self.motion_encoder_optimizer.load_state_dict(
        #     #     states['motion_encoder_optimizer'])
        #     # self.pos_encoder_optimizer.load_state_dict(
        #     #     states['pos_encoder_optimizer'])
        #     self.posterior_optimizer.load_state_dict(
        #         states['posterior_optimizer'])
        #     self.prior_optimizer.load_state_dict(
        #         states['prior_optimizer'])
        #     self.decoder_lstm_optimizer.load_state_dict(
        #         states['decoder_lstm_optimizer'])
