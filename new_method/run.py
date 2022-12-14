from __future__ import print_function, division
import torchvision.transforms as transforms
from utils.loss import PSNR, SSIM
from utils.visualization import visualize
from datasets.dataloader import Gopro, get_transform
from models.variational_gen import Variational_Gen
from models.attention_gen import Attention_Gen
from utils.loss import PSNR, SSIM, KLCriterion, SmoothMSE
from ray import tune
# from fyp.new_method.utils.utils import generation_viz
import configparser
import argparse
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, ASGD
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os
import yaml
import sys
import torchvision.utils as torch_utils
##########################################################
# scheduler for optimizers


def fetch_optimizer(args, model, lr, steps_per_epoch, epochs):
    # optimizer
    if args.optimizer['optimizer_name'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr,
                         weight_decay=args.optimizer['weight_decay'])
    elif args.optimizer['optimizer_name'] == 'AdamW':
        optimizer = AdamW(model.parameters(
        ), lr=lr, weight_decay=args.optimizer['weight_decay'], eps=float(args.optimizer['eps']))
    elif args.optimizer['optimizer_name'] == 'SGD':
        optimizer = SGD(model.parameters(
        ), lr=lr, weight_decay=args.optimizer['weight_decay'], momentum=args.optimizer['momentum'])
    elif args.optimizer['optimizer_name'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=lr,
                            weight_decay=args.optimizer['weight_decay'])
    elif args.optimizer['optimizer_name'] == 'Adagrad':
        optimizer = Adagrad(model.parameters(), lr=lr,
                            weight_decay=args.optimizer['weight_decay'])
    elif args.optimizer['optimizer_name'] == 'Adadelta':
        optimizer = Adadelta(model.parameters(), lr=lr,
                             weight_decay=args.optimizer['weight_decay'])
    elif args.optimizer['optimizer_name'] == 'Adamax':
        optimizer = Adamax(model.parameters(), lr=lr,
                           weight_decay=args.optimizer['weight_decay'])
    elif args.optimizer['optimizer_name'] == 'ASGD':
        optimizer = ASGD(model.parameters(), lr=lr,
                         weight_decay=args.optimizer['weight_decay'])
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))

    # scheduler
    if args.if_scheduler:
        if args.scheduler == 'StepLR':
            scheduler = StepLR(optimizer, step_size=0.0001, gamma=0.9)
        elif args.scheduler == 'MultiStepLR':
            scheduler = MultiStepLR(
                optimizer, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, gamma=0.99)
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode=args.mode, factor=args.factor, patience=args.patience, verbose=True)
        elif args.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.T_max, eta_min=args.eta_min)
        elif args.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
        elif args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
        else:
            raise ValueError('Unknown scheduler: {}'.format(args.scheduler))
    else:
        scheduler = None

    return optimizer, scheduler

######################################################################
# updating data to tensorboardX


class TensorboardWriter(object):
    def __init__(self, args, scheduler, model):
        self.args = args
        if args.train:
            mode = 'train'
        elif args.test:
            mode = 'test'
        elif args.evaluate:
            mode = 'evaluate'
        elif args.sweep:
            mode = 'sweep'
        self.writer = SummaryWriter(
            args.run_dir + args.name + "_" + args.dataset + "_" + mode)
        print(args.run_dir + args.name + "_" + args.dataset + "_" + mode)
        self.model = model
        self.scheduler = scheduler
        self.update_args(self.args)

    # update arguments to tensorboardX
    def update_args(self, args):
        for key, value in args.__dict__.items():
            self.writer.add_text(key, str(value), 0)

    # update model_weights and gradients to tensorboardX
    def update_model(self, model, step):
        for name, param in model.named_parameters():
            self.writer.add_histogram(
                name, param.clone().cpu().data.numpy(), step)
            if param.grad is not None:
                self.writer.add_histogram(
                    name + '/grad', param.grad.clone().cpu().data.numpy(), step)

    def update_loss_and_metric(self, name, loss, loss_prior, step):
        self.writer.add_scalar(name + '/posterier_loss', loss, step)
        self.writer.add_scalar(name + '/prior_loss', loss_prior, step)
        # for key, value in metric.items():
        #     self.writer.add_scalar(name + "/" + key, value, step)

    # updating learning rate
    def update_learning_rate(self, lr, step):
        self.writer.add_scalar('train/learning_rate', lr, step)

    def update(self, model, loss, metric, step, mode, setup=None, seq=None):
        if mode == 'train':
            if self.args.update_training_loss:
                self.update_loss_and_metric(mode, loss, metric, step)
                if self.scheduler is not None:
                    self.update_learning_rate(
                        self.scheduler.get_last_lr()[0], step)
                else:
                    self.update_learning_rate(
                        self.args.training_parameters['lr'], step)
            if self.args.update_weights:
                self.update_model(model, step)
        elif mode == 'test':
            self.update_loss_and_metric(
                mode + "/" + seq + "/" + setup, loss, metric, step)

    def close(self):
        self.writer.close()
######################################################################


######################################################################
# loading model directory
torch.autograd.set_detect_anomaly(True)


def train(model, args):
    model.train()
    # load data
    print("Augmentions used...")
    transform = get_transform(args, 'train')
    print("training augmentation: ", transform)

    training_dataset = Gopro(args, transform, "train")
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=args.training_parameters['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print("loaded data and dataloader")

    writer = TensorboardWriter(args, None, model)
    print("TRAINING STARTED")

    for epoch in range(args.training_parameters['num_epochs']):
        for i, data in enumerate(train_loader):
            seq_len = data['length'].cuda()
            blur_img = data['blur'].cuda()
            gen_seq = data['gen_seq']

            # for varing length generation
            step_size = np.random.randint(3, 6)
            gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
            gen_seq = gen_seq[::step_size]
            gen_seq = gen_seq.cuda()

            # forward pass
            generated_seq, losses, metric = model(
                gen_seq, blur_img, "train", single_image_prediction=True)

            # print(generated_seq[0][1].shape)
            # loss and backprop
            losses = model.update_model()

            writer.update(model, losses[0], losses[-1], epoch *
                          len(train_loader)+i, 'train')

            if i % args.display_step_freq == 0:
                if args.visualize:
                    if not os.path.exists(os.path.join(args.visualization_path, "train")):
                        os.makedirs(os.path.join(
                            args.visualization_path, "train"))

                    if (len(generated_seq) == 3):
                        visualize(generated_seq[1], generated_seq[0], prior=generated_seq[2], path=args.visualization_path + "train/" +
                                  args.name + "_" + args.dataset + "_train" + "_epoch_" + str(epoch) + "_step_" + str(i) + ".png")
                    else:
                        visualize(generated_seq[1], generated_seq[0], path=args.visualization_path + "train/" + args.name +
                                  "_" + args.dataset + "_train" + "_epoch_" + str(epoch) + "_step_" + str(i) + ".png")

                print("epoch: ", epoch, "step: ", i, "gen_seq_length:",
                      len(generated_seq[0]), "losses: ", losses)
                print("metric: ", metric)
            if (i+args.save_step_freq) % args.save_step_freq == 0:
                print("saving model")
                model.save(args.checkpoint_dir + args.name +
                           '_epoch_' + str(epoch) + '_step_' + str(i) + '.pth')
    model.save(args.checkpoint_dir + args.name + '_final' + '.pth')
    writer.close()


def test(model, args):
    model.eval()
    print("Augmentions used...")
    transform = get_transform(args, 'test')
    print("training augmentation: ", transform)

    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1/0.5, 1/0.5, 1/0.5]),
                                   transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                        std=[1., 1., 1.]),
                                   ])

    testing_data = Gopro(args, transform, "train")
    test_loader = torch.utils.data.DataLoader(
        testing_data, batch_size=args.testing_parameters['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print("loaded data and dataloader")

    # writer = TensorboardWriter(args, None, model)
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_blur_psnr = 0
    total_blur_ssim = 0
    for i, data in enumerate(test_loader):
        torch.cuda.empty_cache()
        seq_len = data['length'].cuda()
        blur_img = data['blur'].cuda()
        gen_seq = data['gen_seq']
        # for varing length generation
        step_size = 4
        gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
        gen_seq = gen_seq[::step_size]
        gen_seq = gen_seq.cuda()

        psnr_cri = PSNR()
        ssim_cri = SSIM()
        blur_psnr = psnr_cri(blur_img, gen_seq[len(gen_seq)//2])
        blur_ssim = ssim_cri(blur_img, gen_seq[len(gen_seq)//2])
        total_blur_ssim = total_blur_ssim + blur_ssim.item()
        total_blur_psnr = total_blur_psnr + blur_psnr.item()
        # forward pass
        generated_seq, losses, metric = model(
            gen_seq, blur_img, "test", single_image_prediction=False)
        total_loss = total_loss + losses[0]
        total_psnr = total_psnr + metric[0]
        total_ssim = total_ssim + metric[1]
        # writer.update(model, posterior_loss, prior_loss, epoch *
        # len(train_loader)+i, 'train')
        print("losses: ", losses[0])
        print("psnr", metric[0])
        print("ssim", metric[1])

        if args.visualize:
            if not os.path.exists(os.path.join(args.visualization_path, "test")):
                os.makedirs(os.path.join(args.visualization_path, "test"))

            if not os.path.exists(os.path.join(args.visualization_path, "test/seq")):
                os.makedirs(os.path.join(args.visualization_path, "test/seq"))

            if not os.path.exists(os.path.join(args.visualization_path, "test/blur")):
                os.makedirs(os.path.join(args.visualization_path, "test/blur"))

            blur_img_cpu = invTrans(blur_img.squeeze(0).cpu().detach())
            torch_utils.save_image(blur_img_cpu, args.visualization_path + "test/blur/" +
                                   args.name + "_" + args.dataset + "_train" + "_step_" + str(i) + ".png")

            if (len(generated_seq) == 3):
                visualize(generated_seq[1], generated_seq[0], prior=generated_seq[2], path=args.visualization_path + "test/seq/" +
                          args.name + "_" + args.dataset + "_train" + "_step_" + str(i) + ".png")
            else:
                visualize(generated_seq[1], generated_seq[0], path=args.visualization_path + "test/seq/" + args.name +
                          "_" + args.dataset + "_train" + "_step_" + str(i) + ".png")
    print("total loss: ", total_loss/len(test_loader), "total psnr: ",
          total_psnr/len(test_loader), "total ssim: ", total_ssim/len(test_loader))
    print("total blur psnr: ", total_blur_psnr / len(test_loader),
          "total blur ssim: ", total_blur_ssim/len(test_loader))
    # writer.close()


def evaluate(model, args):
    aug_params = {'crop_size': args.training_augmentations['RandomCrop']
                  ['size'], 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
    train_dataset = datasets.Carla_Dataset(
        aug_params, split='training', root=args.data_root, seq=args.training_seq, setup_type=args.training_setup)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.training_parameters['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(len(train_dataset))
    optimizer, scheduler = fetch_optimizer(args, model, args.training_parameters['lr'], len(
        train_loader), args.training_parameters['num_epochs'])
    scaler = GradScaler(enabled=args.mixed_precision)
    writer = TensorboardWriter(args, scheduler, model)
    # training loop
    for epoch in range(args.training_parameters['num_epochs']):
        torch.cuda.empty_cache()
        for i, data in enumerate(train_loader):
            # data
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data]
            # forward pass
            flow_pred = model(image1, image2)
            # loss
            loss, metric = sequence_loss(flow_pred, image1, image2, flow, valid,
                                         gamma=args.training_parameters['flow_weighting_factor_gamma'], use_matching_loss=args.use_mix_attn)
            if scheduler != None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.training_parameters['clip_grad_norm'])
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                if i % args.display_step_freq == 0:
                    print('Epoch: [{}/{}], Step: [{}/{}], lr: {:.8f}, Loss: {:.8f}, epe:{:.8f}, 1px: {:.8f}, 3px: {:.8f},5px: {:.8f}'.format(
                        epoch+1, args.training_parameters['num_epochs'], i+1, len(train_loader), scheduler.get_last_lr()[0], loss.item(), metric['epe'], metric['1px'], metric['3px'], metric['5px']))

            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.training_parameters['clip_grad_norm'])
                scaler.update()
                if i % args.display_step_freq == 0:
                    print('Epoch: [{}/{}], Step: [{}/{}], lr: {:.8f}, Loss: {:.8f}, epe:{:.8f}, 1px: {:.8f}, 3px: {:.8f},5px: {:.8f}'.format(epoch+1, args.training_parameters['num_epochs'],
                          i+1, len(train_loader), args.training_parameters['lr'], loss.item(), metric['epe'], metric['1px'], metric['3px'], metric['5px']))
            writer.update(model, loss, metric, epoch *
                          len(train_loader)+i, 'train')

            if i % args.save_step_freq == 0:
                print('Saving model...')
                torch.save(model.state_dict(), args.checkpoint_dir + args.name +
                           '_epoch_' + str(epoch) + '_step_' + str(i) + '.pth')

            # evaluate model
            if (i+1) % args.eval_step_freq == 0:
                print('Evaluating model...')
                model.eval()
                for seq in args.test_seq:
                    seq_list = []
                    seq_list.append(seq)
                    for setup in args.test_setup:
                        setuplist = []
                        setuplist.append(setup)
                        aug_params = {
                            'crop_size': args.testing_augmentations['RandomCrop']['size'], 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
                        test_dataset = datasets.Carla_Dataset(
                            aug_params, split='training', root=args.data_root_test, seq=seq_list, setup_type=setuplist)
                        test_loader = torch.utils.data.DataLoader(
                            test_dataset, batch_size=args.testing_parameters['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
                        print(len(test_loader))
                        total_loss = 0
                        total_metric = {}
                        total_metric['epe'] = 0
                        total_metric['1px'] = 0
                        total_metric['3px'] = 0
                        total_metric['5px'] = 0
                        for j, data in enumerate(test_loader):
                            # data
                            image1, image2, flow, valid = [
                                x.cuda() for x in data]
                            # forward pass
                            flow_pred = model(image1, image2)
                            # loss
                            loss, metric = sequence_loss(
                                flow_pred, image1, image2, flow, valid, gamma=args.testing_parameters['flow_weighting_factor_gamma'], use_matching_loss=args.use_mix_attn)
                            # add loss and metric to tensorboard
                            total_loss += loss.item()/len(test_loader)
                            total_metric['epe'] += metric['epe'] / \
                                len(test_loader)
                            total_metric['1px'] += metric['1px'] / \
                                len(test_loader)
                            total_metric['3px'] += metric['3px'] / \
                                len(test_loader)
                            total_metric['5px'] += metric['5px'] / \
                                len(test_loader)
                            if args.visualize:
                                visualize_flow(
                                    flow_pred, image1, image2, flow, valid, args.visualization_path, setup, seq, j)
                        writer.update(model, total_loss/len(test_loader), total_metric,
                                      epoch*len(train_loader)+i, 'test', setup=setup, seq=seq)
                        print('Setup: {}, Loss: {:.8f}, epe:{:.8f}, 1px: {:.8f}, 3px: {:.8f},5px: {:.8f}'.format(
                            setup, total_loss, total_metric['epe'], total_metric['1px'], total_metric['3px'], total_metric['5px']))
                        torch.cuda.empty_cache()
            model.train()
    writer.close()


def run(args):
    print(args)
    if args.model == 'variational_gen':
        model = Variational_Gen(args)
    elif args.model == 'attention_gen':
        model = Attention_Gen(args)

    if args.weights:
        print("weights loaded")
        model.load(args.weights)

    # print cuda device used for training
    print("GPU:", torch.cuda.get_device_name(0))
    model.cuda()

    print("MODEL LOADED SUCCESSFULLY")

    if args.train:
        train(model, args)

    if args.test:
        test(model, args)

    if args.evaluate:
        evaluate(model, args)


def sweep(args):
    # configure parameters
    config = {
        'num_epochs': tune.grid_search(np.arange(args.hyperparameters['num_epochs']['min'], args.hyperparameters['num_epochs']['max'], args.hyperparameters['num_epochs']['step']).tolist()),
        'learning_rate': tune.loguniform(args.hyperparameters['learning_rate']['min'], args.hyperparameters['learning_rate']['max']),
        'batch_size': tune.grid_search(np.arange(args.hyperparameters['batch_size']['min'], args.hyperparameters['batch_size']['max'], args.hyperparameters['batch_size']['step']).tolist()),
        'dropout': tune.uniform(args.hyperparameters['dropout']['min'], args.hyperparameters['dropout']['max']),
        'iterations': tune.grid_search(np.arange(args.hyperparameters['iterations']['min'], args.hyperparameters['iterations']['max'], args.hyperparameters['iterations']['step']).tolist()),
        'flow_weighting_factor_gamma': tune.uniform(args.hyperparameters['flow_weighting_factor_gamma']['min'], args.hyperparameters['flow_weighting_factor_gamma']['max']),
    }
    # configure scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.hyperparameters['num_epochs']['max'],
        grace_period=1,
        reduction_factor=2)

    # configure reporter
    reporter = CLIReporter(
        parameter_columns=["num_epochs", "learning_rate", "batch_size",
                           "dropout", "iterations", "flow_weighting_factor_gamma"],
        metric_columns=["loss", "epe", "acc_1px", "acc_3px", "acc_5px", "training_iteration"])

    results = tune.run(
        partial(train_sweep, args=args),
        config=config,
        resources_per_trial={"cpu": 4, "gpu": 1},
        num_samples=args.num_sweeps,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = results.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


def train_sweep(config, args=None):
    """_summary_

    config = {
        *'num_epochs': tune.grid_search(np.arange(args.hyperparameters['num_epochs']['min'],args.hyperparameters['num_epochs']['max'],args.hyperparameters['num_epochs']['step'])),
        *'learning_rate': tune.loguniform(args.hyperparameters['learning_rate']['min'],args.hyperparameters['learning_rate']['max']),
        *'batch_size': tune.grid_search(np.arange(args.hyperparameters['batch_size']['min'],args.hyperparameters['batch_size']['max'],args.hyperparameters['batch_size']['step'])),
        *'dropout': tune.uniform(args.hyperparameters['dropout']['min'],args.hyperparameters['dropout']['max']),
        *'iterations': tune.grid_search(np.arange(args.hyperparameters['iterations']['min'],args.hyperparameters['iterations']['max'],args.hyperparameters['iterations']['step'])),
        *'flow_weighting_factor_gamma': tune.uniform(args.hyperparameters['flow_weighting_factor_gamma']['min'],args.hyperparameters['flow_weighting_factor_gamma']['max']),
    }
    """
    print(type(config['batch_size']))
    aug_params = {'crop_size': args.training_augmentations['RandomCrop']
                  ['size'], 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
    train_dataset = datasets.Carla_Dataset(
        aug_params, split='training', root=args.data_root, seq=args.training_seq, setup_type=args.training_setup)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # print(len(train_dataset))
    # ,iters=config['iterations'],dropout=config['dropout']
    model = nn.DataParallel(GMFlowNetModel(
        args, iters=config['iterations'], dropout=config['dropout']), device_ids=args.gpus)
    # save initial model
    # torch.save(model.state_dict(), 'test.pth')
    # print("model saved")
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights), strict=True)
        print("all keys matched")
    model.cuda()

    optimizer, scheduler = fetch_optimizer(
        args, model, config['learning_rate'], len(train_loader), config['num_epochs'])
    scaler = GradScaler(enabled=args.mixed_precision)
    writer = TensorboardWriter(args, scheduler, model)
    # training loop
    for epoch in range(config['num_epochs']):
        for i, data in enumerate(train_loader):
            # data
            # torch.cuda.empty_cache()
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data]
            # forward pass
            # print(image1, image2)
            flow_pred = model(image1, image2)
            # print(flow_pred)
            # loss
            loss, metric = sequence_loss(flow_pred, image1, image2, flow, valid,
                                         gamma=config['flow_weighting_factor_gamma'], use_matching_loss=args.use_mix_attn)
            if scheduler != None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.training_parameters['clip_grad_norm'])
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                if i % args.display_step_freq == 0:
                    # tune.report(loss=loss.item(), epe = metric['epe'], acc_1px = metric['1px'] , acc_3px = metric['3px'], acc_5px = metric['5px'])
                    print('Epoch: [{}/{}], Step: [{}/{}], lr: {:.8f}, Loss: {:.8f}, epe:{:.8f}, 1px: {:.8f}, 3px: {:.8f},5px: {:.8f}'.format(
                        epoch+1, args.training_parameters['num_epochs'], i+1, len(train_loader), scheduler.get_last_lr()[0], loss.item(), metric['epe'], metric['1px'], metric['3px'], metric['5px']))

            else:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.training_parameters['clip_grad_norm'])
                scaler.update()
                if i % args.display_step_freq == 0:
                    # tune.report(loss=loss, epe = metric['epe'], acc_1px = metric['1px'] , acc_3px = metric['3px'], acc_5px = metric['5px'])
                    print('Epoch: [{}/{}], Step: [{}/{}], lr: {:.8f}, Loss: {:.8f}, epe:{:.8f}, 1px: {:.8f}, 3px: {:.8f},5px: {:.8f}'.format(epoch+1, args.training_parameters['num_epochs'],
                          i+1, len(train_loader), args.training_parameters['lr'], loss.item(), metric['epe'], metric['1px'], metric['3px'], metric['5px']))
            # writer.update(model, loss, metric, epoch*len(train_loader)+i, 'train')

            if (i+1) % args.save_step_freq == 0:
                print('Saving model...')

                torch.save(model.state_dict(), args.checkpoint_dir + args.name +
                           '_epoch_' + str(epoch) + '_step_' + str(i) + '.pth')

            # evaluate model
            if (i+1) % args.eval_step_freq == 0:
                print('Evaluating model...')
                model.eval()
                for seq in args.test_seq:
                    seq_list = []
                    seq_list.append(seq)
                    loss_overall = 0
                    metric_overall = {}
                    metric_overall['epe'] = 0
                    metric_overall['1px'] = 0
                    metric_overall['3px'] = 0
                    metric_overall['5px'] = 0

                    for setup in args.test_setup:
                        setuplist = []
                        setuplist.append(setup)
                        aug_params = {
                            'crop_size': args.testing_augmentations['RandomCrop']['size'], 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
                        test_dataset = datasets.Carla_Dataset(
                            aug_params, split='training', root=args.data_root_test, seq=seq_list, setup_type=setuplist)
                        test_loader = torch.utils.data.DataLoader(
                            test_dataset, batch_size=args.testing_parameters['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
                        print(len(test_loader))
                        total_loss = 0
                        total_metric = {}
                        total_metric['epe'] = 0
                        total_metric['1px'] = 0
                        total_metric['3px'] = 0
                        total_metric['5px'] = 0
                        for j, data in enumerate(test_loader):
                            # data
                            image1, image2, flow, valid = [
                                x.cuda() for x in data]
                            # forward pass
                            flow_pred = model(image1, image2)
                            # loss
                            loss, metric = sequence_loss(
                                flow_pred, image1, image2, flow, valid, gamma=args.testing_parameters['flow_weighting_factor_gamma'], use_matching_loss=args.use_mix_attn)
                            # add loss and metric to tensorboard
                            total_loss += loss.item()/len(test_loader)
                            total_metric['epe'] += metric['epe'] / \
                                len(test_loader)
                            total_metric['1px'] += metric['1px'] / \
                                len(test_loader)
                            total_metric['3px'] += metric['3px'] / \
                                len(test_loader)
                            total_metric['5px'] += metric['5px'] / \
                                len(test_loader)
                            if args.visualize:
                                visualize_flow(
                                    flow_pred, image1, image2, flow, valid, args.visualization_path, setup, seq, j)
                        # writer.update(model, total_loss/len(test_loader), total_metric, epoch*len(train_loader)+i, 'test',setup=setup,seq=seq)
                        print('Setup: {}, Loss: {:.8f}, epe:{:.8f}, 1px: {:.8f}, 3px: {:.8f},5px: {:.8f}'.format(
                            setup, total_loss, total_metric['epe'], total_metric['1px'], total_metric['3px'], total_metric['5px']))
                        loss_overall += total_loss/len(args.test_setup)
                        metric_overall['epe'] += total_metric['epe'] / \
                            len(args.test_setup)
                        metric_overall['1px'] += total_metric['1px'] / \
                            len(args.test_setup)
                        metric_overall['3px'] += total_metric['3px'] / \
                            len(args.test_setup)
                        metric_overall['5px'] += total_metric['5px'] / \
                            len(args.test_setup)
                    tune.report(loss=loss_overall, epe=metric_overall['epe'], acc_1px=metric_overall[
                                '1px'], acc_3px=metric_overall['3px'], acc_5px=metric_overall['5px'])
                    # torch.cuda.empty_cache()
            model.train()
    writer.close()


if __name__ == '__main__':
    # set the seed
    torch.manual_seed(0)
    np.random.seed(0)

    # parameters
    ######################################################################
    """
    MAX_FLOW: maximum flow value
    LOSS_SUM_FLOW: number of loss to add to the total loss
    DISPLAY_STEP: steps after which the loss is displayed

    train: true if training, false if testing
    update_weights: true if weights are updated, false if not
    update_weight_grad: true if gradients are updated, false if not
    update_loss: true if loss is updated, false if not
    update_accuracy: true if accuracy is updated, false if not

    use_parameter_sweap: true if parameter sweap is used, false if not
    parameter_list : list of parameters to be sweaped
    swep_parameter: parameter to be sweaped (list)

    optimizer_name: name of the optimizer
    grad_scaler: use gradient scaler or not     


    """
    ######################################################################
    parser = argparse.ArgumentParser()
    # about experiment
    parser.add_argument('--config', default=r'C:\Users\Machine Learning GPU\Desktop\fyp\fyp\new_method\config.yml',
                        help="config file", required=False)
    parser.add_argument(
        '--name', help="name of the experiment", required=False)

    # about training and testing
    parser.add_argument('--resume', type=bool,
                        help="resume training", required=False)
    parser.add_argument('--evaluate', type=bool,
                        help="evaluate model", required=False)
    parser.add_argument('--test', type=bool, help="test model", required=False)
    parser.add_argument('--train', type=bool,
                        help="train model", required=False)
    parser.add_argument('--weights', default=None,
                        help="path to weights file", required=False)
    parser.add_argument('--save', default=None,
                        help="path to save weights file", required=False)
    parser.add_argument('--sweep', type=bool,
                        help="sweep parameters", required=False)

    # update and display frequency
    parser.add_argument('--display_step_freq', type=int,
                        help="display step", required=False)
    parser.add_argument('--eval_step_freq', type=int,
                        help="eval step", required=False)
    parser.add_argument('--save_step_freq', type=int,
                        help="save step", required=False)

    # about data update to tensorboard
    parser.add_argument('--update_weights', type=bool,
                        help="update weights", required=False)
    parser.add_argument('--update_weight_grad', type=bool,
                        help="update weight gradients", required=False)
    parser.add_argument('--update_training_loss', type=bool,
                        help="update loss", required=False)
    parser.add_argument('--update_training_metric', type=bool,
                        help="update accuracy", required=False)
    parser.add_argument('--update_validation_loss', type=bool,
                        help="update loss", required=False)
    parser.add_argument('--update_validation_metric',
                        type=bool, help="update accuracy", required=False)
    parser.add_argument('--update_sweep_data', type=bool,
                        help="update sweep data", required=False)

    # visualize data
    parser.add_argument('--visualize', type=bool,
                        help="visualize data", required=False)
    parser.add_argument('--visulaization_path',
                        help="path to save visualization", required=False)

    parser.add_argument('--gpus', default='0',
                        help="gpus to use", required=False)
    ######################################################################
    """folder format
    input
    output
    gt
    epe
    """
    ######################################################################

    # hyperparameters
    # all defined in config file

    sys.argv = ['-f']
    args = parser.parse_args()
    # load arguments from config file
    # print(args.config)
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # load arguments from command line
    # print(config)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    # load arguments from config file
    for key, value in config.items():
        if value is not None:
            config[key] = value

    # update args
    args = argparse.Namespace(**config)
    # print(args.training_augmentations)

    # write args to yml file
    # with open(args.config, 'w') as outfile:
    #     yaml.dump(vars(args), outfile, default_flow_style=False)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.run_dir):
        os.mkdir(args.run_dir)
    if args.visualize:
        if not os.path.exists(args.visualization_path):
            os.mkdir(args.visualization_path)
    # set the gpus
    args.gpus = [i for i in range(len(args.gpus))]

    if not args.sweep:
        run(args)
    else:
        sweep(args)
