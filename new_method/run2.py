from __future__ import print_function, division
import torchvision.transforms as transforms
from utils.loss import PSNR, SSIM
from utils.visualization import visualize
from datasets.dataloader import Gopro, get_transform
from models.variational_gen import Variational_Gen
from models.attention_gen import Attention_Gen
from models.Blur_decoder import Blur_decoder
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


class TensorboardWriter(object):
    def __init__(self, args, model):
        self.args = args
        type = args.mode
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
        # self.scheduler = scheduler
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

    def update_loss_and_metric(self, name, loss, metric, step):
        self.writer.add_scalar(name + '/loss', loss, step)
        # self.writer.add_scalar(name + '/prior_loss', loss_prior, step)
        self.writer.add_scalar(name + '/psnr', metric[0], step)
        self.writer.add_scalar(name + '/ssim', metric[1], step)
        if self.mode == 'train':
            if step % self.args.display_step_freq == 1:
                print('step: {}, loss: {:.4f}, psnr: {:.4f}, ssim: {:.4f}'.format(
                    step, loss, metric[0], metric[1]))
        else:
            print('step: {}, loss: {:.4f}, psnr: {:.4f}, ssim: {:.4f}'.format(
                step, loss, metric[0], metric[1]))
    # updating learning rate

    def update_learning_rate(self, lr, step):
        self.writer.add_scalar('train/learning_rate', lr, step)

    def update(self, model, loss, metric, step, mode, type):
        self.mode = mode
        if mode == 'train':
            if self.args.update_training_loss:
                self.update_loss_and_metric(
                    mode + "/" + type, loss, metric, step)
            # if self.args.update_learning_rate:
            if self.args.update_weights:
                self.update_model(model, step)
        elif mode == 'test':
            self.update_loss_and_metric(
                mode + "/" + type, loss, metric, step)

    def close(self):
        self.writer.close()
######################################################################


######################################################################
# loading model directory
torch.autograd.set_detect_anomaly(True)


def train(model, args):
    model.train()
    psnr = PSNR()
    ssim = SSIM()

    # load data
    print("Augmentions used...")
    transform = get_transform(args, 'train')
    print("training augmentation: ", transform)

    training_dataset = Gopro(args, transform, "train")
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=args.training_parameters['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print("loaded data and dataloader")

    writer = TensorboardWriter(args, model)
    print("TRAINING STARTED")

    for epoch in range(args.training_parameters['num_epochs']):
        for i, data in enumerate(train_loader):
            # seq_len = data['length'].cuda()
            torch.cuda.empty_cache()
            past_img = data['past'].cuda()
            blur_img = data['blur'].cuda()
            gen_seq = data['gen_seq']
            # print(blur_img.min(), blur_img.max())
            # # for varing length generation
            # step_size = np.random.randint(3, 6)
            # gen_seq = gen_seq[::step_size]
            # gen_seq = gen_seq.cuda()
            if args.mode == "train_image_deblurring":
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                gen_seq = gen_seq[0].cuda()
                if i % args.display_step_freq == 0:
                    print(psnr(blur_img, gen_seq).item(),
                          ssim(blur_img, gen_seq).item())
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode)
            elif args.mode == "train_forcaster_sequence":
                # step_size = np.random.randint(3, 6)
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                # gen_seq = gen_seq[::step_size]
                gen_seq = gen_seq.cuda()
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode)
            elif args.mode == "train_sequence":
                # step_size = np.random.randint(3, 6)
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                # gen_seq = gen_seq[::step_size]
                gen_seq = gen_seq.cuda()
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode)
            elif args.mode == "train_forcaster_image":
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                random_index = np.random.randint(0, len(gen_seq))
                gen_seq = data['gen_seq'][random_index].cuda()
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode, gen_index=random_index, gen_length=len(gen_seq))
            elif args.mode == "train_image_pred":
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                random_index = np.random.randint(0, len(gen_seq))
                gen_seq = data['gen_seq'][random_index].cuda()
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode, gen_index=random_index, gen_length=len(gen_seq))

            # loss and backprop
            if args.mode == "train_image_deblurring":
                loss = model.update_deblurring()
            elif args.mode == "train_forcaster_sequence" or args.mode == "train_forcaster_image":
                loss = model.update_forcaster()
            else:
                loss = model.update_model()
            # past_img = blur_img
            # update tensorboard
            writer.update(model, reconstruction_loss, metric,
                          epoch*len(train_loader) + i, 'train', args.mode)

            # update visualization
            if i % args.display_step_freq == 0:
                print("loss:", loss)

            if i % args.visualize_step_freq == 1:
                if args.visualize:
                    if not os.path.exists(os.path.join(args.visualization_path, "train")):
                        os.makedirs(os.path.join(
                            args.visualization_path, "train"))

                    if not os.path.exists(os.path.join(args.visualization_path, "train", args.mode)):
                        os.makedirs(os.path.join(
                            args.visualization_path, "train", args.mode))

                    if not os.path.exists(os.path.join(args.visualization_path, "train", args.mode, "seq")):
                        os.makedirs(os.path.join(
                            args.visualization_path, "train", args.mode, "seq"))

                    if not os.path.exists(os.path.join(args.visualization_path, "train", args.mode, "blur")):
                        os.makedirs(os.path.join(
                            args.visualization_path, "train", args.mode, "blur"))
                    blur_path = os.path.join(
                        args.visualization_path, "train", args.mode, "blur")
                    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                        std=[1/0.5, 1/0.5, 1/0.5]),
                                                   transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                                        std=[1., 1., 1.]),
                                                   ])
                    blur_img_cpu = invTrans(
                        blur_img[0].squeeze(0).cpu().detach())
                    vis_path = os.path.join(
                        args.visualization_path, "train", args.mode, "seq")
                    # generated_seq[1], generated_seq[0], prior={0: blur_img.detach().cpu()}
                    visualize(generated_seq, path=vis_path, name="epoch_{}_iter_{}_loss_{}_psnr_{}_ssim_{}.png".format(
                        epoch, i, reconstruction_loss, metric[0], metric[1]))

                    torch_utils.save_image(blur_img_cpu, os.path.join(blur_path, "epoch_{}_iter_{}_loss_{}_psnr_{}_ssim_{}.png".format(
                        epoch, i, reconstruction_loss, metric[0], metric[1])))

            if i % args.save_step_freq == 1:
                model.save(args.checkpoint_dir + args.name +
                           '_epoch_' + str(epoch) + '_step_' + str(i) + '.pth')

            # test model
        test(model, args, writer, epoch)

    model.save(args.checkpoint_dir + args.name + "final.pth")
    writer.close()


def test(model, args, writer, step):
    model.eval()
    print("Augmentions used...")
    transform = get_transform(args, 'test')
    print("test  augmentation: ", transform)

    testing_data = Gopro(args, transform, "test")
    test_loader = torch.utils.data.DataLoader(
        testing_data, batch_size=args.testing_parameters['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print("loaded data and dataloader")

    # writer = TensorboardWriter(args, None, model)
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    # total_blur_psnr = 0
    # total_blur_ssim = 0
    for i, data in enumerate(test_loader):
        if (i < step):
            # seq_len = data['length'].cuda()
            torch.cuda.empty_cache()
            past_img = data['past'].cuda()
            blur_img = data['blur'].cuda()
            gen_seq = data['gen_seq']
            # print("gen_seq shape: ", gen_seq.shape)
            # print("blur_img shape: ", blur_img.shape)
            # # for varing length generation
            # step_size = np.random.randint(3, 6)
            #
            # gen_seq = gen_seq[::step_size]
            # gen_seq = gen_seq.cuda()
            if args.mode == "train_image_deblurring":
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                gen_seq = gen_seq[0].cuda()
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode)
            elif args.mode == "train_forcaster_sequence":
                # step_size = np.random.randint(3, 6)
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                # gen_seq = gen_seq[::step_size]
                gen_seq = gen_seq.cuda()
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode)
            elif args.mode == "train_sequence":
                # step_size = np.random.randint(3, 6)
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                # gen_seq = gen_seq[::step_size]
                gen_seq = gen_seq.cuda()
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode)
            elif args.mode == "train_forcaster_image":
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                random_index = np.random.randint(0, len(gen_seq))
                gen_seq = data['gen_seq'][random_index].cuda()
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode, gen_index=random_index, gen_length=len(gen_seq))
            elif args.mode == "train_image_pred":
                gen_seq = gen_seq.permute(1, 0, 2, 3, 4)
                random_index = np.random.randint(0, len(gen_seq))
                gen_seq = data['gen_seq'][random_index].cuda()
                generated_seq, reconstruction_loss, metric = model(
                    gen_seq, past_img, blur_img, args.mode, gen_index=random_index, gen_length=len(gen_seq))

            total_loss += reconstruction_loss
            total_psnr += metric[0]
            total_ssim += metric[1]

            # update tensorboard
            # writer.update(model, reconstruction_loss, metric, i, 'test', args.mode)")
            # update visualization
            if i % args.visualize_step_freq == 0:
                print("visualizing")
                if args.visualize:
                    if not os.path.exists(os.path.join(args.visualization_path, "test")):
                        os.makedirs(os.path.join(
                            args.visualization_path, "test"))

                    if not os.path.exists(os.path.join(args.visualization_path, "test", args.mode)):
                        os.makedirs(os.path.join(
                            args.visualization_path, "test", args.mode))

                    if not os.path.exists(os.path.join(args.visualization_path, "test", args.mode, str(step), "seq")):
                        os.makedirs(os.path.join(args.visualization_path,
                                    "test", args.mode, str(step), "seq"))

                    if not os.path.exists(os.path.join(args.visualization_path, "test", args.mode, str(step), "blur")):
                        os.makedirs(os.path.join(args.visualization_path,
                                    "test", args.mode, str(step), "blur"))

                    vis_path = os.path.join(
                        args.visualization_path, "test", args.mode, str(step), "seq")
                    visualize(generated_seq[1], generated_seq[0], path=vis_path, name="iter_{}_loss_{}_psnr_{}_ssim_{}.png".format(
                        i, reconstruction_loss, metric[0], metric[1]))
                    blur_path = os.path.join(
                        args.visualization_path, "test", args.mode, str(step), "blur")
                    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                        std=[1/0.5, 1/0.5, 1/0.5]),
                                                   transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                                        std=[1., 1., 1.]),
                                                   ])
                    blur_img_cpu = invTrans(
                        blur_img[0].squeeze(0).cpu().detach())
                    torch_utils.save_image(blur_img_cpu, os.path.join(
                        blur_path, "iter_{}_loss_{}_psnr_{}_ssim_{}.png".format(i, reconstruction_loss, metric[0], metric[1])))

        else:
            break

    total_loss = total_loss / step
    total_psnr = total_psnr / step
    total_ssim = total_ssim / step
    writer.update(model, total_loss, [
                  total_psnr, total_ssim], i, 'test', args.mode)
    # writer.close()


def run(args):
    print(args)
    if args.model == 'variational_gen':
        model = Variational_Gen(args)
    elif args.model == 'attention_gen':
        model = Attention_Gen(args)
    elif args.model == 'blur_decoder':
        model = Blur_decoder(args)

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


if __name__ == '__main__':
    # set the seed
    torch.manual_seed(12)
    np.random.seed(12)

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
    # generated run name using date and time

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
    import datetime as dt

    now = dt.datetime.now()

    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    args.name = args.name + '_' + dt_string

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
        NotImplementedError
