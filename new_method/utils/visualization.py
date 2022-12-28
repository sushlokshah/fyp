import torch
import configparser
import argparse
from functools import partial
from tqdm import tqdm
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


def visualize(posterior, gt, prior=None, path="output.png"):
    post_seq = []
    prior_seq = []
    gt_seq = []

    for keys in posterior.keys():
        post_seq.append(posterior[keys][0])
        if prior != None:
            prior_seq.append(prior[keys][0])
        gt_seq.append(gt[keys][0])
    post_seq = torch_utils.make_grid(
        post_seq, nrow=len(posterior), padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
    gt_seq = torch_utils.make_grid(
        gt_seq, nrow=len(posterior), padding=0, normalize=False, range=None, scale_each=False, pad_value=0)

    if prior != None:
        prior_seq = torch_utils.make_grid(
            prior_seq, nrow=len(prior), padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
        output = torch_utils.make_grid([gt_seq, post_seq, prior_seq], nrow=3,
                                       padding=0, normalize=False, range=None, scale_each=False, pad_value=0)

    else:
        output = torch_utils.make_grid(
            [gt_seq, post_seq], nrow=len(posterior), padding=0, normalize=False, range=None, scale_each=False, pad_value=0)

    torch_utils.save_image(output, path)
