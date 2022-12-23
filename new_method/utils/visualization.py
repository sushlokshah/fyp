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
import torch.utils as torch_utils


def visualize(posterior, gt, prior = None, path = "output.png"):
    post_seq = []
    for i in range(len(posterior)):
        post_seq.append(posterior[i][0])
    post_seq = torch_utils.make_grid(posterior[0], nrow=1, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
    gt_seq = torch_utils.make_grid(gt[0], nrow=1, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
    
    if prior!=None:
        prior_seq = torch_utils.make_grid(prior[0], nrow=1, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
        output = torch_utils.make_grid([gt_seq,post_seq,prior_seq], nrow=3, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)

    else:
        output = torch_utils.make_grid([gt_seq,post_seq], nrow=2, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)

    torch_utils.save_image(output, path)
