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
import torchvision.transforms as transforms


def visualize(gen_seq, path="output", name="output.png"):
    output_gen = []
    output_sharp = []
    output_gen_edge = []
    output_edge_gt = []
    output_blur = []
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1/0.5, 1/0.5, 1/0.5]),
                                   transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                        std=[1., 1., 1.]),
                                   ])

    for keys in gen_seq[0].keys():
        output_gen.append(invTrans(gen_seq[0][keys][0]))
        output_sharp.append(invTrans(gen_seq[1][keys][0]))
        output_gen_edge.append(invTrans(gen_seq[2][keys][0]))
        output_edge_gt.append(invTrans(gen_seq[3][keys][0]))
        # output_blur.append(invTrans(gen_seq[4][keys][0]))

    output_gen = torch_utils.make_grid(
        output_gen, nrow=1, padding=2, normalize=False, range=None, scale_each=False, pad_value=255)
    output_sharp = torch_utils.make_grid(
        output_sharp, nrow=1, padding=2, normalize=False, range=None, scale_each=False, pad_value=255)

    output_gen_edge = torch_utils.make_grid(
        output_gen_edge, nrow=1, padding=2, normalize=False, range=None, scale_each=False, pad_value=255)
    output_edge_gt = torch_utils.make_grid(
        output_edge_gt, nrow=1, padding=2, normalize=False, range=None, scale_each=False, pad_value=255)

    output1 = torch_utils.make_grid(
        [output_gen, output_sharp], nrow=2, padding=2, normalize=False, range=None, scale_each=False, pad_value=255)

    output2 = torch_utils.make_grid(
        [output_gen_edge, output_edge_gt], nrow=2, padding=2, normalize=False, range=None, scale_each=False, pad_value=255)

    output = torch_utils.make_grid(
        [output1, output2], nrow=1, padding=2, normalize=False, range=None, scale_each=False, pad_value=255)
    torch_utils.save_image(output, os.path.join(path, name))
