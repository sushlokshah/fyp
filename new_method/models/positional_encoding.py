import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch.utils.data

# transformer's positional encoding


class Positional_encoding(nn.Module):
    def __init__(self, d_model):
        super(Positional_encoding, self).__init__()
        self.d_model = d_model

    def forward(self, last_time_stamp, current_time_stamp, max_time_stamp, batch_size):
        # current_time_stamp: (batch_size, 1)
        # max_time_stamp: (batch_size, 1)
        # output: (batch_size, d_model)
        total_time_left = torch.zeros(batch_size, 1).fill_(
            1 - current_time_stamp/max_time_stamp)
        time_of_generation = torch.zeros(batch_size, 1).fill_(
            (current_time_stamp-last_time_stamp)/max_time_stamp)
        current_embedding = torch.zeros(batch_size, self.d_model-2)
        for i in range(self.d_model-2):
            if i % 2 == 0:
                current_embedding[:, i] = torch.sin(torch.tensor(
                    (current_time_stamp/max_time_stamp)/(2**i/(self.d_model-2))))
            if i % 2 == 1:
                current_embedding[:, i] = torch.cos(torch.tensor(
                    (current_time_stamp/max_time_stamp)/(2**(i-1)/(self.d_model-2))))
        return torch.cat((current_embedding, total_time_left, time_of_generation), dim=1)
