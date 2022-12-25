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


def Feature_temporal_encoding(features,time_encoding, directional_information = None):
    # stack features and directional information 
    # features: (batch_size, num_features, H, W)
    # time_encoding: (batch_size, d_model)
    # directional_information: (batch_size, num_directions, H, W)
    
    if directional_information is not None:
        features_new = torch.cat((features, directional_information), dim=1)
        # cat time encoding to each feature
        features_new = torch.cat((features_new, time_encoding.unsqueeze(2).unsqueeze(3).repeat(1,1,features_new.shape[2],features_new.shape[3])), dim=1)
    
    else:
        features_new = torch.cat((features, time_encoding.unsqueeze(2).unsqueeze(3).repeat(1,1,features.shape[2],features.shape[3])), dim=1)
    
    return features_new
    
    
