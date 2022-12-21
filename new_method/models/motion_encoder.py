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


class Corr_Encoder(nn.Module):
    def __init__(self) -> None:
        super(Corr_Encoder, self).__init__()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, features):
        N, D, H, W = features.shape
        print(features.shape)
        list_curr_volumes = []
        for i in range(N):
            current_feature = features[0]
            current_feature = current_feature.squeeze().view(D, -1).transpose(1, 0)
            corr_volume = torch.corrcoef(current_feature)
            corr_volume = self.softmax(corr_volume)
            corr_volume = corr_volume.reshape(H, W, H, W).unsqueeze(0)
            list_curr_volumes.append(corr_volume)

        output = torch.stack(list_curr_volumes)
        print(output.shape)
        # N, H, W, H, W = output.shape

        return output
