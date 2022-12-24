from motion_encoder import Corr_Encoder             # Blur/motion encoder
from encoder import encoder                         # Clear frame encoder
from positional_encoding import Positional_encoding # Position encoder

from unet import UNet                               #model



import torch
import torch.nn as nn

class Organize_encodings_unet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Organize_encodings_unet, self).__init__()
        self.bilinear = bilinear
        self.corr_encoder = Corr_Encoder()
        self.encoder = encoder()
        self.positional_encoder = Positional_encoding(256)
        self.unet = UNet(in_channels, out_channels, bilinear)

    def forward(self, x, last_time_stamp, current_time_stamp, max_time_stamp, batch_size):
        