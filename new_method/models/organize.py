from motion_encoder import Corr_Encoder             # Blur/motion encoder
from encoder import encoder                         # Clear frame encoder
from positional_encoding import Positional_encoding # Position encoder

from unet import UNet                               #model



import torch
import torch.nn as nn

class Organize_encodings_unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Organize_encodings_unet, self).__init__()
        self.corr_encoder = Corr_Encoder()
        self.encoder = encoder()
        self.positional_encoder = Positional_encoding(self.args.model["positional"]['output_channels'])
        self.unet = UNet(in_channels, out_channels)

    def forward(self, x):
