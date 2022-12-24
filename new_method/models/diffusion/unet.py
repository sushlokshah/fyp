# code source: https://amaarora.github.io/2020/09/13/unet.html


#1. to decide interpolation type at last layer.
#2. to decide to manage without center crop.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Two contiguous convolutional layers
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


# Encoder using the convlutional blocks and maxpooling
class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)

    def forward(self, x):
        residual = []
        for block in self.enc_blocks:
            x = block(x)
            residual.append(x)
            x = self.pool(x)
        return residual

# Decoder using the convlutional blocks and upsampling
class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.decoder_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_residual):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            # center crop the encoder residual to match the size of x
            # information at encoder_residual edges is removed. Avoid?
            enc_residual = self.crop(encoder_residual[i], x)
            x        = torch.cat([x, enc_residual], dim=1)
            x        = self.decoder_blocks[i](x)
        return x

    def crop(self, enc_residual, x):
        _, _, H, W = x.shape
        enc_residual   = torchvision.transforms.CenterCrop([H, W])(enc_residual)
        return enc_residual

# UNet model
# Encoder decoder with skip connections
class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=6, retain_dim=False, out_sz=(256,256)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)

        # convert final feature map to num_class channels
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)

        # if True, upsample the output to the same size as input
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x, out_sz):
        enc_residual = self.encoder(x)

        # number of residual connections is one less than the number of encoding steps (no residual connection at the last encoding step)
        out      = self.decoder(enc_residual[::-1][0], enc_residual[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            # is it better to use bilinear or nearest neighbor interpolation?
            out = F.interpolate(out, out_sz)
        return out


# test model

if __name__ == '__main__':
    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    out = model(x, (256, 256))
    print(out.shape)