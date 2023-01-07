import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class KLCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        """KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2))"""
        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum()

class SmoothMSE(nn.Module):
    def __init__(self, opt=None, threshold=0.001):
        super().__init__()
        self.opt = opt
        self.threshold = threshold

    def forward(self, x1, x2):
        _, c, h, w = x1.shape
        mse = ((x1 - x2) ** 2).clamp(min=self.threshold)
        return mse.mean()

class PSNR(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt

    def forward(self, x1, x2):
        mse = (x1 - x2) ** 2
        mse_mean = mse.mean()
        return 20 * torch.log10(1/ mse_mean)



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def image_gradient(img):
    # sobel filter
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3)
    
    # convert image to grayscale
    gray = img.mean(dim=1, keepdim=True)  
    
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    
    return grad_x, grad_y
    
        


if __name__ == "__main__":
    x1 = Variable(torch.randn(1, 3, 256, 256), requires_grad=True)
    # x2 = Variable(torch.randn(1, 3, 256, 256), requires_grad=True)
    x2 = x1.clone()
    criterion = PSNR()
    y = criterion(x1, x2)
    print(y)
    y.backward()
