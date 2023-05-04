#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:45:03 2020

@author: sherbret
"""

import torch
import torch.nn as nn
from basicblock import *

# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
class FDnCNN_NI(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=20, blind_denoising=True):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        """
        super(FDnCNN_NI, self).__init__()
        kernel_size = 3
        padding = 1
        padding_mode = "reflect"

        if not blind_denoising:  # if denoising is not blind, add 1 channel for noisemap
            in_nc += 1

        layers = []
        layers.append(AffineConv2d(in_channels=in_nc, out_channels=nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
        layers.append(SortPool())
        for _ in range(nb - 2):
            layers.append(AffineConv2d(in_channels=nc, out_channels=nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
            layers.append(SortPool())
        layers.append(AffineConv2d(in_channels=nc, out_channels=out_nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
        self.fdncnn = nn.Sequential(*layers)

    def forward(self, x, noisemap=None):
        # Concatenate noisemap as additional input (useless for blind denoising)
        if noisemap is not None:
            x = torch.cat((x, noisemap), dim=1)
        return self.fdncnn(x)


class FDnCNN_NI_old(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=20):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        """
        super(FDnCNN_NI, self).__init__()
        kernel_size = 3
        padding = 1
        padding_mode = "reflect"
        layers = []
        layers.append(AffineConv2d(in_channels=in_nc, out_channels=nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
        layers.append(SortPool())
        for _ in range(nb - 2):
            layers.append(AffineConv2d(in_channels=nc, out_channels=nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
            layers.append(SortPool())
        layers.append(AffineConv2d(in_channels=nc, out_channels=out_nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
        self.fdncnn = nn.Sequential(*layers)

    def forward(self, x, sigma=None):
        # Concatenate noisemap as additional input
        if sigma is not None:
            noisemap = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat((x, noisemap), dim=1)
        return self.fdncnn(x)

if __name__ == "__main__":
    from PIL import Image, ImageOps
    import numpy as np
    from collections import OrderedDict
    import mat73

    torch.manual_seed(99)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for color in [False, True]:
        print("Color:", color)
        m_fdncnn = FDnCNN_NI(in_nc=3 if color else 1, out_nc=3 if color else 1, nc=64, nb=20)
        m_fdncnn.to(device)
    
        # Number of parameters
        model_parameters = filter(lambda p: p.requires_grad, m_fdncnn.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters", params)
        
        # Load image
        img_name = "./../datasets/datasets_testing/BSD68/101087.jpg"
        im = Image.open(img_name)
        if not color:
            im = ImageOps.grayscale(im)              
        img_torch = torch.from_numpy(np.array(im)).float().to(device) / 255
        img_torch = img_torch.permute(2, 0, 1)[None, ...] if color else img_torch[None, None, ...]


        # For incresed precision at inference -> double precision
        img_torch = img_torch.double()
        m_fdncnn.double() 

        # Noise image and denoise
        sigma = 25 / 255
        img_noisy_torch = img_torch + sigma * torch.randn_like(img_torch)
        with torch.no_grad():
            img_den1 = m_fdncnn(img_noisy_torch)
            print("PSNR", 10*torch.log10(1 / torch.mean((img_den1 - img_torch)**2)))
            img_den2 = m_fdncnn(img_noisy_torch * 255) / 255
            print("PSNR", 10*torch.log10(1 / torch.mean((img_den2 - img_torch)**2)))
            img_den3 = m_fdncnn(img_noisy_torch + 100) - 100
            print("PSNR", 10*torch.log10(1 / torch.mean((img_den3 - img_torch)**2)))


