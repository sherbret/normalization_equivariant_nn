#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:26:14 2021

@author: sherbret
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicblock import *

class ResBlock_NI(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(ResBlock_NI, self).__init__()
        self.m_res = nn.Sequential(AffineConv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
                                SortPool2(),
                                AffineConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
        self.alpha = nn.Parameter(0.9 * torch.ones(1))
        
    def forward(self, x):
        return self.alpha * x + (1 - self.alpha) * self.m_res(x)

class DRUnet_NI(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4):
        super(DRUnet_NI, self).__init__()

        self.m_head = AffineConv2d(in_nc, nc[0], 3, stride=1, padding=1, bias=False)
        
        self.m_down = nn.ModuleList([nn.Sequential(
            *[ResBlock_NI(nc[i], nc[i]) for _ in range(nb)],
            AffineConv2d(nc[i], nc[i+1], kernel_size=2, stride=2, padding=0, bias=False))
            for i in range(len(nc)-1)])

        self.m_body = nn.Sequential(*[ResBlock_NI(nc[-1], nc[-1]) for _ in range(nb)])

        self.m_up = nn.ModuleList([nn.Sequential(
            AffineConvTranspose2d(nc[i], nc[i-1]),
            *[ResBlock_NI(nc[i-1], nc[i-1]) for _ in range(nb)])
            for i in range(len(nc)-1, 0, -1)])

        self.m_tail = AffineConv2d(nc[0], out_nc, 3, stride=1, padding=1, bias=False)

        self.alpha = nn.Parameter(0.9 * torch.ones(len(nc)))

    def forward(self, x, sigma=None):
        _, _, h, w = x.size()
        scale = len(self.m_down)
        d = 2**scale
        # Size handling (h and w must divisible by d)
        r1, r2 = h % d, w % d
        x = F.pad(x, pad=(0, d-r2 if r2 > 0 else 0, 0, d-r1 if r1 > 0 else 0), mode='reflect') 
   
        # Concatenate noisemap as additional input (useless for blind denoising)
        if sigma is not None:
            noisemap = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat((x, noisemap), dim=1)

        layers = [self.m_head(x)]
        for i in range(scale): layers.append(self.m_down[i](layers[-1]))
        x = self.m_body(layers[-1])
        for i in range(scale): x = self.m_up[i](self.alpha[i] * x + (1 - self.alpha[i]) * layers[-(1+i)])
        x = self.m_tail(self.alpha[-1] * x + (1 - self.alpha[-1]) * layers[0])
    
        return x[..., :h, :w]

if __name__ == "__main__":
    from PIL import Image, ImageOps
    import numpy as np
    from collections import OrderedDict

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for color in [False, True]:
        print("Color:", color)
        n_channels = 1 if not color else 3
        m_drunet = DRUnet_NI(in_nc=n_channels, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4)
        
        # Number of parameters
        model_parameters = filter(lambda p: p.requires_grad, m_drunet.parameters())
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
        img_torch = img_torch[:, :, :100, :101].double()
        m_drunet.double() 

        # Noise image and denoise
        sigma = 25 / 255
        img_noisy_torch = img_torch + sigma * torch.randn_like(img_torch)
        
        with torch.no_grad():
            img_den1 = m_drunet(img_noisy_torch)
            print("PSNR", 10*torch.log10(1 / torch.mean((img_den1 - img_torch)**2)))
            img_den2 = m_drunet(img_noisy_torch * 255) / 255
            print("PSNR", 10*torch.log10(1 / torch.mean((img_den2 - img_torch)**2)))
            img_den3 = m_drunet(img_noisy_torch + 100) - 100
            print("PSNR", 10*torch.log10(1 / torch.mean((img_den3 - img_torch)**2)))
        
        
        