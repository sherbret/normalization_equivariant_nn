#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:26:14 2021

@author: sherbret
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicblocks import *

# Code inspired from https://github.com/cszn/DPIR/blob/master/models/network_unet.py

# Original DRUNet (it is scale-equivariant if bias is False)
class DRUnet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, bias=False, blind_denoising=True):
        super(DRUnet, self).__init__()

        if not blind_denoising:  # if denoising is not blind, add 1 channel for noisemap
            in_nc += 1

        self.m_head = nn.Conv2d(in_nc, nc[0], 3, stride=1, padding=1, bias=bias)
        
        self.m_down = nn.ModuleList([nn.Sequential(
            *[ResBlock(nc[i], nc[i], bias) for _ in range(nb)],
            nn.Conv2d(nc[i], nc[i+1], kernel_size=2, stride=2, padding=0, bias=bias))
            for i in range(len(nc)-1)])

        self.m_body = nn.Sequential(*[ResBlock(nc[-1], nc[-1], bias) for _ in range(nb)])

        self.m_up = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(nc[i], nc[i-1], kernel_size=2, stride=2, padding=0, bias=bias),
            *[ResBlock(nc[i-1], nc[i-1], bias) for _ in range(nb)])
            for i in range(len(nc)-1, 0, -1)])

        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, stride=1, padding=1, bias=bias)

    def forward(self, x, noisemap=None):
        _, _, h, w = x.size()
        scale = len(self.m_down)
        d = 2**scale
        # Size handling (h and w must divisible by d)
        r1, r2 = h % d, w % d
        x = F.pad(x, pad=(0, d-r2 if r2 > 0 else 0, 0, d-r1 if r1 > 0 else 0), mode='reflect')

        # Concatenate noisemap as additional input (useless for blind denoising)
        if noisemap is not None:
            #noisemap = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            noisemap = F.pad(noisemap, pad=(0, d - r2 if r2 > 0 else 0, 0, d - r1 if r1 > 0 else 0), mode='reflect')
            x = torch.cat((x, noisemap), dim=1)

        layers = [self.m_head(x)]
        for i in range(scale): layers.append(self.m_down[i](layers[-1]))
        x = self.m_body(layers[-1])
        for i in range(scale): x = self.m_up[i](x + layers[-(1+i)])
        x = self.m_tail(x + layers[0])
        
        return x[..., :h, :w]


# Normalization-equivariant DRUNet
class DRUnet_NE(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, blind_denoising=True):
        super(DRUnet_NE, self).__init__()

        if not blind_denoising:  # if denoising is not blind, add 1 channel for noisemap
            in_nc += 1
            noisemap = True
        else:
            noisemap = False

        self.m_head = AffineConv2d(in_nc, nc[0], 3, stride=1, padding=1, bias=False, noisemap=noisemap)

        self.m_down = nn.ModuleList([nn.Sequential(
            *[ResBlock_NE(nc[i], nc[i]) for _ in range(nb)],
            AffineConv2d(nc[i], nc[i + 1], kernel_size=2, stride=2, padding=0, bias=False))
            for i in range(len(nc) - 1)])

        self.m_body = nn.Sequential(*[ResBlock_NE(nc[-1], nc[-1]) for _ in range(nb)])

        self.m_up = nn.ModuleList([nn.Sequential(
            AffineConvTranspose2d(nc[i], nc[i - 1]),
            *[ResBlock_NE(nc[i - 1], nc[i - 1]) for _ in range(nb)])
            for i in range(len(nc) - 1, 0, -1)])

        self.m_tail = AffineConv2d(nc[0], out_nc, 3, stride=1, padding=1, bias=False)

        self.alpha = nn.Parameter(0.9 * torch.ones(len(nc)))

    def forward(self, x, noisemap=None):
        _, _, h, w = x.size()
        scale = len(self.m_down)
        d = 2 ** scale
        # Size handling (h and w must divisible by d)
        r1, r2 = h % d, w % d
        x = F.pad(x, pad=(0, d - r2 if r2 > 0 else 0, 0, d - r1 if r1 > 0 else 0), mode='reflect')

        # Concatenate noisemap as additional input (useless for blind denoising)
        if noisemap is not None:
            # noisemap = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            noisemap = F.pad(noisemap, pad=(0, d - r2 if r2 > 0 else 0, 0, d - r1 if r1 > 0 else 0), mode='reflect')
            x = torch.cat((x, noisemap), dim=1)

        layers = [self.m_head(x)]
        for i in range(scale): layers.append(self.m_down[i](layers[-1]))
        x = self.m_body(layers[-1])
        for i in range(scale): x = self.m_up[i](self.alpha[i] * x + (1 - self.alpha[i]) * layers[-(1 + i)])
        x = self.m_tail(self.alpha[-1] * x + (1 - self.alpha[-1]) * layers[0])

        return x[..., :h, :w]
