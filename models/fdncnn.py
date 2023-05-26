#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:45:03 2020

@author: sherbret
"""

import torch
import torch.nn as nn
from basicblocks import *

# Code inspired from https://github.com/cszn/DPIR/blob/master/models/network_dncnn.py

# Original FDnCNN (it is scale-equivariant if bias is False)
class FDnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=20, bias=True, blind_denoising=True):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        """
        super(FDnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        padding_mode = "zeros"

        if not blind_denoising:  # if denoising is not blind, add 1 channel for noisemap
            in_nc += 1

        layers = []
        layers.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(nb - 2):
            layers.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=bias))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
        self.fdncnn = nn.Sequential(*layers)

    def forward(self, x, noisemap=None):
        # Concatenate noisemap as additional input (useless for blind denoising)
        if noisemap is not None:
            x = torch.cat((x, noisemap), dim=1)
        return self.fdncnn(x)


# Normalization-equivariant FDnCNN
class FDnCNN_NE(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=20, blind_denoising=True):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        """
        super(FDnCNN_NE, self).__init__()
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