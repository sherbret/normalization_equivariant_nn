#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicblocks import *

# Code inspired from https://github.com/cszn/DPIR/blob/master/models/network_dncnn.py

# Original FDnCNN (it is scale-equivariant iif bias is False)
class FDnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=20, bias=True, blind=False):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        """
        super(FDnCNN, self).__init__()

        self.blind = blind
        if not blind: in_nc += 1

        layers = []
        layers.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, padding=1, padding_mode='zeros', bias=bias))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(nb - 2):
            layers.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, padding=1, padding_mode='zeros', bias=bias))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, padding=1, padding_mode='zeros', bias=False))
        self.fdncnn = nn.Sequential(*layers)        

    def forward(self, x, sigma=None):
        if not self.blind: # Concatenate noisemap as additional input
            assert sigma is not None
            noisemap = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat((x, noisemap), dim=1)
        return self.fdncnn(x)


# Normalization-equivariant FDnCNN
class FDnCNN_NE(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=20, blind=False):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        """
        super(FDnCNN_NE, self).__init__()

        self.blind = blind
        if not blind: in_nc += 1

        layers = []
        layers.append(AffineConv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, padding=1, blind=blind))
        layers.append(SortPool())
        for _ in range(nb - 2):
            layers.append(AffineConv2d(in_channels=nc, out_channels=nc, kernel_size=3, padding=1))
            layers.append(SortPool())
        layers.append(AffineConv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, padding=1))
        self.fdncnn = nn.Sequential(*layers)

    def forward(self, x, sigma=None):
        if not self.blind: # Concatenate noisemap as additional input
            assert sigma is not None
            noisemap = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat((x, noisemap), dim=1)
        return self.fdncnn(x)
