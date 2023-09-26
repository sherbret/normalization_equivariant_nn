#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicblocks import *

# Code inspired from https://github.com/cszn/DPIR/blob/master/models/network_dncnn.py

class FDnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=20, blind=False, mode='ordinary'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        """
        super(FDnCNN, self).__init__()

        bias = mode == 'ordinary'
        self.blind = blind
        if not blind: in_nc += 1

        layers = []
        layers.append(conv2d(in_nc, nc, 3, padding=1, bias=bias, blind=blind, mode=mode))
        layers.append(activation(mode))
        for _ in range(nb - 2):
            layers.append(conv2d(nc, nc, 3, padding=1, bias=bias, mode=mode))
            layers.append(activation(mode))
        layers.append(conv2d(nc, out_nc, 3, padding=1, bias=False, mode=mode))
        self.fdncnn = nn.Sequential(*layers)        

    def forward(self, x, sigma=None):
        if not self.blind: # Concatenate noisemap as additional input
            assert sigma is not None
            noisemap = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat((x, noisemap), dim=1)
        return self.fdncnn(x)
