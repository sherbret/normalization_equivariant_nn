#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicblocks import *

# Code inspired from https://github.com/cszn/DPIR/blob/master/models/network_unet.py

class DRUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, bias=False, blind=False, mode="ordinary"):
        super(DRUNet, self).__init__()

        self.blind = blind
        if not blind: in_nc += 1

        self.m_head = conv2d(in_nc, nc[0], 3, stride=1, padding=1, bias=bias, blind=blind, mode=mode)
        
        self.m_down = nn.ModuleList([nn.Sequential(
            *[ResBlock(nc[i], nc[i], bias=bias, mode=mode) for _ in range(nb)],
            conv2d(nc[i], nc[i+1], 2, stride=2, padding=0, bias=bias, mode=mode))
            for i in range(len(nc)-1)])

        self.m_body = nn.Sequential(*[ResBlock(nc[-1], nc[-1], bias=bias, mode=mode) for _ in range(nb)])

        self.m_up = nn.ModuleList([nn.Sequential(
            upscale2(nc[i], nc[i-1], bias=bias, mode=mode),
            *[ResBlock(nc[i-1], nc[i-1], bias=bias, mode=mode) for _ in range(nb)])
            for i in range(len(nc)-1, 0, -1)])

        self.m_tail = conv2d(nc[0], out_nc, 3, stride=1, padding=1, bias=bias, mode=mode)

        self.res = nn.ModuleList([ResidualConnection(mode) for _ in range(len(nc))])
        

    def forward(self, x, sigma=None):
        # Size handling (h and w must divisible by d)
        _, _, h, w = x.size()
        scale = len(self.m_down)
        d = 2**scale
        r1, r2 = h % d, w % d
        x = F.pad(x, pad=(0, d-r2 if r2 > 0 else 0, 0, d-r1 if r1 > 0 else 0), mode='reflect') 
        
        if not self.blind: # Concatenate noisemap as additional input
            assert sigma is not None
            noisemap = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat((x, noisemap), dim=1)

        layers = [self.m_head(x)]
        for i in range(scale): layers.append(self.m_down[i](layers[-1]))
        x = self.m_body(layers[-1])
        for i in range(scale): x = self.m_up[i](self.res[i](x, layers[-(1+i)]))
        x = self.m_tail(self.res[-1](x, layers[0]))
        
        return x[..., :h, :w]