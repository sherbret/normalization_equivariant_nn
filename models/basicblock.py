#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:29:37 2023

@author: sherbret
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='reflect'):
        super(AffineConv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation, 
                                           groups=groups, bias=bias, padding_mode=padding_mode)
        
    def affine_norm(self, w_init):
        w = w_init.view(self.out_channels, -1)
        w_affine = (w.roll(dims=1, shifts=1) - w) + 1 / w.size(1)
        return w_affine.view(w_init.size()).contiguous()
    
    def forward(self, x):
        padding = self.padding[0]
        if padding > 0:
            x = F.pad(x, [padding]*4, mode=self.padding_mode)
        return F.conv2d(x, self.affine_norm(self.weight), bias=self.bias, stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)

class AffineConvTranspose2d(nn.Module):
    """ Affine ConvTranspose2d with kernel=2 and stride=2, implemented using PixelShuffle """
    def __init__(self, in_channels, out_channels):
        super(AffineConvTranspose2d, self).__init__()
        self.conv1x1 = AffineConv2d(in_channels, 4*out_channels, 1, padding=0, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
            
    def forward(self, x):
        return self.pixel_shuffle(self.conv1x1(x))

class SortPool(nn.Module):
    def __init__(self):
        super(SortPool, self).__init__()
        
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).view(N, H*W, C)
        y = torch.empty_like(x)
        y[:, :, ::2] = -F.max_pool1d(-x, 2)
        y[:, :, 1::2] = F.max_pool1d(x, 2) 
        return y.permute(0, 2, 1).view(N, -1, H, W)

