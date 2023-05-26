#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:29:37 2023

@author: sherbret
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineConv2d_alt(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='reflect', noisemap=False):
        super(AffineConv2d_alt, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation,
                                           groups=groups, bias=bias, padding_mode=padding_mode)
        self.noisemap = noisemap

    def affine_norm(self, w_init):
        w_affine = w_init.clone()
        w_affine[:, -1:] = 1 - torch.sum(w_init[:, :-1], dim=1, keepdim=True)
        return w_affine.view(w_init.size()).contiguous()

    def forward(self, x):
        padding = self.padding[0]
        if padding > 0:
            x = F.pad(x, [padding] * 4, mode=self.padding_mode)
        if self.noisemap:
            kernel = torch.cat((self.affine_norm(self.weight[:, :-1, :, :]), self.weight[:, -1:, :, :]), dim=1)
        else:
            kernel = self.affine_norm(self.weight)
        return F.conv2d(x, kernel, bias=self.bias, stride=self.stride, padding=0, dilation=self.dilation,
                        groups=self.groups)


class AffineConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='reflect', noisemap=False):
        super(AffineConv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation, 
                                           groups=groups, bias=bias, padding_mode=padding_mode)
        self.noisemap = noisemap
        
    def affine_norm(self, w_init):
        w = w_init.view(self.out_channels, -1)
        w_affine = (w.roll(dims=1, shifts=1) - w) + 1 / w.size(1)
        return w_affine.view(w_init.size()).contiguous()
    
    def forward(self, x):
        padding = self.padding[0]
        if padding > 0:
            x = F.pad(x, [padding]*4, mode=self.padding_mode)
        if self.noisemap:
            kernel = torch.cat((self.affine_norm(self.weight[:, :-1, :, :]), self.weight[:, -1:, :, :]), dim=1)
        else:
            kernel = self.affine_norm(self.weight)
        return F.conv2d(x, kernel, bias=self.bias, stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)


class AffineConvTranspose2d(nn.Module):
    """ Affine ConvTranspose2d with kernel=2 and stride=2, implemented using PixelShuffle """
    def __init__(self, in_channels, out_channels):
        super(AffineConvTranspose2d, self).__init__()
        self.conv1x1 = AffineConv2d(in_channels, 4*out_channels, 1, padding=0, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
            
    def forward(self, x):
        return self.pixel_shuffle(self.conv1x1(x))


class SortPool(nn.Module):
    """ Channel-wise sort pooling, C must be an even number """
    def __init__(self):
        super(SortPool, self).__init__()
        
    def forward(self, x):
        N, C, H, W = x.size()
        with torch.no_grad():
            sign = (x[:, ::2, :, :] - x[:, 1::2, :, :]) > 0
            index = torch.arange(0, C, 2, device=x.device).repeat_interleave(2).view(1, C, 1, 1).repeat(N, 1, H, W)
            index[:, ::2, :, :] += sign
            index[:, 1::2, :, :] += torch.logical_not(sign)
        return torch.gather(x, dim=1, index=index)


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=False):
        super(ResBlock, self).__init__()
        self.m_res = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=bias))

    def forward(self, x):
        return x + self.m_res(x)


class ResBlock_NE(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(ResBlock_NI, self).__init__()
        self.m_res = nn.Sequential(AffineConv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
                                   SortPool(),
                                   AffineConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
        self.alpha = nn.Parameter(0.9 * torch.ones(1))

    def forward(self, x):
        return self.alpha * x + (1 - self.alpha) * self.m_res(x)