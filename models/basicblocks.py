#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, blind=True):
        super(AffineConv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation, 
                                           groups=groups, bias=False)
        self.blind = blind
        self.reflect_padding = nn.ReflectionPad2d(padding)
        
    def affine(self, w):
        """ returns new kernels that encode affine combinations """
        return w.view(self.out_channels, -1).roll(1, 1).view(w.size()) - w + 1 / w[0, ...].numel()
    
    def forward(self, x):
        kernel = self.affine(self.weight) if self.blind else torch.cat((self.affine(self.weight[:, :-1, :, :]), self.weight[:, -1:, :, :]), dim=1)
        return F.conv2d(self.reflect_padding(x), kernel, stride=self.stride, dilation=self.dilation, groups=self.groups)


class AffineConvTranspose2d(nn.Module):
    """ Affine ConvTranspose2d with kernel=2 and stride=2, implemented using PixelShuffle """
    def __init__(self, in_channels, out_channels):
        super(AffineConvTranspose2d, self).__init__()
        self.conv1x1 = AffineConv2d(in_channels, 4*out_channels, 1)
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
        super(ResBlock_NE, self).__init__()
        self.m_res = nn.Sequential(AffineConv2d(in_channels, in_channels, 3, stride=1, padding=1),
                                SortPool(),
                                AffineConv2d(in_channels, out_channels, 3, stride=1, padding=1))
        self.alpha = nn.Parameter(0.9 * torch.ones(1))
        
    def forward(self, x):
        return self.alpha * x + (1 - self.alpha) * self.m_res(x)
