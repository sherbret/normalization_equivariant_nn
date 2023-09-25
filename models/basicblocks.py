#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode="reflect", blind=True):
        super(AffineConv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation, 
                                           groups=groups, bias=False)
        
        if padding_mode=="reflect":
            self.apply_padding = nn.ReflectionPad2d(padding)
        elif padding_mode=="replicate":
            self.apply_padding = nn.ReplicationPad2d(padding)
        elif padding_mode=="zeros":
            self.apply_padding = nn.ZeroPad2d(padding)
        else:
            raise NotImplementedError("Only padding_mode zero, reflect and replicate are implemented")

        self.blind = blind
        
    def affine(self, w):
        """ returns new kernels that encode affine combinations """
        return w.view(self.out_channels, -1).roll(1, 1).view(w.size()) - w + 1 / w[0, ...].numel()
    
    def forward(self, x):
        kernel = self.affine(self.weight) if self.blind else torch.cat((self.affine(self.weight[:, :-1, :, :]), self.weight[:, -1:, :, :]), dim=1)
        return F.conv2d(self.apply_padding(x), kernel, stride=self.stride, dilation=self.dilation, groups=self.groups)

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
        x1, x2 = torch.split(x.view(N, C//2, 2, H, W), 1, dim=2)
        diff = F.relu(x1 - x2, inplace=True)
        return torch.cat((x1-diff, x2+diff), dim=2).view(x.size())

class ResidualConnection(nn.Module):
    """ Residual connection """
    def __init__(self, mode='ordinary'):
        super(ResidualConnection, self).__init__()

        self.mode = mode
        if mode=='norm-equiv':
            self.alpha = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, x, y):
        if self.mode=='norm-equiv': 
            return self.alpha * x + (1 - self.alpha) * y
        return x + y

def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', blind=True, mode='ordinary'):
    if mode=='ordinary' or mode=='scale-equiv':
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias if mode=='ordinary' else False, padding_mode=padding_mode)
    elif mode=='norm-equiv':
        return AffineConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode='reflect', blind=blind)
    else:
        raise NotImplementedError("Only ordinary, scale-equiv and norm-equiv modes are implemented")

def upscale2(in_channels, out_channels, bias=True, mode='ordinary'):
    """ Upscaling using convtranspose with kernel 2x2 and stride 2"""
    if mode=='ordinary' or mode=='scale-equiv':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=bias if mode=='ordinary' else False)
    elif mode=='norm-equiv':
        return AffineConvTranspose2d(in_channels, out_channels)
    else:
        raise NotImplementedError("Only ordinary, scale-equiv and norm-equiv modes are implemented")

def activation(mode='ordinary'):
    if mode=='ordinary' or mode=='scale-equiv':
        return nn.ReLU(inplace=True)
    elif mode=='norm-equiv':
        return SortPool()
    else:
        raise NotImplementedError("Only ordinary, scale-equiv and norm-equiv modes are implemented")

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=False, mode="ordinary"):
        super(ResBlock, self).__init__()

        self.m_res = nn.Sequential(conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias, mode=mode),
                                activation(mode),
                                conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=bias, mode=mode))

        self.sum = ResidualConnection(mode)
        
    def forward(self, x):
        return self.sum(x, self.m_res(x))
