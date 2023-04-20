#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:45:03 2020

@author: sherbret
"""

import torch
import torch.nn as nn

# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
class FDnCNN(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20):
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
        layers = []
        layers.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(nb - 2):
            layers.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
        self.fdncnn = nn.Sequential(*layers)

    def forward(self, x, sigma):
        # Concatenate noisemap as additional input
        noisemap = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x = torch.cat((x, noisemap), dim=1)
        return self.fdncnn(x)

if __name__ == "__main__":
    from PIL import Image, ImageOps
    import numpy as np
    from collections import OrderedDict
    import mat73

    torch.manual_seed(99)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for color in [False, True]:
        print("Color:", color)
        m_fdncnn = FDnCNN(in_nc=4 if color else 2, out_nc=3 if color else 1, nc=64, nb=20)
        m_fdncnn.to(device)

        # Load trained weights from official implementation: https://github.com/cszn/DnCNN (need to rename the layers)
        layers = mat73.loadmat('./../saved_models/FDnCNN_original/FDnCNN_' + ('color' if color else 'gray') + '.mat')['net']['layers']
        weights = [torch.from_numpy(elt['weights'][i]) for elt in layers if ('weights' in elt and len(elt['weights']) > 0) for i in range(2)]
        weights.pop()
        weights[-1] = weights[-1].reshape(3, 3, 64, -1)
        weights = [elt.permute(3, 2, 1, 0) if len(elt.shape) > 2 else elt for elt in weights]
        renamed_state_dict = OrderedDict()
        for new_val, new_key in zip(weights, m_fdncnn.state_dict().keys()):
            renamed_state_dict[new_key] = new_val
        m_fdncnn.load_state_dict(renamed_state_dict, strict=True)  
        
        # Number of parameters
        model_parameters = filter(lambda p: p.requires_grad, m_fdncnn.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters", params)
        
        # Load image
        img_name = "./../datasets/datasets_testing/BSD68/101087.jpg"
        im = Image.open(img_name)
        if not color:
            im = ImageOps.grayscale(im)              
        img_torch = torch.from_numpy(np.array(im)).float().to(device) / 255
        img_torch = img_torch.permute(2, 0, 1)[None, ...] if color else img_torch[None, None, ...]

        # Noise image and denoise
        sigma = 25 / 255
        img_noisy_torch = img_torch + sigma * torch.randn_like(img_torch)
        with torch.no_grad():
            img_den = m_fdncnn(img_noisy_torch, sigma)
        print("PSNR", 10*torch.log10(1 / torch.mean((img_den - img_torch)**2)))


