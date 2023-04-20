#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:45:03 2020

@author: sherbret
"""

import torch
import torch.nn as nn

class DnCNN(nn.Module): 
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        padding_mode = "zeros"
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
            layers.append(nn.BatchNorm2d(features, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)


if __name__ == "__main__":
    from PIL import Image, ImageOps
    import numpy as np
    from collections import OrderedDict
    import mat73

    torch.manual_seed(99)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    color = False
    for blind in [False, True]:
        print("Blind", blind)
        m_dncnn = DnCNN(channels=1, num_of_layers=20) if blind else DnCNN(channels=1, num_of_layers=17)
        m_dncnn.to(device)

        # Load trained weights from official implementation: https://github.com/cszn/DnCNN (need to rename the layers)
        layers = mat73.loadmat('./../saved_models/DnCNN_original/' + ('blind' if blind else 'sigma=25') +'.mat')['net']['layers']
        weights = [torch.from_numpy(elt['weights'][i]) for elt in layers if 'weights' in elt for i in range(2)]
        weights.pop()
        weights[0] = weights[0].reshape(3, 3, 1, 64)
        weights[-1] = weights[-1].reshape(3, 3, 64, 1)
        weights = [elt.permute(3, 2, 1, 0) if len(elt.shape) > 2 else elt for elt in weights]
        new_weights = []
        for i, elt in enumerate(weights):
            if len(elt.shape) == 1 and i >=2:
                new_weights.append(torch.ones_like(elt)) # weight
                new_weights.append(elt) # bias
                new_weights.append(torch.zeros_like(elt)) # running_mean
                new_weights.append((1-0.0001) * torch.ones_like(elt)) # running_var
                new_weights.append(torch.zeros(1)) # num_batches_tracked
            else:
                new_weights.append(elt)
        renamed_state_dict = OrderedDict()
        for new_val, new_key in zip(new_weights, m_dncnn.state_dict().keys()):
            renamed_state_dict[new_key] = new_val
        m_dncnn.load_state_dict(renamed_state_dict, strict=True)  
        
        
        # Number of parameters
        model_parameters = filter(lambda p: p.requires_grad, m_dncnn.parameters())
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
        m_dncnn.eval()
        with torch.no_grad():
            img_den = m_dncnn(img_noisy_torch)
        print("PSNR", 10*torch.log10(1 / torch.mean((img_den - img_torch)**2)))


