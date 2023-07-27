#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_img(img_path, color=False):
    im = Image.open(img_path)
    if not color:
        im = ImageOps.grayscale(im)              
    img = np.array(im) / 255.0
    if not color:
        img_torch = torch.from_numpy(img).view(1, 1, *img.shape).float()
    else:
        img_torch = torch.from_numpy(img).view(1, *img.shape).permute(0, 3, 1, 2).float()
    return img_torch

def noise_img(img, sigma):
    return img + sigma * torch.randn_like(img)

def psnr(img_true, img_noisy):
    """ assumes that images are in the interval [0, 1] """
    mse = F.mse_loss(img_true.clip(0,1), img_noisy.clip(0,1))
    return float(-10*torch.log10(mse))

def show_img(img_list, titles=None, color=False):
    x_list = []
    with torch.no_grad():
        for img in img_list:
            x_list.append(img[0, 0, ...].numpy() if not color else img[0, ...].permute(1, 2, 0).numpy())

    fig = plt.figure(figsize=(20, 5))
    rows, columns = 1, len(x_list) # setting values to rows and column variables

    for j in range(columns):
        fig.add_subplot(rows, columns, j+1)
        if not color:
            plt.imshow(x_list[j], cmap='gray')
        else:
            # Define colormap for adative filters
            cmap = LinearSegmentedColormap.from_list('my_gradient', \
            ((0.000, (0.722, 0.000, 0.722)), (0.500, (1.000, 1.000, 1.000)), (1.000, (0.118, 0.565, 1.000))))
            maxi = np.max(np.abs(x_list[j]))
            plt.imshow(x_list[j], cmap=cmap, vmin=-maxi, vmax=maxi)
            #plt.colorbar()
        if titles is not None:
            plt.title(titles[j])
    plt.show()

def compute_adaptive_filter(model, img_noise, i, j):
    img_noisy = img_noise.clone()
    model.requires_grad_(False)
    img_noisy.requires_grad_(True)
    den = model(img_noisy)
    den[0, 0, i, j].backward()
    return img_noisy.grad