#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append("./models")
sys.path.append("./utilities")
from fdncnn import *
from drunet import *
from utilities import *

from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from os import listdir
from os.path import isfile

import argparse
parser = argparse.ArgumentParser()

# MODEL
parser.add_argument("--model_name", type=str, dest="model_name", help="Architecture of the model, either fdncnn or drunet.", default="fdncnn")
parser.add_argument("--blind", action='store_true', help="Use a blind model.")
parser.add_argument("--mode", type=str, dest="mode", help="Either ordinary, scale-equiv or norm-equiv.", default="norm-equiv")

# TRAINING SETTINGS
parser.add_argument("--sigma_min", type=float, dest="sigma_min", help="Model is trained for all noise levels between sigma_min and sigma_max.", default=15)
parser.add_argument("--sigma_max", type=float, dest="sigma_max", help="Model is trained for all noise levels between sigma_min and sigma_max.", default=15)
parser.add_argument("--lr", type=float, dest="lr", help="Adam learning rate.", default=1e-4)
parser.add_argument("--num_iterations", type=int, dest="num_iterations", default=900000)
parser.add_argument("--halve_lr_every", type=int, dest="halve_lr_every", default=1800000) # learning rate is kept constant by default
parser.add_argument("--batch_size", type=int, dest="batch_size", default=128)
parser.add_argument("--patch_size", type=int, dest="patch_size", default=70)
parser.add_argument("--loss_function", type=str, dest="loss_function", help="Either mse or l1 loss.", default="mse")
parser.add_argument("--seed", type=int, dest="seed", help="Seed for reproductibility.", default=1234)
parser.add_argument("--save_every", type=int, dest="save_every", default=1000)

# DATA
parser.add_argument("--in_folder", type=str, dest="in_folder", help="Path to the folder containing the images for training.")
parser.add_argument("--out_folder", type=str, dest="out_folder", help="Path to the folder where models are saved.")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_images(in_folder):
    files = [in_folder + "/" + f for f in listdir(in_folder) if isfile(in_folder + "/" + f) and f[:1] != "." and (f[-3:] == "jpg" or f[-3:] == "png" or f[-3:] == "bmp")]
    return [np.array(ImageOps.grayscale(Image.open(elt))).astype(np.uint8) for elt in files]

def augmentation(x, k=0, inverse=False):
    k = k % 8
    if inverse: k = [0, 1, 6, 3, 4, 5, 2, 7][k]
    if k % 2 == 1: x = torch.flip(x, dims=[2])
    return torch.rot90(x, k=(k//2) % 4, dims=[1,2])
    
class TrainingDataset(Dataset):
    def __init__(self, in_folder, sigma_min=15, sigma_max=15, patch_size=70, batch_size=128):
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.patch_size = patch_size
        self.batch_size = batch_size

        self.images_train = load_images(in_folder)
        self.number_of_images = len(self.images_train)
        
    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        sigma = np.random.uniform(low=self.sigma_min/255, high=self.sigma_max/255)

        img_np = self.images_train[np.random.choice(self.number_of_images)]
        h, w = img_np.shape
        i, j = np.random.choice(h - self.patch_size - 1), np.random.choice(w - self.patch_size - 1)
        patch = img_np[i:i+self.patch_size, j:j+self.patch_size]
        
        patch = patch.astype(np.float32) / 255.0
        patch_noisy = patch + sigma * np.random.randn(*patch.shape)

        img_torch = torch.from_numpy(patch).view(1, *patch.shape).float().to(device)
        img_noisy_torch = torch.from_numpy(patch_noisy).view(1, *patch.shape).float().to(device)
        
        k = np.random.randint(8)
        img_torch = augmentation(img_torch, k)
        img_noisy_torch = augmentation(img_noisy_torch, k)
        sigma_torch = sigma * torch.ones(1, 1, 1, device=device)

        return img_torch, img_noisy_torch, sigma_torch
    
### DATASET, MODEL, OPTIMIZER
dataset = TrainingDataset(in_folder=args.in_folder, sigma_min=args.sigma_min, sigma_max=args.sigma_max, patch_size=args.patch_size, batch_size=args.batch_size)
torch.manual_seed(args.seed) # for reproductibility (in particular if the model is loaded several times)
np.random.seed(args.seed)

model = FDnCNN(blind=args.blind, mode=args.mode).to(device) if args.model_name == "fdncnn" else DRUNet(blind=args.blind, mode=args.mode).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=args.halve_lr_every, gamma=0.5)
loss_function = nn.MSELoss() if args.loss_function == "mse" else nn.L1Loss()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of parameters of the model:", params)

### TRAINING LOOP
model.train()
for i in range(1, args.num_iterations+1):
    
    print("Iteration number", i, "/", args.num_iterations)
    
    img_torch, img_noisy_torch, sigma_torch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True)))
    optimizer.zero_grad()
    den = model(img_noisy_torch, sigma=sigma_torch if not args.blind else None)
    loss = loss_function(img_torch, den) 
    loss.backward()
    optimizer.step()
    scheduler.step()

    if i % args.save_every == 0:  
        torch.save(model.state_dict(), args.out_folder + "/" + args.model_name + "_iter" + str(i) + ".pth")
        torch.save(optimizer.state_dict(), args.out_folder + "/adam_" + args.model_name + "_iter" + str(i) + ".pth")
