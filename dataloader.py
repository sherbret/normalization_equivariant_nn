import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as T

import os


def get_fname_list_from_dir(dir_path, ext_list):
    # Get all file names in directory:
    dir_fnames = os.listdir(dir_path)
    fname_list = []
    for fname in dir_fnames:
        file_ext = os.path.splitext(fname)[1]
        file_ext = file_ext[1:]  # remove '.'
        if fname[0] != '.' and file_ext in ext_list:
            fname_list.append(fname)
    fname_list.sort()

    return fname_list


class DenoisingDataset(Dataset):
    def __init__(self, path_list, transform, gray, sigma_low, sigma_high, patches_per_img=1):
        if transform is None:  # minimal transform: convert to torch tensor
            self.transform = T.Compose([T.ToTensor(), ])
        else:
            self.transform = transform
        self.patches_per_img = patches_per_img
        self.gray = gray
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high

        self.img_list = []
        self.fname_list = []
        self.load_images(path_list)

    def load_images(self, path_list):
        for path in path_list:
            self.fname_list = get_fname_list_from_dir(dir_path=path, ext_list=['jpg', 'png', 'bmp'])
            for fname in self.fname_list:
                path_img = os.path.join(path, fname)
                img_pil = Image.open(path_img)

                if self.gray:
                    img_pil = ImageOps.grayscale(img_pil)

                img_np = np.array(img_pil, dtype=np.float32) / 255
                self.img_list.append(img_np)

    def __len__(self):
        return len(self.img_list) * self.patches_per_img

    def __getitem__(self, idx):
        idx_img, idx_patch = np.unravel_index(idx, (len(self.img_list), self.patches_per_img))

        img_np = self.img_list[idx_img]
        img_torch = self.transform(img_np)

        sigma = np.random.uniform(low=self.sigma_low, high=self.sigma_high)
        img_torch_noisy = img_torch + sigma * torch.randn(*img_torch.size())

        # sigma = np.random.uniform(low=self.sigma_low, high=self.sigma_high)
        # img_noisy = img + sigma * np.random.randn(*img.shape)

        # Transfer Data to GPU if available:
        if torch.cuda.is_available():
            img_torch_noisy, img_torch = img_torch_noisy.cuda(), img_torch.cuda()

        return img_torch_noisy, img_torch

# class DenoisingDatasetOld(Dataset):
#
#     def __init__(self, path_list, transform, gray, sigma_low, sigma_high):
#         if transform == None:  # minimal transform is convert to torch tensor
#             self.transform = T.Compose([T.ToTensor(), ])
#         else:
#             self.transform = transform
#         self.gray = gray
#         self.sigma_low = sigma_low
#         self.sigma_high = sigma_high
#
#         self.img_list = []
#         self.fname_list = []
#         self.load_images(path_list)
#
#     def load_images(self, path_list):
#         for path in path_list:
#             self.fname_list = get_fname_list_from_dir(dir_path=path, ext_list=['jpg', 'png', 'bpm'])
#             for fname in self.fname_list:
#                 path_img = os.path.join(path, fname)
#                 img_pil = Image.open(path_img)
#
#                 if self.gray:
#                     img_pil = ImageOps.grayscale(img_pil)
#
#                 img_np = np.array(img_pil, dtype=np.float32) / 255
#                 self.img_list.append(img_np)
#
#     def __len__(self):
#         return len(self.img_list)
#
#     def __getitem__(self, idx):
#         img_np = self.img_list[idx]
#         img_torch = self.transform(img_np)
#
#         sigma = np.random.uniform(low=self.sigma_low, high=self.sigma_high)
#         img_torch_noisy = img_torch + sigma * torch.randn(*img_torch.size())
#
#         # sigma = np.random.uniform(low=self.sigma_low, high=self.sigma_high)
#         # img_noisy = img + sigma * np.random.randn(*img.shape)
#
#         # Transfer Data to GPU if available:
#         if torch.cuda.is_available():
#             img_torch_noisy, img_torch = img_torch_noisy.cuda(), img_torch.cuda()
#
#         return img_torch_noisy, img_torch


class RandomRot90(object):
    def __call__(self, tensor):
        k = np.random.choice([0, 1, 2, 3])
        return torch.rot90(tensor, k=k, dims=[-2, -1])
