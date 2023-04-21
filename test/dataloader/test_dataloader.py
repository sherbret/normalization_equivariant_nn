import sys
sys.path.append('../../')

from dataloader import DenoisingDataset, RandomRot90

import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

patch_size = 128

transform_pipeline = T.Compose([
    T.ToTensor(),
    T.RandomCrop(size=(patch_size, patch_size)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    RandomRot90(),
])
        
data = DenoisingDataset(
    path_list=['/Users/emoebel/serpico-fs2/denoising/paper/data_and_models/datasets/datasets_testing/Set12'],
    transform=transform_pipeline,
    gray=True,
    sigma_low=1/255,
    sigma_high=55/255,
    patches_per_img=1,
)

# Plot:
def plot_augments(idx_list):
    n = 6
    for idx in idx_list:
        fig, axes = plt.subplots(2,n)
        for pidx in range(n):
            img_torch_noisy, img_torch = data[idx]
            
            img_noisy = np.squeeze(img_torch_noisy.numpy())
            img = np.squeeze(img_torch.numpy())
            
            axes[0,pidx].imshow(img, cmap='gray')
            axes[1,pidx].imshow(img_noisy, cmap='gray')
        
        fig.set_size_inches(20,10)
        fig.savefig(f'img{idx}.png', bbox_inches='tight')
        plt.close(fig)
        
plot_augments(idx_list=[0,1,2])
             