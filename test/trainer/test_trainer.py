import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import wandb

import sys
sys.path.append('../../')
from dataloader import DenoisingDataset, RandomRot90
from trainer import DenoisingTrainer

sys.path.append('/net/serpico-fs2/emoebel/github/sdeep')
from sdeep.models import UNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.init(project='test_train_routine_wCARE_newDataloader')

wandb.config.update({
    'patch_size': 96,
    'batch_size': 20,
    'epochs': 100,
    'learning_rate': 3e-4,
})

# Prepare dataset:
transform_pipeline = T.Compose([
    T.ToTensor(),
    T.RandomCrop(size=(wandb.config['patch_size'], wandb.config['patch_size'])),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    RandomRot90(),
])

print('Training set transform pipeline:')
print(transform_pipeline)

data_train = DenoisingDataset(
    path_list=['/net/serpico-fs2/emoebel/denoising/paper/data_and_models/datasets/datasets_training/BSD400'],
    transform=transform_pipeline,
    patches_per_img=800,
    gray=True,
    sigma_low=1/255,
    sigma_high=55/255,
)
    
data_valid = DenoisingDataset(
    path_list=['/net/serpico-fs2/emoebel/denoising/paper/data_and_models/datasets/datasets_testing/Set12'],
    transform=None,
    gray=True,
    sigma_low=25/255,
    sigma_high=25/255,
)

loader_train = DataLoader(data_train, batch_size=wandb.config['batch_size'], shuffle=True, drop_last=True)
loader_valid = DataLoader(data_valid, batch_size=1, shuffle=False)

# Initialize model:
model = UNet(
    n_channels_in=1,
    n_channels_out=1,
    n_feature_first=32,
    use_batch_norm=True,
)
model.to(device)

# Initialize trainer:
trainer = DenoisingTrainer(
    dataloader_train=loader_train,
    dataloader_valid=loader_valid,
    model=model,
    epochs=wandb.config['epochs'],
    learning_rate=wandb.config['learning_rate'],
    loss_function=nn.MSELoss(),
)
trainer.path_model_weights = 'model_weights_wPatchPerImg'

# Launch trainer:
trainer.fit()
