import sys
sys.path.append('../../')

from dataloader import DenoisingDataset

data = DenoisingDataset(
    path_list=[
        '/net/serpico-fs2/emoebel/denoising/paper/data_and_models/datasets/datasets_training/BSD400',
        '/net/serpico-fs2/emoebel/denoising/paper/data_and_models/datasets/datasets_training/DIV2K',
        '/net/serpico-fs2/emoebel/denoising/paper/data_and_models/datasets/datasets_training/Flickr2K',
        '/net/serpico-fs2/emoebel/denoising/paper/data_and_models/datasets/datasets_training/WaterlooExplorationDatabase',
    ],
    transform=None,
    gray=True,
    sigma_low=1/255,
    sigma_high=55/255,
)