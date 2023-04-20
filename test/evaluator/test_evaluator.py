import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('../../')
from dataloader import DenoisingDataset
from evaluator import Evaluator

sys.path.append('/net/serpico-fs2/emoebel/github/sdeep')
from sdeep.models import UNet
from sdeep.models.unet import UNetConvBlock, UNetEncoderBlock, UNetDecoderBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get data:
data_test = DenoisingDataset(
    path_list=['/net/serpico-fs2/emoebel/denoising/paper/data_and_models/datasets/datasets_testing/Set12'],
    transform=None,
    gray=True,
    sigma_low=25/255,
    sigma_high=25/255,
)
loader_test = DataLoader(data_test, batch_size=1, shuffle=False)

# Initialize model:
# model = UNet(
#     n_channels_in=1,
#     n_channels_out=1,
#     n_feature_first=32,
#     use_batch_norm=True,
# )
model = torch.load('/net/serpico-fs2/emoebel/denoising/airbus/care/attempt3_brkl_sdeep/train_seb/logs/care49.p')
model.to(device)

# Initialize evaluator:
eval = Evaluator(
    dataloader_test=loader_test,
    model = model,
)

# Eval:
eval.evaluate()