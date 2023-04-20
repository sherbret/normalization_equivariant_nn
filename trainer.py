import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
import os
import time


class MetricPSNR(nn.Module):
    def __init__(self):
        super(MetricPSNR, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        return -10 * torch.log10(self.mse_loss(output.clip(0, 1), target.clip(0, 1)))


class DenoisingTrainer():
    def __init__(self, dataloader_train, dataloader_valid, model, epochs, learning_rate, loss_function, scheduler=None):
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.model = model
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.metric = MetricPSNR()

        self.current_epoch = 0
        self.save_every_n_epochs = 20

        self.path_model_weights = 'model_weights/'

    def fit(self):
        if not os.path.isdir(self.path_model_weights): os.mkdir(self.path_model_weights)
        # self.validation_step()  # to have initial valid_loss of model
        start = time.time()
        for e in range(self.epochs):
            print(f'Epoch {e + 1}/{self.epochs}:')
            self.current_epoch = e
            # Training
            self.training_step()
            # Validation
            self.validation_step()

            if e % self.save_every_n_epochs == 0:
                torch.save(self.model, os.path.join(self.path_model_weights, f'model_weights_epoch{e}.p'))

        torch.save(self.model, os.path.join(self.path_model_weights, f'model_weights_final.p'))

        stop = time.time()
        print(f'END - Total training time: {np.round(stop - start, 2)} seconds')

    def training_step(self):
        self.model.train()
        running_loss = 0
        running_metric = 0
        start = time.time()
        for idx, data in enumerate(self.dataloader_train):
            img_noisy, img = data

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            img_pred = self.model(img_noisy)
            loss = self.loss_function(img_pred, img)
            loss.backward()
            self.optimizer.step()

            # Get values to log:
            with torch.no_grad():
                running_loss += loss.item()

                # Get train metric:
                metric_value = self.metric(img_pred, img)
                running_metric += metric_value.item()

                wandb.log({'loss_train': loss.item()})
                wandb.log({'metric_train': metric_value.item()})

        stop = time.time()

        if self.scheduler:
            self.scheduler.step()

        loss_train_average = running_loss / len(self.dataloader_train)
        metric_train_average = running_metric / len(self.dataloader_train)
        print(
            f'- Train: loss={np.round(loss_train_average, 6)}, psnr={np.round(metric_train_average, 2)} dB - {np.round(stop - start, 2)} sec')

    def validation_step(self):
        self.model.eval()
        running_loss = 0
        running_metric = 0
        with torch.no_grad():
            start = time.time()
            for idx, data in enumerate(self.dataloader_valid):
                img_noisy, img = data

                img_pred = self.model(img_noisy)

                loss = self.loss_function(img_pred, img)
                running_loss += loss.item()

                if idx == 0:
                    log_img = img[0]
                    log_img_noisy = img_noisy[0]
                    log_img_pred = img_pred[0]

                # Get valid metric:
                metric_value = self.metric(img_pred, img)
                running_metric += metric_value.item()

            stop = time.time()

            loss_valid_average = running_loss / len(self.dataloader_valid)
            metric_valid_average = running_metric / len(self.dataloader_valid)

            wandb.log({'loss_valid': loss_valid_average})
            wandb.log({'metric_valid': metric_valid_average})

            if self.current_epoch % self.save_every_n_epochs == 0:
                images = wandb.Image(
                    torch.cat([log_img_pred, log_img_noisy], dim=-2),
                    caption='Top: prediction, Bottom: Noisy'
                )
                wandb.log({'prediction_sample': images})

            print(
                f'- Valid: loss={np.round(loss_valid_average, 6)}, psnr={np.round(metric_valid_average, 2)} dB - {np.round(stop - start, 2)} sec')
