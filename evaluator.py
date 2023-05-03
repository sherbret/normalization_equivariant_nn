from trainer import MetricPSNR
import time
import numpy as np
import csv
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Evaluator:
    def __init__(self, dataset_test, model, blind_denoising=False):
        self.dataset = dataset_test
        self.model = model
        self.blind_denoising = blind_denoising
        self.metric = MetricPSNR()

        self.metric_average = None
        self.metric_average_noisy = None

    def evaluate(self, eval_fname='scores.csv', eval_noisy=False):
        self.model.eval()
        #running_metric = 0
        metric_dict = {}
        metric_dict_noisy = {}

        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        start = time.time()
        for idx, data in enumerate(dataloader):
            if self.blind_denoising:
                img_noisy, img = data
                img_pred = self.model(img_noisy)  # predict
            else:
                img_noisy, img, noisemap = data
                img_pred = self.model(img_noisy, noisemap)  # predict

            # Get eval metric:
            metric_value = self.metric(img_pred, img)

            key = self.dataset.fname_list[idx]  # retrieve file name
            metric_dict[key] = metric_value.item()

            # running_metric += metric_value.item()

            if eval_noisy:
                metric_value_noisy = self.metric(img_noisy, img)
                metric_dict_noisy[key] = metric_value_noisy.item()


        stop = time.time()
        computing_time = stop-start

        #self.metric_average = running_metric / len(dataloader)
        self.metric_average = np.mean(list(metric_dict.values()))

        if eval_noisy:
            self.metric_average_noisy = np.mean(list(metric_dict_noisy.values()))

        print(f'Total prediction time was {np.round(computing_time,2)} sec, i.e. {np.round(computing_time/len(dataloader),2)} sec/image')
        print(f'Average PSNR value: {np.round(self.metric_average, 2)} dB')

        # Save reval:
        if eval_fname:  # save if fname is specified
            with open(eval_fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['fname', 'psnr'])
                for key, value in metric_dict.items():
                    writer.writerow([key, str(value)])
                writer.writerow(['Average', str(self.metric_average)])