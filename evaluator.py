from trainer import MetricPSNR
import time
import numpy as np
import csv
from torch.utils.data import DataLoader

class Evaluator:
    def __init__(self, dataset_test, model, eval_fname='scores.csv'):
        self.dataset = dataset_test
        self.model = model
        self.metric = MetricPSNR()

        self.metric_average = None

    def evaluate(self, eval_fname='scores.csv'):
        self.model.eval()
        running_metric = 0
        metric_dict = {}

        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        start = time.time()
        for idx, data in enumerate(dataloader):
            img_noisy, img = data

            img_pred = self.model(img_noisy)  # predict

            # Get eval metric:
            metric_value = self.metric(img_pred, img)
            running_metric += metric_value.item()

            key = self.dataset.fname_list[idx]  # retrieve file name
            metric_dict[key] = metric_value.item()


        stop = time.time()
        computing_time = stop-start
        self.metric_average = running_metric / len(dataloader)

        print(f'Total prediction time was {np.round(computing_time,2)} sec, i.e. {np.round(computing_time/len(dataloader),2)} sec/image')
        print(f'Average PSNR value: {self.metric_average} dB')

        # Save reval:
        if eval_fname:  # save if fname is specified
            with open(eval_fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['fname', 'psnr'])
                for key, value in metric_dict.items():
                    writer.writerow([key, str(value)])
                writer.writerow(['Average', str(self.metric_average)])