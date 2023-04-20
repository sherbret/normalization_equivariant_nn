from trainer import MetricPSNR
import time
import numpy as np

class Evaluator:
    def __init__(self, dataloader_test, model):
        self.dataloader = dataloader_test
        self.model = model
        self.metric = MetricPSNR()

    def evaluate(self):
        self.model.eval()
        running_metric = 0

        start = time.time()
        for idx, data in enumerate(self.dataloader):
            img_noisy, img = data

            img_pred = self.model(img_noisy)

            # Get eval metric:
            metric_value = self.metric(img_pred, img)
            running_metric += metric_value.item()

        stop = time.time()
        computing_time = stop-start
        metric_average = running_metric / len(self.dataloader)


        print(f'Total prediction time was {np.round(computing_time,2)} sec, i.e. {np.round(computing_time/len(self.dataloader),2)} sec/image')
        print(f'Average PSNR value: {metric_average} dB')