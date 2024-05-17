import torch
from torchmetrics import Metric, CatMetric


class ClassificationMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preds = CatMetric()
        self.targets = CatMetric().set_dtype(torch.int64)

    def reset(self):
        self.preds.reset()
        self.targets.reset()

    def update(self, preds, target):
        self.preds.update(preds)
        self.targets.update(target)

    def compute(self):
        return {'preds': self.preds.compute(), 'targets': self.targets.compute()}
