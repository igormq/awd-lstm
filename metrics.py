import torch
from pytorch_lightning.metrics import Metric
import torch.nn.functional as F
import math


class PPL(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def forward(self, x, y=None):
        if y is None:
            return torch.exp(torch.as_tensor(x))
        return F.cross_entropy(x,
                               y,
                               weight=None,
                               ignore_index=-100,
                               reduce=None,
                               reduction='mean').exp()


class BPC(TensorMetric):
    def __init__(self):
        super().__init__(name='bpc')

    def forward(self, x, y=None):
        if y is None:
            return torch.as_tensor(x) / math.log(2)
        return F.cross_entropy(x,
                               y,
                               weight=None,
                               ignore_index=-100,
                               reduce=None,
                               reduction='mean') / math.log(2)
