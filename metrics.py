from pytorch_lightning.metrics import TensorMetric
import torch.nn.functional as F
import math


class PPL(TensorMetric):
    def __init__(self):
        super().__init__(name='ppl')

    def forward(self, x, y=None):
        if y is None:
            return x.exp()
        return F.cross_entropy(x, y, weight=None, ignore_index=-100, reduce=None, reduction='mean').exp()


class BPC(TensorMetric):
    def __init__(self):
        super().__init__(name='bpc')

    def forward(self, x, y=None):
        if y is None:
            return x / math.log(2)
        return F.cross_entropy(x, y, weight=None, ignore_index=-100, reduce=None, reduction='mean') / math.log(2)
