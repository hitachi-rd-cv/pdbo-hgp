from __future__ import print_function

import torch
from torch import nn as nn

from module_torch.model.classifier import BinaryClassifierBase


class LogisticRegressionToy(BinaryClassifierBase):
    def __init__(self, *args, ndim_x, **kwargs):
        super().__init__(*args, **kwargs)
        self.ndim_x = ndim_x
        self.fc = nn.Linear(ndim_x, 1)

    def forward(self, x):
        x = torch.squeeze(self.fc(x), dim=1)
        output = torch.sigmoid(x)
        return output
