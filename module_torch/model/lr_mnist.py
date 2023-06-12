from __future__ import print_function

import torch
from torch import nn as nn

from module_torch.model.classifier import BinaryClassifierBase


class LogisticRegressionMNIST(BinaryClassifierBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(784, 1)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = torch.squeeze(self.fc(x), dim=1)
        output = torch.sigmoid(x)
        return output
