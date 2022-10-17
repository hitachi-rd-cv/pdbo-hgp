from __future__ import print_function

from torch import nn as nn
from torch.nn import functional as F

from module_torch.model.classifier import MultiClassifierBase


class CNNEMNIST(MultiClassifierBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # we referred an architecture in https://nextjournal.com/gkoehler/pytorch-mnist
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
