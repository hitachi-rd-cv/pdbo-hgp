from __future__ import print_function

from torch import nn as nn
from torch.nn import functional as F

from module_torch.hyperparameter import HyperLogitsWeightBase
from module_torch.model.classifier import MultiClassifierBase


class CNNMNIST(MultiClassifierBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 20, 5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 20, 5, stride=1, padding=0)
        self.fc = nn.Linear(4 * 4 * 20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 20)
        x = self.fc(x)
        if isinstance(self.hyperparameter_module, HyperLogitsWeightBase):
            return self.hyperparameter_module.weight_outputs(x)
        else:
            return x

    def flatten(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 20)
        return x
