from __future__ import print_function

import torch.nn as nn

from module_torch.model.classifier import MultiClassifierBase


class FCMNIST(MultiClassifierBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        output = self.fc(x)
        return output
