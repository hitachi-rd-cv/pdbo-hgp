from __future__ import division

import torch
import torch.nn as nn


class CrossEntropyLossNormalized(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.nll_loss = nn.NLLLoss(*args, **kwargs)

    def forward(self, input, target):
        input_log = torch.log(input)
        return self.nll_loss(input_log, target)
