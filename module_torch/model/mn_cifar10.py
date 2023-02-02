from __future__ import print_function

from fedem.models import get_mobilenet
from module_torch.hyperparameter import HyperLogitsWeightBase
from module_torch.model.classifier import MultiClassifierBase


class MobileNetCIFAR10(MultiClassifierBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moblienet = get_mobilenet(n_classes=10)

    def forward(self, *args, **kwargs):
        if isinstance(self.hyperparameter_module, HyperLogitsWeightBase):
            return self.hyperparameter_module.weight_outputs(self.moblienet(*args, **kwargs))
        else:
            return self.moblienet(*args, **kwargs)
