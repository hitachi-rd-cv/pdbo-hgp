import torch
import torch.nn.functional as F
from torch import nn

from constants import NamesEvalMetric
from module_torch.hyperparameter import HyperSoftmaxWeightsBase


class ClassifierBase(nn.Module):
    criterion = None

    def __init__(self, weight_decay=1e-2, hyperparameters=None):
        super().__init__()
        self.weight_decay = weight_decay
        self.hyperparameter_module = hyperparameters

    def loss(self, inputs, *args, **kwargs):
        if isinstance(self.hyperparameter_module, HyperSoftmaxWeightsBase):
            losses = self.bare_losses(*inputs)
            losses_weighted = self.hyperparameter_module.weight_losses(losses, **kwargs)
            return torch.mean(losses_weighted) + self.reg_loss()
        else:
            return self.bare_loss(*inputs) + self.reg_loss()

    def reg_loss(self):
        loss = 0.
        for param in self.parameters(recurse=True):
            loss += 0.5 * torch.sum((self.weight_decay * param ** 2))
        return loss


class MultiClassifierBase(ClassifierBase):
    def bare_loss(self, x, y):
        device = next(self.parameters()).device
        x, y = x.to(device), y.to(device)
        return F.cross_entropy(self.forward(x), y)

    def bare_losses(self, x, y):
        device = next(self.parameters()).device
        x, y = x.to(device), y.to(device)
        return F.cross_entropy(self.forward(x), y, reduction='none')

    def _eval_metric(self, metric, loader):
        device = next(self.parameters()).device
        if metric == NamesEvalMetric.LOSS_BARE_MEAN:
            loss_mean = 0.
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                loss_mean += F.cross_entropy(self.forward(x), y, reduction='sum') / len(loader.dataset)

            return loss_mean

        elif metric == NamesEvalMetric.LOSS_MEAN:
            loss_mean = 0.
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                loss_mean += F.cross_entropy(self.forward(x), y, reduction='sum') / len(loader.dataset)

            loss_mean += self.reg_loss()

            return loss_mean

        elif metric == NamesEvalMetric.ACCURACY:
            correct = 0
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                output = self.forward(x)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum()
            acc = 100. * correct / len(loader.dataset)

            return acc

        else:
            raise ValueError(metric)

    def eval_metric_sum_from_x_y(self, metric, x, y):
        device = next(self.parameters()).device
        if metric == NamesEvalMetric.LOSS_BARE_MEAN:
            x, y = x.to(device), y.to(device)
            return F.cross_entropy(self.forward(x), y, reduction='sum')

        else:
            raise ValueError(metric)


class BinaryClassifierBase(ClassifierBase):
    def bare_loss(self, x, y):
        device = next(self.parameters()).device
        x, y = x.to(device), y.type(torch.float32).to(device)
        return F.binary_cross_entropy(self.forward(x), y)

    def bare_losses(self, x, y):
        device = next(self.parameters()).device
        x, y = x.to(device), y.type(torch.float32).to(device)
        return F.binary_cross_entropy(self.forward(x), y, reduction='none')

    def _eval_metric(self, metric, loader):
        device = next(self.parameters()).device
        if metric == NamesEvalMetric.LOSS_BARE_MEAN:
            loss_mean = 0.
            for x, y, _ in loader:
                x, y = x.to(device), y.type(torch.float32).to(device)
                loss_mean += F.binary_cross_entropy(self.forward(x), y, reduction='sum') / len(loader.dataset)

            return loss_mean

        elif metric == NamesEvalMetric.LOSS_MEAN:
            loss_mean = 0.
            for x, y, _ in loader:
                x, y = x.to(device), y.type(torch.float32).to(device)
                loss_mean += F.binary_cross_entropy(self.forward(x), y, reduction='sum') / len(loader.dataset)

            loss_mean += self.reg_loss()

            return loss_mean

        elif metric == NamesEvalMetric.ACCURACY:
            correct = 0
            for x, y, _ in loader:
                x, y = x.to(device), y.type(torch.float32).to(device)
                output = self.forward(x)
                pred = output > .5
                true = y > 0.5
                correct += (pred == true).sum()
            acc = 100. * correct / len(loader.dataset)

            return acc

        else:
            raise ValueError(metric)

    def eval_metric_sum_from_x_y(self, metric, x, y):
        device = next(self.parameters()).device
        if metric == NamesEvalMetric.LOSS_BARE_MEAN:
            x, y = x.to(device), y.type(torch.float32).to(device)
            return F.binary_cross_entropy(self.forward(x), y, reduction='sum')

        else:
            raise ValueError(metric)
