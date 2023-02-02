from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn

from constants import NamesEvalMetric
from module_torch.hyperparameter import HyperLearnersWeightsAndSoftmaxCategoryWeights, HyperLearnersWeights, HyperLearnersWeightsAndLogitsWeightsBase
from module_torch.model.classifier import MultiClassifierBase
from module_torch.model.lstm_shakespeare import NextCharacterLSTM


class MultiLearnersBase(nn.Module):
    def __init__(self,
                 n_learners,
                 name_learner_model,
                 kwargs_learner,
                 hyperparameters=None,
                 ):
        super().__init__()
        self.hyperparameter_module = hyperparameters
        from module_torch.model import D_MODELS  # to avoid recursion

        Learner = D_MODELS[name_learner_model]
        assert issubclass(Learner, (MultiClassifierBase, NextCharacterLSTM)), name_learner_model

        learners = []
        for learner_id in range(n_learners):
            learners.append(Learner(**kwargs_learner))

        self.learners = nn.ModuleList(learners)

    def reg_loss(self):
        loss_sum = 0.
        for learner in self.learners:
            loss_sum += learner.reg_loss()
        return loss_sum

    def forward(self, x):
        y_pred = 0.
        if self.hyperparameter_module is None:
            for learner in self.learners:
                y_pred += F.softmax(learner.forward(x), dim=1)

        elif isinstance(self.hyperparameter_module, (HyperLearnersWeights, HyperLearnersWeightsAndSoftmaxCategoryWeights)):
            for learner, weight in zip(self.learners, self.hyperparameter_module.get_weights()):
                y_pred += weight * F.softmax(learner.forward(x), dim=1)

        elif isinstance(self.hyperparameter_module, HyperLearnersWeightsAndLogitsWeightsBase):
            for idx, (learner, weight) in enumerate(zip(self.learners, self.hyperparameter_module.get_weights())):
                if isinstance(learner, MultiClassifierBase):
                    dim = None
                elif isinstance(learner, NextCharacterLSTM):
                    dim = 1
                else:
                    raise ValueError(self.learner)
                logits = self.hyperparameter_module.weight_outputs(learner.forward(x), idx_learner=idx, dim=dim)
                y_pred += weight * F.softmax(logits, dim=1)

        else:
            raise ValueError(self.hyperparameter_module)

        return y_pred


class MultiLearners(MultiLearnersBase):
    def _eval_metric(self, metric, loader):
        device = next(self.learners[0].parameters()).device
        if metric == NamesEvalMetric.LOSS_BARE_MEAN:
            loss_mean = 0.
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                y_pred = self.forward(x)
                loss_mean += F.nll_loss(torch.log(y_pred), y, reduction='sum') / len(loader.dataset)

            return loss_mean

        elif metric == NamesEvalMetric.LOSS_MEAN:
            loss_mean = 0.
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                y_pred = self.forward(x)
                loss_mean += F.nll_loss(torch.log(y_pred), y, reduction='sum') / len(loader.dataset)

            loss_mean += self.reg_loss()

            return loss_mean

        elif metric == NamesEvalMetric.ACCURACY:
            correct = 0
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                y_pred = self.forward(x)
                pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum()
            acc = 100. * correct / len(loader.dataset)

            return acc

        else:
            raise ValueError(metric)

    def eval_metric_sum_from_x_y(self, metric, x, y):
        device = next(self.learners[0].parameters()).device
        if metric == NamesEvalMetric.LOSS_BARE_MEAN:
            x, y = x.to(device), y.to(device)
            y_pred = self.forward(x)
            return F.nll_loss(torch.log(y_pred), y, reduction='sum')

        else:
            raise ValueError(metric)

    def bare_loss(self, x, y):
        device = next(self.learners[0].parameters()).device
        x, y = x.to(device), y.to(device)
        y_pred = self.forward(x)
        return F.nll_loss(torch.log(y_pred), y)

    def bare_losses(self, x, y):
        device = next(self.learners[0].parameters()).device
        x, y = x.to(device), y.to(device)
        y_pred = self.forward(x)
        return F.nll_loss(torch.log(y_pred), y, reduction="none")

    def loss(self, inputs, **kwargs):
        if isinstance(self.hyperparameter_module, HyperLearnersWeightsAndSoftmaxCategoryWeights):
            losses = self.bare_losses(*inputs)
            losses_weighted = self.hyperparameter_module.weight_losses(losses, **kwargs)
            bare_loss = torch.mean(losses_weighted)
        else:
            bare_loss = self.bare_loss(*inputs)
        reg_loss = self.reg_loss()
        return bare_loss + reg_loss


class MultiLanguageLearners(MultiLearnersBase):
    def loss(self, inputs, ys=None, **kwargs):
        if isinstance(self.hyperparameter_module, HyperLearnersWeightsAndSoftmaxCategoryWeights):
            losses = self.bare_losses(*inputs).flatten()
            ys = ys.flatten()
            bare_loss = self.hyperparameter_module.weight_losses(losses, ys=ys).mean()
        else:
            bare_loss = self.bare_loss(*inputs)
        reg_loss = self.reg_loss()
        return bare_loss + reg_loss

    def bare_loss(self, x, y):
        device = next(self.learners[0].parameters()).device
        x, y = x.to(device), y.to(device)
        y_pred = self.forward(x)
        return F.nll_loss(torch.log(y_pred), y)

    def bare_losses(self, x, y):
        device = next(self.learners[0].parameters()).device
        x, y = x.to(device), y.to(device)
        y_pred = self.forward(x)
        return F.nll_loss(torch.log(y_pred), y, reduction="none")

    def _eval_metric(self, metric, loader):
        device = next(self.learners[0].parameters()).device
        if metric == NamesEvalMetric.LOSS_BARE_MEAN:
            loss_mean = 0.
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                y_pred = self.forward(x)
                loss_mean += F.nll_loss(torch.log(y_pred), y, reduction='sum') / (len(loader.dataset) * y.shape[1])

            return loss_mean

        elif metric == NamesEvalMetric.LOSS_MEAN:
            loss_mean = 0.
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                y_pred = self.forward(x)
                loss_mean += F.nll_loss(torch.log(y_pred), y, reduction='sum') / (len(loader.dataset) * y.shape[1])

            loss_mean += self.reg_loss()

            return loss_mean

        elif metric == NamesEvalMetric.ACCURACY:
            correct = 0
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                output = self.forward(x)
                pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
                correct += pred.eq(y).sum() / pred.shape[1]
            acc = 100. * correct / len(loader.dataset)

            return acc

        else:
            raise ValueError(metric)

    def eval_metric_sum_from_x_y(self, metric, x, y):
        device = next(self.learners[0].parameters()).device
        if metric == NamesEvalMetric.LOSS_BARE_MEAN:
            x, y = x.to(device), y.to(device)
            y_pred = self.forward(x)
            return F.nll_loss(torch.log(y_pred), y, reduction='sum') / y.shape[1]

        else:
            raise ValueError(metric)
