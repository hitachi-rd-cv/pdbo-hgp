from abc import ABCMeta
from abc import abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

from constants import NamesHyperParam, NamesHyperLoss


def normalized_sigmoid(inputs):
    return F.sigmoid(inputs) * 2 / len(inputs)


class HyperParamBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def init_hyperparameters(self):
        raise NotImplementedError

    @abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError


class HyperLogitsWeightBase(HyperParamBase):
    def __init__(self, size, hyper_loss=None, gamma=0., **kwargs):
        self.hyperparameters = nn.ParameterList([nn.Parameter(torch.empty(size))])
        self.hyper_loss = hyper_loss
        self.gamma = gamma

    def init_hyperparameters(self):
        with torch.no_grad():
            self.hyperparameters[0].copy_(torch.zeros_like(self.hyperparameters[0]))

    def loss(self, *args, **kwargs):
        if self.hyper_loss is None:
            return torch.tensor(0., device=self.hyperparameters[0].device)
        elif self.hyper_loss == NamesHyperLoss.L2_REGULARIZER:
            norm = 0.5 * (self.hyperparameters[0] @ self.hyperparameters[0])
            return self.gamma * norm
        else:
            raise ValueError(self.hyper_loss)

    def weight_outputs(self, outputs, **kwargs):
        raise NotImplementedError


class HyperExpLogitsWeights(HyperLogitsWeightBase):
    def __init__(self, n_classes, **kwargs):
        super().__init__(n_classes, **kwargs)
        self.n_classes = n_classes

    def weight_outputs(self, outputs, axis=None, **kwargs):
        assert axis is None
        weights_label = torch.exp(self.hyperparameters[0])
        return outputs * weights_label


class HyperSoftmaxLogitsWeights(HyperLogitsWeightBase):
    def __init__(self, n_classes, **kwargs):
        super().__init__(n_classes, **kwargs)
        self.n_classes = n_classes

    def weight_outputs(self, outputs, dim=None, **kwargs):
        weights_label = F.softmax(self.hyperparameters[0]) * self.n_classes
        if dim is None:
            return outputs * weights_label
        else:
            outputs_perm = torch.movedim(outputs, dim, -1)
            outputs_perm = outputs_perm * weights_label
            return torch.movedim(outputs_perm, -1, dim)


class HyperSoftmaxWeightsBase(HyperParamBase):
    def __init__(self, size, hyper_loss=None, gamma=0., **kwargs):
        self.hyperparameters = nn.ParameterList([nn.Parameter(torch.empty(size))])
        self.hyper_loss = hyper_loss
        self.gamma = gamma

    def init_hyperparameters(self):
        with torch.no_grad():
            self.hyperparameters[0].copy_(torch.zeros_like(self.hyperparameters[0]))

    def weight_losses(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        if self.hyper_loss is None:
            return torch.tensor(0., device=self.hyperparameters[0].device)
        elif self.hyper_loss == NamesHyperLoss.L2_REGULARIZER:
            norm = 0.5 * (self.hyperparameters[0] @ self.hyperparameters[0])
            return self.gamma * norm
        else:
            raise ValueError(self.hyper_loss)


class HyperSoftmaxSampleWeights(HyperSoftmaxWeightsBase):
    def __init__(self, n_samples, **kwargs):
        super().__init__(n_samples, **kwargs)
        self.n_samples = n_samples

    def weight_losses(self, losses, idxs, **kwargs):
        weights = F.softmax(self.hyperparameters[0]) * self.n_samples
        return losses * weights[idxs]


class HyperSoftmaxCategoryWeights(HyperSoftmaxWeightsBase):
    def __init__(self, n_classes, **kwargs):
        super().__init__(n_classes, **kwargs)
        self.n_classes = n_classes

    def weight_losses(self, losses, ys, **kwargs):
        weights_label = F.softmax(self.hyperparameters[0]) * self.n_classes
        weights_sample = torch.hstack([weights_label[y] for y in ys])
        return losses * weights_sample


class HyperLearnersWeights(HyperParamBase):
    def __init__(self, n_learners, hyper_loss=None, gamma=0., **kwargs):
        self.hyperparameters = nn.ParameterList([nn.Parameter(torch.empty(n_learners))])
        self.n_learners = n_learners
        self.hyper_loss = hyper_loss
        self.gamma = gamma

    def init_hyperparameters(self):
        with torch.no_grad():
            self.hyperparameters[0].copy_(torch.zeros_like(self.hyperparameters[0]))

    def get_weights(self):
        return F.softmax(self.hyperparameters[0])

    def loss(self, *args, **kwargs):
        if self.hyper_loss is None:
            return torch.tensor(0., device=self.hyperparameters[0].device)
        elif self.hyper_loss == NamesHyperLoss.L2_REGULARIZER:
            norm = 0.5 * (self.hyperparameters[0] @ self.hyperparameters[0])
            return self.gamma * norm
        else:
            raise ValueError(self.hyper_loss)


class HyperLearnersWeightsAndSoftmaxCategoryWeights(HyperParamBase):
    def __init__(self, n_learners, n_classes, hyper_loss_learners=None, hyper_loss_categories=None, gamma_learners=0.,
                 gamma_categories=0., **kwargs):
        self.hyperparameters = nn.ParameterList([
            nn.Parameter(torch.empty(n_learners)),
            nn.Parameter(torch.empty(n_classes)),
        ])
        self.n_learners = n_learners
        self.n_classes = n_classes
        self.hyper_loss_learners = hyper_loss_learners
        self.hyper_loss_categories = hyper_loss_categories
        self.gamma_learners = gamma_learners
        self.gamma_categories = gamma_categories

    def init_hyperparameters(self):
        with torch.no_grad():
            self.hyperparameters[0].copy_(torch.zeros_like(self.hyperparameters[0]))
            self.hyperparameters[1].copy_(torch.zeros_like(self.hyperparameters[1]))

    def get_weights(self):
        return F.softmax(self.hyperparameters[0])

    def weight_losses(self, losses, ys, **kwargs):
        weights_label = F.softmax(self.hyperparameters[1]) * self.n_classes
        weights_sample = torch.hstack([weights_label[y] for y in ys])
        return losses * weights_sample

    def loss(self, *args, **kwargs):
        result = 0.
        result += self.hyper_loss(self.hyperparameters[0], self.hyper_loss_learners, self.gamma_learners)
        result += self.hyper_loss(self.hyperparameters[1], self.hyper_loss_categories, self.gamma_categories)
        return result

    @staticmethod
    def hyper_loss(hyperparameters, hyper_loss, gamma):
        if hyper_loss is None:
            return torch.tensor(0., device=hyperparameters.device)
        elif hyper_loss == NamesHyperLoss.L2_REGULARIZER:
            norm = 0.5 * (hyperparameters @ hyperparameters)
            return gamma * norm
        else:
            raise ValueError(hyper_loss)


class HyperLearnersWeightsAndLogitsWeightsBase(HyperParamBase):
    def __init__(self, n_learners, n_classes, hyper_loss_learners=None, hyper_loss_categories=None, gamma_learners=0.,
                 gamma_categories=0., **kwargs):
        self.n_learners = n_learners
        self.n_classes = n_classes
        self.hyper_loss_learners = hyper_loss_learners
        self.hyper_loss_categories = hyper_loss_categories
        self.gamma_learners = gamma_learners
        self.gamma_categories = gamma_categories

    def init_hyperparameters(self):
        with torch.no_grad():
            self.hyperparameters[0].copy_(torch.zeros_like(self.hyperparameters[0]))
            self.hyperparameters[1].copy_(torch.zeros_like(self.hyperparameters[1]))

    def get_weights(self):
        return F.softmax(self.hyperparameters[0])

    def loss(self, *args, **kwargs):
        result = 0.
        result += self.hyper_loss(self.hyperparameters[0], self.hyper_loss_learners, self.gamma_learners)
        result += self.hyper_loss(self.hyperparameters[1], self.hyper_loss_categories, self.gamma_categories)
        return result

    def weight_outputs(self, outputs, idx_learner, dim=None, **kwargs):
        raise NotImplementedError

    @staticmethod
    def hyper_loss(hyperparameters, hyper_loss, gamma):
        if hyper_loss is None:
            return torch.tensor(0., device=hyperparameters.device)
        elif hyper_loss == NamesHyperLoss.L2_REGULARIZER:
            norm = 0.5 * torch.sum(hyperparameters * hyperparameters)
            return gamma * norm
        else:
            raise ValueError(hyper_loss)


class HyperLearnersWeightsAndMultiSoftmaxLogitsWeights(HyperLearnersWeightsAndLogitsWeightsBase):
    def __init__(self, n_learners, n_classes, **kwargs):
        super().__init__(n_learners, n_classes, **kwargs)
        self.hyperparameters = nn.ParameterList([
            nn.Parameter(torch.empty(n_learners)),
            nn.Parameter(torch.empty(n_learners, n_classes)),
        ])

    def weight_outputs(self, outputs, idx_learner, dim=None, **kwargs):
        weights_label = F.softmax(self.hyperparameters[1][idx_learner]) * self.n_classes
        if dim is None:
            return outputs * weights_label
        else:
            outputs_perm = torch.movedim(outputs, dim, -1)
            outputs_perm = outputs_perm * weights_label
            return torch.movedim(outputs_perm, -1, dim)


class HyperLearnersWeightsAndSingleSoftmaxLogitsWeights(HyperLearnersWeightsAndLogitsWeightsBase):
    def __init__(self, n_learners, n_classes, **kwargs):
        super().__init__(n_learners, n_classes, **kwargs)
        self.hyperparameters = nn.ParameterList([
            nn.Parameter(torch.empty(n_learners)),
            nn.Parameter(torch.empty(n_classes)),
        ])

    def weight_outputs(self, outputs, idx_learner, dim=None, **kwargs):
        assert idx_learner < self.n_classes, idx_learner
        weights_label = F.softmax(self.hyperparameters[1]) * self.n_classes
        if dim is None:
            return outputs * weights_label
        else:
            outputs_perm = torch.movedim(outputs, dim, -1)
            outputs_perm = outputs_perm * weights_label
            return torch.movedim(outputs_perm, -1, dim)

class HyperDummy(HyperParamBase):
    def __init__(self, *args, **kwargs):
        self.hyperparameters = nn.ParameterList([
            nn.Parameter(torch.empty(1)),
        ])

    def init_hyperparameters(self):
        with torch.no_grad():
            self.hyperparameters[0].copy_(torch.zeros_like(self.hyperparameters[0]))

    def loss(self, *args, **kwargs):
        return torch.tensor(0., device=self.hyperparameters[0].device)


class HyperRegDecay(HyperParamBase):
    def __init__(self, *args, **kwargs):
        self.hyperparameters = nn.ParameterList([
            nn.Parameter(torch.empty(1)),
        ])

    def init_hyperparameters(self):
        with torch.no_grad():
            self.hyperparameters[0].copy_(torch.zeros_like(self.hyperparameters[0]))

    def loss(self, *args, **kwargs):
        return torch.tensor(0., device=self.hyperparameters[0].device)

    def get_reg_loss(self, param):
        return 0.5 * torch.sum(self.hyperparameters[0] * param ** 2)


class HyperLossMasks(HyperParamBase):
    def __init__(self, n_samples, **kwargs):
        self.hyperparameters = nn.ParameterList([nn.Parameter(torch.empty(n_samples))])

    def init_hyperparameters(self):
        with torch.no_grad():
            self.hyperparameters[0].copy_(torch.ones_like(self.hyperparameters[0]))

    def loss(self, *args, **kwargs):
        return torch.tensor(0., device=self.hyperparameters[0].device)

    def weight_losses(self, losses, idxs, **kwargs):
        return losses * self.hyperparameters[0][idxs]


D_HYPER_PARAMETERS = {
    NamesHyperParam.SOFTMAX_CATEGORY_WEIGHTS: HyperSoftmaxCategoryWeights,
    NamesHyperParam.LEARNERS_WEIGHTS: HyperLearnersWeights,
    NamesHyperParam.LEARNERS_WEIGHTS_AND_SOFTMAX_CATEGORY_WEIGHTS: HyperLearnersWeightsAndSoftmaxCategoryWeights,
    NamesHyperParam.HYPER_EXP_LOGITS_WEIGHTS: HyperExpLogitsWeights,
    NamesHyperParam.HYPER_SOFTMAX_LOGITS_WEIGHTS: HyperSoftmaxLogitsWeights,
    NamesHyperParam.LEARNERS_WEIGHTS_AND_MULTI_SOFTMAX_LOGITS_WEIGHTS: HyperLearnersWeightsAndMultiSoftmaxLogitsWeights,
    NamesHyperParam.LEARNERS_WEIGHTS_AND_SINGLE_SOFTMAX_LOGITS_WEIGHTS: HyperLearnersWeightsAndSingleSoftmaxLogitsWeights,
    NamesHyperParam.DUMMY: HyperDummy,
    NamesHyperParam.REG_DECAY: HyperRegDecay,
    NamesHyperParam.LOSS_MASKS: HyperLossMasks,
}
