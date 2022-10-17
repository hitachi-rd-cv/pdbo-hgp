import torch
from torch import nn
from torch.nn import functional as F

from constants import NamesHyperParam, NamesHyperLoss


def normalized_sigmoid(inputs):
    return F.sigmoid(inputs) * 2 / len(inputs)


class HyperSoftmaxWeightsBase:
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


class HyperLearnersWeights:
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


class HyperLearnersWeightsAndSoftmaxCategoryWeights:
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



D_HYPER_PARAMETERS = {
    NamesHyperParam.SOFTMAX_CATEGORY_WEIGHTS: HyperSoftmaxCategoryWeights,
    NamesHyperParam.LEARNERS_WEIGHTS: HyperLearnersWeights,
    NamesHyperParam.LEARNERS_WEIGHTS_AND_SOFTMAX_CATEGORY_WEIGHTS: HyperLearnersWeightsAndSoftmaxCategoryWeights,
}
