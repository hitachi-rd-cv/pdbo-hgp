import torch
import torch.nn.functional as F
from torch import nn

from constants import NamesEvalMetric
from fedem.utils.constants import SHAKESPEARE_CONFIG
from module_torch.hyperparameter import HyperSoftmaxWeightsBase, HyperLogitsWeightBase


class NextCharacterLSTM(nn.Module):
    criterion = None

    def __init__(self, weight_decay=1e-2, hyperparameters=None):
        super().__init__()
        self.weight_decay = weight_decay
        self.hyperparameter_module = hyperparameters

        self.input_size = SHAKESPEARE_CONFIG["input_size"]
        self.embed_size = SHAKESPEARE_CONFIG["embed_size"]
        self.hidden_size = SHAKESPEARE_CONFIG["hidden_size"]
        self.output_size = SHAKESPEARE_CONFIG["output_size"]
        self.n_layers = SHAKESPEARE_CONFIG["n_layers"]
        self.encoder = nn.Embedding(self.input_size, self.embed_size)

        self.rnn = \
            nn.LSTM(
                input_size=self.embed_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def loss(self, inputs, *args, ys=None, **kwargs):
        if isinstance(self.hyperparameter_module, HyperSoftmaxWeightsBase):
            losses = self.bare_losses(*inputs).flatten()
            ys = ys.flatten()
            loss_mean = self.hyperparameter_module.weight_losses(losses, ys=ys).mean()
            return loss_mean + self.reg_loss()
        else:
            return self.bare_loss(*inputs) + self.reg_loss()

    def reg_loss(self):
        loss = 0.
        for param in self.parameters(recurse=True):
            loss += 0.5 * torch.sum((self.weight_decay * param ** 2))
        return loss

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
                loss_mean += F.cross_entropy(self.forward(x), y, reduction='sum') / (len(loader.dataset) * y.shape[1])

            return loss_mean

        elif metric == NamesEvalMetric.LOSS_MEAN:
            loss_mean = 0.
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                loss_mean += F.cross_entropy(self.forward(x), y, reduction='sum') / (len(loader.dataset) * y.shape[1])

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
        device = next(self.parameters()).device
        if metric == NamesEvalMetric.LOSS_BARE_MEAN:
            x, y = x.to(device), y.to(device)
            return F.cross_entropy(self.forward(x), y, reduction='sum') / y.shape[1]

        else:
            raise ValueError(metric)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        if isinstance(self.hyperparameter_module, HyperLogitsWeightBase):
            return self.hyperparameter_module.weight_outputs(output, dim=1)
        else:
            return output
