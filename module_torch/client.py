import numpy as np
import torch.autograd
from prettytable import PrettyTable
from torch import nn

from module_torch.gossip_protocol import D_GOSSIP_CLASS
from module_torch.hyperparameter import D_HYPER_PARAMETERS, HyperSoftmaxSampleWeights, \
    HyperSoftmaxCategoryWeights, HyperLearnersWeightsAndSoftmaxCategoryWeights
from module_torch.model import D_MODELS


class Client(nn.Module):
    def __init__(self, n_nodes, idx_node, mode_gossip, name_model, kwargs_model, name_hyperparam, kwargs_hyperparam, kwargs_gossip):
        super().__init__()
        self.hyperparameter_module = D_HYPER_PARAMETERS[name_hyperparam](n_nodes=n_nodes, idx_node=idx_node, **kwargs_hyperparam)
        self.hyperparameters = self.hyperparameter_module.hyperparameters

        self.gossip = D_GOSSIP_CLASS[mode_gossip](n_nodes, idx_node, **kwargs_gossip)
        self.model = D_MODELS[name_model](**kwargs_model, hyperparameters=self.hyperparameter_module)  # to avoid assigning as a parameter of the model
        self.use_expected_edge = True

        innerparameters = []
        for p in self.model.parameters(recurse=True):
            assert torch.is_tensor(p), p
            innerparameters.append(p)
        self.innerparameters = nn.ParameterList(innerparameters)

        self.params_updated = None
        self.weight_updated = None
        self.current_loss = None
        self.current_lr = None

        self._initialize_biased_innerparameters()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.params_updated = [torch.zeros_like(x) for x in self.innerparameters]
        self.weight_updated = torch.zeros_like(self.gossip.weight)
        self.update_expected_edge = True
        if self.use_expected_edge:
            self.gossip.initialize_expected_values()

    def eval_metric(self, *args, **kwargs):
        return self.model.eval_metric(*args, **kwargs)

    def _initialize_biased_innerparameters(self):
        # biased parameter denoted by x in the paper
        self.params_biased = nn.ParameterList()
        for idx_param, p in enumerate(self.innerparameters):
            biased = nn.Parameter(p.clone().detach())
            self.params_biased.append(biased)

    def get_p_vec(self, are_connected):
        return self.gossip.get_p_vec(are_connected)

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def outer_loss(self, metric, loader):
        loss_model = self.eval_metric(metric, loader)
        loss_hyper = self.hyperparameter_module.loss()
        return loss_model + loss_hyper

    def get_device_of_param(self):
        return next(self.parameters()).device

    def print_parameters_info(self):
        # print out variables information
        nparams_total = 0
        print("Inner-parameters' info:")
        pt_param = PrettyTable(['index', 'param', 'shape', 'size', 'total'])
        for i, (name_p, p) in enumerate(self.innerparameters.named_parameters()):
            param_shape = p.shape
            nparams = np.prod(param_shape)
            nparams_total += nparams
            pt_param.add_row([i, name_p, param_shape, nparams, nparams_total])
        print(pt_param)

        # print out variables information
        nhparams_total = 0
        print("Hyper-parameters' info:")
        pt_hparam = PrettyTable(['index', 'param', 'shape', 'size', 'total'])
        for i, (name_h, h) in enumerate(self.hyperparameters.named_parameters()):
            hparam_shape = h.shape
            nhparams = np.prod(hparam_shape).astype(np.int)
            nhparams_total += nhparams
            pt_hparam.add_row([i, name_h, hparam_shape, nhparams, nhparams_total])
        print(pt_hparam)

    def debias_parameters(self):
        for x, z in zip(self.params_biased, self.innerparameters):
            # this inplace op turns off requires_grad and removes grad_fn but keep itself to be nn.Parameter
            z.detach_()
            # without detach_() copy_ accumulates the grad_fn tied to the attribute. copy_ passes the grad_fn of inputs not only the values.
            z.copy_(x / self.gossip.weight)

    def copy_innerparameters_to_parmas_biased(self):
        for x, z in zip(self.innerparameters, self.params_biased):
            # this inplace op turns off requires_grad and removes grad_fn but keep itself to be nn.Parameter
            z.detach_()
            # without detach_() copy_ accumulates the grad_fn tied to the attribute. copy_ passes the grad_fn of inputs not only the values.
            z.copy_(x)

    def obtain_step(self, step, inputs, idxs, create_graph=False):
        kwargs_loss = self.get_kwargs_loss(inputs, idxs)
        loss = self.model.loss(inputs, **kwargs_loss)
        # TODO(future): loss.backward(create_graph=create_graph)
        # TODO(future): update_steps = self.optimizer.step(create_graph=create_graph)
        grads = torch.autograd.grad(loss, self.innerparameters, create_graph=create_graph)
        lr = self.lr_scheduler(step)
        update_steps = tuple([-lr * grad for grad in grads])

        self.current_loss = loss
        self.current_lr = lr

        return update_steps

    def get_kwargs_loss(self, inputs, idxs):
        if isinstance(self.hyperparameter_module, HyperSoftmaxSampleWeights):
            return {'idxs': idxs}
        elif isinstance(self.hyperparameter_module, (
                HyperSoftmaxCategoryWeights, HyperLearnersWeightsAndSoftmaxCategoryWeights
        )):
            return {'ys': inputs[1].numpy()}
        else:
            return {}

    def mix_sgp_partial(self, sgp_msg, n_steps=None):
        from_node, msg_body = sgp_msg
        if msg_body is not None:
            params_get, weight_get = msg_body
            for x, x_get in zip(self.params_updated, params_get):
                x.add_(x_get.clone().detach())
            self.weight_updated.add_(weight_get.clone().detach())

        # update expected in_neighbors
        if self.use_expected_edge and self.update_expected_edge:
            assert n_steps is not None, n_steps
            is_in_neighbor = float(msg_body is not None)
            self.gossip.mix_are_in_neighbors_expected(is_in_neighbor, from_node, n_steps)

    def step_sgp(self):
        with torch.no_grad():
            # udpate
            for x, x_update in zip(self.params_biased, self.params_updated):
                x.copy_(x_update.clone().detach())
            if self.weight_updated > 0.:
                self.gossip.weight.copy_(self.weight_updated.clone().detach())

            if self.use_expected_edge and self.update_expected_edge:
                self.gossip.step_expected_values()

            # initialize
            for x in self.params_updated:
                x.zero_()
            self.weight_updated.zero_()

            if self.use_expected_edge and self.update_expected_edge:
                self.gossip.initialize_expected_updated_values()

    def estimate(self):
        self.update_expected_edge = False

    def gen_sgp_messages(self, are_connected, step, inputs, idxs, dumping=1., n_steps=None, create_graph=False, no_weight_update=False):
        for to_node, is_connected in enumerate(are_connected):
            p = self.gossip.get_p_vec(are_connected)[to_node]
            if self.use_expected_edge and self.update_expected_edge:
                self.gossip.mix_p_vec_expected(p, to_node, n_steps)

            if is_connected:
                if to_node == self.gossip.idx_node:
                    self.debias_parameters()
                    update_steps = self.obtain_step(step, inputs, idxs, create_graph=create_graph)
                    params_send = [p * x + dumping * s for x, s in zip(self.params_biased, update_steps)]
                else:
                    params_send = [p * x for x in self.params_biased]

                if no_weight_update:
                    weight_send = torch.zeros_like(self.gossip.weight)
                else:
                    weight_send = p * self.gossip.weight

                yield self.gossip.idx_node, (params_send, weight_send)

            else:
                yield self.gossip.idx_node, None

    def gen_expected_sgp_messages(self, step, inputs, idxs, dumping=1., create_graph=False, no_weight_update=False, true_p_vec_expected=None, true_are_in_neighbors_expected=None):
        if true_p_vec_expected is not None and true_are_in_neighbors_expected is not None:
            p_vec_expected = true_p_vec_expected
            are_in_neighbors_expected = true_are_in_neighbors_expected
        else:
            p_vec_expected = self.gossip.p_vec_expected
            are_in_neighbors_expected = self.gossip.are_in_neighbors_expected
        for to_node, (is_in_neighbor_expected, p_expect) in enumerate(zip(are_in_neighbors_expected, p_vec_expected)):
            if to_node == self.gossip.idx_node:
                self.debias_parameters()
                update_steps = self.obtain_step(step, inputs, idxs, create_graph=create_graph)
                params_send = [(p_expect * x + dumping * s) / is_in_neighbor_expected for x, s in zip(self.params_biased, update_steps)]
            else:
                params_send = [(p_expect / is_in_neighbor_expected) * x for x in self.params_biased]

            if no_weight_update:
                weight_send = torch.zeros_like(self.gossip.weight)
            else:
                weight_send = (p_expect / is_in_neighbor_expected) * self.gossip.weight

            yield self.gossip.idx_node, (params_send, weight_send)
