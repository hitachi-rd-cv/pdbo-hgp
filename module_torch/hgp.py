import torch
import torch.autograd
import torch.optim
from torch import nn

from lib_common.torch.autograd import mygrad, myjvp
from module_torch.model.learners_ensemble import MultiLanguageLearners
from module_torch.model.learners_ensemble import MultiLearners
from module_torch.model.lstm_shakespeare import NextCharacterLSTM
from module_torch.model.mn_cifar10 import MobileNetCIFAR10
from module_torch.model.mn_cifar100 import MobileNetCIFAR100


class HyperGradEstimatorDummy(nn.Module):
    def __init__(self, client):
        super().__init__()
        self.client = client


class HyperGradEstimator(nn.Module):
    def __init__(self, client, loader_train, loader_valid, dumping, use_iid_samples):
        super().__init__()
        assert 0. <= dumping <= 1., f"dumping is expedted in [0., 1] but dumping={dumping} is given."

        self.client = client
        self.client.estimate()

        self.idx_node = self.client.gossip.idx_node
        self.innerparameters = self.client.innerparameters
        self.hyperparameters = self.client.hyperparameters

        self.loader_train = loader_train
        self.loader_valid = loader_valid
        self.dumping = dumping
        self.use_iid_samples = use_iid_samples

        self.iter_train = iter(loader_train)
        self.us = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.innerparameters])
        self.vs = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.hyperparameters])
        self.ws = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.innerparameters])
        self.ws_step = [torch.zeros_like(x) for x in self.innerparameters]


    def compute_outer_gradients(self, metric):
        grad_biased = [torch.zeros_like(x) for x in self.innerparameters]
        grads_hyper = [torch.zeros_like(x) for x in self.hyperparameters]

        flag_cudnn = not isinstance(self.client.model, (MultiLanguageLearners, NextCharacterLSTM))

        if isinstance(self.client.model, (MobileNetCIFAR10, MobileNetCIFAR100)):
            has_bn = True
        elif isinstance(self.client.model, MultiLearners):
            has_bn = isinstance(self.client.model.learners[0], (MobileNetCIFAR10, MobileNetCIFAR100))
        else:
            has_bn = False

        if has_bn:
            self.client.model.eval()

        with torch.backends.cudnn.flags(enabled=flag_cudnn):  # Double backwards is not supported for CuDNN RNNs due to limitations in the CuDNN API
            for x, y, _ in self.loader_valid:
                self.client.debias_parameters()
                outer_model_loss = self.client.model.eval_metric_sum_from_x_y(metric, x, y) / (len(self.loader_valid.dataset) * self.client.gossip.n_nodes)
                grad_inner_tmp, grads_hyper_tmp = mygrad(outer_model_loss, (tuple(self.client.innerparameters), tuple(self.hyperparameters)), allow_unused=True)
                for g, g_t in zip(grad_biased, grad_inner_tmp):
                    if g_t is not None:
                        g += g_t
                for g, g_t in zip(grads_hyper, grads_hyper_tmp):
                    if g_t is not None:
                        g += g_t

            outer_hyper_loss = self.client.hyperparameter_module.loss() / self.client.gossip.n_nodes
            if not (outer_hyper_loss.grad is None and outer_hyper_loss.grad_fn is None):
                grad_inner_tmp, grads_hyper_tmp = mygrad(outer_hyper_loss, (tuple(self.client.innerparameters), tuple(self.hyperparameters)), allow_unused=True)
                for g, g_t in zip(grad_biased, grad_inner_tmp):
                    if g_t is not None:
                        g += g_t
                for g, g_t in zip(grads_hyper, grads_hyper_tmp):
                    if g_t is not None:
                        g += g_t

        if has_bn:
            self.client.model.train()

        return grad_biased, grads_hyper

    def init_hypergrads(self, metric):
        # initialize us and u_w
        grad_biased, grads_hyper = self.compute_outer_gradients(metric)

        for u_x, w_x, g_x in zip(self.us, self.ws, grad_biased):
            u_x.copy_(g_x.clone().detach())
            w_x.copy_(g_x.clone().detach())

        for v, g_h in zip(self.vs, grads_hyper):
            v.copy_(g_h.clone().detach())

        # initialize debias weight aggregation placeholder
        self.client.gossip.weight_updated = torch.zeros_like(self.client.gossip.weight)

    def get_inputs_and_idxs(self):
        try:
            *inputs, idxs = next(self.iter_train)
        except StopIteration:
            self.iter_train = iter(self.loader_train)  # initialize iteration
            *inputs, idxs = next(self.iter_train)
        return inputs, idxs

    # @profile
    def get_hgp_msg_generator(self, are_connected, p_vec_true=None):
        return self.send_hgp_msgs(are_connected, p_vec_true=p_vec_true)

    # @profile
    def send_hgp_msgs(self, are_connected, p_vec_true=None):
        if p_vec_true is None:
            p_vec = self.client.gossip.get_p_vec(are_connected)
        else:
            p_vec = p_vec_true
        for to_node, p in enumerate(p_vec):
            if p > 0.:
                yield self.client.gossip.idx_node, ((p * w for w in self.ws), p * self.client.gossip.weight)
            else:
                yield self.client.gossip.idx_node, None

    # @profile
    def init_s(self):
        # update w
        for w, u in zip(self.ws, self.us):
            w.copy_(u.clone().detach())

    # @profile
    def mix_s(self, hgp_msg_in):  # TODO(future): make jvp compulation once in single depth by when jacobian is node-independent (p is not hyper-parameterized)
        _, hgp_msg_in_body = hgp_msg_in
        if hgp_msg_in_body is not None:
            ws_neigh, weight_neigh = hgp_msg_in_body
            for w, w_neigh in zip(self.ws_step, ws_neigh):
                w.add_(w_neigh.clone().detach())
            self.client.gossip.weight_updated.add_(weight_neigh.clone().detach())

    # @profile
    def step_s(self):
        with torch.no_grad():
            # update w
            for w, w_step in zip(self.ws, self.ws_step):
                w.copy_(w_step.clone().detach())

            # update debias weight
            self.client.gossip.weight.copy_(self.client.gossip.weight_updated.clone().detach())

            # initialize temporary parameters
            for w in self.ws_step:
                w.zero_()
            self.client.gossip.weight_updated.zero_()

    # @profile
    def step_m(self, weight_true=None):
        # compute train loss
        if self.use_iid_samples:
            inputs, idxs = self.get_inputs_and_idxs()
            kwargs_loss = self.client.get_kwargs_loss(inputs, idxs)
            loss = self.client.model.loss(inputs, **kwargs_loss)

            with torch.backends.cudnn.flags(enabled=False):
                grads_x = mygrad(loss, self.innerparameters, create_graph=True)

            jvp_u_hyper = myjvp(
                outputs=tuple(grads_x),
                inputs=tuple(self.client.hyperparameters),
                v=tuple(self.ws),
            )

            inputs, idxs = self.get_inputs_and_idxs()
            kwargs_loss = self.client.get_kwargs_loss(inputs, idxs)
            loss = self.client.model.loss(inputs, **kwargs_loss)

            with torch.backends.cudnn.flags(enabled=False):
                grads_x = mygrad(loss, self.innerparameters, create_graph=True)

            jvp_u_param = myjvp(
                outputs=tuple(grads_x),
                inputs=tuple(self.client.innerparameters),
                v=tuple(self.ws),
            )

        else:
            inputs, idxs = self.get_inputs_and_idxs()
            kwargs_loss = self.client.get_kwargs_loss(inputs, idxs)
            loss = self.client.model.loss(inputs, **kwargs_loss)

            with torch.backends.cudnn.flags(enabled=False):
                grads_x = mygrad(loss, self.innerparameters, create_graph=True)

            jvp_u_param, jvp_u_hyper = myjvp(
                outputs=tuple(grads_x),
                inputs=(tuple(self.client.innerparameters), tuple(self.client.hyperparameters)),
                v=tuple(self.ws),
            )

        if weight_true is None:
            weight = self.client.gossip.weight.clone().detach()
        else:
            weight = weight_true

        with torch.no_grad():
            # update v
            for v, v_step in zip(self.vs, jvp_u_hyper):
                v.add_(- self.dumping * v_step.clone().detach() / weight.clone().detach())

            # update u
            for u, w, u_step in zip(self.us, self.ws, jvp_u_param):
                u.copy_((w.clone().detach() - self.dumping * u_step.clone().detach()) / weight.clone().detach())
