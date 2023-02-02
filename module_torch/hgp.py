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


class HyperGradEstimatorBase(nn.Module):
    def __init__(self, client, loader_train, loader_valid, lr_scheduler, dumping, alpha_v, alpha_w):
        super().__init__()
        assert 0. <= dumping <= 1., f"dumping is expedted in [0., 1] but dumping={dumping} is given."
        assert 0. <= alpha_v <= 1., f"alpha_v is expedted in [0., 1] but alpha_v={alpha_v} is given."
        assert 0. <= alpha_w <= 1., f"alpha_w is expedted in [0., 1] but alpha_w={alpha_v} is given."

        self.client = client
        self.client.lr_scheduler = lr_scheduler
        self.client.estimate()

        self.idx_node = self.client.gossip.idx_node
        self.innerparameters = self.client.innerparameters
        self.hyperparameters = self.client.hyperparameters

        self.loader_train = loader_train
        self.loader_valid = loader_valid
        self.dumping = dumping

        self.iter_train = iter(loader_train)

        self.alpha_v = alpha_v
        self.alpha_w = alpha_w

    def compute_outer_gradients(self, metric):
        grad_biased = [torch.zeros_like(x) for x in self.innerparameters]
        grads_hyper = [torch.zeros_like(x) for x in self.hyperparameters]
        grad_weight = torch.zeros_like(self.client.gossip.weight)

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
                grad_biased_tmp, grad_weight_tmp, grads_hyper_tmp = mygrad(outer_model_loss, (tuple(self.client.params_biased), self.client.gossip.weight, tuple(self.hyperparameters)), allow_unused=True)
                for g, g_t in zip(grad_biased, grad_biased_tmp):
                    if g_t is not None:
                        g += g_t
                for g, g_t in zip(grads_hyper, grads_hyper_tmp):
                    if g_t is not None:
                        g += g_t
                if grad_weight_tmp is not None:
                    grad_weight += grad_weight_tmp

            outer_hyper_loss = self.client.hyperparameter_module.loss() / self.client.gossip.n_nodes
            if not (outer_hyper_loss.grad is None and outer_hyper_loss.grad_fn is None):
                grad_biased_tmp, grad_weight_tmp, grads_hyper_tmp = mygrad(outer_hyper_loss, (tuple(self.client.params_biased), self.client.gossip.weight, tuple(self.hyperparameters)), allow_unused=True)
                for g, g_t in zip(grad_biased, grad_biased_tmp):
                    if g_t is not None:
                        g += g_t
                for g, g_t in zip(grads_hyper, grads_hyper_tmp):
                    if g_t is not None:
                        g += g_t
                if grad_weight_tmp is not None:
                    grad_weight += grad_weight_tmp

        if has_bn:
            self.client.model.train()

        return grad_biased, grad_weight, grads_hyper

    def get_inputs_and_idxs(self):
        try:
            *inputs, idxs = next(self.iter_train)
        except StopIteration:
            self.iter_train = iter(self.loader_train)  # initialize iteration
            *inputs, idxs = next(self.iter_train)
        return inputs, idxs

    # @profile
    def get_hgp_msg_generator(self, are_connected):
        return self.send_hgp_msgs(are_connected)

    # @profile
    def send_hgp_msgs(self, are_connected):
        for to_node, is_connected in enumerate(are_connected):
            if is_connected:
                yield self.client.gossip.idx_node, (self.us_x, self.u_w, self.ws_x, self.w_w)

            else:
                yield self.client.gossip.idx_node, None

    def init_hypergrads(self, metric):
        raise NotImplementedError

    def get_sgp_msg_generator(self, step, no_weight_update=False, true_p_vec_expected=None, true_are_in_neighbors_expected=None):
        raise NotImplementedError

    def mix_hgp_partial(self, hgp_msg_in, sgp_msg_out, values_update=('u', 'v')):
        raise NotImplementedError

    def step(self, step, no_weight_update, values_update=('u', 'v')):
        raise NotImplementedError


class HyperGradEstimatorAssran(HyperGradEstimatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.us_x = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.innerparameters])
        self.u_w = nn.Parameter(torch.zeros_like(self.client.gossip.weight), requires_grad=False)
        self.vs = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.hyperparameters])
        self.us_x_step = None
        self.u_w_step = None

        if self.alpha_v < 1.0:
            self.ws_x = nn.ParameterList(
                [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.innerparameters])
            self.cs_x = nn.ParameterList(
                [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.innerparameters])
            self.w_w = nn.Parameter(torch.zeros_like(self.client.gossip.weight), requires_grad=False)
            self.c_w = nn.Parameter(torch.zeros_like(self.client.gossip.weight), requires_grad=False)
        else:
            self.ws_x = None
            self.cs_x = None
            self.w_w = None
            self.c_w = None

        self.ws_x_step = None
        self.w_w_step = None

    def init_hypergrads(self, metric):
        self.us_x_step = [torch.zeros_like(x) for x in self.innerparameters]
        self.u_w_step = torch.zeros_like(self.client.gossip.weight)

        if self.alpha_v < 1.0:
            self.ws_x_step = [torch.zeros_like(x) for x in self.innerparameters]
            self.w_w_step = torch.zeros_like(self.client.gossip.weight)

        # initialize us_xand u_w
        grad_biased, grad_weight, grads_hyper = self.compute_outer_gradients(metric)

        for u_x, g_x in zip(self.us_x, grad_biased):
            u_x.copy_(g_x.clone().detach())

        if self.alpha_v < 1.0:
            for w_x, c_x, g_x in zip(self.ws_x, self.cs_x, grad_biased):
                w_x.copy_(g_x.clone().detach())
                c_x.copy_(g_x.clone().detach())

        self.u_w.copy_(grad_weight.clone().detach())

        if self.alpha_v < 1.0:
            self.w_w.copy_(grad_weight.clone().detach())
            self.c_w.copy_(grad_weight.clone().detach())

        for v, g_h in zip(self.vs, grads_hyper):
            v.copy_(g_h.clone().detach())

    def get_sgp_msg_generator(self, step, no_weight_update=False, true_p_vec_expected=None, true_are_in_neighbors_expected=None):
        return self.client.gen_expected_ps(
            true_p_vec_expected=true_p_vec_expected,
            true_are_in_neighbors_expected=true_are_in_neighbors_expected
        )

    # @profile
    def mix_hgp_partial(self, hgp_msg_in, sgp_msg_out, values_update=('u', 'v')):  # TODO(future): make jvp compulation once in single depth by when jacobian is node-independent (p is not hyper-parameterized)
        _, hgp_msg_in_body = hgp_msg_in
        _, sgp_msg_out_body = sgp_msg_out
        if hgp_msg_in_body is not None and sgp_msg_out_body is not None:
            us_x_neigh, u_w_neigh, ws_x_neigh, w_w_neigh = hgp_msg_in_body
            p = sgp_msg_out_body

            for u, jvp_u in zip(self.us_x_step, us_x_neigh):
                u.add_(p.clone().detach() * jvp_u.clone().detach())
            self.u_w_step.add_(p.clone().detach() * u_w_neigh.clone().detach())

            if self.alpha_v < 1.0:
                for w, jvp_w in zip(self.ws_x_step, ws_x_neigh):
                    w.add_(p.clone().detach() * jvp_w.clone().detach())
                self.w_w_step.add_(p.clone().detach() * w_w_neigh.clone().detach())

    # @profile
    def step(self, step, no_weight_update, values_update=('u', 'v')):
        inputs, idxs = self.get_inputs_and_idxs()

        with torch.backends.cudnn.flags(enabled=False):
            update_steps = self.client.get_biased_parameter_step(step, inputs, idxs, create_graph=True)
            params_send = [x + self.dumping * s for x, s in zip(self.client.params_biased, update_steps)]
            if no_weight_update:
                weight_send = torch.zeros_like(self.client.gossip.weight)
            else:
                weight_send = self.client.gossip.weight

        if self.alpha_v < 1.0:
            jvp_u_param, jvp_u_weight, jvp_u_hypers = myjvp(
                outputs=(*tuple(params_send), weight_send),
                inputs=(tuple(self.client.params_biased), self.client.gossip.weight, tuple(self.hyperparameters)),
                v=(*tuple(self.us_x_step), self.u_w_step),
                create_graph=True
            )

            jvp_w_param, jvp_w_weight, jvp_w_hypers = myjvp(
                outputs=(*tuple(params_send), weight_send),
                inputs=(tuple(self.client.params_biased), self.client.gossip.weight, tuple(self.hyperparameters)),
                v=(*tuple(self.ws_x_step), self.w_w_step),
            )

        else:
            jvp_u_param, jvp_u_weight, jvp_u_hypers = myjvp(
                outputs=(*tuple(params_send), weight_send),
                inputs=(tuple(self.client.params_biased), self.client.gossip.weight, tuple(self.hyperparameters)),
                v=(*tuple(self.us_x_step), self.u_w_step),
            )

        if 'v' in values_update:
            if self.alpha_v < 1.0:
                for v, v_u_step, v_w_step in zip(self.vs, jvp_u_hypers, jvp_w_hypers):
                    # compute alpha * (v + B @ u) + (1 - alpha) * B @ w for each element
                    v_updated = self.alpha_v * (v + v_u_step) + (1. - self.alpha_v) * v_w_step
                    v.copy_(v_updated.clone().detach())

            else:
                for v, v_u_step in zip(self.vs, jvp_u_hypers):
                    # compute  v + B @ u for each element
                    v_updated = v + v_u_step
                    v.copy_(v_updated.clone().detach())

        if 'u' in values_update:
            # update u
            for u, u_step in zip(self.us_x, jvp_u_param):
                u.copy_(u_step)
            self.u_w.copy_(jvp_u_weight)

            # update w
            if self.alpha_v < 1.0:
                for w, w_step, u, c in zip(self.ws_x, jvp_w_param, self.us_x, self.cs_x):
                    # compute alpha * (A @ w + c) + (1 - alpha) * (w + u) for each element
                    w_updated = self.alpha_w * (w_step + c) + (1. - self.alpha_w) * (w + u)
                    w.copy_(w_updated.clone().detach())
                w_w_updated = self.alpha_w * (jvp_w_weight + self.c_w) + (1. - self.alpha_w) * (self.w_w + self.u_w)
                self.w_w.copy_(w_w_updated.clone().detach())

        # initialize temporary parameters
        for u in self.us_x_step:
            u.zero_()
        self.u_w_step.zero_()

        if self.alpha_v < 1.0:
            for w in self.ws_x_step:
                w.zero_()
            self.w_w_step.zero_()


class HyperGradEstimatorNedic(HyperGradEstimatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.us_x = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.innerparameters])
        self.u_w = nn.Parameter(torch.zeros_like(self.client.gossip.weight), requires_grad=False)
        self.vs = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.hyperparameters])
        self.us_x_step = None
        self.u_w_step = None

        if self.alpha_v < 1.0:
            self.ws_x = nn.ParameterList(
                [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.innerparameters])
            self.cs_x = nn.ParameterList(
                [nn.Parameter(torch.zeros_like(x), requires_grad=False) for x in self.innerparameters])
            self.w_w = nn.Parameter(torch.zeros_like(self.client.gossip.weight), requires_grad=False)
            self.c_w = nn.Parameter(torch.zeros_like(self.client.gossip.weight), requires_grad=False)
        else:
            self.ws_x = None
            self.cs_x = None
            self.w_w = None
            self.c_w = None

        self.ws_x_step = None
        self.w_w_step = None

    def init_hypergrads(self, metric):
        self.us_x_step = [torch.zeros_like(x) for x in self.innerparameters]
        self.u_w_step = torch.zeros_like(self.client.gossip.weight)
        self.vs_u_step = [torch.zeros_like(x) for x in self.hyperparameters]

        if self.alpha_v < 1.0:
            self.ws_x_step = [torch.zeros_like(x) for x in self.innerparameters]
            self.w_w_step = torch.zeros_like(self.client.gossip.weight)
            self.vs_w_step = [torch.zeros_like(x) for x in self.hyperparameters]

        # initialize us_xand u_w
        grad_biased, grad_weight, grads_hyper = self.compute_outer_gradients(metric)

        for u_x, g_x in zip(self.us_x, grad_biased):
            u_x.copy_(g_x.clone().detach())

        if self.alpha_v < 1.0:
            for w_x, c_x, g_x in zip(self.ws_x, self.cs_x, grad_biased):
                w_x.copy_(g_x.clone().detach())
                c_x.copy_(g_x.clone().detach())

        self.u_w.copy_(grad_weight.clone().detach())

        if self.alpha_v < 1.0:
            self.w_w.copy_(grad_weight.clone().detach())
            self.c_w.copy_(grad_weight.clone().detach())

        for v, g_h in zip(self.vs, grads_hyper):
            v.copy_(g_h.clone().detach())

    # @profile
    def get_sgp_msg_generator(self, step, no_weight_update=False, true_p_vec_expected=None, true_are_in_neighbors_expected=None):
        inputs, idxs = self.get_inputs_and_idxs()
        return self.client.gen_expected_nedic_sgp_messages(step, inputs, idxs, dumping=self.dumping, create_graph=True, no_weight_update=no_weight_update, true_p_vec_expected=true_p_vec_expected,
                                                           true_are_in_neighbors_expected=true_are_in_neighbors_expected)

    # @profile
    def mix_hgp_partial(self, hgp_msg_in, sgp_msg_out, values_update=('u', 'v')):  # TODO(future): make jvp compulation once in single depth by when jacobian is node-independent (p is not hyper-parameterized)
        _, hgp_msg_in_body = hgp_msg_in
        _, sgp_msg_out_body = sgp_msg_out
        if hgp_msg_in_body is not None and sgp_msg_out_body is not None:

            us_x_neigh, u_w_neigh, ws_x_neigh, w_w_neigh = hgp_msg_in_body
            params_send, weight_send = sgp_msg_out_body

            if self.alpha_v < 1.0:
                jvp_u_param, jvp_u_weight, jvp_u_hypers = myjvp(
                    outputs=(*tuple(params_send), weight_send),
                    inputs=(tuple(self.client.params_biased), self.client.gossip.weight, tuple(self.hyperparameters)),
                    v=(*tuple(us_x_neigh), u_w_neigh),
                    create_graph=True
                )

                jvp_w_param, jvp_w_weight, jvp_w_hypers = myjvp(
                    outputs=(*tuple(params_send), weight_send),
                    inputs=(tuple(self.client.params_biased), self.client.gossip.weight, tuple(self.hyperparameters)),
                    v=(*tuple(ws_x_neigh), w_w_neigh),
                )
            else:
                jvp_u_param, jvp_u_weight, jvp_u_hypers = myjvp(
                    outputs=(*tuple(params_send), weight_send),
                    inputs=(tuple(self.client.params_biased), self.client.gossip.weight, tuple(self.hyperparameters)),
                    v=(*tuple(us_x_neigh), u_w_neigh),
                )

            if 'v' in values_update:
                for v_u, jvp_u, in zip(self.vs_u_step, jvp_u_hypers):
                    v_u.add_(jvp_u.clone().detach())

                if self.alpha_v < 1.0:
                    for v_w, jvp_w in zip(self.vs_w_step, jvp_w_hypers):
                        v_w.add_(jvp_w.clone().detach())

            if 'u' in values_update:
                for u, jvp_u in zip(self.us_x_step, jvp_u_param):
                    u.add_(jvp_u.clone().detach())
                self.u_w_step.add_(jvp_u_weight.clone().detach())

                if self.alpha_v < 1.0:
                    for w, jvp_w in zip(self.ws_x_step, jvp_w_param):
                        w.add_(jvp_w.clone().detach())
                    self.w_w_step.add_(jvp_w_weight.clone().detach())

    # @profile
    def step(self, step, no_weight_update, values_update=('u', 'v')):
        if 'v' in values_update:
            if self.alpha_v < 1.0:
                for v, v_u_step, v_w_step in zip(self.vs, self.vs_u_step, self.vs_w_step):
                    # compute alpha * (v + B @ u) + (1 - alpha) * B @ w for each element
                    v_updated = self.alpha_v * (v + v_u_step) + (1. - self.alpha_v) * v_w_step
                    v.copy_(v_updated.clone().detach())

            else:
                for v, v_u_step in zip(self.vs, self.vs_u_step):
                    # compute  v + B @ u for each element
                    v_updated = v + v_u_step
                    v.copy_(v_updated.clone().detach())

        if 'u' in values_update:
            # update u
            for u, u_step in zip(self.us_x, self.us_x_step):
                u.copy_(u_step)
            self.u_w.copy_(self.u_w_step)

            # update w
            if self.alpha_v < 1.0:
                for w, w_step, u, c in zip(self.ws_x, self.ws_x_step, self.us_x, self.cs_x):
                    # compute alpha * (A @ w + c) + (1 - alpha) * (w + u) for each element
                    w_updated = self.alpha_w * (w_step + c) + (1. - self.alpha_w) * (w + u)
                    w.copy_(w_updated.clone().detach())
                w_w_updated = self.alpha_w * (self.w_w_step + self.c_w) + (1. - self.alpha_w) * (self.w_w + self.u_w)
                self.w_w.copy_(w_w_updated.clone().detach())

        # initialize temporary parameters
        for v_u in self.vs_u_step:
            v_u.zero_()

        for u in self.us_x_step:
            u.zero_()
        self.u_w_step.zero_()

        if self.alpha_v < 1.0:
            for v_w in self.vs_w_step:
                v_w.zero_()
            for w in self.ws_x_step:
                w.zero_()
            self.w_w_step.zero_()
