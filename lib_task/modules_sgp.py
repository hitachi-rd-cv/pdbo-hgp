from _bisect import bisect_right
from collections import OrderedDict, Counter
from copy import deepcopy

import numpy as np
import torch

from constants import NamesLrScheduler


class TrainerBase:
    def __init__(self, client, loader_train, lr_scheduler, train_logger):
        self.client = client
        self.idx_node = client.gossip.idx_node
        self.loader_train = loader_train
        self.lr_scheduler = lr_scheduler
        self.train_logger = train_logger
        self.iter_train = iter(loader_train)
        self.client.lr_scheduler = lr_scheduler
        self.client.train()

    def get_msg_generator(self, are_connected, step, no_weight_update, n_steps):
        try:
            *inputs, idxs = next(self.iter_train)
        except StopIteration:
            self.iter_train = iter(self.loader_train)  # initialize iteration
            *inputs, idxs = next(self.iter_train)

        return self.gen_sgp_messages(are_connected, step, inputs, idxs, n_steps=n_steps, no_weight_update=no_weight_update)

    def log(self, are_connected):
        self.train_logger.record_step_value({
            'lr': self.client.current_lr,
            'loss': self.client.current_loss.item(),
            'weight': self.client.gossip.weight.item(),
            'deg_out': torch.sum(are_connected).item(),
        })

    def eval(self):
        self.train_logger.record_step_metric()

    def step(self):
        with torch.no_grad():
            # udpate
            for x, x_update in zip(self.client.params_biased, self.client.params_updated):
                x.copy_(x_update.clone().detach())

            if self.client.weight_updated > 0.:
                self.client.gossip.weight.copy_(self.client.weight_updated.clone().detach())

            if self.client.update_expected_edge:
                self.client.gossip.step_expected_values()

            # initialize
            for x in self.client.params_updated:
                x.zero_()
            self.client.weight_updated.zero_()

            if self.client.update_expected_edge:
                self.client.gossip.initialize_expected_updated_values()

    def mix_sgp_partial(self, sgp_msg, n_steps=None):
        raise NotImplementedError

    def gen_sgp_messages(self, are_connected, step, inputs, idxs, n_steps=None, create_graph=False, no_weight_update=False):
        raise NotImplementedError


class TrainerAssran(TrainerBase):
    def mix_sgp_partial(self, sgp_msg, n_steps=None):
        from_node, msg_body = sgp_msg
        if msg_body is not None:
            params_get, weight_get, p = msg_body
            for x, x_get in zip(self.client.params_updated, params_get):
                x.add_(p.clone().detach() * x_get.clone().detach())
            self.client.weight_updated.add_(p.clone().detach() * weight_get.clone().detach())

        # update expected in_neighbors
        if self.client.update_expected_edge:
            assert n_steps is not None, n_steps
            is_in_neighbor = float(msg_body is not None)
            self.client.gossip.mix_are_in_neighbors_expected(is_in_neighbor, from_node, n_steps)

    def gen_sgp_messages(self, are_connected, step, inputs, idxs, n_steps=None, create_graph=False, no_weight_update=False):
        with torch.backends.cudnn.flags(enabled=False):
            update_steps = self.client.get_biased_parameter_step(step, inputs, idxs, create_graph=create_graph)
            params_send = [x + s for x, s in zip(self.client.params_biased, update_steps)]

            if no_weight_update:
                weight_send = torch.zeros_like(self.client.gossip.weight)
            else:
                weight_send = self.client.gossip.weight

            for to_node, is_connected in enumerate(are_connected):
                p = self.client.gossip.get_p_vec(are_connected)[to_node]
                if self.client.update_expected_edge:
                    self.client.gossip.mix_p_vec_expected(p, to_node, n_steps)
                if is_connected:
                    yield self.client.gossip.idx_node, (params_send, weight_send, p)

                else:
                    yield self.client.gossip.idx_node, None


class TrainerNedic(TrainerBase):
    def mix_sgp_partial(self, sgp_msg, n_steps=None):
        from_node, msg_body = sgp_msg
        if msg_body is not None:
            params_get, weight_get = msg_body
            for x, x_get in zip(self.client.params_updated, params_get):
                x.add_(x_get.clone().detach())
            self.client.weight_updated.add_(weight_get.clone().detach())

        # update expected in_neighbors
        if self.client.update_expected_edge:
            assert n_steps is not None, n_steps
            is_in_neighbor = float(msg_body is not None)
            self.client.gossip.mix_are_in_neighbors_expected(is_in_neighbor, from_node, n_steps)

    def gen_sgp_messages(self, are_connected, step, inputs, idxs, n_steps=None, create_graph=False, no_weight_update=False):
        with torch.backends.cudnn.flags(enabled=False):
            for to_node, is_connected in enumerate(are_connected):
                p = self.client.gossip.get_p_vec(are_connected)[to_node]
                if self.client.update_expected_edge:
                    self.client.gossip.mix_p_vec_expected(p, to_node, n_steps)

                if is_connected:
                    if to_node == self.client.gossip.idx_node:
                        update_steps = self.client.get_biased_parameter_step(step, inputs, idxs, create_graph=create_graph)
                        params_send = [p * x + s for x, s in zip(self.client.params_biased, update_steps)]
                    else:
                        params_send = [p * x for x in self.client.params_biased]

                    if no_weight_update:
                        weight_send = torch.zeros_like(self.client.gossip.weight)
                    else:
                        weight_send = p * self.client.gossip.weight

                    yield self.client.gossip.idx_node, (params_send, weight_send)

                else:
                    yield self.client.gossip.idx_node, None


class NodeTrainLogger(object):
    def __init__(self, client, n_steps, loader, names_eval_metric=None):
        self.names_train_values = [
            'loss',
            'weight',
            'lr',
            'deg_out',
            # 'norm_vs_center',
            # 'norm_mean_vs_others',
            # 'norm_std_vs_others',
            # 'weighted_squared_norm_mean_vs_all',
            # 'weighted_squared_norm_std_vs_all',
        ]
        if names_eval_metric is None:
            self.names_metric = self.names_train_values
        else:
            self.names_metric = self.names_train_values + list(names_eval_metric)

        self.names_eval_metric = names_eval_metric
        self.client = client
        self.loader = loader
        self.n_steps = n_steps
        self.log_train_step = None

        self.init_log_train_step()

    def init_log_train_step(self):
        self.log_train_step = {metric: None for metric in self.names_metric}

    def record_step_value(self, d):
        self.log_train_step.update(d)

    def record_step_metric(self):
        if self.names_eval_metric is not None:
            for name_metric in self.names_eval_metric:
                val = self.client.eval_metric(name_metric, self.loader).item()
                self.log_train_step[name_metric] = val


class CentralTrainLogger(object):
    def __init__(self, n_steps, train_loggers, names_eval_metric=None, _run=None):
        self.names_train_values = [
            'loss',
            'weight',
            'lr',
            'deg_out',
            # 'norm_vs_center',
            # 'norm_mean_vs_others',
            # 'norm_std_vs_others',
            # 'weighted_squared_norm_mean_vs_all',
            # 'weighted_squared_norm_std_vs_all',
        ]
        if names_eval_metric is None:
            self.names_metric = self.names_train_values
        else:
            self.names_metric = self.names_train_values + list(names_eval_metric)

        self.n_nodes = len(train_loggers)
        self.n_steps = n_steps
        self._run = _run
        self.train_loggers = train_loggers

        # define log formatters
        self.d_format = OrderedDict((name, '{}: {:.6f}') for name in self.names_metric)

        self.log_train = {
            'mean': {metric: [] for metric in self.names_metric},
            'std': {metric: [] for metric in self.names_metric},
            'node': [{metric: [] for metric in self.names_metric} for _ in range(self.n_nodes)],
        }

    def log_step(self, step):
        # # TODO(future): for debug
        # # check degrees of consensus
        # innerparameters_nodes = [train_logger.client.innerparameters for train_logger in self.train_loggers]
        # innerparameters_mean = [torch.mean(torch.vstack(param_biased_nodes), dim=0) for param_biased_nodes in zip(*innerparameters_nodes)]
        # for train_logger in self.train_loggers:
        #     norm = compute_parameter_norm(train_logger.client.innerparameters, innerparameters_mean)
        #     train_logger.record_step_value({'norm_vs_center': norm.item()})
        #
        # # TODO(future): for debug
        # # compute the second term of the potential function of DSGD
        # for train_logger in self.train_loggers:
        #     norms = []
        #     for idx_other_node, innerparameters_other in enumerate(innerparameters_nodes):
        #         if idx_other_node != train_logger.client.gossip.idx_node:
        #             norm = compute_parameter_norm(train_logger.client.innerparameters, innerparameters_other).item()
        #             norms.append(norm)
        #     train_logger.record_step_value({'norm_mean_vs_others': np.mean(norms)})
        #     train_logger.record_step_value({'norm_std_vs_others': np.std(norms)})
        #
        # # TODO(future): for debug
        # # compute the second term of the potential function of DSGD
        # for train_logger in self.train_loggers:
        #     squared_norms = []
        #     for idx_other_node, (train_logger_other, innerparameters_other) in enumerate(zip(self.train_loggers, innerparameters_nodes)):
        #         p = train_logger_other.client.gossip.get_p_vec(are_connected=torch.ones(self.n_nodes))[train_logger.client.gossip.idx_node]
        #         squared_norm = compute_parameter_norm(train_logger.client.innerparameters, innerparameters_other, squared=True)
        #         weighted_squared_norm = p * squared_norm
        #         squared_norms.append(weighted_squared_norm.item())
        #     train_logger.record_step_value({'weighted_squared_norm_mean_vs_all': np.mean(squared_norms)})
        #     train_logger.record_step_value({'weighted_squared_norm_std_vs_all': np.std(squared_norms)})

        log_train_step_nodes = []
        for train_logger in self.train_loggers:
            log_train_step_nodes.append(deepcopy(train_logger.log_train_step))
            train_logger.init_log_train_step()

        self.log_train_step = {
            'mean': {metric: None for metric in self.names_metric},
            'std': {metric: None for metric in self.names_metric},
            'node': log_train_step_nodes,
        }

        # trace train and valid score
        for idx_node in range(self.n_nodes):
            # compute mean over the nodes
            for name_metric in self.names_metric:
                vals = []
                for idx_node in range(self.n_nodes):
                    if self.log_train_step['node'][idx_node][name_metric] is not None:
                        vals.append(self.log_train_step['node'][idx_node][name_metric])
                if len(vals) > 0:
                    self.log_train_step['mean'][name_metric] = np.mean(vals)
                    self.log_train_step['std'][name_metric] = np.std(vals)

        # print stdout log
        log_str = f'step: [{step:0{len(str(self.n_steps))}d}/{self.n_steps}]'
        for name_metric in self.names_metric:
            if self.log_train_step['mean'][name_metric] is not None:
                log_str += ', ' + self.d_format[name_metric].format(name_metric, self.log_train_step['mean'][name_metric])
        print(log_str)

        # log for sacred
        if self._run is not None:
            for name_metric in self.names_metric:
                # for idx_node in range(self.n_nodes):
                #     if self.log_train_step['node'][idx_node][name_metric] is not None:
                #         self._run.log_scalar(f'{name_metric}_node_{idx_node:02d}', self.log_train_step['node'][idx_node][name_metric], step)
                if self.log_train_step['mean'][name_metric] is not None:
                    self._run.log_scalar(f'{name_metric}_mean', self.log_train_step['mean'][name_metric], step)
                    self._run.log_scalar(f'{name_metric}_std', self.log_train_step['std'][name_metric], step)

        # append step log to log_train
        for name_metric in self.names_metric:
            for idx_node in range(self.n_nodes):
                self.log_train['node'][idx_node][name_metric].append(self.log_train_step['node'][idx_node][name_metric])
            self.log_train['mean'][name_metric].append(self.log_train_step['mean'][name_metric])
            self.log_train['std'][name_metric].append(self.log_train_step['std'][name_metric])


class LRSchedulerBase(object):
    def __init__(self, lr, **kwargs):
        self.lr = lr


class LRSchedulerConst(LRSchedulerBase):
    def __call__(self, step):
        return self.lr


class LRSchedulerMulti(LRSchedulerBase):
    def __init__(self, lr, n_steps, milestones, gamma=0.1, **kwargs):
        super().__init__(lr, **kwargs)
        assert all([n_steps > m for m in milestones]), (n_steps, milestones)

        self.milestones = Counter(milestones)
        self.gamma = gamma

    def __call__(self, step):
        return self._get_closed_form_lr(step)

    def _get_closed_form_lr(self, step):
        milestones = list(sorted(self.milestones.elements()))
        return self.lr * self.gamma ** bisect_right(milestones, step)


D_LR_SCHEDULER_NODE = {
    NamesLrScheduler.CONST: LRSchedulerConst,
    NamesLrScheduler.MULTI_STEP: LRSchedulerMulti,
}
