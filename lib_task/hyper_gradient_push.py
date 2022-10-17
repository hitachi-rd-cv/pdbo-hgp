import random

import numpy as np
import torch
from tqdm import tqdm

from constants import KeysOptionHGP, KeysOptionTrainSig, KeysOptionEval, KeysOptionHGPInsig, ModesHGPUpdate, \
    NamesHGPMetric
from lib_task.concat_hgp import compute_expected_p_mat
from lib_task.stochastic_gradient_push import D_LR_SCHEDULER_NODE
from module_torch.hgp import HyperGradEstimatorHGP


def hyper_gradient_push(clients, graph, loaders_train, loaders_valid, option_eval_metric, n_steps, batch_sizes, lrs,
                        option_train_significant, option_hgp=None, seed=None, hypergrads_nodes_true=None,
                        true_backward_mode=False,
                        option_hgp_insignificant=None, save_intermediate_hypergradients=False, _run=None):
    # argument check
    if true_backward_mode:
        assert option_hgp[
            KeysOptionHGP.USE_TRUE_EXPECTED_EDGES], f'"true_backward_mode" option is only available with {KeysOptionHGP.USE_TRUE_EXPECTED_EDGES}'

    # fix seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    estimators = []
    for client, loader_train, loader_valid, batch_size, lr in zip(clients, loaders_train, loaders_valid, batch_sizes,
                                                                  lrs):
        estimator = HyperGradEstimatorHGP(
            client=client,
            loader_train=loader_train,
            loader_valid=loader_valid,
            lr_scheduler=D_LR_SCHEDULER_NODE[option_train_significant[KeysOptionTrainSig.LR_SCHEDULER]](lr,
                                                                                                        n_steps=n_steps,
                                                                                                        **
                                                                                                        option_train_significant[
                                                                                                            KeysOptionTrainSig.KWARGS_LR_SCHEDULER]),
            dumping=option_hgp[KeysOptionHGP.DUMPING],
            alpha_v=option_hgp[KeysOptionHGP.ALPHA_V],
            alpha_w=option_hgp[KeysOptionHGP.ALPHA_W],
        )
        estimators.append(estimator)

    print("Initializing hgp estimators ...")
    for esitmator in tqdm(estimators):
        esitmator.init_hypergrads(option_eval_metric[KeysOptionEval.NAME])

    logger = HGPLogger(
        estimators=estimators,
        _run=_run,
        hypergrads_nodes_true=hypergrads_nodes_true,
        hgp_metrics=option_hgp_insignificant[KeysOptionHGPInsig.NAMES_METRIC_LOG],
        save_intermediate_hypergradients=save_intermediate_hypergradients,
    )

    if option_hgp[KeysOptionHGP.USE_TRUE_EXPECTED_EDGES]:
        p_mat = compute_expected_p_mat(graph, clients)
        if true_backward_mode:
            # transposing rate_connected_mat results in passing expected out_neighbors to clients, letting them compute true backward mode
            adj_mat = graph.rate_connected_mat.T
        else:
            adj_mat = graph.rate_connected_mat
    else:
        p_mat = None
        adj_mat = None

    print('Start implicit approximation ...')
    neumann_depth = option_hgp[KeysOptionHGP.DEPTH]
    for m in tqdm(range(neumann_depth)):
        names_hypergrad_to_update = get_names_hypergrad_to_update(option_hgp[KeysOptionHGP.MODE_UPDATE], m, neumann_depth)

        # sample edges
        graph.sample()
        msg_generators = []
        for estimator in estimators:
            are_connected_hgp = graph.are_connected(estimator.idx_node)
            if true_backward_mode:
                # make graph bidirected only for evaluating estimation error
                are_connected_hgp = torch.hstack([graph.are_connected(idx_from)[estimator.idx_node] for idx_from in range(len(estimators))])
            msg_generators.append(estimator.get_hgp_msg_generator(are_connected_hgp))

        for from_node, estimator in enumerate(estimators):
            are_connected_sgp = graph.are_connected(estimator.idx_node)
            if option_hgp[KeysOptionHGP.USE_TRUE_EXPECTED_EDGES]:
                sgp_msgs = estimator.get_sgp_msg_generator(
                    are_connected_sgp,
                    step=n_steps,
                    no_weight_update=option_train_significant[KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT],
                    true_p_vec_expected=p_mat[:, from_node],
                    true_are_in_neighbors_expected=adj_mat[from_node, :])
            else:
                sgp_msgs = estimator.get_sgp_msg_generator(are_connected_sgp, step=n_steps, no_weight_update=option_train_significant[KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT])
            for msg_generator in msg_generators:
                sgp_msg = next(sgp_msgs)
                hgp_msg = next(msg_generator)
                estimator.mix_hgp_partial(hgp_msg, sgp_msg, values_update=names_hypergrad_to_update)
            sgp_msgs.close()

        for estimator in estimators:
            estimator.step(
                values_update=names_hypergrad_to_update
            )

        # log norms of hypergradient
        logger.log(m)

    # reshape hypergrads to original tensor shapes
    hypergrads_nodes = [[v.clone().detach() for v in e.vs] for e in estimators]

    return hypergrads_nodes, estimators, logger.hypergrads_nodes_steps


def get_names_hypergrad_to_update(mode, m, neumann_depth):
    if mode == ModesHGPUpdate.SIMULTANEOUS:
        names_hypergrad_to_update = ('u', 'v')
    elif mode == ModesHGPUpdate.ALT_U_V:
        if m % 2 == 0:
            names_hypergrad_to_update = ('u',)
        else:
            names_hypergrad_to_update = ('v',)
    elif mode == ModesHGPUpdate.ALT_V_U:
        if m % 2 == 0:
            names_hypergrad_to_update = ('v',)
        else:
            names_hypergrad_to_update = ('u',)
    elif mode == ModesHGPUpdate.U_TO_V:
        if m < neumann_depth / 2:
            names_hypergrad_to_update = ('u',)
        else:
            names_hypergrad_to_update = ('v',)
    else:
        raise ValueError(mode)
    return names_hypergrad_to_update


class HGPLogger:
    def __init__(self, estimators, _run, hgp_metrics, hypergrads_nodes_true=None, save_intermediate_hypergradients=False):
        self.estimators = estimators
        self.n_nodes = len(estimators)
        self._run = _run
        self.hypergrads_nodes_true = hypergrads_nodes_true
        self.hgp_metrics = hgp_metrics
        if save_intermediate_hypergradients:
            self.hypergrads_nodes_steps = []
        else:
            self.hypergrads_nodes_steps = None

    def log(self, step):
        with torch.no_grad():
            for name_metric in self.hgp_metrics:
                sum_squared_norm = 0.
                for idx_node, estimator in enumerate(self.estimators):
                    if self.hypergrads_nodes_true is None:
                        squared_norm = self.get_hypergrad_squared_norm(estimator, name_metric)
                    else:
                        squared_norm = self.get_hypergrad_squared_norm(estimator, name_metric, self.hypergrads_nodes_true[idx_node])
                    sum_squared_norm += squared_norm
                norm = sum_squared_norm ** 0.5

                if self._run is not None:
                    self._run.log_scalar(name_metric, norm.item(), step)

        if self.hypergrads_nodes_steps is not None:
            hypergrads_nodes = []
            for estimator in self.estimators:
                hypergrads = [v.clone().detach() for v in estimator.vs]
                hypergrads_nodes.append(hypergrads)
            self.hypergrads_nodes_steps.append(hypergrads_nodes)

    @staticmethod
    def get_hypergrad_squared_norm(estimator, name_metric, hypergrads_nodes_true=None):
        if name_metric == NamesHGPMetric.U_NORM:
            squared_norm_u_x = 0.
            for u_x in estimator.us_x:
                squared_norm_u_x += torch.sum(u_x ** 2)
            squared_norm_u_w = estimator.u_w ** 2
            return squared_norm_u_x + squared_norm_u_w
        elif name_metric == NamesHGPMetric.V_NORM:
            squared_norm_v = 0.
            for v in estimator.vs:
                squared_norm_v += torch.sum(v ** 2)
            return squared_norm_v
        elif name_metric == NamesHGPMetric.V_DIFF_NORM:
            assert hypergrads_nodes_true is not None, hypergrads_nodes_true
            squared_norm_v = 0.
            for idx_v, v in enumerate(estimator.vs):
                squared_norm_v += torch.sum((v - hypergrads_nodes_true[idx_v]) ** 2)
            return squared_norm_v
        else:
            raise ValueError(name_metric)
