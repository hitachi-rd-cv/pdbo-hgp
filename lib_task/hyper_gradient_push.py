import random

import numpy as np
import torch
from tqdm import tqdm

from constants import KeysOptionHGP, KeysOptionEval, KeysOptionHGPInsig, \
    NamesHGPMetric
from lib_task.concat_hgp import compute_expected_p_mat
from module_torch.hgp import HyperGradEstimator


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
    for client, loader_train, loader_valid, batch_size, lr in zip(clients, loaders_train, loaders_valid, batch_sizes, lrs):
        estimator = HyperGradEstimator(
            client=client,
            loader_train=loader_train,
            loader_valid=loader_valid,
            dumping=option_hgp[KeysOptionHGP.DUMPING],
            use_iid_samples=option_hgp[KeysOptionHGP.INDEPENDENT_JACOBS]
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

    p_vecs_true = [None] * len(clients)
    weights_true = [None] * len(clients)

    if option_hgp[KeysOptionHGP.USE_TRUE_EXPECTED_EDGES] or option_hgp[KeysOptionHGP.USE_TRUE_DEBIAS_WEIGHTS]:
        # get true expected mixing matrix
        p_mat = compute_expected_p_mat(graph, clients)
        # get true debias weight
        eig_vec = torch.linalg.eig(p_mat)[1][:, 0].real

        if option_hgp[KeysOptionHGP.USE_TRUE_EXPECTED_EDGES]:
            p_vecs_true = p_mat.T

        if option_hgp[KeysOptionHGP.USE_TRUE_DEBIAS_WEIGHTS]:
            weights_true = len(clients) * eig_vec / eig_vec.sum()

    print('Start implicit approximation ...')
    neumann_depth = option_hgp[KeysOptionHGP.DEPTH]
    for m in tqdm(range(neumann_depth)):
        # init w
        for estimator in estimators:
            estimator.init_s()

        if option_hgp[KeysOptionHGP.INIT_DEBIAS_WEIGHT]:
            for esitmator in estimators:
                esitmator.client.gossip.initialize_weight()

        for s in range(option_hgp[KeysOptionHGP.N_PUSH]):
            # sample edges
            graph.sample()

            for estimator, p_vec_true in zip(estimators, p_vecs_true):
                are_connected_hgp = graph.are_connected(estimator.idx_node)
                msg_generator = estimator.get_hgp_msg_generator(are_connected_hgp, p_vec_true=p_vec_true)

                # send hgp messages to the neighbors
                for receiver, msg in zip(estimators, msg_generator):
                    receiver.mix_s(msg)

            # update w
            for estimator in estimators:
                estimator.step_s()

        # update u and v
        for estimator, weight_true in zip(estimators, weights_true):
            estimator.step_m(weight_true=weight_true)

        # log norms of hypergradient
        logger.log(m)

    # reshape hypergrads to original tensor shapes
    hypergrads_nodes = [[v.clone().detach() for v in e.vs] for e in estimators]

    return hypergrads_nodes, estimators, logger.hypergrads_nodes_steps



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
            for u_x in estimator.us:
                squared_norm_u_x += torch.sum(u_x ** 2)
            return squared_norm_u_x
        elif name_metric == NamesHGPMetric.W_NORM:
            squared_norm_w_x = 0.
            for w_x in estimator.ws:
                squared_norm_w_x += torch.sum(w_x ** 2)
            return squared_norm_w_x
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
