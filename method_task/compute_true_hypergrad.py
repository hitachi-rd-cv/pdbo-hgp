import itertools

import numpy as np
import torch

from constants import KeysOptionTrainSig, KeysOptionEval
from lib_common.torch.autograd import myjacobian
from lib_task.common import assign_concatenated_parameters, reshape_hypergrads_cat
from lib_task.concat_hgp import concat_params_and_weights, split_params_and_weights, kron_diag_matrix_vector_prod, \
    compute_expected_p_mat, update_param_cat_biased
from lib_task.modules_sgp import D_LR_SCHEDULER_NODE
from module_torch.hgp import HyperGradEstimatorDummy


def compute_true_hypergrad(clients, graph, loaders_train, loaders_valid, option_eval_metric, n_steps, batch_sizes, lrs, option_train_significant, option_train_insignificant,
                           compute_hg=False, option_hgp=None, shuffle_train=True, seed=None, option_hgp_insignificant=None, _run=None):
    # set train mode
    for m in clients:
        m.train()

    # recover model and hyperparameter class
    device = clients[0].get_device_of_param()
    n_nodes = len(clients)

    # make concatenated matrix and vectors over the nodes
    x_nodes = [[tensor.detach() for tensor in m.params_biased] for m in clients]
    lambda_nodes = [[h.detach() for h in m.hyperparameters] for m in clients]
    weight_nodes = [m.gossip.weight.detach() for m in clients]

    # concat values
    lambda_cat = torch.hstack([torch.hstack([torch.flatten(h) for h in hyperparameters]) for hyperparameters in lambda_nodes])
    weights = torch.hstack(weight_nodes)
    x_cat = torch.hstack([torch.flatten(param) for param in list(itertools.chain.from_iterable(x_nodes))])

    # set scheduler
    lrs = torch.tensor(lrs).to(device)
    Scheduler = D_LR_SCHEDULER_NODE[option_train_significant[KeysOptionTrainSig.LR_SCHEDULER]]
    lrs_latest = torch.zeros_like(lrs)
    for i, lr in enumerate(lrs):
        lrs_latest[i] = Scheduler(lr, n_steps=n_steps, **option_train_significant[KeysOptionTrainSig.KWARGS_LR_SCHEDULER])(n_steps)

    # start training
    iters_train = [iter(loader) for loader in loaders_train]

    # concat param and weight
    if option_train_significant[KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT]:
        y_cat = torch.nn.Parameter(x_cat)
    else:
        y_cat = concat_params_and_weights(x_cat, weights, n_nodes)

    # convert to parameters for autograd
    y_cat = torch.nn.Parameter(y_cat.clone().detach())
    lambda_cat = torch.nn.Parameter(lambda_cat.clone().detach())

    # compute grad u_0 = F(lambda) vs param_weight_cat_opt
    if option_train_significant[KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT]:
        z_cat = y_cat
    else:
        x_cat, weights = split_params_and_weights(y_cat, n_nodes)
        z_cat = kron_diag_matrix_vector_prod(x_cat, 1 / weights)
    assign_concatenated_parameters(clients, param_cat_debiased=z_cat, hyperparam_cat=lambda_cat)

    val_eval_mean = 0.
    for client, loader_valid in zip(clients, loaders_valid):
        val_eval_mean += client.model._eval_metric(option_eval_metric[KeysOptionEval.NAME], loader_valid) / n_nodes
    C_x, C_y = torch.autograd.grad(val_eval_mean, (y_cat, lambda_cat), allow_unused=True)
    if C_y is None:
        C_y = torch.zeros_like(lambda_cat)

    print('Start implicit approximation ...')
    # to unsure gradient and graph is freed
    y_cat = torch.nn.Parameter(y_cat.clone().detach())
    lambda_cat = torch.nn.Parameter(lambda_cat.clone().detach())

    # assign parameters on models
    if option_train_significant[KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT]:
        z_cat = y_cat
        x_cat = y_cat
    else:
        x_cat, weights = split_params_and_weights(y_cat, n_nodes)
        z_cat = kron_diag_matrix_vector_prod(x_cat, 1 / weights)

    assign_concatenated_parameters(clients, param_cat_debiased=z_cat, hyperparam_cat=lambda_cat)

    P = compute_expected_p_mat(graph, models=clients)

    idxs_sample_nodes = []
    inputs_nodes = []
    for idx_node in range(n_nodes):
        try:
            *inputs, idxs = next(iters_train[idx_node])
        except StopIteration:
            iters_train[idx_node] = iter(loaders_train[idx_node])  # initialize iteration
            *inputs, idxs = next(iters_train[idx_node])
        inputs_nodes.append(inputs)
        idxs_sample_nodes.append(idxs)

    x_cat_updated = update_param_cat_biased(clients, x_cat, z_cat, inputs_nodes, idxs_sample_nodes, P, lrs_latest,
                                            mode_sgp=option_train_significant[KeysOptionTrainSig.MODE_SGP],
                                            create_graph=True)

    if option_train_significant[KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT]:
        y_cat_updated = x_cat_updated
    else:
        weights_updated = P @ weights
        y_cat_updated = concat_params_and_weights(x_cat_updated, weights_updated, n_nodes)

    # compute jacobian y+1 vs lambda
    A, B = myjacobian(y_cat_updated, (y_cat, lambda_cat))

    # hypergrad = B (I - A)^{-1} C_x + C_y
    I = torch.eye(len(y_cat), dtype=A.dtype, device=device)
    # hypergrads_cat = B @ torch.linalg.inv(I - A) @ C_x + C_y # TODO(future): Error raises with torch's inv with no reason: RuntimeError: torch.linalg.inv: The diagonal element 2358 is zero, the inversion could not be completed because the input matrix is singular.
    hypergrads_cat_tmp = B.cpu().numpy() @ np.linalg.inv(I.cpu().numpy() - A.cpu().numpy()) @ C_x.cpu().numpy() + C_y.cpu().numpy()
    hypergrads_cat = torch.tensor(hypergrads_cat_tmp, dtype=A.dtype, device=device)

    # reshape hypergrads to original tensor shapes
    hypergrads_nodes = reshape_hypergrads_cat(clients, hypergrads_cat, clone=True, detach=True)

    # empty estimators to return
    estimators = [HyperGradEstimatorDummy(client) for client in clients]
    return hypergrads_nodes, estimators, None
