import itertools

import numpy as np
import torch

from constants import KeysOptionEval, KeysOptionHGP
from lib_common.torch.autograd import myjacobian
from lib_task.common import assign_concatenated_parameters, reshape_hypergrads_cat
from lib_task.concat_hgp import compute_loss_sum
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
    x_nodes = [torch.hstack([torch.flatten(x.detach()) for x in m.innerparameters]) for m in clients]
    lambda_nodes = [torch.hstack([torch.flatten(h.detach()) for h in m.hyperparameters]) for m in clients]
    weight_nodes = [m.gossip.weight.detach() for m in clients]

    # check consensus
    for x_node in x_nodes:
        assert torch.allclose(x_nodes[0], x_node), n_nodes

    # check identitiy of two implicit differentiaitons

    # concat values
    lambda_cat = torch.hstack([torch.hstack([torch.flatten(h) for h in hyperparameters]) for hyperparameters in lambda_nodes])
    weights = torch.hstack(weight_nodes)
    x_cat = torch.hstack([param for param in list(itertools.chain.from_iterable(x_nodes))])

    # make parameters for autograd
    x_cat = torch.nn.Parameter(x_cat.clone().detach())
    lambda_cat = torch.nn.Parameter(lambda_cat.clone().detach())

    # compute grad u_0 = F(lambda) vs param_weight_cat_opt
    assign_concatenated_parameters(clients, param_cat_debiased=x_cat, hyperparam_cat=lambda_cat)

    val_eval_mean = 0.
    for client, loader_valid in zip(clients, loaders_valid):
        val_eval_mean += client.model._eval_metric(option_eval_metric[KeysOptionEval.NAME], loader_valid) / n_nodes
    C_x, C_lambda = torch.autograd.grad(val_eval_mean, (x_cat, lambda_cat), allow_unused=True)
    if C_lambda is None:
        C_lambda = torch.zeros_like(lambda_cat)

    print('Start implicit approximation ...')
    # to unsure gradient and graph is freed
    x_cat = torch.nn.Parameter(x_cat.clone().detach())
    lambda_cat = torch.nn.Parameter(lambda_cat.clone().detach())
    assign_concatenated_parameters(clients, param_cat_debiased=x_cat, hyperparam_cat=lambda_cat)

    # load train data
    iters_train = [iter(loader) for loader in loaders_train]
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

    # compute loss
    loss_sum = compute_loss_sum(clients, inputs_nodes, idxs_sample_nodes=idxs_sample_nodes)

    # compute grad
    grad = torch.autograd.grad(loss_sum, x_cat, create_graph=True)[0]

    # compute jacobians
    A, B = myjacobian(grad, (x_cat, lambda_cat))
    len_x = len(x_nodes[0])
    len_lambda = len(lambda_nodes[0])
    A_blocks = [torch.split(A_col, split_size_or_sections=len_x, dim=0)[i] for i, A_col in enumerate(torch.split(A, split_size_or_sections=len_x, dim=1))]
    B_blocks = [torch.split(B_col, split_size_or_sections=len_lambda, dim=0)[i] for i, B_col in enumerate(torch.split(B, split_size_or_sections=len_x, dim=1))]

    # true weight and consensus
    I = torch.eye(len_x, dtype=A.dtype, device=device)
    I_kron = torch.block_diag(*([I] * n_nodes))
    ones = torch.ones(n_nodes, dtype=A.dtype, device=device)
    alpha = option_hgp[KeysOptionHGP.DUMPING]

    # a * 1.T @ diag(∇f)
    B_c = (alpha / n_nodes) * torch.vstack([torch.kron(ones, B_block) for B_block in B_blocks])
    # I - (1/n) * 1.T @ diag(I - a * ∇_x f))
    A_c = I_kron - (1 / n_nodes) * torch.vstack([torch.kron(ones, I - alpha * A_block) for A_block in A_blocks])

    # - a * (1/n) * 1.T @ diag(∇_lambda f) @ (I - (1/n) * (1.T @ diag(I - a * ∇_x f)))^(-1) @ C_x + C_lambda
    hypergrads_cat_push_true_W_consensus = - B_c.cpu().numpy() @ np.linalg.inv(A_c.cpu().numpy()) @ C_x.cpu().numpy() + C_lambda.cpu().numpy()
    hypergrads_cat_push_true_W_consensus = torch.tensor(hypergrads_cat_push_true_W_consensus, dtype=A.dtype, device=device)

    # reshape hypergrads to original tensor shapes
    hypergrads_nodes = reshape_hypergrads_cat(clients, hypergrads_cat_push_true_W_consensus, clone=True, detach=True)

    # empty estimators to return
    estimators = [HyperGradEstimatorDummy(client) for client in clients]
    return hypergrads_nodes, estimators, None
