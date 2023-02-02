import torch
from tqdm import tqdm

from constants import ModesSGP
from lib_task.common import compute_expected_p_vec


def concat_params_and_weights(param_cat, weights, n_nodes):
    param_opt_nodes = torch.reshape(param_cat, (n_nodes, -1))
    return torch.cat((param_opt_nodes, torch.reshape(weights, (-1, 1))), dim=1).reshape(-1)


def split_params_and_weights(param_weight_cat, n_nodes):
    param_weight_nodes = torch.reshape(param_weight_cat, (n_nodes, -1))
    weights = param_weight_nodes[:, -1]
    params_biased = param_weight_nodes[:, :-1]
    return torch.flatten(params_biased), weights


def kron_matrix_vector_prod(v, A, B=None):
    # kron(A, B) @ v = vec(B @ vec^-1(v) @ A.T)
    # V =  vec^-1(v)
    # V is transposed between numpy implementation and math in the paper
    V = torch.reshape(v, (A.shape[1], -1))
    if B is None:  # B is assumed to be identity matrix
        return torch.flatten(A @ V)
    else:
        raise NotImplementedError


def kron_diag_matrix_vector_prod(v, u):
    # kron(diag(u), I) @ v = vec(vec^-1(v) * u)
    # V =  vec^-1(v)
    V = torch.reshape(v, (u.shape[0], -1))
    return torch.flatten((V.T.contiguous() * u).T.contiguous())


def compute_expected_p_mat(graph, models):
    # sample edges
    p_vecs = []
    for idx_from, model in tqdm(enumerate(models)):
        p_vec = compute_expected_p_vec(model, graph)
        p_vecs.append(p_vec)
    return torch.vstack(p_vecs).T.contiguous()

def get_current_p_mat(graph, models):
    # sample edges
    p_vecs = []
    for idx_from, model in enumerate(models):
        are_connected = graph.are_connected(idx_from)
        p_vec = model.get_p_vec(are_connected)
        p_vecs.append(p_vec)
    return torch.vstack(p_vecs).T.contiguous()

def compute_loss_sum(clients, inputs_nodes, idxs_sample_nodes=None, to_node=None, train_logger=None):
    loss_sum = 0.
    for client, inputs, idxs_sample in zip(clients, inputs_nodes, idxs_sample_nodes):
        kwargs_loss = client.get_kwargs_loss(inputs, idxs_sample)
        loss = client.model.loss(inputs, **kwargs_loss)
        loss_sum += loss
        if train_logger is not None:
            train_logger.record_step_value(client.gossip.idx_node, 'loss', loss.item())

    return loss_sum


def update_param_cat_biased(clients, param_cat_biased, param_cat_debiased, inputs_nodes, idxs_sample_nodes, P, lr_cat, create_graph, mode_sgp, train_logger=None):
    if train_logger is not None:
        for idx_node, lr in enumerate(lr_cat):
            train_logger.record_step_value(idx_node, 'lr', lr.item())

    loss_sum = compute_loss_sum(clients, inputs_nodes, idxs_sample_nodes=idxs_sample_nodes, train_logger=train_logger)
    # param update
    grad = torch.autograd.grad(loss_sum, param_cat_debiased, create_graph=create_graph)[0]
    if mode_sgp == ModesSGP.ASSRAN:
        # x = P (x - A * g (z, lambda))
        param_cat_biased_updated = kron_matrix_vector_prod(param_cat_biased - kron_diag_matrix_vector_prod(grad, lr_cat), P)
    elif mode_sgp == ModesSGP.NEDIC:
        # x = P x - A * g (z, lambda)
        param_cat_biased_updated = kron_matrix_vector_prod(param_cat_biased, P) - kron_diag_matrix_vector_prod(grad, lr_cat)
    else:
        raise ValueError(mode_sgp)

    return param_cat_biased_updated
