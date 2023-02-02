import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from module_torch.client import Client
from module_torch.communication_graph import D_DYNAMIC_GRAPH


def load_clients(name_model, kwargs_build_nodes, n_nodes, state_dicts=None, device=None, seed=None):
    clients = []
    for idx_node, kwargs_build in enumerate(kwargs_build_nodes):
        if state_dicts is None:
            state_dict = None
        else:
            state_dict = state_dicts[idx_node]
        client = load_client(device, idx_node, kwargs_build, n_nodes, name_model, seed, state_dict)
        clients.append(client)
    return clients


def load_client(device, idx_node, kwargs_build, n_nodes, name_model, seed=None, state_dict=None):
    if seed is not None:
        torch.manual_seed(seed)  # Followed FedEM implementation

    client = Client(
        n_nodes=n_nodes,
        idx_node=idx_node,
        name_model=name_model,
        **kwargs_build,
    )
    if state_dict is not None:
        client.load_state_dict(state_dict)
    if device is not None:
        client = client.to(device)
    return client

def load_graph(mode_graph, n_nodes, state_dict=None, device=None):
    graph = D_DYNAMIC_GRAPH[mode_graph](n_nodes)
    if state_dict is not None:
        graph.load_state_dict(state_dict)
    if device is not None:
        graph = graph.to(device)
    return graph


def get_loaders(batch_sizes, datasets, shuffle, drop_last):
    loaders = []
    for dataset, batch_size in zip(datasets, batch_sizes):
        drop_last_tmp = drop_last and (len(dataset) > batch_size)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last_tmp))
    return loaders

def compute_expected_p_vec(model, graph):
    device = model.get_device_of_param()
    p_vec_expectation = torch.zeros(graph.n_nodes, device=device, dtype=torch.float32)
    events = torch.tensor(np.array(list(itertools.product([0, 1], repeat=graph.n_nodes))), device=device, dtype=torch.float32)
    p_cum = 0.
    for event in events:
        prob_event = graph.compute_prob_event(event, model.gossip.idx_node)  # assuming gossip is time invariant expectation
        if prob_event > 0.:
            p_vec_event = model.get_p_vec(event)
            p_vec_expectation += p_vec_event * prob_event
        p_cum += prob_event
    # assert probabilistic constraint
    assert torch.isclose(p_cum, torch.tensor(1.)), p_cum
    # assert column stochastic
    assert torch.isclose(torch.sum(p_vec_expectation), torch.tensor(1.)), f"sum({p_vec_expectation}) = {torch.sum(p_vec_expectation)}"
    return p_vec_expectation


def get_state_dicts_from_concatenated_parameters(models, param_cat_debiased=None, param_cat_biased=None, hyperparam_cat=None, clone=False, detach=False):
    n_nodes = len(models)
    shapes_of_param_blocks = [p.shape for p in models[0].innerparameters]
    shapes_of_hyper_blocks = [p.shape for p in models[0].hyperparameters]
    n_elems_params = [np.prod(shape).astype(np.int) for shape in shapes_of_param_blocks]
    n_elems_hypers = [np.prod(shape).astype(np.int) for shape in shapes_of_hyper_blocks]
    n_params = np.sum(n_elems_params)
    n_hypers = np.sum(n_elems_hypers)

    state_dicts_partial = [{} for _ in range(n_nodes)]

    if param_cat_debiased is not None:
        debiased_params_nodes_out = reshape_global_level_vector_into_tensors(param_cat_debiased, ([n_params] * n_nodes, n_elems_params), shapes_of_param_blocks)
        for idx_node_i, params in enumerate(debiased_params_nodes_out):
            state_dict_partial = state_dicts_partial[idx_node_i]
            for idx_param, param in enumerate(params):
                if clone:
                    param = torch.clone(param)
                if detach:
                    param = torch.detach(param)
                state_dict_partial[f'innerparameters.{idx_param}'] = param

    if param_cat_biased is not None:
        biased_params_nodes_out = reshape_global_level_vector_into_tensors(param_cat_biased, ([n_params] * n_nodes, n_elems_params), shapes_of_param_blocks)
        for idx_node_i, biased_params in enumerate(biased_params_nodes_out):
            state_dict_partial = state_dicts_partial[idx_node_i]
            for idx_biased_param, biased_param in enumerate(biased_params):
                if clone:
                    biased_param = torch.clone(biased_param)
                if detach:
                    biased_param = torch.detach(biased_param)
                state_dict_partial[f'params_biased.{idx_biased_param}'] = biased_param


    if hyperparam_cat is not None:
        hyperparam_nodes_out = reshape_global_level_vector_into_tensors(hyperparam_cat, ([n_hypers] * n_nodes, n_elems_hypers), shapes_of_hyper_blocks)
        for idx_node_i, hparams in enumerate(hyperparam_nodes_out):
            state_dict_partial = state_dicts_partial[idx_node_i]
            for idx_hparam, hparam in enumerate(hparams):
                if clone:
                    hparam = torch.clone(hparam)
                if detach:
                    hparam = torch.detach(hparam)
                state_dict_partial[f'hyperparameters.{idx_hparam}'] = hparam


    return state_dicts_partial


def assign_concatenated_parameters(models, param_cat_debiased=None, param_cat_biased=None, hyperparam_cat=None, clone=False, detach=False):
    state_dicts = get_state_dicts_from_concatenated_parameters(models, param_cat_debiased, param_cat_biased, hyperparam_cat, clone=clone, detach=detach)
    assign_parameters(models, state_dicts)


def assign_parameters(models, state_dicts):
    for model, state_dict in zip(models, state_dicts):
        for name, param_updated in state_dict.items():
            param_ori = model.get_parameter(name)
            # this inplace op turns off requires_grad and removes grad_fn but keep itself to be nn.Parameter
            param_ori.detach_()
            # without detach_() copy_ accumulates the grad_fn tied to the attribute. copy_ passes the grad_fn of inputs not only the values.
            param_ori.copy_(param_updated)


def reshape_hypergrads_cat(models, hypergrads_cat, clone=False, detach=False):
    n_nodes = len(models)
    shapes_of_hyper_blocks = [p.shape for p in models[0].hyperparameters]
    n_elems_hypers = [np.prod(shape).astype(np.int) for shape in shapes_of_hyper_blocks]
    n_hypers = np.sum(n_elems_hypers)

    hypergrads_nodes = reshape_global_level_vector_into_tensors(hypergrads_cat, ([n_hypers] * n_nodes, n_elems_hypers), shapes_of_hyper_blocks)
    hypergrads_nodes_out = []
    for hypergrads in hypergrads_nodes:
        hypergrads_out = []
        for hypergrad in hypergrads:
            if clone:
                hypergrad = torch.clone(hypergrad)
            if detach:
                hypergrad = hypergrad.detach()
            hypergrads_out.append(hypergrad)
        hypergrads_nodes_out.append(hypergrads_out)
    return hypergrads_nodes_out


def reshape_global_level_vector_into_tensors(structure, n_elements_of_blocks_list, shapes_of_blocks):
    s_nodes_out_flat = decompose_into_blocks_recur(structure, n_elements_of_blocks_list)
    s_nodes_out = [[param_flat.reshape(shape) for param_flat, shape in zip(s, shapes_of_blocks)] for s in s_nodes_out_flat]
    return s_nodes_out

def decompose_into_blocks_recur(block, n_elements_of_blocks_list):
    n_elements = len(block)
    if not n_elements == sum(n_elements_of_blocks_list[0]):
        raise ValueError(f'len(block) != sum(n_elements_of_blocks_list[0]), {len(block)} != {sum(n_elements_of_blocks_list[0])}')

    decomposed_blocks_out = []
    idx_start = 0
    for n_elements_of_block in n_elements_of_blocks_list[0]:
        block_tmp = block[idx_start:idx_start + n_elements_of_block].T.contiguous()
        if len(n_elements_of_blocks_list) > 1:
            decomposed_blocks_tmp = decompose_into_blocks_recur(block_tmp, n_elements_of_blocks_list[1:])
            decomposed_blocks_out.append(decomposed_blocks_tmp)
        else:
            decomposed_blocks_out.append(block_tmp)
        idx_start += n_elements_of_block

    return decomposed_blocks_out