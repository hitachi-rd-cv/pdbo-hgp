from copy import deepcopy

import torch

from constants import ModesGossip, TypesDevice
from lib_task.common import load_clients, load_graph
from module_torch.communication_graph import GraphErdosRenyi, GraphFullyConnected


def update_kwargs_build(
        n_nodes,
        datasets_train,
        kwargs_build_base,
        mode_graph,
        state_dict_graph,
):
    graph = load_graph(mode_graph, n_nodes=n_nodes, state_dict=state_dict_graph)
    kwargs_build_nodes = []
    for idx_node, dataset_train in enumerate(datasets_train):
        kwargs_build = deepcopy(kwargs_build_base)
        kwargs_build['kwargs_hyperparam']['n_classes'] = len(dataset_train.classes)
        kwargs_build['kwargs_hyperparam']['n_samples'] = len(dataset_train)
        if not 'kwargs_gossip' in kwargs_build:
            kwargs_build['kwargs_gossip'] = {}
        if isinstance(graph, (GraphErdosRenyi, GraphFullyConnected)):
            if kwargs_build["mode_gossip"] == ModesGossip.WEIGHT_GIVEN:
                kwargs_build['kwargs_gossip']['p_vec'] = graph.mixing_matrix.numpy()[:, idx_node]

        kwargs_build_nodes.append(kwargs_build)

    return kwargs_build_nodes


def build_nodes(
        n_nodes,
        name_model,
        kwargs_build_nodes,
        use_cuda,
        seed,
        kwargs_init_hparams=None,
        _run=None
):
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)

    # build network
    # According to assran2019,
    # "Each node i must only know/choose its outgoing mixing weights, which correspond to the i-th column of P(k). Each node can consequently choose its mixing weights independently of the other nodes in the network".

    models = load_clients(
        name_model=name_model,
        kwargs_build_nodes=kwargs_build_nodes,
        n_nodes=n_nodes,
        device=device,
        seed=seed,
    )

    if kwargs_init_hparams is not None:
        for model in models:
            model.hyperparameter_module.init_hyperparameters(**kwargs_init_hparams)

    return [m.cpu().state_dict() for m in models]


def build_graph(n_nodes, mode_graph, kwargs_init_graph, use_cuda):
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)
    graph = load_graph(mode_graph=mode_graph, n_nodes=n_nodes, device=device)
    graph.init_params(**kwargs_init_graph)

    return graph.cpu().state_dict()
