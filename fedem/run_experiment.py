# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of FedEM (https://github.com/omarfoq/FedEM), Copyright {copyright_marfoq}.
# The portions of the following codes are licensed under the {license_type_marfoq}.
# The full license text is available at ({license_url_marfoq}).
# ------------------------------------------------------------------------
import os
from argparse import Namespace

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fedem.utils.constants import CLIENT_TYPE, AGGREGATOR_TYPE
from fedem.utils.utils import get_learners_ensemble, get_client, get_loaders, get_aggregator
from lib_common.torch.etc import optimizer_to_, lr_scheduler_to_
from module_torch.communication_graph import GraphErdosRenyi, GraphFullyConnected
from module_torch.gossip_protocol import GossipGivenWeights
from module_torch.hyperparameter import D_HYPER_PARAMETERS


def init_clients_hgp(args_, logs_dir, train_iterators, val_iterators, test_iterators, kwargs_model=None):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        learners_ensemble = \
            get_learners_ensemble(
                n_learners=args_.n_learners,
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                kwargs_model=kwargs_model,
                weight_decay=args_.weight_decay,
            )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients
        )

        clients_.append(client)

    return clients_


def get_state_dicts_of_clients(aggregator, decentralized, clients):
    state_dicts_models = []
    state_dicts_optimizers = []
    state_dicts_lr_schedulers = []
    state_dicts_gossips = []
    for id_client, client in enumerate(clients):
        state_dicts_model = []
        state_dicts_optimizer = []
        state_dicts_lr_scheduler = []
        for learner in client.learners_ensemble.learners:
            # maps to gpu to free memories
            state_dicts_model.append(learner.model.cpu().state_dict())
            optimizer_to_(learner.optimizer, "cpu")
            state_dicts_optimizer.append(learner.optimizer.state_dict())
            lr_scheduler_to_(learner.lr_scheduler, "cpu")
            state_dicts_lr_scheduler.append(learner.lr_scheduler.state_dict())

        if decentralized:
            mean_adjacency_matrix = aggregator.sum_adjacency_matrix / aggregator.c_round
            mean_weighted_adjacency_matrix = aggregator.sum_weighted_adjacency_matrix / aggregator.c_round
        else:
            mean_adjacency_matrix = aggregator.sum_adjacency_matrix
            mean_weighted_adjacency_matrix = aggregator.sum_weighted_adjacency_matrix
        state_dicts_gossips.append({'are_in_neighbors_expected': mean_adjacency_matrix[id_client, :], 'p_vec_expected': mean_weighted_adjacency_matrix[:, id_client]})
        state_dicts_models.append(state_dicts_model)
        state_dicts_optimizers.append(state_dicts_optimizer)
        state_dicts_lr_schedulers.append(state_dicts_lr_scheduler)

    return state_dicts_lr_schedulers, state_dicts_models, state_dicts_optimizers, state_dicts_gossips


def get_state_dicts_model_in_fedem_experiment(
        n_nodes,
        kwargs_fedem,
        lrs,
        n_steps,
        kwargs_model=None,
):
    assert all([lr == lr for lr in lrs]), lrs
    args_ = Namespace(**kwargs_fedem, lr=lrs[0], n_rounds=n_steps)
    torch.manual_seed(args_.seed)

    # save state_dicts of models and optimizers
    state_dicts_models = []
    for _ in range(n_nodes):
        learners_ensemble = get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            kwargs_model=kwargs_model,
            weight_decay=args_.weight_decay,
        )
        state_dicts_model = []
        for learner in learners_ensemble.learners:
            state_dicts_model.append(learner.model.cpu().state_dict())
        state_dicts_models.append(state_dicts_model)

    return state_dicts_models


def run_experiment(
        args_,
        logs_dir,
        state_dicts_models_init,
        datasets_train,
        datasets_valid,
        datasets_test,
        batch_sizes,
        n_nodes=None,
        kwargs_model=None,
        state_dicts_hyperparameters=None,
        kwargs_build_nodes=None,
        graph=None,
        device=None,
        use_train_as_valid=True,
):
    train_iterators, val_iterators, test_iterators = get_loaders(args_.validation, datasets_train, datasets_valid, datasets_test, batch_sizes, use_train_as_valid=use_train_as_valid)

    torch.manual_seed(args_.seed)

    print("==> Clients initialization..")
    clients = init_clients_hgp(args_, logs_dir, train_iterators, val_iterators, test_iterators, kwargs_model=kwargs_model)

    print("==> Loading Initial Model Parameters..")
    for client, state_dicts in zip(clients, state_dicts_models_init):
        for learner, state_dict in zip(client.learners_ensemble.learners, state_dicts):
            learner.model.load_state_dict(state_dict)

    if state_dicts_hyperparameters is not None:
        print("==> Loading Hyperparameters..")
        for idx_node, (client, state_dict_hyper, kwargs_build) in enumerate(zip(clients, state_dicts_hyperparameters, kwargs_build_nodes)):
            hyperparameter_module = D_HYPER_PARAMETERS[kwargs_build['name_hyperparam']](n_nodes=n_nodes, idx_node=idx_node, **kwargs_build['kwargs_hyperparam'])
            hyperparameter_module.hyperparameters.load_state_dict(state_dict_hyper)
            hyperparameter_module.hyperparameters = hyperparameter_module.hyperparameters.to(device)

            for learner in client.learners_ensemble.learners:
                learner.model.hyperparameter_module = hyperparameter_module

    if args_.decentralized:
        assert isinstance(graph, (GraphErdosRenyi, GraphFullyConnected)), graph
        for idx_node, client in enumerate(clients):
            p_vec = graph.mixing_matrix[:, idx_node].cpu().numpy()
            client.gossip = GossipGivenWeights(len(clients), idx_node, p_vec).to(graph.rate_connected_mat.device)

    print("==> Test Clients initialization..")
    test_clients = []
    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)
    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)
    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            kwargs_model=kwargs_model,
            weight_decay=args_.weight_decay,
        )
    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]
    aggregator = \
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            graph=graph,
            seed=args_.seed
        )
    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    while current_round <= args_.n_rounds:

        aggregator.mix()

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round
    if "save_dir" in args_:
        save_dir = os.path.join(args_.save_dir)

        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)

    # save state_dicts of models and optimizers
    state_dicts_lr_schedulers, state_dicts_models, state_dicts_optimizers, state_dicts_gossips = get_state_dicts_of_clients(aggregator, args_.decentralized, clients)

    return state_dicts_models, state_dicts_optimizers, state_dicts_lr_schedulers, state_dicts_gossips, aggregator.last_log_train_mean, aggregator.last_log_train_bottom
