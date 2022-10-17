import torch

from constants import KeysOptionTrainSig, TypesDevice
from lib_task.common import load_clients, load_graph, get_loaders
from lib_task.stochastic_gradient_push import stochastic_gradient_push


def run_stochastic_gradient_push(n_nodes, name_model, kwargs_build_nodes, state_dicts, lrs, batch_sizes, n_steps,
                                 datasets_train, datasets_valid, shuffle_train, seed, option_train_insignificant,
                                 state_dict_graph, option_train_significant,
                                 mode_graph, hyperparam_state_dicts=None, use_cuda=True, _run=None):
    # build network
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)

    models = load_clients(
        name_model=name_model,
        kwargs_build_nodes=kwargs_build_nodes,
        n_nodes=n_nodes,
        state_dicts=state_dicts,
        device=device
    )
    graph = load_graph(mode_graph=mode_graph, n_nodes=n_nodes, state_dict=state_dict_graph, device=device)

    if hyperparam_state_dicts is not None:
        # load trained hyper parameters
        for model, hyper_state_dict in zip(models, hyperparam_state_dicts):
            model.hyperparameters.load_state_dict(hyper_state_dict)

    loaders_train = get_loaders(batch_sizes, datasets_train, shuffle_train,
                                option_train_significant[KeysOptionTrainSig.DROP_LAST])
    loaders_valid = get_loaders(batch_sizes, datasets_valid, shuffle=False, drop_last=False)

    models, log_train = stochastic_gradient_push(models, graph=graph, loaders_train=loaders_train,
                                                 loaders_valid=loaders_valid, n_steps=n_steps, batch_sizes=batch_sizes,
                                                 lrs=lrs, option_train_significant=option_train_significant,
                                                 option_train_insignificant=option_train_insignificant, seed=seed,
                                                 _run=_run)
    state_dicts_trained = [m.cpu().state_dict() for m in models]

    return state_dicts_trained, log_train

