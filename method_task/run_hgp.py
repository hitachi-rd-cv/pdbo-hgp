import torch

from constants import KeysOptionTrainSig, TypesDevice
from lib_task.common import load_clients, load_graph, get_loaders
from lib_task.hyper_gradient_push import hyper_gradient_push
from method_task.compute_true_hypergrad import compute_true_hypergrad


def run_hyper_gradient_push(n_nodes, name_model, kwargs_build_nodes, state_dicts, option_eval_metric, lrs, batch_sizes, n_steps, datasets_train, datasets_valid, shuffle_train, seed, option_train_insignificant,
                            state_dict_graph, option_train_significant, mode_graph, use_train_for_outer_loss, compute_hg=False, option_hgp=None, option_hgp_insignificant=None, use_cuda=True, _run=None, mode='hgp',
                            hypergrads_nodes_true=None, save_intermediate_hypergradients=False, true_backward_mode=False, ):
    # build network
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)

    clients = load_clients(
        name_model=name_model,
        kwargs_build_nodes=kwargs_build_nodes,
        n_nodes=n_nodes,
        state_dicts=state_dicts,
        device=device,
    )
    graph = load_graph(mode_graph=mode_graph, n_nodes=n_nodes, state_dict=state_dict_graph, device=device)
    # make dataloaders for hgp
    loaders_train = get_loaders(batch_sizes, datasets_train, shuffle_train, option_train_significant[KeysOptionTrainSig.DROP_LAST])
    if use_train_for_outer_loss:
        loaders_valid = get_loaders(batch_sizes, datasets_train, shuffle=False, drop_last=False)
    else:
        loaders_valid = get_loaders(batch_sizes, datasets_valid, shuffle=False, drop_last=False)

    if mode == 'true':
        hypergrads_nodes, estimators, hypergrads_nodes_steps = compute_true_hypergrad(clients, graph=graph, loaders_train=loaders_train, loaders_valid=loaders_valid, option_eval_metric=option_eval_metric, n_steps=n_steps,
                                                                                      batch_sizes=batch_sizes,
                                                                                      lrs=lrs,
                                                                                      option_train_significant=option_train_significant, option_train_insignificant=option_train_insignificant,
                                                                                      option_hgp_insignificant=option_hgp_insignificant,
                                                                                      option_hgp=option_hgp, shuffle_train=shuffle_train, compute_hg=compute_hg, seed=seed, _run=_run)
    elif mode == 'hgp':
        hypergrads_nodes, estimators, hypergrads_nodes_steps = hyper_gradient_push(clients, graph=graph,
                                                                                   loaders_train=loaders_train,
                                                                                   loaders_valid=loaders_valid,
                                                                                   option_eval_metric=option_eval_metric,
                                                                                   n_steps=n_steps,
                                                                                   batch_sizes=batch_sizes, lrs=lrs,
                                                                                   option_train_significant=option_train_significant,
                                                                                   option_hgp=option_hgp, seed=seed,
                                                                                   hypergrads_nodes_true=hypergrads_nodes_true,
                                                                                   true_backward_mode=true_backward_mode,
                                                                                   option_hgp_insignificant=option_hgp_insignificant,
                                                                                   save_intermediate_hypergradients=save_intermediate_hypergradients,
                                                                                   _run=_run)
    else:
        raise ValueError(mode)

    state_dicts_estimator = [e.cpu().state_dict() for e in estimators]

    return hypergrads_nodes, state_dicts_estimator, hypergrads_nodes_steps
