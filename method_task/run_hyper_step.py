from datetime import datetime

import torch

from constants import TypesDevice, KeysOptionTrainSig
from lib_task.common import load_clients, load_graph, get_loaders
from lib_task.hyper_gradient_push import hyper_gradient_push
from lib_task.hyper_step import log_hyper_sgd_step, get_hyper_optimizer, any_nan_hypergrad
from lib_task.stochastic_gradient_push import stochastic_gradient_push


def hyper_opts_yield(
        n_nodes,
        name_model,
        kwargs_build_nodes,
        state_dicts_init,
        option_eval_metric,
        lrs,
        batch_sizes,
        n_steps,
        datasets_train,
        datasets_valid,
        datasets_test,
        shuffle_train,
        options_eval_metric_hyper,
        seed,
        save_state_dicts,
        option_train_insignificant,
        option_train_significant,
        state_dict_graph,
        mode_graph,
        t_h,
        use_train_for_outer_loss,
        state_dicts_hyperparameters,
        state_dicts_optimizer,
        option_hgp_insignificant,
        option_hgp,
        hyper_learning_rate,
        hyper_optimizer,
        kwargs_hyper_optimizer,
        lrs_per_hyperparameter=None,
        use_cuda=True,
        _run=None,
):
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)
    # get models
    clients = load_clients(name_model, kwargs_build_nodes, n_nodes, state_dicts=state_dicts_init, device=device)
    graph = load_graph(mode_graph=mode_graph, n_nodes=n_nodes, state_dict=state_dict_graph, device=device)

    optimizers = [get_hyper_optimizer(m.hyperparameters, hyper_optimizer, hyper_learning_rate, lrs_per_hyperparameter,
                                      **kwargs_hyper_optimizer) for m in clients]

    for model, optimizer, d_h, d_o in zip(clients, optimizers, state_dicts_hyperparameters, state_dicts_optimizer):
        # load hyper parameter of previous step
        model.hyperparameters.load_state_dict(d_h)
        # load optimizer
        optimizer.load_state_dict(d_o)

    # make dataloaders for hgp
    loaders_train = get_loaders(batch_sizes, datasets_train, shuffle_train,
                                option_train_significant[KeysOptionTrainSig.DROP_LAST])
    if use_train_for_outer_loss:
        loaders_valid = get_loaders(batch_sizes, datasets_train, shuffle=False, drop_last=False)
    else:
        loaders_valid = get_loaders(batch_sizes, datasets_valid, shuffle=False, drop_last=False)

    time_start = datetime.now()
    # TODO(future): dry training at the last hyper step
    clients, _ = stochastic_gradient_push(clients, graph=graph, loaders_train=loaders_train,
                                          loaders_valid=loaders_valid, n_steps=n_steps, batch_sizes=batch_sizes,
                                          lrs=lrs, option_train_significant=option_train_significant,
                                          option_train_insignificant=option_train_insignificant, seed=seed, _run=_run)

    # log metrics
    d_d_val_eval_bottom, d_d_val_eval_mean, d_d_val_eval_nodes = log_hyper_sgd_step(clients, datasets_train,
                                                                                    datasets_valid, datasets_test,
                                                                                    batch_sizes, n_nodes,
                                                                                    options_eval_metric_hyper, t_h,
                                                                                    _run)

    hypergrads_nodes, _, _ = hyper_gradient_push(clients, graph=graph, loaders_train=loaders_train,
                                                 loaders_valid=loaders_valid, option_eval_metric=option_eval_metric,
                                                 n_steps=n_steps, batch_sizes=batch_sizes, lrs=lrs,
                                                 option_train_significant=option_train_significant,
                                                 option_hgp=option_hgp, seed=seed,
                                                 option_hgp_insignificant=option_hgp_insignificant, _run=_run)

    if not any_nan_hypergrad(hypergrads_nodes):
        for model, optimizer, hypergrads in zip(clients, optimizers, hypergrads_nodes):
            # hyper parameter update
            for idx_hyper, (hyperparam, hypergrad) in enumerate(zip(model.hyperparameters, hypergrads)):
                hyperparam.grad = hypergrad
                # update hyper-parameters using assigned grad
                optimizer.step()

    if save_state_dicts:
        # save trained model
        state_dicts = [m.cpu().state_dict() for m in clients]
    else:
        state_dicts = None

    # save hyperparam
    state_dicts_hyperparameters = [m.hyperparameters.cpu().state_dict() for m in clients]
    state_dicts_optimizer = [o.state_dict() for o in optimizers]

    time_past = datetime.now() - time_start

    return state_dicts_hyperparameters, hypergrads_nodes, state_dicts_optimizer, state_dicts, time_past, d_d_val_eval_nodes, d_d_val_eval_mean, d_d_val_eval_bottom
