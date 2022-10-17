import numpy as np
import torch

from constants import KeysOptionEval, NamesLoader, HyperOptimizers
from lib_task.common import get_loaders
from module_torch.hyperparameter import D_HYPER_PARAMETERS


def init_hyperparams_of_nodes(
        n_nodes,
        kwargs_build_nodes,
        bounds_hparam,
        hyper_optimizer,
        kwargs_hyper_optimizer,
        hyper_learning_rate=None,
        lrs_per_hyperparameter=None,
):
    state_dicts_hyperparameters = []
    state_dicts_hyper_optimizer = []
    for idx_node, kwargs_build in enumerate(kwargs_build_nodes):
        module = D_HYPER_PARAMETERS[kwargs_build["name_hyperparam"]](n_nodes=n_nodes, idx_node=idx_node,
                                                                     **kwargs_build["kwargs_hyperparam"])
        with torch.no_grad():
            for idx_hyper, hparam in enumerate(module.hyperparameters):
                torch.nn.init.uniform_(hparam, *bounds_hparam)

        hyperparameters = module.hyperparameters

        optimizer = get_hyper_optimizer(hyperparameters, hyper_optimizer, hyper_learning_rate, lrs_per_hyperparameter,
                                        **kwargs_hyper_optimizer)

        state_dicts_hyperparameters.append(module.hyperparameters.state_dict())
        state_dicts_hyper_optimizer.append(optimizer.state_dict())

    return state_dicts_hyperparameters, state_dicts_hyper_optimizer


def get_hyper_optimizer(hyperparameters, hyper_optimizer, hyper_learning_rate, lrs_per_hyperparameter, **kwargs):
    if lrs_per_hyperparameter is None:
        assert hyper_learning_rate is not None, hyper_learning_rate
        kwargs_updated = dict(
            params=hyperparameters,
            lr=hyper_learning_rate,
            **kwargs
        )
    else:
        assert hyper_learning_rate is None, f"hyper_learning_rate must be None when lrs_per_hyperparameter is set. hyper_learning_rate={hyper_learning_rate}, lrs_per_hyperparameter={lrs_per_hyperparameter}"
        assert len(hyperparameters) == len(
            lrs_per_hyperparameter), f"Mismatch of length. len(module.hyperparameters)={len(hyperparameters)}, len(lrs_per_hyperparameter)={len(lrs_per_hyperparameter)}"
        kwargs_updated = dict(
            params=[{'params': h, 'lr': lr} for h, lr in zip(hyperparameters, lrs_per_hyperparameter)],
            **kwargs
        )
    if hyper_optimizer == HyperOptimizers.SGD:
        optimizer = torch.optim.SGD(**kwargs_updated)
    elif hyper_optimizer == HyperOptimizers.ADAM:
        optimizer = torch.optim.Adam(**kwargs_updated)
    else:
        raise ValueError(hyper_optimizer)
    return optimizer


def log_hyper_eval_metric(val_eval, name_metric, name_loader, t_h, suffix, _run=None, n_hyper_steps='?'):
    print(f'hyper-step: [{t_h}/{n_hyper_steps}], node: {suffix}, {name_metric}_{name_loader}: {val_eval:.6f}')
    if _run is not None:
        _run.log_scalar(f'hyper_{name_metric}_{name_loader}_{suffix}', val_eval, t_h)


def log_hyper_sgd_step(clients, datasets_train, datasets_valid, datasets_test, batch_sizes, n_nodes,
                       options_eval_metric_hyper, t_h, _run):
    d_loaders = {
        NamesLoader.TRAIN: get_loaders(batch_sizes, datasets_train, shuffle=False, drop_last=False),
        NamesLoader.VALID: get_loaders(batch_sizes, datasets_valid, shuffle=False, drop_last=False),
        NamesLoader.TEST: get_loaders(batch_sizes, datasets_test, shuffle=False, drop_last=False),
    }

    idx_bottom = max(int(n_nodes / 10) - 1, 0)
    d_d_val_eval_nodes = {}
    d_d_val_eval_mean = {}
    d_d_val_eval_bottom = {}
    for option_eval_hyper in options_eval_metric_hyper:
        d_d_val_eval_nodes[option_eval_hyper[KeysOptionEval.NAME]] = {}
        d_d_val_eval_mean[option_eval_hyper[KeysOptionEval.NAME]] = {}
        d_d_val_eval_bottom[option_eval_hyper[KeysOptionEval.NAME]] = {}
        for name_loader, loaders in d_loaders.items():
            d_d_val_eval_nodes[option_eval_hyper[KeysOptionEval.NAME]][name_loader] = []
            val_eval_nodes = []
            for idx_node, (model, loader) in enumerate(zip(clients, loaders)):
                val_eval = model.eval_metric(metric=option_eval_hyper[KeysOptionEval.NAME], loader=loader).item()
                val_eval_nodes.append(val_eval)
                d_d_val_eval_nodes[option_eval_hyper[KeysOptionEval.NAME]][name_loader].append(val_eval)
            ## mean
            val_eval_mean = np.mean(val_eval_nodes)
            log_hyper_eval_metric(val_eval_mean, option_eval_hyper[KeysOptionEval.NAME], name_loader, t_h, "mean", _run)
            d_d_val_eval_mean[option_eval_hyper[KeysOptionEval.NAME]][name_loader] = val_eval_mean
            ## 10% percentile
            val_eval_bottom_decile = np.sort(val_eval_nodes)[idx_bottom]
            log_hyper_eval_metric(val_eval_bottom_decile, option_eval_hyper[KeysOptionEval.NAME], name_loader, t_h, "bottom", _run)
            d_d_val_eval_bottom[option_eval_hyper[KeysOptionEval.NAME]][name_loader] = val_eval_bottom_decile

    return d_d_val_eval_bottom, d_d_val_eval_mean, d_d_val_eval_nodes

def any_nan_hypergrad(hypergrads_nodes):
    if hypergrads_nodes is None:
        return False
    else:
        for hypergrads in hypergrads_nodes:
            for hypergrad in hypergrads:
                if torch.any(torch.isnan(hypergrad)):
                    return True
        return False
