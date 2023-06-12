import numpy as np
import torch

from constants import KeysOptionTrainSig, TypesDevice, KeysOptionEval
from lib_task.common import load_clients, load_graph, get_loaders
from lib_task.stochastic_gradient_push import stochastic_gradient_push
from module_torch.hyperparameter import HyperLossMasks


def gen_uniform_hyper_perturbs_of_nodes_of_iters(n_nodes, name_model, kwargs_build_nodes, use_cuda, scales_uniform_hyper_perturbs):
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)
    # get models
    models = load_clients(name_model=name_model, kwargs_build_nodes=kwargs_build_nodes, n_nodes=n_nodes, device=device)
    hyper_perturbs_of_nodes_of_iters = []
    for scale in scales_uniform_hyper_perturbs:
        hyper_perturbs_of_nodes = []
        for model in models:
            hyper_perturbs = []
            for hyperparam in model.hyperparameters:
                hyper_perturbs.append(torch.ones_like(hyperparam) * scale)
            hyper_perturbs_of_nodes.append(hyper_perturbs)
        hyper_perturbs_of_nodes_of_iters.append(hyper_perturbs_of_nodes)
    return hyper_perturbs_of_nodes_of_iters


def gen_random_hyper_perturbs_of_nodes_of_iters(n_nodes, name_model, kwargs_build_nodes, use_cuda, scales_random_hyper_perturbs):
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)
    # get models
    models = load_clients(name_model=name_model, kwargs_build_nodes=kwargs_build_nodes, n_nodes=n_nodes, device=device)
    hyper_perturbs_of_nodes_of_iters = []
    for scale in scales_random_hyper_perturbs:
        hyper_perturbs_of_nodes = []
        for model in models:
            hyper_perturbs = []
            for hyperparam in model.hyperparameters:
                hyper_perturbs.append((torch.rand_like(hyperparam) - 0.5) * 2. * scale)
            hyper_perturbs_of_nodes.append(hyper_perturbs)
        hyper_perturbs_of_nodes_of_iters.append(hyper_perturbs_of_nodes)
    return hyper_perturbs_of_nodes_of_iters


def gen_steepest_step_of_nodes_of_iters(n_nodes, name_model, kwargs_build_nodes, use_cuda, hyper_learning_rate, hypergrads_nodes):
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)
    # get models
    models = load_clients(name_model=name_model, kwargs_build_nodes=kwargs_build_nodes, n_nodes=n_nodes, device=device)
    hyper_perturbs_of_nodes = []
    for hypergrads, model in zip(hypergrads_nodes, models):
        hyper_perturbs = []
        for hypergrad, hyperparam in zip(hypergrads, model.hyperparameters):
            hyper_perturbs.append(- hyper_learning_rate * hypergrad)
        hyper_perturbs_of_nodes.append(hyper_perturbs)
    return [hyper_perturbs_of_nodes]

def gen_most_influential_perturbs_of_nodes_of_iters(n_nodes, name_model, kwargs_build_nodes, use_cuda, hypergrads_nodes, n_out):
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)
    # get models
    models = load_clients(name_model=name_model, kwargs_build_nodes=kwargs_build_nodes, n_nodes=n_nodes, device=device)
    # make concatenated matrix and vectors over the nodes
    hypergrads_flat = torch.hstack([torch.hstack([torch.flatten(h.detach()) for h in hs]) for hs in hypergrads_nodes])
    idx_samples = torch.argsort(-torch.abs(hypergrads_flat))[:n_out]

    hyper_perturbs_of_nodes_of_iters = []
    for idx_remove in idx_samples:
        idx_current = 0
        # randomly pick a client
        hyper_perturbs_of_nodes = []
        for model in models:
            hyper_perturbs = []
            for hyperparam in model.hyperparameters:
                mask = torch.zeros_like(hyperparam)
                for i in range(len(hyperparam)):
                    if idx_current == idx_remove:
                        mask[i] = torch.tensor(-1., dtype=mask.dtype, device=mask.device)
                    idx_current += 1
                hyper_perturbs.append(mask)
            hyper_perturbs_of_nodes.append(hyper_perturbs)
        hyper_perturbs_of_nodes_of_iters.append(hyper_perturbs_of_nodes)
        assert idx_current == len(hypergrads_flat)

    return hyper_perturbs_of_nodes_of_iters


def gen_leave_one_out_perturbs_of_iters(n_nodes, name_model, kwargs_build_nodes, use_cuda, n_out):
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)
    # get models
    models = load_clients(name_model=name_model, kwargs_build_nodes=kwargs_build_nodes, n_nodes=n_nodes, device=device)
    assert isinstance(models[0].hyperparameter_module, HyperLossMasks), models[0].hyperparameter_module

    # get total number of instances over models
    n_samples = np.sum([len(m.hyperparameters[0]) for m in models])
    # select index of n_out samples randomly without replacement
    idx_samples = np.random.choice(n_samples, n_out, replace=False)

    hyper_perturbs_of_nodes_of_iters = []
    for idx_remove in idx_samples:
        idx_current = 0
        # randomly pick a client
        hyper_perturbs_of_nodes = []
        for model in models:
            hyper_perturbs = []
            for hyperparam in model.hyperparameters:
                mask = torch.zeros_like(hyperparam)
                for i in range(len(hyperparam)):
                    if idx_current == idx_remove:
                        mask[i] = torch.tensor(-1., dtype=mask.dtype, device=mask.device)
                    idx_current += 1
                hyper_perturbs.append(mask)
            hyper_perturbs_of_nodes.append(hyper_perturbs)
        hyper_perturbs_of_nodes_of_iters.append(hyper_perturbs_of_nodes)
    return hyper_perturbs_of_nodes_of_iters


def compute_actual_diff(n_nodes, name_model, kwargs_build_nodes, state_dicts_init, state_dicts_trained, datasets_train,
                        datasets_valid, option_eval_metric, lrs, batch_sizes, n_steps, shuffle_train, seed,
                        option_train_insignificant, hyper_perturbs_of_nodes, state_dict_graph,
                        option_train_significant, mode_graph,
                        use_cuda=True, _run=None):
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)

    # get models
    models = load_clients(name_model=name_model, kwargs_build_nodes=kwargs_build_nodes, n_nodes=n_nodes, device=device)
    graph = load_graph(mode_graph=mode_graph, n_nodes=n_nodes, state_dict=state_dict_graph, device=device)

    # dataset
    loaders_train = get_loaders(batch_sizes, datasets_train, shuffle_train, option_train_significant[KeysOptionTrainSig.DROP_LAST])
    loaders_valid = get_loaders(batch_sizes, datasets_valid, shuffle=False, drop_last=False)

    # mean eval metric of trained model
    val_eval_mean_current = 0.
    for model, state_dict_trained, dataloader in zip(models, state_dicts_trained, loaders_valid):
        model.load_state_dict(state_dict_trained, strict=False)
        val_eval_mean_current += model.eval_metric(option_eval_metric[KeysOptionEval.NAME], dataloader).detach().cpu().numpy().item() / n_nodes

    # load initial parameter and hyperparameters and add perturbation on hyperparameters
    vals_eval_mean_perturb = []
    for model, state_dict_init, hyper_perturbs_of_hypers in zip(models, state_dicts_init, hyper_perturbs_of_nodes):
        model.load_state_dict(state_dict_init, strict=False)
        for hyperparam, hyper_perturb in zip(model.hyperparameters, hyper_perturbs_of_hypers):
            with torch.no_grad():
                hyperparam += hyper_perturb

    models, _ = stochastic_gradient_push(models, graph=graph, loaders_train=loaders_train, loaders_valid=loaders_valid, n_steps=n_steps, batch_sizes=batch_sizes, lrs=lrs,
                                         option_train_significant=option_train_significant, option_train_insignificant=option_train_insignificant, seed=seed, _run=_run)

    val_eval_mean_perturb = 0.
    for model, dataloader in zip(models, loaders_valid):
        val_eval_perturb = model.eval_metric(option_eval_metric[KeysOptionEval.NAME], dataloader)
        val_eval_mean_perturb += val_eval_perturb.clone().detach().cpu().numpy().item() / n_nodes

    diff_val_eval = val_eval_mean_perturb - val_eval_mean_current

    return diff_val_eval
