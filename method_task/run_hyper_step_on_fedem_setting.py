from argparse import Namespace

import torch

from constants import KeysOptionTrainSig, NamesLrScheduler, TypesDevice, AbbrModels
from fedem.run_experiment import run_experiment
from lib_task.common import load_graph, load_clients, get_loaders
from lib_task.hyper_gradient_push import hyper_gradient_push
from lib_task.hyper_step import get_hyper_optimizer, any_nan_hypergrad


def run_hyper_sgd(
        n_nodes,
        name_model,
        kwargs_build_nodes,
        option_eval_metric,
        lrs,
        batch_sizes,
        n_steps,
        use_cuda,
        shuffle_train,
        datasets_valid,
        datasets_train,
        datasets_test,
        use_train_for_outer_loss,
        seed,
        option_train_insignificant,
        option_train_significant,
        state_dict_graph,
        mode_graph,
        hyper_learning_rate,
        hyper_optimizer,
        kwargs_hyper_optimizer,
        option_hgp,
        option_hgp_insignificant,
        kwargs_fedem,
        logs_dir,
        state_dicts_models_init,
        kwargs_model,
        t_h,
        state_dicts_hyperparameters,
        state_dicts_hyper_optimizer,
        lrs_per_hyperparameter=None,
        _run=None,
):
    assert option_train_significant[KeysOptionTrainSig.LR_SCHEDULER] == NamesLrScheduler.CONST, option_train_significant[KeysOptionTrainSig.LR_SCHEDULER]
    assert all([lrs[0] == lr for lr in lrs]), lrs
    assert all([batch_sizes[0] == bs for bs in batch_sizes]), batch_sizes

    # check weight decay is zero or None to avoid it is double applied by our model and optimizers in fedem settings
    for kwargs_build in kwargs_build_nodes:
        if name_model == AbbrModels.LEARNERS_ENSEMBLE:
            kwargs_learner = kwargs_build["kwargs_model"]["kwargs_learner"]
            if "weight_decay" in kwargs_learner:
                assert kwargs_learner["weight_decay"] in (0.0, None), kwargs_learner
        else:
            for kwargs_build in kwargs_build_nodes:
                if "weight_decay" in kwargs_build["kwargs_model"]:
                    assert kwargs_build["kwargs_model"]["weight_decay"] in (0.0, None), kwargs_build

    args_ = Namespace(**kwargs_fedem, lr=lrs[0], n_rounds=n_steps, bz=batch_sizes[0], seed=seed)

    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)
    graph = load_graph(mode_graph, n_nodes=n_nodes, state_dict=state_dict_graph).to(device)

    state_dicts_models, state_dicts_optimizers, state_dicts_lr_schedulers, state_dicts_gossips, d_metric_mean, d_metric_bottom = run_experiment(
        args_,
        logs_dir=logs_dir,
        state_dicts_models_init=state_dicts_models_init,
        datasets_train=datasets_train,
        datasets_valid=datasets_valid,
        datasets_test=datasets_test,
        batch_sizes=batch_sizes,
        n_nodes=n_nodes,
        kwargs_model=kwargs_model,
        state_dicts_hyperparameters=state_dicts_hyperparameters,
        kwargs_build_nodes=kwargs_build_nodes,
        graph=graph,
        device=device,
        use_train_as_valid=False
    )

    # log trained model metrics
    print(f'hyper-step: [{t_h}], {", ".join([f"{k}(Mean):{v:03f}" for k, v in d_metric_mean.items()])}, {", ".join([f"{k}(Bottom):{v:03f}" for k, v in d_metric_bottom.items()])}')
    if _run is not None:
        for suffix, d_metric in zip(('Mean', 'Bottom'), (d_metric_mean, d_metric_bottom)):
            for name, value in d_metric.items():
                _run.log_scalar(f'{name}({suffix})', value, t_h)

    # update kwargs if optimizer in fedem uses weight decay
    lrs_last = []
    for s_o, s_lr, kwargs_build in zip(state_dicts_optimizers, state_dicts_lr_schedulers, kwargs_build_nodes):
        assert len(s_o) == 1 and len(s_lr) == 1, (len(s_o) == 1, len(s_lr))
        if name_model == AbbrModels.LEARNERS_ENSEMBLE:
            kwargs_build['kwargs_model']['kwargs_learner']['weight_decay'] = s_o[0]['param_groups'][0]['weight_decay']
        else:
            kwargs_build['kwargs_model']['weight_decay'] = s_o[0]['param_groups'][0]['weight_decay']
        lrs_last.append(s_lr[0]['_last_lr'][0])

    state_dicts_hyperparameters, hypergrads_nodes, state_dicts_hyper_optimizer = run_hgp(
        state_dicts_models=state_dicts_models,
        n_nodes=n_nodes,
        name_model=name_model,
        kwargs_build_nodes=kwargs_build_nodes,
        option_eval_metric=option_eval_metric,
        last_lrs=lrs_last,
        batch_sizes=batch_sizes,
        n_steps=n_steps,
        use_cuda=use_cuda,
        shuffle_train=shuffle_train,
        datasets_train=datasets_train,
        datasets_valid=datasets_valid,
        use_train_for_outer_loss=use_train_for_outer_loss,
        seed=seed,
        option_train_insignificant=option_train_insignificant,
        option_train_significant=option_train_significant,
        graph=graph,
        hyper_learning_rate=hyper_learning_rate,
        hyper_optimizer=hyper_optimizer,
        kwargs_hyper_optimizer=kwargs_hyper_optimizer,
        state_dicts_hyper_optimizer=state_dicts_hyper_optimizer,
        option_hgp=option_hgp,
        option_hgp_insignificant=option_hgp_insignificant,
        state_dicts_hyperparameters=state_dicts_hyperparameters,
        state_dicts_gossips=state_dicts_gossips,
        lrs_per_hyperparameter=lrs_per_hyperparameter,
        _run=_run,
    )

    return state_dicts_hyperparameters, hypergrads_nodes, state_dicts_hyper_optimizer, d_metric_mean, d_metric_bottom



def run_hgp(state_dicts_models,
            n_nodes,
            name_model,
            kwargs_build_nodes,
            option_eval_metric,
            last_lrs,
            batch_sizes,
            n_steps,
            use_cuda,
            shuffle_train,
            datasets_train,
            datasets_valid,
            use_train_for_outer_loss,
            seed,
            option_train_insignificant,
            option_train_significant,
            graph,
            hyper_learning_rate,
            hyper_optimizer,
            kwargs_hyper_optimizer,
            state_dicts_hyper_optimizer,
            option_hgp,
            option_hgp_insignificant,
            state_dicts_gossips,
            state_dicts_hyperparameters,
            lrs_per_hyperparameter=None,
            _run=None,
            ):
    # suppose dsgd not sgp
    assert option_train_significant[KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT], option_train_insignificant

    # init clients
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)

    # get models
    clients = load_clients(name_model, kwargs_build_nodes, n_nodes, device=device)

    # load trained innerparameters and expected values of network
    for client, state_dicts_model, state_dict_gossip in zip(clients, state_dicts_models, state_dicts_gossips):
        assert len(state_dicts_model) == 1, len(state_dicts_model)
        client.gossip.load_state_dict(state_dict_gossip, strict=False)
        assert torch.any(client.gossip.p_vec_expected) > 0., client.gossip.p_vec_expected
        assert torch.any(client.gossip.are_in_neighbors_expected) > 0., client.gossip.are_in_neighbors_expected
        client.model.load_state_dict(state_dicts_model[0])
        client.copy_innerparameters_to_parmas_biased()

    optimizers = [get_hyper_optimizer(m.hyperparameters, hyper_optimizer, hyper_learning_rate, lrs_per_hyperparameter,
                                      **kwargs_hyper_optimizer) for m in clients]

    for client, optimizer, state_dict in zip(clients, optimizers, state_dicts_hyperparameters):
        # load hyper parameter of previous step
        client.hyperparameters.load_state_dict(state_dict)

    for client, optimizer, state_dict in zip(clients, optimizers, state_dicts_hyper_optimizer):
        # load optimizer
        optimizer.load_state_dict(state_dict)

    # make dataloaders for hgp
    loaders_train = get_loaders(batch_sizes, datasets_train, shuffle_train,
                                option_train_significant[KeysOptionTrainSig.DROP_LAST])

    if use_train_for_outer_loss:
        loaders_valid = get_loaders(batch_sizes, datasets_train, shuffle=False, drop_last=False)
    else:
        loaders_valid = get_loaders(batch_sizes, datasets_valid, shuffle=False, drop_last=False)

    hypergrads_nodes, _, _ = hyper_gradient_push(clients, graph=graph, loaders_train=loaders_train,
                                                 loaders_valid=loaders_valid, option_eval_metric=option_eval_metric,
                                                 n_steps=n_steps, batch_sizes=batch_sizes, lrs=last_lrs,
                                                 option_train_significant=option_train_significant,
                                                 option_hgp=option_hgp, seed=seed,
                                                 option_hgp_insignificant=option_hgp_insignificant, _run=_run)

    if not any_nan_hypergrad(hypergrads_nodes):
        for client, optimizer, hypergrads in zip(clients, optimizers, hypergrads_nodes):
            # hyper parameter update
            for idx_hyper, (hyperparam, hypergrad) in enumerate(zip(client.hyperparameters, hypergrads)):
                hyperparam.grad = hypergrad
            # update hyper-parameters using assigned grad
            optimizer.step()

    # save hyperparam
    state_dicts_hyperparameters = [m.hyperparameters.cpu().state_dict() for m in clients]
    state_dicts_hyper_optimizer = [o.state_dict() for o in optimizers]

    return state_dicts_hyperparameters, hypergrads_nodes, state_dicts_hyper_optimizer
