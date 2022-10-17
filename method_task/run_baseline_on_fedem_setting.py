from argparse import Namespace

import torch

from constants import TypesDevice
from fedem.run_experiment import run_experiment
from lib_task.common import load_graph


def run_baselines(
        n_nodes,
        batch_sizes,
        datasets_valid,
        datasets_train,
        datasets_test,
        kwargs_fedem,
        logs_dir,
        lrs,
        n_steps,
        state_dict_graph,
        mode_graph,
        use_cuda,
        state_dicts_models_init,
        _run=None,
):
    assert all([lrs[0] == lr for lr in lrs]), lrs
    assert all([batch_sizes[0] == bs for bs in batch_sizes]), batch_sizes
    args_ = Namespace(**kwargs_fedem, lr=lrs[0], n_rounds=n_steps, bz=batch_sizes[0])

    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)
    graph = load_graph(mode_graph, n_nodes=n_nodes, state_dict=state_dict_graph).to(device)

    state_dicts_models, state_dicts_optimizers, state_dicts_lr_schedulers, state_dicts_gossips, d_metric_mean, d_metric_bottom = run_experiment(
        args_,
        logs_dir=logs_dir,
        state_dicts_models_init=state_dicts_models_init,
        graph=graph,
        datasets_train=datasets_train,
        datasets_valid=datasets_valid,
        datasets_test=datasets_test,
        batch_sizes=batch_sizes,
    )

    # log trained model metrics
    print(f'{", ".join([f"{k}(Mean):{v:03f}" for k, v in d_metric_mean.items()])}, {", ".join([f"{k}(Bottom):{v:03f}" for k, v in d_metric_bottom.items()])}')
    if _run is not None:
        for suffix, d_metric in zip(('Mean', 'Bottom'), (d_metric_mean, d_metric_bottom)):
            for name, value in d_metric.items():
                _run.log_scalar(f'{name}({suffix})', value)

    return state_dicts_models, state_dicts_optimizers, state_dicts_lr_schedulers, state_dicts_gossips, d_metric_mean, d_metric_bottom
