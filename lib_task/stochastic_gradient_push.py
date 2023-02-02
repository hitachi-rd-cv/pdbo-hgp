import random

import numpy as np
import torch
from tqdm import tqdm

from constants import KeysOptionTrainInsig, KeysOptionTrainSig, ModesSGP
from lib_task.modules_sgp import TrainerAssran, TrainerNedic, NodeTrainLogger, CentralTrainLogger, D_LR_SCHEDULER_NODE


def stochastic_gradient_push(clients, graph, loaders_train, loaders_valid, n_steps, batch_sizes, lrs,
                             option_train_significant, option_train_insignificant, seed=None, _run=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if option_train_significant[KeysOptionTrainSig.MODE_SGP] == ModesSGP.ASSRAN:
        Trainer = TrainerAssran
    elif option_train_significant[KeysOptionTrainSig.MODE_SGP] == ModesSGP.NEDIC:
        Trainer = TrainerNedic
    else:
        raise ValueError(option_train_significant[KeysOptionTrainSig.MODE_SGP])

    trainers = []
    train_loggers = []
    for client, loader_train, loader_valid, batch_size, lr in zip(clients, loaders_train, loaders_valid, batch_sizes,
                                                                  lrs):
        lr_scheduler = D_LR_SCHEDULER_NODE[option_train_significant[KeysOptionTrainSig.LR_SCHEDULER]](lr,
                                                                                                      n_steps=n_steps,
                                                                                                      **
                                                                                                      option_train_significant[
                                                                                                          KeysOptionTrainSig.KWARGS_LR_SCHEDULER])
        loader_train = loader_train
        loader_valid = loader_valid
        train_logger = NodeTrainLogger(
            client,
            n_steps=n_steps,
            loader=loader_valid,
            names_eval_metric=option_train_insignificant[KeysOptionTrainInsig.NAMES_METRIC_LOG],
        )
        trainer = Trainer(
            client=client,
            loader_train=loader_train,
            lr_scheduler=lr_scheduler,
            train_logger=train_logger,
        )
        train_loggers.append(train_logger)
        trainers.append(trainer)

    # create logger
    central_train_logger = CentralTrainLogger(
        n_steps=n_steps,
        train_loggers=train_loggers,
        names_eval_metric=option_train_insignificant[KeysOptionTrainInsig.NAMES_METRIC_LOG],
        _run=_run
    )

    for t in tqdm(range(n_steps)):
        graph.sample()  # TODO(future): assert time-invariant
        msg_generators = []
        for trainer in trainers:
            are_connected = graph.are_connected(trainer.idx_node)
            msg_generators.append(trainer.get_msg_generator(are_connected, t, no_weight_update=option_train_significant[KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT], n_steps=n_steps))

        for msg_generator in msg_generators:
            for sgp_msg, trainer in zip(msg_generator, trainers):
                trainer.mix_sgp_partial(sgp_msg, n_steps=n_steps)

        # step
        for trainer in trainers:
            trainer.step()

        for trainer in trainers:
            trainer.log(are_connected=graph.are_connected(trainer.idx_node))
            if (t + 1) % option_train_insignificant[KeysOptionTrainInsig.LOG_EVERY] == 0:
                # compute validation evaluation metrics
                trainer.eval()

        # print, append, send and refresh the step logs
        central_train_logger.log_step(step=t)

    with torch.no_grad():
        for trainer in trainers:
            trainer.client.debias_parameters()

    return [t.client for t in trainers], central_train_logger.log_train


