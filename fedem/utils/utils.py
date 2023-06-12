# ------------------------------------------------------------------------
# The following codes are copied from parts of FedEM (https://github.com/omarfoq/FedEM).
# The following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/omarfoq/FedEM/blob/main/LICENSE).
# ------------------------------------------------------------------------
from torch.utils.data import DataLoader

from fedem.aggregator import *
from fedem.client import *
from fedem.models import *
from lib_common.torch.module import CrossEntropyLossNormalized
from module_torch.model import MultiLearners, MobileNetCIFAR10, CNNEMNIST
from .metrics import accuracy
from .optim import *
from .optim import get_optimizer, get_lr_scheduler
from ..learners.learner import LearnerHyperParameterized, Learner
from ..learners.learners_ensemble import LearnersEnsemble, LearnersEnsembleHyperParameterized
from ..models import FemnistCNN


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_client(
        client_type,
        learners_ensemble,
        q,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally,
):
    """

    :param client_type:
    :param learners_ensemble:
    :param q: fairness hyper-parameter, ony used for FFL client
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param tune_locally

    :return:

    """
    if client_type == "mixture":
        return MixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    elif client_type == "AFL":
        return AgnosticFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    elif client_type == "FFL":
        return FFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            q=q
        )
    else:
        return Client(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )


def get_learners_ensemble(
        n_learners,
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        weight_decay,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        kwargs_model=None,
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param n_learners: number of learners in the ensemble
    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: LearnersEnsemble

    """
    learners = [
        get_learner(
            name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed + learner_id,
            mu=mu,
            kwargs_model=kwargs_model,
            weight_decay=weight_decay,
        ) for learner_id in range(n_learners)
    ]

    learners_weights = torch.ones(n_learners) / n_learners
    if name in ["emnist_learners_ensemble", "cifar10_learners_ensemble"]:
        return LearnersEnsembleHyperParameterized(learners=learners, learners_weights=learners_weights)
    else:
        return LearnersEnsemble(learners=learners, learners_weights=learners_weights)


def get_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        weight_decay,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        kwargs_model=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: Learner

    """
    torch.manual_seed(seed)

    if name in ["emnist", "femnist"]:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=62).to(device)
        is_binary_classification = False
    elif name == "emnist_softmax_weight":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = CNNEMNIST(**kwargs_model).to(device)
        is_binary_classification = False
    elif name == "cifar10_softmax_weight":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = MobileNetCIFAR10(**kwargs_model).to(device)
        is_binary_classification = False
    elif name in ["emnist_learners_ensemble", "cifar10_learners_ensemble"]:
        criterion = CrossEntropyLossNormalized(reduction="none").to(device)
        metric = accuracy
        model = MultiLearners(**kwargs_model).to(device)
        is_binary_classification = False
    else:
        raise ValueError(name)

    optimizer = \
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu,
            weight_decay=weight_decay
        )
    lr_scheduler = \
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    if name in ["emnist_softmax_weight", "cifar10_softmax_weight"]:
        return LearnerHyperParameterized(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    elif name in ["emnist_learners_ensemble", "cifar10_learners_ensemble"]:
        return LearnerHyperParameterized(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    else:
        return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )


def get_loader(dataset, batch_size, train):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=False)


def get_loaders(validation, datasets_train, datasets_valid, datasets_test, batch_sizes, use_train_as_valid=False):
    print("===> Building data iterators..")
    train_iterators = [get_loader(dataset, batch_size, True) for dataset, batch_size in zip(datasets_train, batch_sizes)]
    if use_train_as_valid:
        val_iterators = [get_loader(dataset, batch_size, False) for dataset, batch_size in zip(datasets_train, batch_sizes)]
    else:
        val_iterators = [get_loader(dataset, batch_size, False) for dataset, batch_size in zip(datasets_valid, batch_sizes)]
    if validation:
        test_iterators = [get_loader(dataset, batch_size, False) for dataset, batch_size in zip(datasets_valid, batch_sizes)]
    else:
        test_iterators = [get_loader(dataset, batch_size, False) for dataset, batch_size in zip(datasets_test, batch_sizes)]

    return train_iterators, val_iterators, test_iterators


def get_aggregator(
        aggregator_type,
        clients,
        global_learners_ensemble,
        lr,
        lr_lambda,
        mu,
        communication_probability,
        q,
        sampling_rate,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        verbose,
        graph=None,
        seed=None
):
    """
    `personalized` corresponds to pFedMe

    :param aggregator_type:
    :param clients:
    :param global_learners_ensemble:
    :param lr: oly used with FLL aggregator
    :param lr_lambda: only used with Agnostic aggregator
    :param mu: penalization term, only used with L2SGD
    :param communication_probability: communication probability, only used with L2SGD
    :param q: fairness hyper-parameter, ony used for FFL client
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "personalized":
        return PersonalizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "clustered":
        return ClusteredAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "L2SGD":
        return LoopLessLocalSGDAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            communication_probability=communication_probability,
            penalty_parameter=mu,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "AFL":
        return AgnosticAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr_lambda=lr_lambda,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "FFL":
        return FFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr=lr,
            q=q,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )

    elif aggregator_type == "decentralized":
        return DecentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            graph=graph,
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized`,"
            " `personalized`, `clustered`, `fednova`, `AFL`,"
            " `FFL` and `decentralized`."
        )
