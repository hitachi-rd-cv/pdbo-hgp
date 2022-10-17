from constants import *
from task import *


def get_config():
    n_nodes = 100

    configs_overwrite = []
    names_option = []

    names_option.append('PDBO-DA (centralized)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.CNN_EMNIST,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.SOFTMAX_CATEGORY_WEIGHTS,
            'kwargs_hyperparam': {},
            'kwargs_model': {'weight_decay': 0.},
        },
        kwargs_fedem={
            'experiment': 'emnist_softmax_weight',
            'decentralized': False,
        },
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
        lr=0.01,
        mode_graph=ModesGraph.CONSTANTS,
        kwargs_init_graph={},
    ))

    names_option.append('PDBO-MTL (centralized)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.LEARNERS_ENSEMBLE,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.LEARNERS_WEIGHTS,
            'kwargs_hyperparam': {'n_learners': 3, "hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": 1e-2},
            'kwargs_model': {
                'n_learners': 3,
                'name_learner_model': AbbrModels.CNN_EMNIST,
                'kwargs_learner': {'weight_decay': 0.},
            },
        },
        kwargs_fedem={
            'experiment': 'emnist_learners_ensemble',
            'decentralized': False,
        },
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
        lr=0.05,
        mode_graph=ModesGraph.CONSTANTS,
        kwargs_init_graph={},
        hyper_learning_rate=1e0,
    ))

    names_option.append('PDBO-DA&MTL (centralized)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.LEARNERS_ENSEMBLE,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.LEARNERS_WEIGHTS_AND_SOFTMAX_CATEGORY_WEIGHTS,
            'kwargs_hyperparam': {
                'n_learners': 3,
                "hyper_loss_learners": NamesHyperLoss.L2_REGULARIZER,
                "gamma_learners": 1e-2,
                'hyper_loss_categories': NamesHyperLoss.L2_REGULARIZER,
                'gamma_categories': 5e-4,
            },
            'kwargs_model': {
                'n_learners': 3,
                'name_learner_model': AbbrModels.CNN_EMNIST,
                'kwargs_learner': {'weight_decay': 0.},
            },
        },
        kwargs_fedem={
            'experiment': 'emnist_learners_ensemble',
            'decentralized': False,
        },
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
        lr=0.05,
        mode_graph=ModesGraph.CONSTANTS,
        kwargs_init_graph={},
        lrs_per_hyperparameter=[1e0, 1e-1],
        hyper_learning_rate=None,
    ))

    names_option.append('PDBO-DA (decentralized)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.CNN_EMNIST,
        kwargs_build_base={
            'mode_gossip': ModesGossip.WEIGHT_GIVEN,
            'name_hyperparam': NamesHyperParam.SOFTMAX_CATEGORY_WEIGHTS,
            'kwargs_hyperparam': {},
            'kwargs_model': {'weight_decay': 0.},
        },
        kwargs_fedem={
            'experiment': 'emnist_softmax_weight',
            'decentralized': True,
        },
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
        lr=0.01,
        mode_graph=ModesGraph.ERDOS_RENYI,
        kwargs_init_graph={'p': 0.4},
    )),

    names_option.append('PDBO-MTL (decentralized)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.LEARNERS_ENSEMBLE,
        kwargs_build_base={
            'mode_gossip': ModesGossip.WEIGHT_GIVEN,
            'name_hyperparam': NamesHyperParam.LEARNERS_WEIGHTS,
            'kwargs_hyperparam': {'n_learners': 3, "hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": 1e-2},
            'kwargs_model': {
                'n_learners': 3,
                'name_learner_model': AbbrModels.CNN_EMNIST,
                'kwargs_learner': {'weight_decay': 0.},
            },
        },
        kwargs_fedem={
            'experiment': 'emnist_learners_ensemble',
            'decentralized': True,
        },
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
        lr=0.05,
        mode_graph=ModesGraph.ERDOS_RENYI,
        kwargs_init_graph={'p': 0.4},
        hyper_learning_rate=1e0,
    ))

    names_option.append('PDBO-DA&MTL (decentralized)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.LEARNERS_ENSEMBLE,
        kwargs_build_base={
            'mode_gossip': ModesGossip.WEIGHT_GIVEN,
            'name_hyperparam': NamesHyperParam.LEARNERS_WEIGHTS_AND_SOFTMAX_CATEGORY_WEIGHTS,
            'kwargs_hyperparam': {
                'n_learners': 3,
                "hyper_loss_learners": NamesHyperLoss.L2_REGULARIZER,
                "gamma_learners": 1e-2,
                'hyper_loss_categories': NamesHyperLoss.L2_REGULARIZER,
                'gamma_categories': 5e-4,
            },
            'kwargs_model': {
                'n_learners': 3,
                'name_learner_model': AbbrModels.CNN_EMNIST,
                'kwargs_learner': {'weight_decay': 0.},
            },
        },
        kwargs_fedem={
            'experiment': 'emnist_learners_ensemble',
            'decentralized': True,
        },
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
        lr=0.05,
        mode_graph=ModesGraph.ERDOS_RENYI,
        kwargs_init_graph={'p': 0.4},
        lrs_per_hyperparameter=[1e0, 1e-1],
        hyper_learning_rate=None,
    ))

    return dict(
        workspace_directory="./processed",
        db_name=os.environ.get('MONGO_DB', "local"),
        mongo_auth=os.environ.get('MONGO_AUTH', None),
        n_nodes=n_nodes,
        name_model=None,
        kwargs_build_base={},
        kwargs_init_hparams={},
        mode_graph=None,
        kwargs_init_graph={},
        option_eval_metric={
            KeysOptionEval.NAME: NamesEvalMetric.LOSS_BARE_MEAN},
        option_train_significant={
            KeysOptionTrainSig.LR_SCHEDULER: NamesLrScheduler.CONST,
            KeysOptionTrainSig.KWARGS_LR_SCHEDULER: {},
            KeysOptionTrainSig.DROP_LAST: False,
            KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT: True,
        },
        option_train_insignificant={
            KeysOptionTrainInsig.LOG_EVERY: 5,
            KeysOptionTrainInsig.NAMES_METRIC_LOG: [
                # NamesEvalMetric.ACCURACY
            ]},
        option_hgp={
            KeysOptionHGP.USE_TRUE_EXPECTED_EDGES: False,
            KeysOptionHGP.MODE_UPDATE: ModesHGPUpdate.SIMULTANEOUS,
            KeysOptionHGP.DUMPING: 1.0,
            KeysOptionHGP.ALPHA_V: 1.0,
            KeysOptionHGP.ALPHA_W: 0.0,
        },
        option_hgp_insignificant={
            KeysOptionHGPInsig.NAMES_METRIC_LOG: [],
        },
        lr=None,
        batch_size=128,
        n_steps=200,
        n_hyper_steps=21,
        use_cuda=True,
        option_dataset={
            'n_components': -1,
            'alpha': 0.4,
            's_frac': 0.1,
            'tr_frac': 0.8,
            'val_frac': 0.25,
            'n_classes': 62,
            'n_shards': 2,
            'pathological_split': False,
            'test_tasks_frac': 0.0,
        },
        name_dataset=NamesDataset.EMNIST,
        shuffle_train=True,
        fix_random_seed_value=1234,
        bounds_hparam=[0., 0.],
        save_state_dicts=False,
        configs_overwrite=configs_overwrite,
        names_option=names_option,
        hyper_optimizer=HyperOptimizers.ADAM,
        kwargs_hyper_optimizer={},
        hyper_learning_rate=1e-1,
        kwargs_fedem={
            'weight_decay': 1e-3,
            'method': 'FedAvg',
            'sampling_rate': 1.0,
            'input_dimension': None,
            'output_dimension': None,
            'n_learners': 1,
            'local_steps': 1,
            'log_freq': 5,
            'device': 'cuda',
            'optimizer': 'sgd',
            'lr_lambda': 0.0,
            'lr_scheduler': 'multi_step',
            'mu': 0,
            'communication_probability': 0.1,
            'q': 1.0,
            'locally_tune_clients': False,
            'validation': False,
            'verbose': 1,
            'seed': 1234
        },
        seed_dataset=12345,
        use_train_for_outer_loss=True,
    )
