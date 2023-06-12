from constants import *
from task import *


def get_config():
    n_nodes = 100

    configs_overwrite = []
    names_option = []

    for M in [10]:
        for S in [1, 10]:
            names_option.append(f'PDBO-AT (centralized, M={M}, S={S})')
            configs_overwrite.append(dict(
                name_model=AbbrModels.CNN_EMNIST,
                kwargs_build_base={
                    'mode_gossip': ModesGossip.NORMALIZE,
                    'name_hyperparam': NamesHyperParam.HYPER_SOFTMAX_LOGITS_WEIGHTS,
                    'kwargs_hyperparam': {"hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": 1e-2},
                    'kwargs_model': {'weight_decay': 0.},
                },
                kwargs_fedem={
                    'experiment': 'emnist_softmax_weight',
                    'decentralized': False,
                },
                option_hgp={
                    KeysOptionHGP.DEPTH: M,
                    KeysOptionHGP.N_PUSH: S,
                },
                lr=0.03,
                mode_graph=ModesGraph.CONSTANTS,
                kwargs_init_graph={},
                hyper_learning_rate=1e-1,
            ))

            names_option.append(f'PDBO-AT (decentralized, M={M}, S={S})')
            configs_overwrite.append(dict(
                name_model=AbbrModels.CNN_EMNIST,
                kwargs_build_base={
                    'mode_gossip': ModesGossip.WEIGHT_GIVEN,
                    'name_hyperparam': NamesHyperParam.HYPER_SOFTMAX_LOGITS_WEIGHTS,
                    'kwargs_hyperparam': {"hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": 1e-2},
                    'kwargs_model': {'weight_decay': 0.},
                },
                kwargs_fedem={
                    'experiment': 'emnist_softmax_weight',
                    'decentralized': True,
                },
                option_hgp={
                    KeysOptionHGP.DEPTH: M,
                    KeysOptionHGP.N_PUSH: S,
                },
                lr=0.03,
                mode_graph=ModesGraph.ERDOS_RENYI,
                kwargs_init_graph={'p': 0.5},
                hyper_learning_rate=1e-1,
            ))

            names_option.append(f'PDBO-MTL (centralized, M={M}, S={S})')
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
                    KeysOptionHGP.DEPTH: M,
                    KeysOptionHGP.N_PUSH: S,
                },
                lr=0.1,
                mode_graph=ModesGraph.CONSTANTS,
                kwargs_init_graph={},
                hyper_learning_rate=1e0,
            ))

            names_option.append(f'PDBO-MTL (decentralized, M={M}, S={S})')
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
                    KeysOptionHGP.DEPTH: M,
                    KeysOptionHGP.N_PUSH: S,
                },
                lr=0.1,
                mode_graph=ModesGraph.ERDOS_RENYI,
                kwargs_init_graph={'p': 0.5},
                hyper_learning_rate=1e0,
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
            KeysOptionTrainSig.MODE_SGP: ModesSGP.ASSRAN,
        },
        option_train_insignificant={
            KeysOptionTrainInsig.LOG_EVERY: 5,
            KeysOptionTrainInsig.NAMES_METRIC_LOG: [
                # NamesEvalMetric.ACCURACY
            ]},
        option_hgp={
            KeysOptionHGP.DUMPING: 1e-2,
            KeysOptionHGP.USE_TRUE_EXPECTED_EDGES: False,
            KeysOptionHGP.USE_TRUE_DEBIAS_WEIGHTS: False,
        },
        option_hgp_insignificant={
            KeysOptionHGPInsig.NAMES_METRIC_LOG: [],
        },
        lr=None,
        batch_size=128,
        n_steps=200,
        n_hyper_steps=6,
        use_cuda=True,
        option_dataset={
            'n_components': 3,
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
        },
        seed_dataset=12345,
        use_train_for_outer_loss=True,
    )


if __name__ == "__main__":
    config = get_config()
    print(*[f'"{x}"' for x in config["names_option"]], sep=",\n")
