from constants import *
from task import *


def get_config():
    n_nodes = 100

    configs_overwrite = []
    names_option = []

    for M in [10]:
        for S in [1, 10]:
            lr = 1e-1
            names_option.append(f'PDBO-AT (undirected, M={M}, S={S})')
            configs_overwrite.append(dict(
                name_model=AbbrModels.CNN_EMNIST,
                kwargs_build_base={
                    'mode_gossip': ModesGossip.NORMALIZE,
                    'name_hyperparam': NamesHyperParam.HYPER_SOFTMAX_LOGITS_WEIGHTS,
                    'kwargs_hyperparam': {"hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": 1e-3},
                    'kwargs_model': {'weight_decay': 1e-3},
                },
                mode_graph=ModesGraph.STOCHASTIC_BIDIRECTED,
                lr=lr,
                option_hgp={
                    KeysOptionHGP.DEPTH: M,
                    KeysOptionHGP.N_PUSH: S,
                },
                hyper_learning_rate=1e-1,
            ))

            names_option.append(f'PDBO-AT (directed, M={M}, S={S})')
            configs_overwrite.append(dict(
                name_model=AbbrModels.CNN_EMNIST,
                kwargs_build_base={
                    'mode_gossip': ModesGossip.NORMALIZE,
                    'name_hyperparam': NamesHyperParam.HYPER_SOFTMAX_LOGITS_WEIGHTS,
                    'kwargs_hyperparam': {"hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": 1e-3},
                    'kwargs_model': {'weight_decay': 1e-3},
                },
                mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
                lr=lr,
                option_hgp={
                    KeysOptionHGP.DEPTH: M,
                    KeysOptionHGP.N_PUSH: S,
                },
                hyper_learning_rate=1e-1,
            ))

            gamma = 1e-2
            lr = 0.5
            lr_h = 1e-1

            names_option.append(f'PDBO-MTL (undirected, M={M}, S={S})')
            configs_overwrite.append(dict(
                name_model=AbbrModels.LEARNERS_ENSEMBLE,
                kwargs_build_base={
                    'mode_gossip': ModesGossip.NORMALIZE,
                    'name_hyperparam': NamesHyperParam.LEARNERS_WEIGHTS,
                    'kwargs_hyperparam': {'n_learners': 3, "hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": gamma},
                    'kwargs_model': {
                        'n_learners': 3,
                        'name_learner_model': AbbrModels.CNN_EMNIST,
                        'kwargs_learner': {'weight_decay': 1e-3},
                    },
                },
                mode_graph=ModesGraph.STOCHASTIC_BIDIRECTED,
                lr=lr,
                option_hgp={
                    KeysOptionHGP.DEPTH: M,
                    KeysOptionHGP.N_PUSH: S,
                },
                hyper_learning_rate=lr_h,
            ))

            names_option.append(f'PDBO-MTL (directed, M={M}, S={S})')
            configs_overwrite.append(dict(
                name_model=AbbrModels.LEARNERS_ENSEMBLE,
                kwargs_build_base={
                    'mode_gossip': ModesGossip.NORMALIZE,
                    'name_hyperparam': NamesHyperParam.LEARNERS_WEIGHTS,
                    'kwargs_hyperparam': {'n_learners': 3, "hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": gamma},
                    'kwargs_model': {
                        'n_learners': 3,
                        'name_learner_model': AbbrModels.CNN_EMNIST,
                        'kwargs_learner': {'weight_decay': 1e-3},
                    },
                },
                mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
                lr=lr,
                option_hgp={
                    KeysOptionHGP.DEPTH: M,
                    KeysOptionHGP.N_PUSH: S,
                },
                hyper_learning_rate=lr_h,
            ))

    return dict(
        fix_random_seed_value=1234,
        workspace_directory="./processed",
        db_name=os.environ.get('MONGO_DB', "local"),
        mongo_auth=os.environ.get('MONGO_AUTH', None),
        n_nodes=n_nodes,
        name_model=None,
        kwargs_build_base={
        },
        option_hgp={
            KeysOptionHGP.DUMPING: 1e-1,
            KeysOptionHGP.USE_TRUE_EXPECTED_EDGES: False,
            KeysOptionHGP.USE_TRUE_DEBIAS_WEIGHTS: False,
        },
        lr=None,
        kwargs_init_hparams={},
        mode_graph=None,
        kwargs_init_graph={'low': 0.4, 'high': 0.8, 'p': 1.0, 'force_sparse': False},
        option_eval_metric={
            KeysOptionEval.NAME: NamesEvalMetric.LOSS_BARE_MEAN
        },
        option_train_significant={
            KeysOptionTrainSig.LR_SCHEDULER: NamesLrScheduler.MULTI_STEP,
            KeysOptionTrainSig.KWARGS_LR_SCHEDULER: {"milestones": [500, 550]},
            KeysOptionTrainSig.DROP_LAST: True,
            KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT: False,
            KeysOptionTrainSig.MODE_SGP: ModesSGP.NEDIC,
        },
        option_train_insignificant={
            KeysOptionTrainInsig.LOG_EVERY: 5,
            KeysOptionTrainInsig.NAMES_METRIC_LOG: [
                # NamesEvalMetric.ACCURACY
            ]},
        option_hgp_insignificant={
            KeysOptionHGPInsig.NAMES_METRIC_LOG: [],
        },
        batch_size=128,
        n_steps=600,
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
        seed_dataset=12345,
        shuffle_train=True,
        configs_overwrite=configs_overwrite,
        names_option=names_option,
        bounds_hparam=[0., 0.],
        names_checked_dataset=[NamesLoader.VALID, NamesLoader.TEST],
        options_eval_metric_hyper=[
            {KeysOptionEval.NAME: NamesEvalMetric.LOSS_BARE_MEAN},
            {KeysOptionEval.NAME: NamesEvalMetric.ACCURACY},
        ],
        save_state_dicts=False,
        n_hyper_steps=6,
        hyper_optimizer=HyperOptimizers.ADAM,
        kwargs_hyper_optimizer={},
        hyper_learning_rate=None,
        use_train_for_outer_loss=True,
    )


if __name__ == "__main__":
    config = get_config()
    print(*[f'"{x}"' for x in config["names_option"]], sep=",\n")
