from constants import *
from task import *


def get_config():
    n_nodes = 100

    configs_overwrite = []
    names_option = []


    names_option.append('PDBO-DA (undirected)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.CNN_EMNIST,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.SOFTMAX_CATEGORY_WEIGHTS,
            'kwargs_hyperparam': {},
            'kwargs_model': {'weight_decay': 1e-3},

        },
        mode_graph=ModesGraph.STOCHASTIC_BIDIRECTED,
        lr=0.05,
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
    ))

    names_option.append('PDBO-MTL (undirected)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.LEARNERS_ENSEMBLE,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.LEARNERS_WEIGHTS,
            'kwargs_hyperparam': {'n_learners': 3, "hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": 1e-2},
            'kwargs_model': {
                'n_learners': 3,
                'name_learner_model': AbbrModels.CNN_EMNIST,
                'kwargs_learner': {'weight_decay': 1e-3},
            },
        },
        mode_graph=ModesGraph.STOCHASTIC_BIDIRECTED,
        lr=0.25,
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
    ))

    names_option.append('PDBO-DA&MTL (undirected)')
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
                'kwargs_learner': {'weight_decay': 1e-3},
            },
        },
        mode_graph=ModesGraph.STOCHASTIC_BIDIRECTED,
        lr=0.25,
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
    ))

    names_option.append('PDBO-DA (directed)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.CNN_EMNIST,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.SOFTMAX_CATEGORY_WEIGHTS,
            'kwargs_hyperparam': {},
            'kwargs_model': {'weight_decay': 1e-3},
        },
        mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
        lr=0.05,
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
    ))

    names_option.append('PDBO-MTL (directed)')
    configs_overwrite.append(dict(
        name_model=AbbrModels.LEARNERS_ENSEMBLE,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.LEARNERS_WEIGHTS,
            'kwargs_hyperparam': {'n_learners': 3, "hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": 1e-2},
            'kwargs_model': {
                'n_learners': 3,
                'name_learner_model': AbbrModels.CNN_EMNIST,
                'kwargs_learner': {'weight_decay': 1e-3},
            },
        },
        mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
        lr=0.25,
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
    ))

    names_option.append('PDBO-DA&MTL (directed)')
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
                'kwargs_learner': {'weight_decay': 1e-3},
            },
        },
        mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
        lr=0.25,
        option_hgp={
            KeysOptionHGP.DEPTH: 200,
        },
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
            KeysOptionHGP.USE_TRUE_EXPECTED_EDGES: False,
            KeysOptionHGP.MODE_UPDATE: ModesHGPUpdate.SIMULTANEOUS,
            KeysOptionHGP.DUMPING: 1.0,
            KeysOptionHGP.ALPHA_V: 0.9,
            KeysOptionHGP.ALPHA_W: 0.1,
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
        n_hyper_steps=21,
        hyper_optimizer=HyperOptimizers.ADAM,
        kwargs_hyper_optimizer={},
        hyper_learning_rate=1e-1,
        use_train_for_outer_loss=True,
        n_steps_logged_train=600,
        option_train_insignificant_logged_train={
            KeysOptionTrainInsig.LOG_EVERY: 5,
            KeysOptionTrainInsig.NAMES_METRIC_LOG: [
                NamesEvalMetric.ACCURACY,
                NamesEvalMetric.LOSS_BARE_MEAN,
            ]},
        map_dataset_logged_train={
            KeysTarget.TRAIN: KeysTarget.TRAIN,
            KeysTarget.VALID: KeysTarget.TRAIN,
        },
    )
