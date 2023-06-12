from constants import *
from task import *


def get_config():
    return dict(
        workspace_directory="./processed",
        db_name=os.environ.get('MONGO_DB', "local"),
        mongo_auth=os.environ.get('MONGO_AUTH', None),
        n_nodes=3,
        name_model=AbbrModels.CNN_MNIST,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.LOSS_MASKS,
            'kwargs_hyperparam': {},
            'kwargs_model': {'weight_decay': 1e-3},
        },
        kwargs_init_hparams={},
        mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
        kwargs_init_graph={'low': 0.4, 'high': 0.8, 'p': 1.0, 'force_sparse': False},
        option_eval_metric={KeysOptionEval.NAME: NamesEvalMetric.LOSS_BARE_MEAN},
        shuffle_train=True,
        option_train_significant={
            KeysOptionTrainSig.LR_SCHEDULER: NamesLrScheduler.MULTI_STEP,
            KeysOptionTrainSig.KWARGS_LR_SCHEDULER: {'milestones': [3000, 4000, 4500, 4600, 4700, 4800, 4900, 4950, 4990, 4999]},
            KeysOptionTrainSig.DROP_LAST: True,
            KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT: False,
            KeysOptionTrainSig.MODE_SGP: ModesSGP.ASSRAN,
        },
        option_train_insignificant={
            KeysOptionTrainInsig.LOG_EVERY: 5,
            KeysOptionTrainInsig.NAMES_METRIC_LOG: [
                # NamesEvalMetric.ACCURACY
            ]},
        option_hgp_insignificant={
            KeysOptionHGPInsig.NAMES_METRIC_LOG: [
                NamesHGPMetric.V_NORM,
                NamesHGPMetric.U_NORM,
            ],
        },
        lr=1e-2,
        batch_size=999999,
        n_steps=5000,
        use_cuda=True,
        option_dataset={
            'n_components': 3,
            'alpha': 0.4,
            's_frac': 0.01,
            'tr_frac': 0.8,
            'val_frac': 0.25,
            'n_classes': 62,
            'n_shards': 2,
            'pathological_split': False,
            'test_tasks_frac': 0.0,
        },
        name_dataset=NamesDataset.EMNIST_DIGITS,
        option_hgp={
            KeysOptionHGP.DEPTH: 1000,
            KeysOptionHGP.DUMPING: 5e-2,
            KeysOptionHGP.N_PUSH: 100,
            KeysOptionHGP.USE_TRUE_EXPECTED_EDGES: False,
            KeysOptionHGP.USE_TRUE_DEBIAS_WEIGHTS: False,
            KeysOptionHGP.INIT_DEBIAS_WEIGHT: True,
            KeysOptionHGP.INDEPENDENT_JACOBS: True,
        },
        fix_random_seed_value=1234,
        score=NamesMetricDiff.KENDALL_TAU,
        kwargs_isclose_diff={'rtol': 1e-1},
        rate_ignore_nonclose_param=0.0,
        scale='linear',
        n_out=50,
    )
