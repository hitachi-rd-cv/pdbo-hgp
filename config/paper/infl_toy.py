from constants import *
from task import *


def get_config():
    n_nodes = 3
    return dict(
        workspace_directory="./processed",
        db_name=os.environ.get('MONGO_DB', "local"),
        mongo_auth=os.environ.get('MONGO_AUTH', None),
        n_nodes=n_nodes,
        name_model=AbbrModels.LR_TOY,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.LOSS_MASKS,
            'kwargs_hyperparam': {},
            'kwargs_model': {'ndim_x': 5},
        },
        kwargs_init_hparams={},
        mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
        kwargs_init_graph={'low': 0.4, 'high': 0.8, 'p': 1.0, 'force_sparse': False},
        option_eval_metric={KeysOptionEval.NAME: NamesEvalMetric.LOSS_BARE_MEAN},
        option_hgp_insignificant={
            KeysOptionHGPInsig.NAMES_METRIC_LOG: [
            ],
        },
        shuffle_train=True,
        option_train_significant={
            KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT: False,
            KeysOptionTrainSig.LR_SCHEDULER: NamesLrScheduler.MULTI_STEP,
            KeysOptionTrainSig.KWARGS_LR_SCHEDULER: {'milestones': [3000, 4000, 4500, 4600, 4700, 4800, 4900, 4950, 4990, 4999]},
            KeysOptionTrainSig.DROP_LAST: False,
            KeysOptionTrainSig.MODE_SGP: ModesSGP.ASSRAN,
        },
        option_train_insignificant={
            KeysOptionTrainInsig.LOG_EVERY: 1,
            KeysOptionTrainInsig.NAMES_METRIC_LOG: []
        },
        lr=2e0,
        batch_size=99999,
        n_steps=5000,
        use_cuda=True,
        option_dataset={
            'n_classes': 2,
            'n_components': 3,
            'dimension': 5,
            'noise_level': 0.1,
            'n_test': 5000,
            'alpha': 0.4,
            'uniform_marginal': True,
            'box': (-1.0, 1.0),
            'min_num_samples': 100,
            'max_num_samples': 100,
        },
        name_dataset=NamesDataset.SYNTHETIC,
        option_hgp={
            KeysOptionHGP.DEPTH: 500,
            KeysOptionHGP.DUMPING: 1e0,
            KeysOptionHGP.USE_TRUE_EXPECTED_EDGES: False,
            KeysOptionHGP.USE_TRUE_DEBIAS_WEIGHTS: False,
            KeysOptionHGP.INIT_DEBIAS_WEIGHT: True,
            KeysOptionHGP.INDEPENDENT_JACOBS: True,
            KeysOptionHGP.N_PUSH: 100,
        },
        fix_random_seed_value=1234,
        score=NamesMetricDiff.R2,
        kwargs_isclose_diff={'rtol': 1e-1},
        rate_ignore_nonclose_param=0.0,
        scale='linear',
        n_out=50,
    )
