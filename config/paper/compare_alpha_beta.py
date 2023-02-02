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
            'name_hyperparam': NamesHyperParam.SOFTMAX_CATEGORY_WEIGHTS,
            'kwargs_hyperparam': {},
            'kwargs_model': {'ndim_x': 1},
        },
        kwargs_init_hparams={},
        mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
        kwargs_init_graph={'low': 0.4, 'high': 0.8, 'p': 1.0, 'force_sparse': False},
        option_eval_metric={KeysOptionEval.NAME: NamesEvalMetric.LOSS_BARE_MEAN},
        option_hgp_insignificant={
            KeysOptionHGPInsig.NAMES_METRIC_LOG: [
                NamesHGPMetric.V_NORM,
                NamesHGPMetric.V_DIFF_NORM,
                NamesHGPMetric.U_NORM,
            ],
        },
        option_train_significant={
            KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT: False,
            KeysOptionTrainSig.LR_SCHEDULER: NamesLrScheduler.CONST,
            KeysOptionTrainSig.KWARGS_LR_SCHEDULER: {},
            KeysOptionTrainSig.DROP_LAST: False,
            KeysOptionTrainSig.MODE_SGP: ModesSGP.NEDIC,
        },
        option_train_insignificant={KeysOptionTrainInsig.LOG_EVERY: 1, KeysOptionTrainInsig.NAMES_METRIC_LOG: []},
        lr=2e0,
        batch_size=99999,
        n_steps=500,
        use_cuda=True,
        check_true_consistency=False,
        option_dataset={
            'n_classes': 2,
            'n_samples_train': 100,
            'n_samples_valid': 100,
        },
        name_dataset=NamesDataset.TOY_MNIST,
        option_hgp={
            KeysOptionHGP.DEPTH: 500,
            KeysOptionHGP.DUMPING: 1.,
            KeysOptionHGP.USE_TRUE_EXPECTED_EDGES: True,
            KeysOptionHGP.MODE_UPDATE: ModesHGPUpdate.ALT_V_U,
        },
        configs_overwrite=[
            dict(
                option_hgp={
                    KeysOptionHGP.ALPHA_V: 1.0,
                    KeysOptionHGP.ALPHA_W: 0.0,
                },
            ),
            dict(
                option_hgp={
                    KeysOptionHGP.ALPHA_V: 0.0,
                    KeysOptionHGP.ALPHA_W: 1.0,
                },
            ),
            dict(
                option_hgp={
                    KeysOptionHGP.ALPHA_V: 0.5,
                    KeysOptionHGP.ALPHA_W: 0.5,
                },
            ),
            dict(
                option_hgp={
                    KeysOptionHGP.ALPHA_V: 0.9,
                    KeysOptionHGP.ALPHA_W: 0.1,
                },
            ),
            dict(
                option_hgp={
                    KeysOptionHGP.ALPHA_V: 0.1,
                    KeysOptionHGP.ALPHA_W: 0.9,
                },
            ),
        ],
        names_option=[
            "alpha=1.0, beta=0.0",
            "alpha=0.0, beta=1.0",
            "alpha=0.5, beta=0.5",
            "alpha=0.9, beta=0.1",
            "alpha=0.1, beta=0.9",
        ],
        fix_random_seed_value=1234,
    )
