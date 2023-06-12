from constants import *
from task import *


def get_config():
    n_nodes = 100

    configs_overwrite = []
    names_option = []

    names_option.append('Local')
    configs_overwrite.append(dict(
        mode_graph=ModesGraph.SELF_LOOP,
        kwargs_init_graph={},
    ))

    names_option.append('SGP (undirected)')
    configs_overwrite.append(dict(
        mode_graph=ModesGraph.STOCHASTIC_BIDIRECTED,
        kwargs_init_graph={'low': 0.4, 'high': 0.8, 'p': 1.0, 'force_sparse': False},
    ))

    names_option.append('SGP (directed)')
    configs_overwrite.append(dict(
        mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
        kwargs_init_graph={'low': 0.4, 'high': 0.8, 'p': 1.0, 'force_sparse': False},
    ))

    return dict(
        workspace_directory="./processed",
        db_name=os.environ.get('MONGO_DB', "local"),
        mongo_auth=os.environ.get('MONGO_AUTH', None),
        n_nodes=n_nodes,
        name_model=AbbrModels.CNN_EMNIST,
        kwargs_build_base={
            'mode_gossip': ModesGossip.NORMALIZE,
            'name_hyperparam': NamesHyperParam.DUMMY,
            'kwargs_hyperparam': {},
            'kwargs_model': {'weight_decay': 1e-3},
        },
        mode_graph=None,
        kwargs_init_graph={},
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
        lr=0.1,
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
        option_hgp={
            KeysOptionHGP.DEPTH: 1,
            KeysOptionHGP.USE_TRUE_EXPECTED_EDGES: False,
            KeysOptionHGP.DUMPING: 1.0,
        },
        name_dataset=NamesDataset.EMNIST,
        shuffle_train=True,
        fix_random_seed_value=1234,
        bounds_hparam=[0., 0.],
        names_checked_dataset=[NamesLoader.VALID, NamesLoader.TEST],
        options_eval_metric_hyper=[
            {KeysOptionEval.NAME: NamesEvalMetric.LOSS_BARE_MEAN},
            {KeysOptionEval.NAME: NamesEvalMetric.ACCURACY},
        ],
        save_state_dicts=False,
        configs_overwrite=configs_overwrite,
        names_option=names_option,
        hyper_optimizer=HyperOptimizers.ADAM,
        kwargs_hyper_optimizer={},
        hyper_learning_rate=1e0,
        seed_dataset=12345,
        n_hyper_steps=1,
        use_train_for_outer_loss=True,
    )


if __name__ == "__main__":
    config = get_config()
    print(*[f'"{x}"' for x in config["names_option"]], sep=",\n")
