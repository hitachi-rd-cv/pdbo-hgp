from constants import *
from task import *


def get_config():
    n_nodes = 100
    return dict(
        workspace_directory="./processed",
        db_name=os.environ.get('MONGO_DB', "local"),
        mongo_auth=os.environ.get('MONGO_AUTH', None),
        n_nodes=n_nodes,
        name_model=AbbrModels.CNN_EMNIST,
        kwargs_build_base={},
        mode_graph=ModesGraph.ERDOS_RENYI,
        kwargs_init_graph={'p': 0.4},
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
        lr=0.01,
        batch_size=128,
        n_steps=200,
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
        fix_random_seed_value=1234,
        configs_overwrite=[
            # "FedAvg",
            dict(
                kwargs_fedem={
                    'communication_probability': 0.1,
                    'decentralized': False,
                    'device': 'cuda',
                    'experiment': 'emnist',
                    'input_dimension': None,
                    'local_steps': 1,
                    'locally_tune_clients': False,
                    'log_freq': 5,
                    'lr_lambda': 0.0,
                    'lr_scheduler': 'multi_step',
                    'method': 'FedAvg',
                    'mu': 0,
                    'n_learners': 1,
                    'optimizer': 'sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'seed': 1234,
                    'validation': False,
                    'verbose': 1
                }
            ),
            # "FedAvg + local adaption",
            dict(
                kwargs_fedem={
                    'communication_probability': 0.1,
                    'decentralized': False,
                    'device': 'cuda',
                    'experiment': 'emnist',
                    'input_dimension': None,
                    'local_steps': 1,
                    'locally_tune_clients': True,
                    'log_freq': 5,
                    'lr_lambda': 0.0,
                    'lr_scheduler': 'multi_step',
                    'method': 'FedAvg',
                    'mu': 0,
                    'n_learners': 1,
                    'optimizer': 'sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'seed': 1234,
                    'validation': False,
                    'verbose': 1
                }
            ),
            # "Local",
            dict(
                kwargs_fedem={
                    'communication_probability': 0.1,
                    'decentralized': False,
                    'device': 'cuda',
                    'experiment': 'emnist',
                    'input_dimension': None,
                    'local_steps': 1,
                    'locally_tune_clients': False,
                    'log_freq': 5,
                    'lr_lambda': 0.0,
                    'lr_scheduler': 'multi_step',
                    'method': 'local',
                    'mu': 0,
                    'n_learners': 1,
                    'optimizer': 'sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'seed': 1234,
                    'validation': False,
                    'verbose': 1
                }
            ),
            # "Clustered FL",
            dict(
                kwargs_fedem={
                    'communication_probability': 0.1,
                    'decentralized': False,
                    'device': 'cuda',
                    'experiment': 'emnist',
                    'input_dimension': None,
                    'local_steps': 1,
                    'locally_tune_clients': False,
                    'log_freq': 5,
                    'lr_lambda': 0.0,
                    'lr_scheduler': 'multi_step',
                    'method': 'clustered',
                    'mu': 0,
                    'n_learners': 1,
                    'optimizer': 'sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'seed': 1234,
                    'validation': False,
                    'verbose': 1
                }
            ),
            # "FedProx",
            dict(
                kwargs_fedem={
                    'communication_probability': 0.1,
                    'decentralized': False,
                    'device': 'cuda',
                    'experiment': 'emnist',
                    'input_dimension': None,
                    'local_steps': 1,
                    'locally_tune_clients': False,
                    'log_freq': 5,
                    'lr_lambda': 0.0,
                    'lr_scheduler': 'multi_step',
                    'method': 'FedProx',
                    'mu': 0.1,
                    'n_learners': 1,
                    'optimizer': 'prox_sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'seed': 1234,
                    'validation': False,
                    'verbose': 1
                }
            ),
            # "FedEM",
            dict(
                kwargs_fedem={
                    'communication_probability': 0.1,
                    'decentralized': False,
                    'device': 'cuda',
                    'experiment': 'emnist',
                    'input_dimension': None,
                    'local_steps': 1,
                    'locally_tune_clients': False,
                    'log_freq': 5,
                    'lr_lambda': 0.0,
                    'lr_scheduler': 'multi_step',
                    'method': 'FedEM',
                    'mu': 0,
                    'n_learners': 3,
                    'optimizer': 'sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'seed': 1234,
                    'validation': False,
                    'verbose': 1
                },
                lr=0.05,
            ),
            # "FedEM (Decentralized)",
            dict(
                kwargs_fedem={
                    'communication_probability': 0.1,
                    'decentralized': True,
                    'device': 'cuda',
                    'experiment': 'emnist',
                    'input_dimension': None,
                    'local_steps': 1,
                    'locally_tune_clients': False,
                    'log_freq': 5,
                    'lr_lambda': 0.0,
                    'lr_scheduler': 'multi_step',
                    'method': 'FedEM',
                    'mu': 0,
                    'n_learners': 3,
                    'optimizer': 'sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'seed': 1234,
                    'validation': False,
                    'verbose': 1
                },
                lr=0.05,
            ),
            # "FedAvg (Decentralized)",
            dict(
                kwargs_fedem={
                    'communication_probability': 0.1,
                    'decentralized': True,
                    'device': 'cuda',
                    'experiment': 'emnist',
                    'input_dimension': None,
                    'local_steps': 1,
                    'locally_tune_clients': False,
                    'log_freq': 5,
                    'lr_lambda': 0.0,
                    'lr_scheduler': 'multi_step',
                    'method': 'FedAvg',
                    'mu': 0,
                    'n_learners': 1,
                    'optimizer': 'sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'seed': 1234,
                    'validation': False,
                    'verbose': 1
                }
            ),
        ],
        names_option=[
            "FedAvg",
            "FedAvg + local adaption",
            "Local",
            "Clustered FL",
            "FedProx",
            "FedEM",
            "FedEM (Decentralized)",
            "FedAvg (Decentralized)",
        ],
        kwargs_fedem={
            'weight_decay': 1e-3,
        },
        seed_dataset=12345,
        use_train_for_outer_loss=True,
    )
