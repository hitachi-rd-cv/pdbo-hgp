from constants import *
from task import *


def get_config():
    n_nodes = 100
    configs_overwrite = []
    names_option = []

    for lr in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
        names_option.append(f"(lr={lr}) FedAvg")
        # "FedAvg",
        configs_overwrite.append(dict(
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
                'validation': False,
                'verbose': 1
            },
            lr=lr,
        ))
        # "FedAvg + local adaption",
        names_option.append(f"(lr={lr}) FedAvg + local adaption")
        configs_overwrite.append(dict(
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
                'validation': False,
                'verbose': 1
            },
            lr=lr,
        ))

        # "Local",
        names_option.append(f"(lr={lr}) Local")
        configs_overwrite.append(dict(
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
                'validation': False,
                'verbose': 1
            },
            lr=lr,
        ))
        # "Clustered FL",
        names_option.append(f"(lr={lr}) Clustered FL")
        configs_overwrite.append(dict(
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
                'validation': False,
                'verbose': 1
            },
            lr=lr,
        ))

        # "FedEM",
        names_option.append(f"(lr={lr}) FedEM")
        configs_overwrite.append(dict(
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
                'validation': False,
                'verbose': 1
            },
            lr=lr,
        ))

        # "FedEM (Decentralized)",
        names_option.append(f"(lr={lr}) FedEM (Decentralized)")
        configs_overwrite.append(dict(
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
                'validation': False,
                'verbose': 1
            },
            lr=lr,
        ))

        # "FedAvg (Decentralized)",
        names_option.append(f"(lr={lr}) FedAvg (Decentralized)")
        configs_overwrite.append(dict(
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
                'validation': False,
                'verbose': 1
            },
            lr=lr,
        ))

        for mu in [1e1, 1e0, 1e-1, 1e-2]:
            # "Personalized (Richtarek's Formulation)",
            names_option.append(f"(lr={lr}, mu={mu}) Personalized (Richtarek's Formulation)")
            configs_overwrite.append(dict(
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
                    # 'method': 'personalized', ??
                    'method': 'pFedMe',
                    'mu': mu,
                    'n_learners': 1,
                    'optimizer': 'prox_sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'validation': False,
                    'verbose': 1
                },
                lr=lr,
            ))

            # "FedProx",
            names_option.append(f"(lr={lr}, mu={mu}) FedProx")
            configs_overwrite.append(dict(
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
                    'mu': mu,
                    'n_learners': 1,
                    'optimizer': 'prox_sgd',
                    'output_dimension': None,
                    'q': 1.0,
                    'sampling_rate': 1.0,
                    'validation': False,
                    'verbose': 1
                },
                lr=lr,
            ))

    return dict(
        workspace_directory="./processed",
        db_name=os.environ.get('MONGO_DB', "local"),
        mongo_auth=os.environ.get('MONGO_AUTH', None),
        n_nodes=n_nodes,
        name_model=AbbrModels.CNN_EMNIST,
        kwargs_build_base={},
        mode_graph=ModesGraph.ERDOS_RENYI,
        kwargs_init_graph={'p': 0.5},
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
        lr=None,
        batch_size=128,
        n_steps=200,
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
        fix_random_seed_value=1234,
        configs_overwrite=configs_overwrite,
        names_option=names_option,
        kwargs_fedem={
            'weight_decay': 1e-3,
        },
        seed_dataset=12345,
        use_train_for_outer_loss=True,
    )


if __name__ == "__main__":
    config = get_config()
    print(*[f'"{x}"' for x in config["names_option"]], sep=",\n")
