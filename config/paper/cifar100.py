from config.paper.base import get_config_overwrite, merge_base_config
from constants import NamesDataset, AbbrModels, NamesHyperParam


def get_config():
    n_nodes = 20
    n_steps = 1200
    name_dataset = NamesDataset.CIFAR100
    option_dataset = {
        'pathological_split': False,
        'pachinko_allocation_split': True,
        'n_shards': 2,
        'n_components': 3,
        'alpha': 0.4,
        'beta': 10.0,
        's_frac': 1.0,
        'tr_frac': 0.4,
        'val_frac': 0.25,
        'test_tasks_frac': 0.0,
    }
    model = AbbrModels.MN_CIFAR100

    configs_overwrite = []
    names_option = []

    hyper_learning_rate = 0.1
    conf = dict(name_hyperparam=NamesHyperParam.HYPER_SOFTMAX_LOGITS_WEIGHTS, hyper_learning_rate=hyper_learning_rate, model=model, n_steps=n_steps, lr=0.1)
    names_option.append(", ".join([f"{k}={v}" for k, v in conf.items()]))
    configs_overwrite.append(get_config_overwrite(**conf))

    hyper_learning_rate = 1e0
    conf = dict(name_hyperparam=NamesHyperParam.LEARNERS_WEIGHTS, hyper_learning_rate=hyper_learning_rate, model=model, n_steps=n_steps, lr=0.2)
    names_option.append(", ".join([f"{k}={v}" for k, v in conf.items()]))
    configs_overwrite.append(get_config_overwrite(**conf))

    hyper_learning_rate = 1e-1
    conf = dict(name_hyperparam=NamesHyperParam.LEARNERS_WEIGHTS_AND_SINGLE_SOFTMAX_LOGITS_WEIGHTS, hyper_learning_rate=hyper_learning_rate, model=model, n_steps=n_steps, lr=0.2)
    names_option.append(", ".join([f"{k}={v}" for k, v in conf.items()]))
    configs_overwrite.append(get_config_overwrite(**conf))

    return merge_base_config(configs_overwrite, n_nodes, name_dataset, names_option, option_dataset)

if __name__ == "__main__":
    config = get_config()
    print(*[f'"{x}"' for x in config["names_option"]], sep=",\n")
