from config.paper.base import get_config_overwrite, merge_base_config
from constants import NamesDataset, AbbrModels, NamesHyperParam


def get_config():
    n_nodes = 20
    n_steps = 1200
    name_dataset = NamesDataset.SHAKE_SPEARE
    option_dataset = {
        's_frac': 1.0,
        'tr_frac': 0.8,
        'val_frac': 0.25,
        'train_tasks_frac': 1.0,
        'thres_n_lines': 2,
    }
    model = AbbrModels.LSTM_SHAKESPEARE

    configs_overwrite = []
    names_option = []

    hyper_learning_rate = 0.1
    conf = dict(name_hyperparam=NamesHyperParam.HYPER_SOFTMAX_LOGITS_WEIGHTS, hyper_learning_rate=hyper_learning_rate, model=model, n_steps=n_steps, lr=1.)
    names_option.append(", ".join([f"{k}={v}" for k, v in conf.items()]) + " (Local)")
    configs_overwrite.append(get_config_overwrite(**conf))

    conf = dict(name_hyperparam=NamesHyperParam.LEARNERS_WEIGHTS, hyper_learning_rate=hyper_learning_rate, model=model, n_steps=n_steps, lr=10.)
    names_option.append(", ".join([f"{k}={v}" for k, v in conf.items()]) + " (Local)")
    configs_overwrite.append(get_config_overwrite(**conf))

    return merge_base_config(configs_overwrite, n_nodes, name_dataset, names_option, option_dataset, local=True, n_hyper_steps=1)

if __name__ == "__main__":
    config = get_config()
    print(*[f'"{x}"' for x in config["names_option"]], sep=",\n")
