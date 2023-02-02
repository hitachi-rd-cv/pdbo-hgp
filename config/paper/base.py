from constants import *
from constants import NamesHyperParam, ModesGossip, NamesHyperLoss, AbbrModels
from task import *


def merge_base_config(configs_overwrite, n_nodes, name_dataset, names_option, option_dataset, hgp=True, use_train_for_outer_loss=True, local=False, n_hyper_steps=11):
    if hgp:
        conf_vrhgp = {
            KeysOptionHGP.ALPHA_V: 1.0,
            KeysOptionHGP.ALPHA_W: 0.0,
        }
    else:
        conf_vrhgp = {
            KeysOptionHGP.ALPHA_V: 0.9,
            KeysOptionHGP.ALPHA_W: 0.1,
        }

    if local:
        conf_graph = dict(
            mode_graph=ModesGraph.SELF_LOOP,
            kwargs_init_graph={},
        )
    else:
        conf_graph = dict(
            mode_graph=ModesGraph.STOCHASTIC_DIRECTED,
            kwargs_init_graph={'low': 0.4, 'high': 0.8, 'p': 1.0, 'force_sparse': False},
        )

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
            **conf_vrhgp,
            KeysOptionHGP.DEPTH: 20,
        },
        lr=None,
        kwargs_init_hparams={},
        **conf_graph,
        option_eval_metric={
            KeysOptionEval.NAME: NamesEvalMetric.LOSS_BARE_MEAN},
        option_train_significant={
            KeysOptionTrainSig.LR_SCHEDULER: NamesLrScheduler.MULTI_STEP,
            KeysOptionTrainSig.DROP_LAST: True,
            KeysOptionTrainSig.DISABLE_DEBIAS_WEIGHT: False,
            KeysOptionTrainSig.MODE_SGP: ModesSGP.NEDIC,
        },
        option_train_insignificant={
            KeysOptionTrainInsig.LOG_EVERY: 5,
            KeysOptionTrainInsig.NAMES_METRIC_LOG: [
                # NamesEvalMetric.LOSS_BARE_MEAN,
                # NamesEvalMetric.LOSS_MEAN,
                # NamesEvalMetric.ACCURACY,
            ]},
        option_hgp_insignificant={
            KeysOptionHGPInsig.NAMES_METRIC_LOG: [
                # NamesHGPMetric.V_NORM,
                # NamesHGPMetric.U_NORM,
                # NamesHGPMetric.W_NORM,
            ],
        },
        batch_size=128,
        n_steps=None,
        use_cuda=True,
        option_dataset=option_dataset,
        name_dataset=name_dataset,
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
        n_hyper_steps=n_hyper_steps,
        hyper_optimizer=HyperOptimizers.ADAM,
        kwargs_hyper_optimizer={},
        hyper_learning_rate=1e-1,
        use_train_for_outer_loss=use_train_for_outer_loss,
        n_steps_logged_train=None,
    )


def get_config_overwrite(name_hyperparam, hyper_learning_rate, model, n_steps, lr, gamma=1e-3):
    if name_hyperparam in [NamesHyperParam.SOFTMAX_CATEGORY_WEIGHTS, NamesHyperParam.HYPER_SOFTMAX_LOGITS_WEIGHTS]:
        if gamma is None:
            kwargs_hyperparam = {}
        else:
            kwargs_hyperparam = {"hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": gamma}

        return dict(
            name_model=model,
            kwargs_build_base={
                'mode_gossip': ModesGossip.NORMALIZE,
                'name_hyperparam': name_hyperparam,
                'kwargs_hyperparam': kwargs_hyperparam,
                'kwargs_model': {'weight_decay': 1e-3},
            }
            ,
            lr=lr,
            hyper_learning_rate=hyper_learning_rate,
            option_train_significant={
                KeysOptionTrainSig.KWARGS_LR_SCHEDULER: {"milestones": [n_steps - 100, n_steps - 50]},
            },
            n_steps=n_steps,
        )

    elif name_hyperparam == NamesHyperParam.LEARNERS_WEIGHTS:
        if model == AbbrModels.LSTM_SHAKESPEARE:
            ensemble_model = AbbrModels.LANGUAGE_LEARNERS_ENSEMBLE
        else:
            ensemble_model = AbbrModels.LEARNERS_ENSEMBLE

        return dict(
            name_model=ensemble_model,
            kwargs_build_base={
                'mode_gossip': ModesGossip.NORMALIZE,
                'name_hyperparam': name_hyperparam,
                'kwargs_hyperparam': {'n_learners': 3, "hyper_loss": NamesHyperLoss.L2_REGULARIZER, "gamma": 1e-2},
                'kwargs_model': {
                    'n_learners': 3,
                    'name_learner_model': model,
                    'kwargs_learner': {'weight_decay': 1e-3},
                }
            },
            lr=lr,
            hyper_learning_rate=hyper_learning_rate,
            option_train_significant={
                KeysOptionTrainSig.KWARGS_LR_SCHEDULER: {"milestones": [n_steps - 100, n_steps - 50]},
            },
            n_steps=n_steps,
        )

    elif name_hyperparam in [NamesHyperParam.LEARNERS_WEIGHTS_AND_SOFTMAX_CATEGORY_WEIGHTS, NamesHyperParam.LEARNERS_WEIGHTS_AND_MULTI_SOFTMAX_LOGITS_WEIGHTS, NamesHyperParam.LEARNERS_WEIGHTS_AND_SINGLE_SOFTMAX_LOGITS_WEIGHTS]:
        if gamma is None:
            kwargs_hyperparam = {"hyper_loss_categories": NamesHyperLoss.L2_REGULARIZER, "gamma_categories": 5e-4}
        else:
            kwargs_hyperparam = {"hyper_loss_categories": NamesHyperLoss.L2_REGULARIZER, "gamma_categories": gamma}

        if model == AbbrModels.LSTM_SHAKESPEARE:
            ensemble_model = AbbrModels.LANGUAGE_LEARNERS_ENSEMBLE
        else:
            ensemble_model = AbbrModels.LEARNERS_ENSEMBLE

        return dict(
            name_model=ensemble_model,
            kwargs_build_base={
                'mode_gossip': ModesGossip.NORMALIZE,
                'name_hyperparam': name_hyperparam,
                'kwargs_hyperparam': {
                    'n_learners': 3,
                    "hyper_loss_learners": NamesHyperLoss.L2_REGULARIZER,
                    "gamma_learners": 1e-2,
                    **kwargs_hyperparam,
                },
                'kwargs_model': {
                    'n_learners': 3,
                    'name_learner_model': model,
                    'kwargs_learner': {'weight_decay': 1e-3},
                }
            },
            lr=lr,
            lrs_per_hyperparameter=[hyper_learning_rate, 1e-1],
            hyper_learning_rate=None,
            option_train_significant={
                KeysOptionTrainSig.KWARGS_LR_SCHEDULER: {"milestones": [n_steps - 100, n_steps - 50]},
            },
            n_steps=n_steps,
        )

    else:
        raise ValueError(name_hyperparam)
