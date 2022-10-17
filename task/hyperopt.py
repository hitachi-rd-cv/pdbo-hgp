from datetime import timedelta

import luigi
from luigi import Parameter, DictParameter, IntParameter, BoolParameter, ListParameter, FloatParameter
from luigi.util import requires, inherits

from constants import KeysTarget
from lib_common import waluigi
from lib_task.hyper_step import init_hyperparams_of_nodes, any_nan_hypergrad
from method_task.run_hyper_step import hyper_opts_yield
from method_task.run_hyper_step_on_fedem_setting import run_hyper_sgd
from task import InitModelsForFedEmSetting
from task.init import MakeDatasets, BuildClientsWOHyperParams, BuildGraph, MakeKwargsBuildClients


@requires(MakeKwargsBuildClients)
class InitHyperParams(waluigi.TaskBase):
    bounds_hparam = ListParameter()
    hyper_optimizer = Parameter()
    hyper_learning_rate = FloatParameter()
    kwargs_hyper_optimizer = DictParameter()
    lrs_per_hyperparameter = ListParameter(None)

    def run(self):
        kwargs_build_nodes = self.load()
        state_dicts_hyper, state_dicts_hyper_optimizer = init_hyperparams_of_nodes(
            n_nodes=self.n_nodes,
            kwargs_build_nodes=kwargs_build_nodes,
            bounds_hparam=self.bounds_hparam,
            hyper_optimizer=self.hyper_optimizer,
            hyper_learning_rate=self.hyper_learning_rate,
            kwargs_hyper_optimizer=self.kwargs_hyper_optimizer,
            lrs_per_hyperparameter=self.lrs_per_hyperparameter,
        )
        self.dump({
            KeysTarget.STATE_DICTS_HYPER: state_dicts_hyper,
            KeysTarget.STATE_DICTS_HYPER_OPTIMIZER: state_dicts_hyper_optimizer,
            KeysTarget.TIME_HYPER_UPDATE: timedelta(0),
            KeysTarget.HYPER_GRADIENTS_NODES: None,
        })


@inherits(BuildClientsWOHyperParams, MakeDatasets, BuildGraph, MakeKwargsBuildClients, InitHyperParams)
class HyperSGDStep(waluigi.TaskBase):
    option_eval_metric: dict = DictParameter()
    lr: float = FloatParameter()
    batch_size: int = IntParameter()
    n_steps: int = IntParameter()
    shuffle_train = BoolParameter()
    options_eval_metric_hyper = ListParameter()
    save_state_dicts = BoolParameter()
    option_train_insignificant = DictParameter(significant=False)
    option_train_significant = DictParameter()
    t_h: int = luigi.IntParameter(None)

    option_hgp_insignificant: dict = DictParameter(significant=False)
    option_hgp: dict = DictParameter()
    use_train_for_outer_loss = BoolParameter()

    def requires(self):
        tasks = [
            self.clone(BuildClientsWOHyperParams),
            self.clone(MakeDatasets),
            self.clone(BuildGraph),
            self.clone(MakeKwargsBuildClients),
        ]

        if self.t_h == 0:
            tasks.append(self.clone(InitHyperParams))
        else:
            tasks.append(self.clone(self.__class__, t_h=self.t_h - 1))

        return tasks

    def get_method_kwargs(self):
        raise NotImplementedError

    def run(self, **kwargs):
        ds_loaded = self.load()

        if any_nan_hypergrad(ds_loaded[4][KeysTarget.HYPER_GRADIENTS_NODES]):
            # bypass the result to the end of the last hyper-step
            self.dump(ds_loaded[4])

        else:
            state_dicts_hyperparameters = ds_loaded[4][KeysTarget.STATE_DICTS_HYPER]
            state_dicts_optimizer = ds_loaded[4][KeysTarget.STATE_DICTS_HYPER_OPTIMIZER]

            state_dicts_hyperparameters, hypergrads_nodes, state_dicts_optimizer, state_dicts, time_past, d_d_val_eval_nodes, d_d_val_eval_mean, d_d_val_eval_bottom = self.run_in_sacred_experiment(
                hyper_opts_yield,
                n_nodes=self.n_nodes,
                name_model=self.name_model,
                kwargs_build_nodes=ds_loaded[3],
                option_eval_metric=self.option_eval_metric,
                lrs=[self.lr] * self.n_nodes,
                batch_sizes=[self.batch_size] * self.n_nodes,
                n_steps=self.n_steps,
                use_cuda=self.use_cuda,
                shuffle_train=self.shuffle_train,
                datasets_valid=ds_loaded[1][KeysTarget.VALID],
                datasets_train=ds_loaded[1][KeysTarget.TRAIN],
                datasets_test=ds_loaded[1][KeysTarget.TEST],
                options_eval_metric_hyper=self.options_eval_metric_hyper,
                state_dicts_init=ds_loaded[0][KeysTarget.STATE_DICTS],
                seed=self.fix_random_seed_value,
                save_state_dicts=self.save_state_dicts,
                option_train_insignificant=self.option_train_insignificant,
                option_train_significant=self.option_train_significant,
                state_dict_graph=ds_loaded[2][KeysTarget.STATE_DICT_GRAPH],
                mode_graph=self.mode_graph,
                t_h=self.t_h,
                state_dicts_hyperparameters=state_dicts_hyperparameters,
                state_dicts_optimizer=state_dicts_optimizer,
                hyper_learning_rate=self.hyper_learning_rate,
                option_hgp_insignificant=self.option_hgp_insignificant,
                option_hgp=self.option_hgp,
                hyper_optimizer=self.hyper_optimizer,
                kwargs_hyper_optimizer=self.kwargs_hyper_optimizer,
                use_train_for_outer_loss=self.use_train_for_outer_loss,
                lrs_per_hyperparameter=self.lrs_per_hyperparameter,
            )

            self.dump({
                KeysTarget.STATE_DICTS: state_dicts,
                KeysTarget.STATE_DICTS_HYPER: state_dicts_hyperparameters,
                KeysTarget.D_D_VAL_EVAL_NODES: d_d_val_eval_nodes,
                KeysTarget.D_D_VAL_EVAL_MEAN: d_d_val_eval_mean,
                KeysTarget.D_D_VAL_EVAL_BOTTOM: d_d_val_eval_bottom,
                KeysTarget.TIME_HYPER_UPDATE: time_past,
                KeysTarget.HYPER_GRADIENTS_NODES: hypergrads_nodes,
                KeysTarget.STATE_DICTS_HYPER_OPTIMIZER: state_dicts_optimizer,
            })


@inherits(HyperSGDStep, InitHyperParams)
class HyperSGD(waluigi.TaskBase):
    n_hyper_steps: int = IntParameter()

    def requires(self):
        return [self.clone(InitHyperParams)] + [self.clone(HyperSGDStep, t_h=step) for step in
                                                range(self.n_hyper_steps)]

    def run(self):
        results = self.load()

        self.dump({
            KeysTarget.HYPER_STATE_DICTS_OF_HYPER_STEPS: [r[KeysTarget.STATE_DICTS_HYPER] for r in results],
            KeysTarget.STATE_DICTS_HYPER: results[-1][KeysTarget.STATE_DICTS_HYPER],
            KeysTarget.D_D_VAL_EVAL_MEAN_HYPER_STEPS: [r[KeysTarget.D_D_VAL_EVAL_MEAN] for r in results[1:]],
            KeysTarget.D_D_VAL_EVAL_BOTTOM_HYPER_STEPS: [r[KeysTarget.D_D_VAL_EVAL_BOTTOM] for r in results[1:]],
        })


@inherits(MakeDatasets, MakeKwargsBuildClients, InitModelsForFedEmSetting, InitHyperParams)
class HyperSGDStepOnFedEmSetting(waluigi.TaskBase):
    name_model = Parameter()
    option_eval_metric: dict = DictParameter()
    lr: float = FloatParameter()
    batch_size: int = IntParameter()
    n_steps: int = IntParameter()
    shuffle_train = BoolParameter()
    option_train_insignificant = DictParameter(significant=False)
    option_train_significant = DictParameter()
    save_state_dicts = BoolParameter()
    option_hgp_insignificant: dict = DictParameter(significant=False)
    option_hgp: dict = DictParameter()
    t_h: int = IntParameter(None)
    use_train_for_outer_loss = BoolParameter()

    def requires(self):
        tasks = [
            self.clone(MakeDatasets),
            self.clone(BuildGraph),
            self.clone(MakeKwargsBuildClients),
            self.clone(InitModelsForFedEmSetting),
        ]
        if self.t_h == 0:
            tasks.append(self.clone(InitHyperParams))
        else:
            tasks.append(self.clone(self.__class__, t_h=self.t_h - 1))

        return tasks

    def run(self, **kwargs):
        d_dataset, d_graph, kwargs_build_nodes, d_state_dicts_models, d_hyper = self.load()

        if any_nan_hypergrad(d_hyper[KeysTarget.HYPER_GRADIENTS_NODES]):
            # bypass the result to the end of the last hyper-step
            self.dump(d_hyper)

        else:
            state_dicts_hyperparameters = d_hyper[KeysTarget.STATE_DICTS_HYPER]
            state_dicts_hyper_optimizer = d_hyper[KeysTarget.STATE_DICTS_HYPER_OPTIMIZER]

            state_dicts_hyperparameters, hypergrads_nodes, state_dicts_hyper_optimizer, d_metric_mean, d_metric_bottom = self.run_in_sacred_experiment(
                run_hyper_sgd,
                n_nodes=self.n_nodes,
                name_model=self.name_model,
                kwargs_build_nodes=kwargs_build_nodes,
                option_eval_metric=self.option_eval_metric,
                lrs=[self.lr] * self.n_nodes,
                batch_sizes=[self.batch_size] * self.n_nodes,
                n_steps=self.n_steps,
                use_cuda=self.use_cuda,
                shuffle_train=self.shuffle_train,
                datasets_valid=d_dataset[KeysTarget.VALID],
                datasets_train=d_dataset[KeysTarget.TRAIN],
                datasets_test=d_dataset[KeysTarget.TEST],
                seed=self.fix_random_seed_value,
                option_train_insignificant=self.option_train_insignificant,
                option_train_significant=self.option_train_significant,
                state_dict_graph=d_graph[KeysTarget.STATE_DICT_GRAPH],
                mode_graph=self.mode_graph,
                hyper_learning_rate=self.hyper_learning_rate,
                option_hgp_insignificant=self.option_hgp_insignificant,
                option_hgp=self.option_hgp,
                hyper_optimizer=self.hyper_optimizer,
                kwargs_hyper_optimizer=self.kwargs_hyper_optimizer,
                kwargs_fedem=self.kwargs_fedem,
                state_dicts_models_init=d_state_dicts_models[KeysTarget.STATE_DICTS_LEARNERS],
                kwargs_model=self.kwargs_build_base['kwargs_model'],
                logs_dir=self.make_and_get_temporary_directory(),
                t_h=self.t_h,
                state_dicts_hyperparameters=state_dicts_hyperparameters,
                state_dicts_hyper_optimizer=state_dicts_hyper_optimizer,
                use_train_for_outer_loss=self.use_train_for_outer_loss,
                lrs_per_hyperparameter=self.lrs_per_hyperparameter,
            )

            self.dump({
                KeysTarget.STATE_DICTS_HYPER: state_dicts_hyperparameters,
                KeysTarget.HYPER_GRADIENTS_NODES: hypergrads_nodes,
                KeysTarget.STATE_DICTS_HYPER_OPTIMIZER: state_dicts_hyper_optimizer,
                KeysTarget.D_METRIC_FEDEM: (d_metric_mean, d_metric_bottom),
            })


@inherits(HyperSGDStepOnFedEmSetting, InitHyperParams)
class HyperSGDOnFedEmSetting(waluigi.TaskBase):
    n_hyper_steps: int = IntParameter()

    def requires(self):
        return [self.clone(InitHyperParams)] + [self.clone(HyperSGDStepOnFedEmSetting, t_h=t_h) for t_h in
                                                range(self.n_hyper_steps)]

    def run(self):
        results = self.load()

        self.dump({
            KeysTarget.HYPER_STATE_DICTS_OF_HYPER_STEPS: [r[KeysTarget.STATE_DICTS_HYPER] for r in results],
            KeysTarget.STATE_DICTS_HYPER: results[-1][KeysTarget.STATE_DICTS_HYPER],
            KeysTarget.D_METRIC_FEDEM_HYPER_STEPS: [r[KeysTarget.D_METRIC_FEDEM] for r in results[1:]],
        })
