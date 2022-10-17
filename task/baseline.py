from luigi import DictParameter, IntParameter, FloatParameter
from luigi.util import requires

from constants import KeysTarget
from fedem.run_experiment import get_state_dicts_model_in_fedem_experiment
from lib_common import waluigi
from method_task.run_baseline_on_fedem_setting import run_baselines
from task.init import MakeDatasets, BuildGraph


class InitModelsForFedEmSetting(waluigi.TaskBase):
    n_nodes: int = IntParameter()
    kwargs_fedem = DictParameter()
    kwargs_build_base: dict = DictParameter()
    lr: float = FloatParameter()
    n_steps = IntParameter()

    def run(self):
        state_dicts_models = get_state_dicts_model_in_fedem_experiment(
            kwargs_fedem=self.kwargs_fedem,
            n_nodes=self.n_nodes,
            kwargs_model=self.kwargs_build_base['kwargs_model'],
            lrs=[self.lr] * self.n_nodes,
            n_steps=self.n_steps,
        )
        self.dump({
            KeysTarget.STATE_DICTS_LEARNERS: state_dicts_models,
        })


class InitBaselineModelsForFedEmSetting(waluigi.TaskBase):
    n_nodes: int = IntParameter()
    kwargs_fedem = DictParameter()
    lr: float = FloatParameter()
    n_steps = IntParameter()

    def run(self):
        state_dicts_models = get_state_dicts_model_in_fedem_experiment(
            kwargs_fedem=self.kwargs_fedem,
            n_nodes=self.n_nodes,
            lrs=[self.lr] * self.n_nodes,
            n_steps=self.n_steps,
        )
        self.dump({
            KeysTarget.STATE_DICTS_LEARNERS: state_dicts_models,
        })


@requires(MakeDatasets, InitBaselineModelsForFedEmSetting, BuildGraph)
class RunBaselineOnFedEmSetting(waluigi.TaskBase):
    batch_size: int = IntParameter()

    def run(self):
        d_dataset, d_state_dicts_models, d_graph = self.load()
        state_dicts_model, state_dicts_optimizer, _, _, d_metric, _ = self.run_in_sacred_experiment(
            run_baselines,
            n_nodes=self.n_nodes,
            batch_sizes=[self.batch_size] * self.n_nodes,
            datasets_valid=d_dataset[KeysTarget.VALID],
            datasets_train=d_dataset[KeysTarget.TRAIN],
            datasets_test=d_dataset[KeysTarget.TEST],
            state_dicts_models_init=d_state_dicts_models[KeysTarget.STATE_DICTS_LEARNERS],
            logs_dir=self.make_and_get_temporary_directory(),
            kwargs_fedem=self.kwargs_fedem,
            lrs=[self.lr] * self.n_nodes,
            n_steps=self.n_steps,
            mode_graph=self.mode_graph,
            state_dict_graph=d_graph[KeysTarget.STATE_DICT_GRAPH],
            use_cuda=self.use_cuda,
        )
        self.dump({
            KeysTarget.STATE_DICTS_MODEL: state_dicts_model,
            KeysTarget.STATE_DICTS_HYPER_OPTIMIZER: state_dicts_optimizer,
            KeysTarget.D_METRIC_FEDEM: d_metric,
        })


