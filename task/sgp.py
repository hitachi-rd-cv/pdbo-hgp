from luigi import DictParameter, IntParameter, BoolParameter, FloatParameter
from luigi.util import requires

from constants import KeysTarget
from lib_common import waluigi
from method_task.check_consensus import assert_consensus
from method_task.run_sgp import run_stochastic_gradient_push
from task.init import BuildClients, MakeDatasets, BuildGraph, MakeKwargsBuildClients


@requires(BuildClients, MakeDatasets, BuildGraph, MakeKwargsBuildClients)
class StochasticGradPush(waluigi.TaskBase):
    lr: float = FloatParameter()
    batch_size: int = IntParameter()
    n_steps: int = IntParameter()
    shuffle_train = BoolParameter()
    option_train_insignificant = DictParameter(significant=False)
    option_train_significant = DictParameter()

    def run(self):
        ds_loaded = self.load()

        state_dicts, log_train = self.run_in_sacred_experiment(
            run_stochastic_gradient_push,
            n_nodes=self.n_nodes,
            name_model=self.name_model,
            kwargs_build_nodes=ds_loaded[3],
            lrs=[self.lr] * self.n_nodes,
            batch_sizes=[self.batch_size] * self.n_nodes,
            n_steps=self.n_steps,
            use_cuda=self.use_cuda,
            shuffle_train=self.shuffle_train,
            datasets_valid=ds_loaded[1][KeysTarget.VALID],
            datasets_train=ds_loaded[1][KeysTarget.TRAIN],
            state_dicts=ds_loaded[0][KeysTarget.STATE_DICTS],
            seed=self.fix_random_seed_value,
            option_train_insignificant=self.option_train_insignificant,
            option_train_significant=self.option_train_significant,
            state_dict_graph=ds_loaded[2][KeysTarget.STATE_DICT_GRAPH],
            mode_graph=self.mode_graph,
        )
        self.dump({
            KeysTarget.STATE_DICTS: state_dicts,
            KeysTarget.LOG_TRAIN: log_train,
        })

@requires(StochasticGradPush, MakeKwargsBuildClients)
class CheckConsensus(waluigi.TaskBase):
    def run(self):
        ds_loaded = self.load()

        assert_consensus(
            n_nodes=self.n_nodes,
            name_model=self.name_model,
            kwargs_build_nodes=ds_loaded[1],
            state_dicts=ds_loaded[0][KeysTarget.STATE_DICTS],
            use_cuda=self.use_cuda,
        )

        self.dump(None)