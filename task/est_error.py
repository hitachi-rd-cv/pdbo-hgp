from luigi import DictParameter, BoolParameter, IntParameter, FloatParameter
from luigi.util import requires

from constants import KeysTarget, NamesHGPMetric
from lib_common import waluigi
from method_task.run_hgp import run_hyper_gradient_push
from task.init import MakeDatasets, BuildGraph, MakeKwargsBuildClients
from task.sgp import StochasticGradPush


@requires(StochasticGradPush, MakeDatasets, BuildGraph, MakeKwargsBuildClients)
class ComputeTrueHyperGrad(waluigi.TaskBase):
    option_eval_metric: dict = DictParameter()
    option_hgp_insignificant: dict = DictParameter(significant=False)
    lr: float = FloatParameter()
    batch_size: int = IntParameter()
    n_steps: int = IntParameter()
    option_hgp: dict = DictParameter()
    shuffle_train = BoolParameter()
    option_train_insignificant = DictParameter(significant=False)
    option_train_significant = DictParameter()
    use_train_for_outer_loss = BoolParameter()

    def run(self):
        ds_loaded = self.load()

        hypergrads_nodes, _, state_dicts_estimator = run_hyper_gradient_push(
            n_nodes=self.n_nodes,
            name_model=self.name_model,
            kwargs_build_nodes=ds_loaded[3],
            option_eval_metric=self.option_eval_metric,
            option_hgp_insignificant=self.option_hgp_insignificant,
            lrs=[self.lr] * self.n_nodes,
            batch_sizes=[self.batch_size] * self.n_nodes,
            n_steps=self.n_steps,
            use_cuda=self.use_cuda,
            option_hgp=self.option_hgp,
            shuffle_train=self.shuffle_train,
            datasets_valid=ds_loaded[1][KeysTarget.VALID],
            datasets_train=ds_loaded[1][KeysTarget.TRAIN],
            state_dicts=ds_loaded[0][KeysTarget.STATE_DICTS],
            seed=self.fix_random_seed_value,
            option_train_insignificant=self.option_train_insignificant,
            option_train_significant=self.option_train_significant,
            state_dict_graph=ds_loaded[2][KeysTarget.STATE_DICT_GRAPH],
            mode_graph=self.mode_graph,
            use_train_for_outer_loss=self.use_train_for_outer_loss,
            compute_hg=True,
            mode='true',
        )
        self.dump({
            KeysTarget.HYPER_GRADIENTS_NODES: hypergrads_nodes,
            KeysTarget.STATE_DICTS_ESTIMATOR: state_dicts_estimator,
        })


@requires(StochasticGradPush, MakeDatasets, BuildGraph, MakeKwargsBuildClients, ComputeTrueHyperGrad)
class RecordStepHyperGrads(waluigi.TaskBase):
    option_eval_metric: dict = DictParameter()
    option_hgp_insignificant: dict = DictParameter(significant=False)
    lr: float = FloatParameter()
    batch_size: int = IntParameter()
    n_steps: int = IntParameter()
    option_hgp: dict = DictParameter()
    shuffle_train = BoolParameter()
    option_train_insignificant = DictParameter(significant=False)
    option_train_significant = DictParameter()
    seed_hgp = IntParameter(None)
    use_train_for_outer_loss = BoolParameter()

    def run(self):
        if self.seed_hgp is None:
            seed = self.fix_random_seed_value
        else:
            seed = self.seed_hgp

        ds_loaded = self.load()
        hypergrads_nodes, state_dicts_estimator, hypergrads_nodes_steps = self.run_in_sacred_experiment(
            run_hyper_gradient_push,
            n_nodes=self.n_nodes,
            name_model=self.name_model,
            kwargs_build_nodes=ds_loaded[3],
            option_eval_metric=self.option_eval_metric,
            option_hgp_insignificant=self.option_hgp_insignificant,
            lrs=[self.lr] * self.n_nodes,
            batch_sizes=[self.batch_size] * self.n_nodes,
            n_steps=self.n_steps,
            use_cuda=self.use_cuda,
            option_hgp=self.option_hgp,
            shuffle_train=self.shuffle_train,
            datasets_valid=ds_loaded[1][KeysTarget.VALID],
            datasets_train=ds_loaded[1][KeysTarget.TRAIN],
            state_dicts=ds_loaded[0][KeysTarget.STATE_DICTS],
            seed=seed,
            option_train_insignificant=self.option_train_insignificant,
            option_train_significant=self.option_train_significant,
            state_dict_graph=ds_loaded[2][KeysTarget.STATE_DICT_GRAPH],
            mode_graph=self.mode_graph,
            compute_hg=True,
            hypergrads_nodes_true=ds_loaded[4][KeysTarget.HYPER_GRADIENTS_NODES],
            save_intermediate_hypergradients=True,
            true_backward_mode=False,
            use_train_for_outer_loss=self.use_train_for_outer_loss
        )
        self.dump({
            KeysTarget.HYPER_GRADIENTS_NODES: hypergrads_nodes,
            KeysTarget.HYPER_GRADIENTS_NODES_STEPS: hypergrads_nodes_steps,
            KeysTarget.STATE_DICTS_ESTIMATOR: state_dicts_estimator,
        })


@requires(RecordStepHyperGrads, ComputeTrueHyperGrad)
class ComputeHyperGradErrorOfSteps(waluigi.TaskBase):
    def run(self):
        result_hgp, result_true = self.load()
        error_norm_steps = self.run_in_sacred_experiment(
            self.compute_hypergrads_error,
            hypergrads_nodes_steps=result_hgp[KeysTarget.HYPER_GRADIENTS_NODES_STEPS],
            hypergrads_nodes_true=result_true[KeysTarget.HYPER_GRADIENTS_NODES],
        )

        self.dump({
            KeysTarget.ERROR_NORM_HYPERGRAD_STEPS: error_norm_steps,
        })

    @staticmethod
    def compute_hypergrads_error(hypergrads_nodes_steps, hypergrads_nodes_true, _run=None):
        error_norm_steps = []
        for step, hypergrads_nodes in enumerate(hypergrads_nodes_steps):
            # compute estimation error
            sum_squared_norm = 0.
            for hypergrads, hypergrads_true in zip(hypergrads_nodes, hypergrads_nodes_true):
                for v, v_true in zip(hypergrads, hypergrads_true):
                    sum_squared_norm += (v - v_true) @ (v - v_true)
            error_norm = sum_squared_norm ** 0.5
            error_norm_steps.append(error_norm)

            # send log to sacred
            if _run is not None:
                _run.log_scalar(NamesHGPMetric.V_DIFF_NORM, error_norm.item(), step)

        return error_norm_steps
