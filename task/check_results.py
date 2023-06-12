import luigi
from luigi import DictParameter, IntParameter, Parameter, BoolParameter, FloatParameter

from constants import KeysTarget
from lib_common import waluigi
from lib_common.waluigi.util import inherits, requires
from method_task.actual_diff import compute_actual_diff, gen_most_influential_perturbs_of_nodes_of_iters
from method_task.compare_val_eval_diffs import compare_val_eval_diffs
from method_task.linear_approx import linear_approx
from method_task.run_hgp import run_hyper_gradient_push
from task.init import MakeDatasets, BuildClients, BuildGraph, MakeKwargsBuildClients
from task.sgp import StochasticGradPush, CheckConsensus


@requires(StochasticGradPush, MakeDatasets, BuildGraph, MakeKwargsBuildClients, CheckConsensus)
class HyperGradPush(waluigi.TaskBase):
    option_eval_metric: str = DictParameter()
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

        hypergrads_nodes, state_dicts_estimator, _ = self.run_in_sacred_experiment(
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
            seed=self.fix_random_seed_value,
            option_train_insignificant=self.option_train_insignificant,
            option_train_significant=self.option_train_significant,
            state_dict_graph=ds_loaded[2][KeysTarget.STATE_DICT_GRAPH],
            mode_graph=self.mode_graph,
            use_train_for_outer_loss=self.use_train_for_outer_loss,
            compute_hg=True,
        )
        self.dump({
            KeysTarget.HYPER_GRADIENTS_NODES: hypergrads_nodes,
            KeysTarget.STATE_DICTS_ESTIMATOR: state_dicts_estimator,
        })


@requires(HyperGradPush, MakeKwargsBuildClients)
class GenMostInfluentialPerturbs(waluigi.TaskBase):
    n_out = IntParameter()
    n_nodes: int = IntParameter()
    name_model: str = Parameter()
    use_cuda: bool = BoolParameter()

    def run(self):
        d_hgp, kwargs_build_nodes = self.load()

        result = gen_most_influential_perturbs_of_nodes_of_iters(
            n_nodes=self.n_nodes,
            name_model=self.name_model,
            kwargs_build_nodes=kwargs_build_nodes,
            use_cuda=self.use_cuda,
            hypergrads_nodes=d_hgp[KeysTarget.HYPER_GRADIENTS_NODES],
            n_out=self.n_out,
        )
        self.dump({KeysTarget.HYPER_PERTURBS_ITERS: result})


@requires(BuildClients, StochasticGradPush, MakeDatasets, GenMostInfluentialPerturbs, BuildGraph, MakeKwargsBuildClients)
class ComputeActualDiffByMostInfluentialPerturbs(waluigi.TaskBase):
    sample_rate_tol_actual_diff_floating_point_error = luigi.FloatParameter(0.5)
    option_eval_metric: str = DictParameter()
    idx_hyper_perturbs = IntParameter(None)

    def run(self):
        ds_result = self.load()
        state_dicts_init = ds_result[0][KeysTarget.STATE_DICTS]
        state_dicts_trained = ds_result[1][KeysTarget.STATE_DICTS]
        hyper_perturbs_of_nodes = ds_result[3][KeysTarget.HYPER_PERTURBS_ITERS][self.idx_hyper_perturbs]
        state_dict_graph = ds_result[4][KeysTarget.STATE_DICT_GRAPH]
        kwargs_build_nodes = ds_result[5]
        diff_val_eval_approx = compute_actual_diff(n_nodes=self.n_nodes, name_model=self.name_model, kwargs_build_nodes=kwargs_build_nodes, state_dicts_init=state_dicts_init, state_dicts_trained=state_dicts_trained,
                                                   datasets_train=ds_result[2][KeysTarget.TRAIN], datasets_valid=ds_result[2][KeysTarget.VALID], option_eval_metric=self.option_eval_metric, lrs=[self.lr] * self.n_nodes,
                                                   batch_sizes=[self.batch_size] * self.n_nodes,
                                                   n_steps=self.n_steps, shuffle_train=self.shuffle_train, seed=self.fix_random_seed_value, option_train_insignificant=self.option_train_insignificant,
                                                   option_train_significant=self.option_train_significant,
                                                   hyper_perturbs_of_nodes=hyper_perturbs_of_nodes, state_dict_graph=state_dict_graph, mode_graph=self.mode_graph,
                                                   use_cuda=self.use_cuda)
        self.dump({KeysTarget.DIFFS_VAL_EVAL: diff_val_eval_approx})


@inherits(ComputeActualDiffByMostInfluentialPerturbs)
class RangeComputeActualDiffByMostInfluentialPerturbs(waluigi.TaskBase):
    def requires(self):
        return [self.clone(ComputeActualDiffByMostInfluentialPerturbs, idx_hyper_perturbs=idx_hyper_perturbs) for idx_hyper_perturbs in range(self.n_out)]

    def run(self):
        results = self.load()
        diffs_val_eval_approx = [result[KeysTarget.DIFFS_VAL_EVAL] for result in results]

        self.dump({KeysTarget.DIFFS_VAL_EVAL: diffs_val_eval_approx})


@requires(HyperGradPush, GenMostInfluentialPerturbs)
class LinearApproxDiffByMostInfluentialPerturbs(waluigi.TaskBase):
    def run(self):
        ds_result = self.load()
        diffs_val_eval_approx = linear_approx(
            hyper_perturbs_of_nodes_of_iters=ds_result[1][KeysTarget.HYPER_PERTURBS_ITERS],
            hypergrads_nodes=ds_result[0][KeysTarget.HYPER_GRADIENTS_NODES]
        )
        self.dump({KeysTarget.DIFFS_VAL_EVAL: diffs_val_eval_approx})


@requires(LinearApproxDiffByMostInfluentialPerturbs, RangeComputeActualDiffByMostInfluentialPerturbs)
class CompareApproxActualDiffByMostInfluentialPerturbs(waluigi.TaskBase):
    ver_compare_approx_actual_diff_by_most_influential_perturbs = luigi.Parameter("no text")
    kwargs_isclose_diff = luigi.DictParameter()
    scale = luigi.Parameter()
    score = luigi.Parameter()
    lim = luigi.FloatParameter(None)

    def run(self):
        ds_result = self.load()
        diffs_val_eval_approx = ds_result[0][KeysTarget.DIFFS_VAL_EVAL]
        diffs_val_eval_actual = ds_result[1][KeysTarget.DIFFS_VAL_EVAL]
        self.dump(self.run_in_sacred_experiment(
            compare_val_eval_diffs,
            diffs_val_eval_approx=diffs_val_eval_approx,
            diffs_val_eval_actual=diffs_val_eval_actual,
            scale=self.scale,
            score=self.score,
            lim=self.lim,
            kwargs_isclose=self.kwargs_isclose_diff
        ))
