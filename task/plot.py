import os

import luigi
import numpy as np
import pandas as pd
import torch
from luigi.util import inherits
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from constants import KeysOptionHGP
from constants import KeysTarget
from task.baseline import RunBaselineOnFedEmSetting
from task.est_error import ComputeHyperGradErrorOfSteps
from task.hyperopt import HyperSGD, HyperSGDOnFedEmSetting
from task.utils import ZipTaskBase


# plt.rcParams['text.usetex'] = True
# plt.rc('text.latex', preamble=r'\usepackage{bm}')

class MakeAccuracyTableBase(ZipTaskBase):
    @staticmethod
    def pretty_print_table(df_out):
        # pretty print table
        pt_param = PrettyTable(["Method", *df_out.columns.to_list()])
        for name, ss in df_out.iterrows():
            pt_param.add_row([name, *[f"{x:.2f}" for x in ss.to_list()]])

        print(pt_param)


@inherits(HyperSGD)
class MakeAccuracyTableHyperSGD(MakeAccuracyTableBase):
    name_mean_valid = 'hyper_accuracy_valid_mean'
    name_mean_test = 'hyper_accuracy_test_mean'
    name_bottom_test = 'hyper_accuracy_test_bottom'
    rate = 1.

    def run(self):
        d_methods = self.load()
        sss_out = []
        for name, d_steps in d_methods.items():
            ss_valid = pd.Series([d_mean["accuracy"]["valid"] for d_mean in d_steps[KeysTarget.D_D_VAL_EVAL_MEAN_HYPER_STEPS]])
            ss_test_mean = pd.Series([d_mean["accuracy"]["test"] for d_mean in d_steps[KeysTarget.D_D_VAL_EVAL_MEAN_HYPER_STEPS]])
            ss_test_bottom = pd.Series([d_bottom["accuracy"]["test"] for d_bottom in d_steps[KeysTarget.D_D_VAL_EVAL_BOTTOM_HYPER_STEPS]])

            assert len(ss_valid) == len(ss_test_mean) == len(ss_test_bottom)

            # use the result of hyper step whose average validation accuracy is the best
            idx_max = ss_valid.index[ss_valid.argmax()]
            sss_out.append(pd.Series({"Average": ss_test_mean[idx_max] * self.rate, "Bottom 10% percentile": ss_test_bottom[idx_max] * self.rate}, name=name))

        df_out = pd.concat(sss_out, axis=1).T
        self.pretty_print_table(df_out)

        self.dump(df_out)


@inherits(HyperSGDOnFedEmSetting)
class MakeAccuracyTableHyperSGDOnFedEmSetting(MakeAccuracyTableBase):
    name_mean_valid = 'Train/Metric'
    name_mean_test = 'Test/Metric'
    name_bottom_test = 'Test/Metric'
    rate = 100.

    def run(self):
        d_methods = self.load()
        sss_out = []
        for name, d_steps in d_methods.items():
            ss_valid = pd.Series([d_mean[self.name_mean_valid] for d_mean, _ in d_steps[KeysTarget.DS_METRIC_FEDEM_HYPER_STEPS]])
            ss_test_mean = pd.Series([d_mean[self.name_mean_test] for d_mean, _ in d_steps[KeysTarget.DS_METRIC_FEDEM_HYPER_STEPS]])
            ss_test_bottom = pd.Series([d_bottom[self.name_bottom_test] for _, d_bottom in d_steps[KeysTarget.DS_METRIC_FEDEM_HYPER_STEPS]])

            assert len(ss_valid) == len(ss_test_mean) == len(ss_test_bottom)

            # use the result of hyper step whose average validation accuracy is the best
            idx_max = ss_valid.index[ss_valid.argmax()]
            sss_out.append(pd.Series({"Average": ss_test_mean[idx_max] * self.rate, "Bottom 10% percentile": ss_test_bottom[idx_max] * self.rate}, name=name))

        df_out = pd.concat(sss_out, axis=1).T
        self.pretty_print_table(df_out)

        self.dump(df_out)


@inherits(RunBaselineOnFedEmSetting)
class MakeAccuracyTableBaselineOnFedEmSetting(MakeAccuracyTableBase):
    name_mean_valid = 'Train/Metric'
    name_mean_test = 'Test/Metric'
    name_bottom_test = 'Test/Metric'
    rate = 100.

    def run(self):
        d_methods = self.load()
        sss_out = []
        for name, d_result in d_methods.items():
            d_metric = d_result[KeysTarget.DS_METRIC_FEDEM]
            sss_out.append(pd.Series({"Average": d_metric[self.name_mean_test] * self.rate, "Bottom 10% percentile": d_metric[self.name_bottom_test] * self.rate}, name=name))

        df_out = pd.concat(sss_out, axis=1).T
        self.pretty_print_table(df_out)

        self.dump(df_out)

@inherits(ComputeHyperGradErrorOfSteps)
class PlotZipComputeHyperGradErrorOfSteps(ZipTaskBase):
    yscale = luigi.Parameter("log")
    xscale = luigi.Parameter("linear")

    def run(self):
        d_result = self.load()
        fig = self.run_in_sacred_experiment(
            self.plot_errors,
            d_result=d_result,
            dir_output=self.make_and_get_temporary_directory(),
            xlim=self.option_hgp[KeysOptionHGP.DEPTH],
            xscale=self.xscale,
            yscale=self.yscale,
        )
        self.dump(fig)

    @staticmethod
    def plot_errors(d_result, dir_output, xlim, xscale, yscale, _run=None):
        ylim = 0.
        for name, result in d_result.items():
            errors = result[KeysTarget.ERROR_NORM_HYPERGRAD_STEPS]
            label = name.replace("alpha", r"$\alpha$").replace("beta", r"$\beta$").replace("name=", "")
            plt.plot(errors, label=label, linewidth=1.0)
            ylim = max(ylim, max(*errors))

        plt.legend()
        plt.xscale(xscale)
        plt.yscale(yscale)
        # plt.xlabel(r"$m$")
        plt.xlabel("Depth of Neumann Approximation")
        plt.xlim((0, xlim))
        plt.ylim((0., ylim.cpu()))
        # plt.ylabel(r"$\|\|\mathbf{v}^{(m)} - \mathrm{d}_{\mathbf{\lambda}}\bar{F}(\mathbf{x}^{*}, \mathbf{\lambda})\|\|$")
        plt.ylabel("L2 Error of Hyper-gradient Estimation")
        plt.tight_layout()

        path = os.path.join(dir_output, f'hypergrad_errors.png')
        plt.savefig(path)
        if _run is not None:
            _run.add_artifact(path)

        path = os.path.join(dir_output, f'hypergrad_errors.pdf')
        plt.savefig(path)
        if _run is not None:
            _run.add_artifact(path)

        return plt.figure()


@inherits(ComputeHyperGradErrorOfSteps)
class PlotComputeHyperGradErrorOfStepsWithErrorBar(ZipTaskBase):
    ver_plot_compute_hyper_grad_error_of_steps = luigi.Parameter("v1")
    yscale = luigi.Parameter("log")
    xscale = luigi.Parameter("linear")
    seeds_hgp = luigi.ListParameter()
    ns_push = luigi.ListParameter()
    error_bar = luigi.BoolParameter()
    y_max = luigi.FloatParameter(None)
    y_min = luigi.FloatParameter(None)
    no_ylabel = luigi.BoolParameter()
    scale = luigi.FloatParameter()

    def run(self):
        d_result = self.load()
        fig = self.run_in_sacred_experiment(
            self.plot_errors,
            d_result=d_result,
            dir_output=self.make_and_get_temporary_directory(),
            xlim=self.option_hgp[KeysOptionHGP.DEPTH],
            xscale=self.xscale,
            yscale=self.yscale,
            seeds_hgp=self.seeds_hgp,
            ns_push=self.ns_push,
            add_error_bar=self.error_bar,
            y_max=self.y_max,
            y_min=self.y_min,
            no_ylabel=self.no_ylabel,
            scale=self.scale,
        )
        self.dump(fig)

    @staticmethod
    def plot_errors(d_result, dir_output, xlim, xscale, yscale, seeds_hgp, ns_push, add_error_bar, scale=1.0, no_ylabel=False, y_max=None, y_min=None, _run=None):
        plt.rcParams["figure.figsize"] = (8 * scale, 5 * scale)

        results_ns_push = {n_push: {seed: None for seed in seeds_hgp} for n_push in ns_push}
        for k, v in d_result.items():
            for n_push in ns_push:
                for seed in seeds_hgp:
                    if f'seed={seed}, S={n_push}' == k:
                        results_ns_push[n_push][seed] = v

        for n_push in ns_push:
            results = results_ns_push[n_push]
            # compute mean of results and std of results then plot the mean with error bar
            errors = []
            for seed, result in results.items():
                error = result[KeysTarget.ERROR_NORM_HYPERGRAD_STEPS]
                error = torch.hstack(error)
                errors.append(error)
            errors = torch.stack(errors, dim=0).cpu().numpy()
            mean = errors.mean(axis=0)
            min = np.percentile(errors, 10, axis=0)
            max = np.percentile(errors, 90, axis=0)
            # min = errors.min(axis=0)
            # max = errors.max(axis=0)

            label = rf"$S={n_push}$"
            plt.plot(mean, label=label, linewidth=1.0)
            if add_error_bar:
                plt.fill_between(torch.arange(len(mean)), min, max, alpha=0.2)

        # plot legend at top above the figure with 2 columns
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2)
        # plot legend at right next to the figure
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xscale(xscale)
        plt.yscale(yscale)
        # plt.xlabel(r"$m$")
        plt.xlabel(r"Depth of approximation $m$")

        plt.xlim((0, xlim))
        # force the minimum of y to be 0
        if y_min is None:
            y_min = 0.

        plt.ylim((y_min, y_max))
        # plt.ylabel(r"$\|\|\bm{v}^{(m)} - \mathrm{d}_{\bm{\lambda}}\bar{f}(\bm{x}(\bm{\lambda}), \bm{\lambda})\|\|$")
        if not no_ylabel:
            plt.ylabel("$\ell$-2 error of hyper-grad.")
        plt.tight_layout()

        print("ylim: ", plt.ylim())
        path = os.path.join(dir_output, f'hypergrad_errors.png')
        plt.savefig(path)
        if _run is not None:
            _run.add_artifact(path)

        path = os.path.join(dir_output, f'hypergrad_errors.pdf')
        plt.savefig(path)
        if _run is not None:
            _run.add_artifact(path)

        return plt.figure()