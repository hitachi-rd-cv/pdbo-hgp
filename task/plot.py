import os

import pandas as pd
from luigi.util import inherits
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from constants import KeysTarget
from task.baseline import RunBaselineOnFedEmSetting
from task.est_error import ComputeHyperGradErrorOfSteps
from task.hyperopt import HyperSGD, HyperSGDOnFedEmSetting
from task.utils import ZipTaskBase


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
            ss_valid = pd.Series([d_mean[self.name_mean_valid] for d_mean, _ in d_steps[KeysTarget.D_METRIC_FEDEM_HYPER_STEPS]])
            ss_test_mean = pd.Series([d_mean[self.name_mean_test] for d_mean, _ in d_steps[KeysTarget.D_METRIC_FEDEM_HYPER_STEPS]])
            ss_test_bottom = pd.Series([d_bottom[self.name_bottom_test] for _, d_bottom in d_steps[KeysTarget.D_METRIC_FEDEM_HYPER_STEPS]])

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
            d_metric = d_result[KeysTarget.D_METRIC_FEDEM]
            sss_out.append(pd.Series({"Average": d_metric[self.name_mean_test] * self.rate, "Bottom 10% percentile": d_metric[self.name_bottom_test] * self.rate}, name=name))

        df_out = pd.concat(sss_out, axis=1).T
        self.pretty_print_table(df_out)

        self.dump(df_out)


@inherits(ComputeHyperGradErrorOfSteps)
class PlotZipComputeHyperGradErrorOfSteps(ZipTaskBase):
    def run(self):
        d_result = self.load()
        for name, result in d_result.items():
            errors = result[KeysTarget.ERROR_NORM_HYPERGRAD_STEPS]
            label = name.replace("alpha", r"$\alpha$").replace("beta", r"$\beta$").replace("name=", "")
            plt.plot(errors, label=label, linewidth=1.0)

        plt.legend()
        plt.yscale("log")
        plt.xlabel(r"$m$")
        plt.xlim((0, 500))
        plt.ylabel(r"$\|\|\mathbf{v}^{(m)} - \mathrm{d}_{\mathbf{\lambda}}\bar{F}(\mathbf{x}^{*}, \mathbf{\lambda})\|\|$")

        dir_output = self.make_and_get_temporary_directory()
        path = os.path.join(dir_output, f'hypergrad_errors.png')
        plt.savefig(path)

        self.dump(None)
