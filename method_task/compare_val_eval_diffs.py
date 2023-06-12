import os

import numpy as np
import sklearn
from scipy.stats import kendalltau

from constants import NamesMetricDiff
from lib_common.utils import gen_error_fig


def compare_val_eval_diffs(
        diffs_val_eval_approx,
        diffs_val_eval_actual,
        scale,
        score,
        kwargs_isclose,
        lim=None,
        _run=None
):
    diffs_val_eval_actual = np.array(diffs_val_eval_actual)
    diffs_val_eval_approx = np.array(diffs_val_eval_approx)

    val = sklearn.metrics.r2_score(diffs_val_eval_actual, diffs_val_eval_approx)
    print('R2 score = {:.03}'.format(val))
    _run.log_scalar(NamesMetricDiff.R2, val)

    val, _ = kendalltau(diffs_val_eval_actual, diffs_val_eval_approx)
    print("Kendall's tau = {:.03}".format(val))
    _run.log_scalar(NamesMetricDiff.KENDALL_TAU, val)

    are_negative_actual = diffs_val_eval_actual < 0
    are_negative_approx = diffs_val_eval_approx < 0
    val = sklearn.metrics.jaccard_score(are_negative_actual, are_negative_approx)
    print("Jaccard score = {:.03}".format(val))
    _run.log_scalar(NamesMetricDiff.JACCARD_INDEX, val)

    val = sklearn.metrics.f1_score(are_negative_actual, are_negative_approx)
    print("F1 score = {:.03}".format(val))
    _run.log_scalar(NamesMetricDiff.F1_SCORE, val)

    if lim is None:
        range_ = None
    else:
        range_ = (-lim, lim)
    fig, _ = gen_error_fig(diffs_val_eval_actual, diffs_val_eval_approx,
                           _run,
                           scale=scale,
                           title=None,
                           range_=range_,
                           # text=text,
                           xlabel='Actuall diff in loss',
                           ylabel='Predicted diff in loss')

    path = os.path.join('lie_error.png')
    fig.savefig(path)
    if _run is not None:
        _run.add_artifact(path)

    path = os.path.join('lie_error.pdf')
    fig.savefig(path)
    if _run is not None:
        _run.add_artifact(path)

    print('Approx diffs:')
    print(diffs_val_eval_approx)
    print('Actual diffs:')
    print(diffs_val_eval_actual)

    if score == NamesMetricDiff.R2:
        val_optimal = 1.0
    elif score == NamesMetricDiff.KENDALL_TAU:
        val_optimal = 1.0
    elif score == NamesMetricDiff.JACCARD_INDEX:
        val_optimal = 1.0
    elif score == NamesMetricDiff.JACCARD_INDEX:
        val_optimal = 1.0
    else:
        raise ValueError(score)

    if not np.isclose(val, val_optimal, **kwargs_isclose):
        raise ValueError(f'{score}={val} !~ {val_optimal}')

    return fig
