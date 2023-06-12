from __future__ import division

import itertools
import json
import os
import pickle
from collections.abc import Iterable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import comb

import torch

sns.set()
sns.set_style('ticks')


def normalize_lists(list_):
    '''
    get 1-level nested list and returns un-nested list
    Args:
        list_:

    Returns: list

    '''
    return list(itertools.chain(*list_))


def normalize_list_recursively(list_):
    '''
    get n-level nested list and returns un-nested list
    Args:
        list_:

    Returns:

    '''
    items = []
    for item in list_:
        if isinstance(item, Iterable):
            items.extend(normalize_list_recursively(item))
        else:
            items.append(item)
    return items


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_lowest_or_highest_score_indices(scores, nsamples_low=10, nsamples_high=10):
    if len(scores) <= nsamples_low + nsamples_high:
        print('Number of scores equals to or is smaller than the total number of samples for jaccard score')
    original_indices = np.arange(len(scores))
    sorted_indices = np.argsort(scores)
    target_indices = np.concatenate([sorted_indices[:nsamples_low], sorted_indices[-nsamples_high:]])
    return np.isin(original_indices, target_indices)


def digits_to_one_hot(x, nclasses):
    x = np.array(x, dtype=np.int)
    nsamples = len(x)
    onehot = np.zeros((nsamples, nclasses), dtype=np.float)
    for i in range(nsamples):
        onehot[i, x[i]] = 1.0
    return onehot


def parse_json(path, *args, **kwargs):
    with open(path, "r") as f:
        return json.load(f, *args, **kwargs)


def dump_json(_dict, path, *args, **kwargs):
    with open(path, "w") as f:
        return json.dump(_dict, f)


def get_smallest_largest_val_indices(values, nsamples):
    sorted_indices = np.argsort(values)
    large_indices = sorted_indices[-nsamples:]
    small_indices = sorted_indices[:nsamples]
    return small_indices, large_indices


def get_minibatch_indices(nsamples, bsize, append_remainder=True, original_order=False, indices=None,
                          number_of_same_batches=1):
    nsteps, nremainders = divmod(nsamples, bsize)
    if indices is None:
        if original_order:
            perm_indices = np.arange(nsamples)
        else:
            perm_indices = np.random.permutation(np.arange(nsamples))
    else:
        perm_indices = indices
    indices_without_remainder = [perm_indices[i * bsize:(i + 1) * bsize] for i in range(nsteps)]
    if append_remainder and nremainders > 0:
        indices_remainder = [perm_indices[-nremainders:]]
        minibatch_indices = indices_without_remainder + indices_remainder
    else:
        minibatch_indices = indices_without_remainder

    if number_of_same_batches > 1:
        alt_minibatch_indices = []
        for indices in minibatch_indices:
            for _ in range(number_of_same_batches):
                alt_minibatch_indices.append(indices)
        return alt_minibatch_indices
    else:
        return minibatch_indices


def order(n):
    return str(n) + ("th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


def merge_dicts(dicts):
    return dict(itertools.chain(*[dic.items() for dic in dicts]))


def expected_indepencent_jaccard(nsamples, j_size):
    # thank you joriki! https://math.stackexchange.com/a/1770628
    n = nsamples
    m = j_size
    expected_jac = 0
    for k in np.arange(m + 1):
        expected_jac += k / (2 * m - k) * comb(m, k) * comb(n - m, m - k) / comb(n, m)
    return expected_jac


def write_str(s, fp, verbose=True):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, 'w') as f:
        f.write(s)
    if verbose:
        print(s)


def are_tensors(arrays):
    return all(isinstance(x, torch.Tensor) for x in arrays)


def are_ndarrays(arrays):
    return all(isinstance(x, np.ndarray) for x in arrays)


def normalize_list_and_array_recursively(obj):
    '''
    get n-level nested list and returns un-nested list
    Args:
        obj:

    Returns:

    '''
    items = []
    for item in obj:
        if isinstance(item, (tuple, list, torch.nn.ParameterList, torch.nn.ModuleList)):
            items.extend(normalize_list_and_array_recursively(item))
        elif isinstance(item, (np.ndarray, np.generic, torch.Tensor)):
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            if len(item.shape) == 0:
                if isinstance(item, torch.Tensor):
                    items.append(item)
                else:
                    items.append(item)
            else:
                items.extend(normalize_list_and_array_recursively(item))
        else:
            items.append(item)
    return np.array(items)


def is_any_element_in_b(a, b):
    a_flat = normalize_list_and_array_recursively(a)
    b_flat = normalize_list_and_array_recursively(b)
    return np.all([np.any(np.isclose(aa, b_flat)) for aa in a_flat])


def get_are_close_recursively(xs, ys, kwargs_isclose={}):
    xs_flat = normalize_list_and_array_recursively(xs)
    ys_flat = normalize_list_and_array_recursively(ys)
    if len(xs_flat) != len(ys_flat):
        raise ValueError(f'Mismatch in the lengths {len(xs_flat)} != {len(ys_flat)}')
    return np.isclose(xs_flat, ys_flat, **kwargs_isclose)


def log_formatter(y, pos):
    return '{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0))).format(y)


def gen_error_fig(actual, approx, _run=None, zero_centered=True,
                  scale='linear',
                  title='Errors between actual and approximated',
                  xlabel='Actual',
                  ylabel='Approximated',
                  text=None,
                  index_colored=False,
                  plot_index=False,
                  fig=None,
                  ax=None,
                  range_=None,
                  s=20,
                  label=None,
                  figsize=(4, 3.5),
                  ):
    assert len(actual) == len(approx)

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.0E"))

    if index_colored:
        c = np.arange(len(actual))
    else:
        c = None
    ax.scatter(actual, approx, label=label, c=c, zorder=2, s=s)

    if plot_index:
        for idx, (x, y) in enumerate(zip(actual, approx)):
            ax.annotate(str(idx), (x, y))

    if zero_centered:
        max_abs = np.max([np.abs(actual), np.abs(approx)])
        min_, max_ = -max_abs * 1.1, max_abs * 1.1
        pos_text = [max_abs, -max_abs]
    else:
        min_ = np.min([actual, approx])
        max_ = np.max([actual, approx])
        m = np.abs(max_ - min_) * 0.1
        min_ -= m
        max_ += m
        pos_text = [max_, min_]

    if range_ is None:
        range_ = [min_, max_]

    # adjust axes
    try:
        ax.set_xlim(*range_)
        ax.set_ylim(*range_)
    except ValueError as e:
        print(f'actual={actual}, approx={approx}')
        raise e

    # ticks
    if scale in ['log', 'symlog']:
        ax.set_xscale(scale, linthresh=1e-10)
        ax.set_yscale(scale, linthresh=1e-10)
        # ax.xaxis.set_major_formatter(LogFormatterExponent())
        # ax.yaxis.set_major_formatter(LogFormatterExponent())
    else:
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        # ax.ticklabel_format(useOffset=False, axis='both', style='sci', scilimits=(0, 0), useMathText=True)

    # texts
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.plot(range_, range_, 'k-', alpha=0.2, zorder=1)

    if text is not None:
        ax.text(*pos_text, text, verticalalignment='bottom', horizontalalignment='right')

    fig.subplots_adjust(bottom=0.2)
    # fit figure
    fig.tight_layout()

    return fig, ax


def maybe_cast_one_len_list_to_n_len_list(xs, n):
    if xs is None:
        return xs
    else:
        if len(xs) == n:
            return xs
        else:
            if len(xs) > 1:
                raise ValueError(f'Invalid length, {len(xs)}')
            else:
                return xs * n
