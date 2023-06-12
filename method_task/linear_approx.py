import torch


def linear_approx(
        hyper_perturbs_of_nodes_of_iters,
        hypergrads_nodes,
        _run=None
):
    diffs_val_eval_approx = []
    for hyper_perturbs_of_nodes in hyper_perturbs_of_nodes_of_iters:
        diff_val_eval_approx = 0.
        for hypergrads, hyper_perturbs_of_hypers in zip(hypergrads_nodes, hyper_perturbs_of_nodes):
            for hypergrad, hyper_perturb in zip(hypergrads, hyper_perturbs_of_hypers):
                diff_val_eval_approx += torch.sum(hypergrad.detach().cpu() * hyper_perturb.detach().cpu()).numpy().item()
        diffs_val_eval_approx.append(diff_val_eval_approx)

    return diffs_val_eval_approx
