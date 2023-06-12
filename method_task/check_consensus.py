import torch

from constants import TypesDevice
from lib_task.common import load_clients


def assert_consensus(n_nodes, name_model, kwargs_build_nodes, state_dicts, use_cuda=True):
    # build network
    device = torch.device(TypesDevice.CUDA if use_cuda else TypesDevice.CPU)

    clients = load_clients(
        name_model=name_model,
        kwargs_build_nodes=kwargs_build_nodes,
        n_nodes=n_nodes,
        state_dicts=state_dicts,
        device=device,
    )
    for m in clients:
        m.train()

    # make concatenated matrix and vectors over the nodes
    x_nodes = [torch.hstack([torch.flatten(x.detach()) for x in m.innerparameters]) for m in clients]

    # check consensus
    for x_node in x_nodes:
        assert torch.allclose(x_nodes[0], x_node)
