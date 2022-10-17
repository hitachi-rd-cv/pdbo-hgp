import torch
from torch import nn

from constants import ModesGossip


class GossipBase(nn.Module):
    def __init__(self, n_nodes, idx_node):
        super().__init__()
        self.n_nodes = n_nodes
        self.idx_node = idx_node
        # define weight for SGP
        self.weight = nn.Parameter(torch.tensor(1.))


class GossipConstBase(GossipBase):
    def __init__(self, n_nodes, idx_node):
        super().__init__(n_nodes, idx_node)
        self.p_vec_expected = nn.Parameter(torch.zeros(n_nodes), requires_grad=False)
        self.are_in_neighbors_expected = nn.Parameter(torch.zeros(n_nodes), requires_grad=False)

    def initialize_expected_values(self):
        self.p_vec_expected_updated = torch.zeros_like(self.p_vec_expected)
        self.are_in_neighbors_expected_updated = torch.zeros_like(self.are_in_neighbors_expected)

    def mix_p_vec_expected(self, p, to_node, n_steps):
        self.p_vec_expected_updated[to_node] = p / n_steps  # TODO(future): p_expect accumulate numerical error

    def mix_are_in_neighbors_expected(self, is_in_neighbor, from_node, n_steps):
        self.are_in_neighbors_expected_updated[from_node] = is_in_neighbor / n_steps

    def step_expected_values(self):
        self.p_vec_expected.add_(self.p_vec_expected_updated)
        self.are_in_neighbors_expected.add_(self.are_in_neighbors_expected_updated)

    def initialize_expected_updated_values(self):
        self.p_vec_expected_updated.zero_()
        self.are_in_neighbors_expected_updated.zero_()


class GossipGivenWeights(GossipConstBase):
    def __init__(self, n_nodes, idx_node, p_vec):
        super().__init__(n_nodes, idx_node)
        self.p_vec = nn.Parameter(torch.tensor(p_vec), requires_grad=False)

    def get_p_vec(self, are_connected):
        assert torch.isclose(are_connected @ self.p_vec, torch.tensor(1., device=self.p_vec.device)), (are_connected, self.p_vec)
        return self.p_vec * are_connected



class GossipNormalize(GossipConstBase):
    def get_p_vec(self, are_connected):
        p_vec = torch.zeros_like(are_connected)
        are_connected_bool = are_connected > 0
        p_vec[are_connected_bool] = 1. / torch.sum(are_connected)
        return p_vec



D_GOSSIP_CLASS = {
    ModesGossip.WEIGHT_GIVEN: GossipGivenWeights,
    ModesGossip.NORMALIZE: GossipNormalize,
}
