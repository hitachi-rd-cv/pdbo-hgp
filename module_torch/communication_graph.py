import networkx as nx
import numpy as np
import torch
from torch import nn

from constants import ModesGraph
from fedem.utils.decentralized import compute_mixing_matrix


class DynamicGraphBase(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.n_nodes = nn.Parameter(torch.tensor(n_nodes, dtype=torch.int32), requires_grad=False)
        # for device assign
        self.one = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.zero = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def current_adjacency_matrix(self):
        return torch.vstack([self.are_connected(idx_from) for idx_from in range(self.n_nodes)]).T

    def are_connected(self, from_node):
        return torch.hstack([self.is_connected(from_node, idx_to) for idx_to in range(self.n_nodes)])

    @classmethod
    def get_communication_graph(cls, n, p, seed=None):
        return nx.generators.random_graphs.binomial_graph(n=n, p=p, seed=seed)

    def add_self_edges_(self, mat):
        for i in range(self.n_nodes):
            mat[i, i] = torch.tensor(1., dtype=mat.dtype, device=mat.device)


class GraphFullyConnected(DynamicGraphBase):
    def __init__(self, n_nodes):
        super().__init__(n_nodes)
        self.rate_connected_mat = nn.Parameter(torch.ones((n_nodes, n_nodes), dtype=torch.float32), requires_grad=False)
        self.mixing_matrix = nn.Parameter(torch.ones((n_nodes, n_nodes)) / n_nodes, requires_grad=False)

    def sample(self):
        pass

    def init_params(self, *args, **kwargs):
        return

    def is_connected(self, from_node, to_node, ):
        return self.one

    def compute_prob_event(self, event, idx_from):
        prob_joint = torch.tensor(1., device=self.n_nodes.device)
        for idx_to, e in enumerate(event):
            if e > 0:
                prob_joint *= 1.
            else:
                prob_joint *= 0.
        return prob_joint


class GraphErdosRenyi(DynamicGraphBase):
    def __init__(self, n_nodes):
        super().__init__(n_nodes)
        self.rate_connected_mat = nn.Parameter(torch.zeros((n_nodes, n_nodes)), requires_grad=False)
        self.mixing_matrix = nn.Parameter(torch.zeros((n_nodes, n_nodes)), requires_grad=False)
        self.G_current = None

    def init_params(self, p):
        adjacency_mat, mixing_mat = self.gen_mixing_matrix(p)
        self.rate_connected_mat.copy_(torch.tensor(adjacency_mat))
        self.mixing_matrix.copy_(torch.tensor(mixing_mat))

    def sample(self):
        G = nx.from_numpy_matrix(self.rate_connected_mat.cpu().numpy())
        self.G_current = G.copy()

    def is_connected(self, from_node, to_node):
        edges = list(self.G_current.edges)
        if (to_node, from_node) in edges or (from_node, to_node) in edges:
            return self.one
        else:
            return self.zero

    def gen_mixing_matrix(self, p, max_trial=1000, tol_rate=1e-2):
        tol_weight = tol_rate / self.n_nodes.item()
        mixing_matrix = None
        adjacency_matrix_bool = None
        n_trial = 0
        retry = True
        while n_trial < max_trial and retry:
            n_trial += 1
            graph = self.get_communication_graph(self.n_nodes, p)
            adjacency_matrix_bool = nx.adjacency_matrix(graph, weight=None).todense()
            # add self loop
            for i in range(self.n_nodes):
                adjacency_matrix_bool[i, i] = 1.
            mixing_matrix = compute_mixing_matrix(adjacency_matrix_bool)

            # for numerical stability
            retry = np.any(mixing_matrix[np.nonzero(mixing_matrix)] < tol_weight) or not nx.is_connected(graph)

        assert not retry, mixing_matrix

        return adjacency_matrix_bool, mixing_matrix


    def compute_prob_event(self, event, idx_from):
        prob_joint = torch.tensor(1., device=self.n_nodes.device)
        for idx_to, e in enumerate(event):
            if e > 0:
                prob_joint *= self.rate_connected_mat[idx_to, idx_from]
            else:
                prob_joint *= 1 - self.rate_connected_mat[idx_to, idx_from]

        return prob_joint


class StochasticGraphBase(DynamicGraphBase):
    def compute_prob_event(self, event, idx_from):
        prob_joint = torch.tensor(1., device=self.n_nodes.device)
        for idx_to, e in enumerate(event):
            if e > 0:
                prob_joint *= self.adjacency_matrix[idx_to, idx_from] * self.rate_connected_mat[idx_to, idx_from]
            else:
                prob_joint *= 1. - self.adjacency_matrix[idx_to, idx_from] * self.rate_connected_mat[idx_to, idx_from]
        return prob_joint

    @classmethod
    def gen_adjacency_matrix(cls, n, p, force_sparse=False, seed=None, n_trials=1000):
        n_edges_complete = nx.complete_graph(n, nx.DiGraph())
        for i in range(n_trials):
            try:
                G = cls.get_communication_graph(n, p, seed=seed)
                if force_sparse:
                    # base network assumed NOT to be NOT fully connected
                    assert not G.number_of_edges() == n_edges_complete, nx.adjacency_matrix(G, weight=None).todense()
                # add selfloops
                G.add_edges_from([(i, i) for i in range(n)])
                # base network assumed to be strongly connected
                assert nx.is_connected(G), nx.adjacency_matrix(G, weight=None).todense()
            except:
                if i + 1 < n_trials:
                    continue
                else:
                    raise AssertionError(f'{i} == {n_trials}')
            else:
                break

        return nx.adjacency_matrix(G, weight=None).todense()


class GraphStochasticBase(StochasticGraphBase):
    def __init__(self, n_nodes):
        super().__init__(n_nodes)
        self.n_edges = nn.Parameter(torch.tensor(-1, dtype=torch.int32), requires_grad=False)
        self.adjacency_matrix = nn.Parameter(torch.zeros((n_nodes, n_nodes)), requires_grad=False)
        self.rate_connected_mat = nn.Parameter(torch.zeros_like(self.adjacency_matrix), requires_grad=False)
        self.force_sparse = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.G_current = None
        self.step_current = None

    def init_params(self, low, high, p, force_sparse, *args, **kwargs):
        adjacency_mat = self.gen_adjacency_matrix(self.n_nodes.item(), p, force_sparse=force_sparse)
        self.adjacency_matrix.copy_(torch.tensor(adjacency_mat))
        self.force_sparse.copy_(torch.tensor(force_sparse, dtype=torch.bool))
        rate_connected_mat = torch.tensor(np.random.uniform(low, high, size=self.rate_connected_mat.shape))
        self.add_self_edges_(rate_connected_mat)
        self.rate_connected_mat.copy_(rate_connected_mat)


class StochasticDirectedGraph(GraphStochasticBase):
    def sample(self):
        # uniform sampling of edges
        G_base = nx.from_numpy_matrix(self.adjacency_matrix.cpu().numpy())
        self.G_current = G_base.copy().to_directed()
        edges = list(self.G_current.edges)
        for u, v in edges:
            if u != v:
                if np.random.random() > self.rate_connected_mat[u, v]:
                    self.G_current.remove_edge(u, v)

    def is_connected(self, from_node, to_node):
        edges = list(self.G_current.edges)
        if (to_node, from_node) in edges:
            return self.one
        else:
            return self.zero


class StochasticBidirectedGraph(GraphStochasticBase):
    def init_params(self, low, high, p, force_sparse, *args, **kwargs):
        super().init_params(low, high, p, force_sparse, *args, **kwargs)
        # use symmetric matrix made from upper matrix of possibly asymmetric rate_connected_mat
        rate_connected_mat_T = self.rate_connected_mat.T
        rate_connected_mat_sym = torch.tril(rate_connected_mat_T) + torch.tril(rate_connected_mat_T).T - torch.diag(torch.diagonal(rate_connected_mat_T))
        self.rate_connected_mat.copy_(rate_connected_mat_sym)

    def sample(self):
        # uniform sampling of edges
        G_base = nx.from_numpy_matrix(self.adjacency_matrix.cpu().numpy())
        self.G_current = G_base.copy()
        edges = list(self.G_current.edges)
        for u, v in edges:
            if np.random.random() > self.rate_connected_mat[u, v]:
                self.G_current.remove_edge(u, v)

    def is_connected(self, from_node, to_node):
        edges = list(self.G_current.edges)
        if (to_node, from_node) in edges or (from_node, to_node) in edges:
            return self.one
        else:
            return self.zero


class SelfLoopedGraph(DynamicGraphBase):
    def __init__(self, n_nodes):
        super().__init__(n_nodes)

    def sample(self):
        pass

    def init_params(self, *args, **kwargs):
        return

    def is_connected(self, from_node, to_node, ):
        if from_node == to_node:
            return self.one
        else:
            return self.zero


D_DYNAMIC_GRAPH = {
    ModesGraph.CONSTANTS: GraphFullyConnected,
    ModesGraph.STOCHASTIC_DIRECTED: StochasticDirectedGraph,
    ModesGraph.STOCHASTIC_BIDIRECTED: StochasticBidirectedGraph,
    ModesGraph.ERDOS_RENYI: GraphErdosRenyi,
    ModesGraph.SELF_LOOP: SelfLoopedGraph,
}
