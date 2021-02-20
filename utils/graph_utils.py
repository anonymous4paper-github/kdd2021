'''
	Graph utils.
'''
import numpy as np
import torch

class NeibSampler:
    '''
        Neighbor sampling.
    '''
    def __init__(self, graph, n_nb, include_self=False):
        n_nodes = graph.number_of_nodes()
        if include_self:
            nb_all = torch.zeros(n_nodes, n_nb + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n_nodes)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n_nodes, n_nb, dtype=torch.int64)
            nb = nb_all
        hots = []
        for v in range(n_nodes):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= n_nb:
                nb_v.extend([-1] * (n_nb - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                hots.append(v)
        self.include_self = include_self
        self.g = graph
        self.nb_all, self.hots = nb_all, hots

    def to(self, device):
        self.nb_all = self.nb_all.to(device)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        n_nb = nb.size(1)
        hots_nb = np.zeros((len(self.hots), n_nb), dtype=np.int64)
        for i, v in enumerate(self.hots):
            hots_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), n_nb)
        nb[self.hots] = torch.from_numpy(hots_nb).to(nb.device)
        return self.nb_all
