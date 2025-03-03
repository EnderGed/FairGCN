import dgl
import torch

from ModelGCN import GCNBaseN
from consts import SEED


class EwBaseN(GCNBaseN):
    """
    Edge weightening GCN Base class

    Base class for edge weightening based GCNs, inspired by FairWalk '19
    Implements weight calculation and saving them on the graph edges.
    """

    fair_prefix = "Ew"

    def __init__(self, data: dgl.DGLGraph, alpha: float, inductive: bool = False, validation: bool = True,
                 gpu: str = 'cpu', debug_mode: bool = False, seed: int = SEED):
        """
        :param data: dgl.graph
        :param alpha: float [0, 1] edge weightening strength parameter, 0 - no weights, 1 - complete counterweights
        :param inductive: bool, whether inductive setting should be used, transductive by default.
        :param validation: bool, whether validation set should be used, if False, add it to test set
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param debug_mode: bool, if yes, then print instead of log to wandb
        :param seed: randomness seed
        """
        self.data = data
        self.alpha = alpha
        self.add_edge_weights()
        self.device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
        GCNBaseN.__init__(self, data=self.data, inductive=inductive, validation=validation, gpu=gpu,
                          debug_mode=debug_mode, seed=seed)

    def add_edge_weights(self):
        def calc_weight(edges):
            same = torch.where(edges.src['sens'] == 1, edges.dst['inc_1'], edges.dst['inc_0'])
            other = torch.where(edges.src['sens'] == 0, edges.dst['inc_1'], edges.dst['inc_0'])
            return {'w': (self.alpha * other + (1 - self.alpha) * same) / same}

        self.data.apply_edges(calc_weight)
        self.data.edata['w'] = self.data.edata['w'].double()
        # find which nodes have only one incoming edges group and reset their weights to 1
        self.data.apply_edges(lambda edges: {'leave': (edges.dst['inc_1'] * edges.dst['inc_0']) == 0})
        self.data.edata['w'][self.data.edata['leave']] = torch.ones(self.data.edata['leave'].sum(),
                                                                    dtype=torch.double).to(self.device)
        # all the u -> u edges should keep the weight 1
        self.data.edata['w'][:len(self.data.ndata['label'])] = torch.ones(len(self.data.ndata['label']),
                                                                          dtype=torch.double).to(self.device)
