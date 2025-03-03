import dgl
import torch

from ModelGCN import EwBaseN, KipfNNet
from consts import SEED


class EwKipfN(EwBaseN):
    """
    Kipf's GCN model with edge weightening.

    Be sure to modify this file if you modify KipfN - bad practice here, but I'm not sure how to make it better.
    """

    def __init__(self, data: dgl.DGLGraph, alpha: float, hid_feats: int, dropout: (float, float) = (0.5, 0.),
                 lr: float = 1e-3, weight_decay: float = 0., inductive: bool = False, validation: bool = True,
                 gpu: str = 'cpu', run_name: str = '', debug_mode: bool = False, seed: int = SEED):
        """
        :param data: dgl.graph,
        :param alpha: float [0, 1] edge weightening strength parameter, 0 - no weights, 1 - complete counterweights
        :param hid_feats: int, size of each hidden layer
        :param dropout: (float, float), probabilities of zeroeing a value after first and second Convolution layers
        :param lr: float, learning rate of the optimizer
        :param weight_decay: float, weight decay (L2 penalty) of the optimizer
        :param inductive: bool, whether inductive setting should be used, transductive by default.
        :param validation: bool, whether validation set should be used, if False, add it to test set
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param run_name: str, name of the run, for easy reading in wandb and saving results on the device
        :param debug_mode: bool, if yes, then print instead of log to wandb
        :param seed: randomness seed
        """
        EwBaseN.__init__(self, data=data, alpha=alpha, inductive=inductive, validation=validation, gpu=gpu,
                         debug_mode=debug_mode, seed=seed)
        assert len(dropout) == 2
        self.model = EwKipfNNet(data.ndata['feat'].shape[1], hid_feats, data.num_classes, dropout).to(
            self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.set_run_config(run_name=run_name, lr=lr, weight_decay=weight_decay)


class EwKipfNNet(KipfNNet):
    def get_embedding(self, g, h):
        h = self.drop0(self.conv0(g, h, edge_weight=g.edata['w']))
        h = self.drop1(self.conv1(g, h, edge_weight=g.edata['w']))
        return h
