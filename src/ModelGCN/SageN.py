import dgl
from dgl.nn.pytorch import SAGEConv
import torch
import torch.nn.functional as F

from ModelGCN import GCNBaseN, GCNBaseNNet
from consts import SEED


class SageN(GCNBaseN):
    """
    GraphSAGE model using DGL.
    Has 2 Convolutional layers and an MLP out layer
    """

    def __init__(self, data: dgl.DGLGraph, hid_feats: int, dropout: (float, float) = (0.5, 0.),
                 lr: float = 1e-3, weight_decay: float = 0., inductive: bool = False, validation: bool = True,
                 gpu: str = 'cpu', run_name: str = None, debug_mode: bool = False, seed: int = SEED):
        """
        :param data: dgl.graph,
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
        GCNBaseN.__init__(self, data, gpu, inductive, validation, debug_mode, seed)
        assert len(dropout) == 2
        self.model = SageNNet(data.ndata['feat'].shape[1], hid_feats, data.num_classes, dropout).to(
            self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.set_run_config(run_name=run_name, lr=lr, weight_decay=weight_decay)


class SageNNet(GCNBaseNNet):

    name = "Sage"
    config = None

    def __init__(self, in_feats, hid_feats, out_feats, dropout):
        GCNBaseNNet.__init__(self, hid_feats, out_feats)
        self.config = {"hid_feats": hid_feats, "num_hid_layers": 1, "dropout_0": dropout[0], "dropout_1": dropout[1]}
        self.conv0 = SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean', activation=F.relu)
        self.drop0 = torch.nn.Dropout(dropout[0])
        self.conv1 = SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean', activation=F.relu)
        self.drop1 = torch.nn.Dropout(dropout[1])

    def get_embedding(self, g, h):
        h = self.drop0(self.conv0(g, h))
        h = self.drop1(self.conv1(g, h))
        return h
