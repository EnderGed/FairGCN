import dgl
import dgl.nn as dglnn
import torch
import torch.nn.functional as F
import wandb

from ModelGCN import GCNBaseN, GCNBaseNNet
from consts import SEED


class GatN(GCNBaseN):
    """
    Graph Attention Network model using DGL.
    Has 2 Convolutional layers and an MLP out layer
    """

    def __init__(self, data: dgl.DGLGraph, hid_feats: int, num_heads: (int, int) = (1, 1),
                 feat_drop: (float, float) = (0.5, 0.), att_drop: (float, float) = (0.5, 0.),
                 negative_slope: float = .2, lr: float = 1e-3, weight_decay: float = 0., inductive: bool = False,
                 validation: bool = True, gpu: str = 'cpu', run_name: str = None, debug_mode: bool = False,
                 seed: int = SEED):
        """
        :param data: dgl.graph,
        :param hid_feats: int, size of each hidden layer
        :param num_heads: (int, int), number of attention heads for each Conv layer
        :param feat_drop: (float, float), probabilities of zeroeing feature value after each Conv layer
        :param att_drop: (float, float), attention dropout for each Conv layer
        :param negative_slope: float, LeakyReLU angle of negative slope, same for both Conv layers
        :param lr: float, learning rate of the optimizer
        :param weight_decay: float, weight decay (L2 penalty) of the optimizer
        :param inductive: bool, whether inductive setting should be used, transductive by default.
        :param validation: bool, whether validation set should be used, if False, add it to test set
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param run_name: str, name of the run, for easy reading in wandb
        :param debug_mode: bool, if yes, then print instead of log to wandb
        :param seed: randomness seed
        """
        GCNBaseN.__init__(self, data, gpu, inductive, validation, debug_mode, seed)
        assert len(num_heads) == len(feat_drop) == len(att_drop) == 2
        self.model = GatNNet(data.ndata['feat'].shape[1], hid_feats, data.num_classes, num_heads, feat_drop, att_drop,
                             negative_slope).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.set_run_config(run_name=run_name, lr=lr, weight_decay=weight_decay)


class GatNNet(GCNBaseNNet):

    name = "Gat"
    config = None

    def __init__(self, in_feats, hid_feats, out_feats, num_heads, feat_drop, att_drop, negative_slope):
        GCNBaseNNet.__init__(self, hid_feats, out_feats)
        self.config = {
            "hid_feats": hid_feats,
            "heads_0": num_heads[0],
            "heads_1": num_heads[1],
            "feat_drop_0": feat_drop[0],
            "feat_drop_1": feat_drop[1],
            "att_drop_0": att_drop[0],
            "att_drop_1": att_drop[1],
            "neg_slope": negative_slope,
        }
        self.num_heads = num_heads
        self.conv0 = dglnn.GATConv(in_feats=in_feats, out_feats=hid_feats, num_heads=num_heads[0], feat_drop=feat_drop[0],
                                   attn_drop=att_drop[0], negative_slope=negative_slope, activation=F.relu)
        self.conv1 = dglnn.GATConv(in_feats=hid_feats * num_heads[0], out_feats=hid_feats, num_heads=num_heads[1],
                                   feat_drop=feat_drop[1], attn_drop=att_drop[1], negative_slope=negative_slope,
                                   activation=F.relu)

    def get_embedding(self, g, h):
        h = self.conv0(g, h)
        # concatenate results of the attention heads
        h = torch.cat([h[:, i, :] for i in range(self.num_heads[0])], dim=1)
        h = self.conv1(g, h)
        # average results of the attention heads
        h = torch.mean(torch.stack([h[:, i, :] for i in range(self.num_heads[1])]), dim=0)
        return h
