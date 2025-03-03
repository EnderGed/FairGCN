import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn.functional as F
import wandb

from ModelGCN import GCNBase
from ModelGCN.SageN import SageNNet
from consts import SEED, PROJECT_NAME


class SageE(GCNBase):
    """
    GraphSAGE for Edge regression.
    It uses DotProduct for the Edge Score
    """

    def __init__(self, data, hid_feats, gpu='cpu', run_name=None, debug_mode=False, seed=SEED):
        GCNBase.__init__(self, data, gpu, debug_mode, seed)
        if data.num_classes != 1:
            raise Exception("Only regression is supported, but number of classes is: {}. Expected 1.".format(
                data.num_classes))
        self.model = SageENet(data.ndata['feat'].shape[1], hid_feats, data.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # wandb config
        if not self.debug_mode:
            if run_name is None:
                run_name = 'S{}'.format(self.data.split_name)
            wandb.init(project=PROJECT_NAME, name=run_name, config={
                "layer1": hid_feats,
            }, tags=['sage', 'edge', self.data.split_name], reinit=True)
            wandb.watch(self.model)

    def train_step(self):
        self.model.train()
        logits = self.model(self.data)
        loss = F.mse_loss(logits[self.data.edata['train_mask']],
                          self.data.edata['label'][self.data.edata['train_mask']])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        logits, accs = self.model(self.data), []
        for mask in [self.data.edata[mask_name] for mask_name in ['train_mask', 'val_mask', 'test_mask']]:
            pred = logits[mask]
            labels = self.data.edata['label'][mask]
            correct = torch.sum(torch.abs(labels - pred) < 0.125)  # 0.125 works only for MovieLens
            accs.append(correct.item() * 1.0 / len(labels))
        return accs


# Define a Dot Product Predictor from https://docs.dgl.ai/guide/training-edge.html
class DotProductPredictor(torch.nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class SageENet(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.sage = SageNNet(in_feats, hid_feats, out_feats)
        self.pred = DotProductPredictor()

    def forward(self, g):
        h = self.sage(g)
        return self.pred(g, h)
