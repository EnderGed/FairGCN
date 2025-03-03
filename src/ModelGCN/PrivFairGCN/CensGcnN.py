import dgl
import torch

from ModelGCN import GCNBaseN, GCNBaseNNet, CensorMLPNet
from consts import SEED


class CensGcnN(GCNBaseN):
    """
    Censoring GCN class following the "Overlearning reveals sensitive attributes" paper
    It's very similar to AdGcnN, except:
     it uses a different adversarial network (Censor MLP),
     applies it on a privacy sensitive attribute rather than fairness sensitive,
     doesn't have covariance constraint
    """


    fair_prefix = 'Cens'

    def __init__(self, data: dgl.DGLGraph, base_net: GCNBaseNNet, priv_lambda: float,
                 lr: float = 0.001, weight_decay: float = 0., run_name: str = None, gpu: str = 'cpu',
                 debug_mode: bool = False, seed: int = SEED, *args, **kwargs):
        """
        :param data: dgl.graph,
        :param base_net: GCNBaseNNet, base GCN network to be made private (eg. KipfNNet, SageNNet),
        :param priv_lambda: float, importance of adversarial loss to the loss function
        :param lr: float, learning rate of the optimizers
        :param weight_decay: float, weight decay of the optimizers
        :param run_name: str, name of the run for easy reading in wandb and saving ressults on the device
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param debug_mode: bool, if True then print instead of log to wandb
        :param seed: randomness seed
        """
        GCNBaseN.__init__(self, data=data, gpu=gpu, debug_mode=debug_mode, seed=seed)
        self.model = base_net.to(self.device).double()

        # Unsqueeze sens feature once, for faster loss calculation
        if self.data.priv_classes == 1:
            self.data.ndata['priv_u'] = self.data.ndata['priv'].unsqueeze(1).double()
        else:
            self.data.ndata['priv_u'] = self.data.ndata['priv']

        # Create the privacy adversarial network
        self.priv_adv = CensorMLPNet(base_net.emb_size, max(1, self.data.priv_classes)).to(self.device).double()
        self.priv_lambda = priv_lambda
        self.optimizer_P = torch.optim.Adam(self.priv_adv.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_G = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion_G = torch.nn.BCEWithLogitsLoss()
        self.criterion_P = torch.nn.BCEWithLogitsLoss() if self.data.priv_classes == 1 else torch.nn.CrossEntropyLoss()

        # set wandb configuration
        self.set_run_config(run_name=run_name, lr=lr, weight_decay=weight_decay)
        self.config['priv_lambda'] = self.priv_lambda

    def train_step(self):
        self.model.train()

        # update GNN (self.model)
        self.priv_adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        emb = self.model.get_embedding(self.data, self.data.ndata['feat'])
        logits_G = self.model.classify_emb(emb)
        logits_P = self.priv_adv(emb)

        priv_sens = self.data.ndata['priv_u']
        y_score = torch.sigmoid(logits_G)
        cls_loss = self.criterion_G(logits_G[self.data.ndata['train_mask']],
                                    self.data.ndata['label'][self.data.ndata['train_mask']])
        priv_adv_loss = self.criterion_P(logits_P, priv_sens)

        G_loss = cls_loss - self.priv_lambda * priv_adv_loss
        G_loss.backward()
        self.optimizer_G.step()

        # update priv Adv (self.priv_adv)
        self.priv_adv.requires_grad_(True)
        self.optimizer_P.zero_grad()

        logits_P = self.priv_adv(emb.detach())
        P_loss = self.criterion_P(logits_P, priv_sens)
        P_loss.backward()
        self.optimizer_P.step()

        return float(G_loss.item())
