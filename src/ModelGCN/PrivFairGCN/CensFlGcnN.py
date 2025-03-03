import dgl
import torch

from ModelGCN import FlGcnN, CensGcnN, GCNBaseNNet
from consts import SEED


class CensFlGcnN(FlGcnN, CensGcnN):
    """
    Fair learning GCN with privacy censoring.
    """

    fair_prefix = CensGcnN.fair_prefix + FlGcnN.fair_prefix

    def __init__(self, data: dgl.DGLGraph, base_net: GCNBaseNNet, alpha: float,
                 priv_lambda: float, fairnes_type: str = 'par', lr: float = 0.001, weight_decay: float = 0.,
                 run_name: str = None, gpu: str = 'cpu', debug_mode: bool = False, seed: int = SEED, *args, **kwargs):
        """
        :param data: dgl.graph,
        :param base_net: base GCN network to be made fair
        :param alpha: float, importance of covariance constraint to the loss function
        :param priv_lambda: float, importance of adversarial loss to the loss function
        :param fairnes_type: str, 'eoo' or 'par', definition of fairnes used for calculating the loss
        :param lr: float, learning rate of the optimizers
        :param weight_decay: float, weight decay of the optimizers
        :param run_name: str, name of the run for easy reading in wandb and saving results on the device
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param debug_mode: bool, if True then print instead of log to wandb
        :param seed: randomness seed
        """
        FlGcnN.__init__(self, data=data, base_net=base_net, alpha=alpha, fairnes_type=fairnes_type, lr=lr,
                        weight_decay=weight_decay, run_name=run_name, gpu=gpu, debug_mode=debug_mode, seed=seed)
        CensGcnN.__init__(self, data=self.data, base_net=self.model, priv_lambda=priv_lambda, lr=lr,
                          weight_decay=weight_decay, run_name=run_name, gpu=gpu, debug_mode=debug_mode, seed=seed)

    def train_step(self):
        self.model.train()

        self.priv_adv.requires_grad_(False)

        # model classification loss
        emb = self.model.get_embedding(self.data, self.data.ndata['feat'])
        logits_G = self.model.classify_emb(emb)
        logits_P = self.priv_adv(emb)

        cls_loss = self.criterion_G(logits_G[self.data.ndata['train_mask']],
                                  self.data.ndata['label'][self.data.ndata['train_mask']])
        # model fairness loss
        fair_loss = self.get_fair_loss(logits_G)

        # model privacy loss
        priv_sens = self.data.ndata['priv_u']
        priv_adv_loss = self.criterion_P(logits_P, priv_sens)

        loss = cls_loss + self.alpha * fair_loss - self.priv_lambda * priv_adv_loss
        self.optimizer_G.zero_grad()
        loss.backward()
        self.optimizer_G.step()

        # update priv Adv (self.priv_adv)
        self.priv_adv.requires_grad_(True)
        self.optimizer_P.zero_grad()

        logits_P = self.priv_adv(emb.detach())
        P_loss = self.criterion_P(logits_P, priv_sens)
        P_loss.backward()
        self.optimizer_P.step()

        return float(loss.item())