import dgl
import torch

from ModelGCN import AdGcnN, CensGcnN, GCNBaseNNet
from consts import SEED


class CensAdGcnN(AdGcnN, CensGcnN):
    """
    Adversarial debiasing with privacy censoring.
    """

    fair_prefix = CensGcnN.fair_prefix + AdGcnN.fair_prefix

    def __init__(self, data: dgl.DGLGraph, base_net: GCNBaseNNet, alpha: float, beta: float, priv_lambda: float,
                 lr: float = 0.001, weight_decay: float = 0., run_name: str = None, gpu: str = 'cpu',
                 debug_mode: bool = False, seed: int = SEED, *args, **kwargs):
        """
        :param data: dgl.graph,
        :param base_net: base GCN network to be made fair
        :param alpha: float, importance of covariance constraint to the loss function
        :param beta: float, importance of adversarial loss to the loss function
        :param priv_lambda: float, importance of adversarial loss to the loss function
        :param lr: float, learning rate of the optimizers
        :param weight_decay: float, weight decay of the optimizers
        :param run_name: str, name of the run for easy reading in wandb and saving results on the device
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param debug_mode: bool, if True then print instead of log to wandb
        :param seed: randomness seed
        """
        AdGcnN.__init__(self, data=data, base_net=base_net, alpha=alpha, beta=beta, lr=lr, weight_decay=weight_decay,
                        run_name=run_name, gpu=gpu, debug_mode=debug_mode, seed=seed)
        CensGcnN.__init__(self, data=self.data, base_net=self.model, priv_lambda=priv_lambda, lr=lr,
                          weight_decay=weight_decay, run_name=run_name, gpu=gpu, debug_mode=debug_mode, seed=seed)

    def train_step(self):
        self.model.train()

        # update GNN (self.model)
        self.adv.requires_grad_(False)
        self.priv_adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        emb = self.model.get_embedding(self.data, self.data.ndata['feat'])
        logits_G = self.model.classify_emb(emb)
        logits_A = self.adv(emb)
        logits_P = self.priv_adv(emb)

        sens = self.data.ndata['sens_u']
        priv_sens = self.data.ndata['priv_u']
        y_score = torch.sigmoid(logits_G)
        cov = torch.abs(torch.mean((sens - torch.mean(sens)) * (y_score - torch.mean(y_score))))
        cls_loss = self.criterion_G(logits_G[self.data.ndata['train_mask']],
                                  self.data.ndata['label'][self.data.ndata['train_mask']])
        adv_loss = self.criterion_G(logits_A, sens)
        priv_adv_loss = self.criterion_P(logits_P, priv_sens)

        G_loss = cls_loss + self.alpha * cov - self.beta * adv_loss - self.priv_lambda * priv_adv_loss
        G_loss.backward()
        self.optimizer_G.step()

        # update Adv (self.adv)
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()

        logits_A = self.adv(emb.detach())
        A_loss = self.criterion_G(logits_A, sens)
        A_loss.backward()
        self.optimizer_A.step()

        # update priv Adv (self.priv_adv)
        self.priv_adv.requires_grad_(True)
        self.optimizer_P.zero_grad()

        logits_P = self.priv_adv(emb.detach())
        P_loss = self.criterion_P(logits_P, priv_sens)
        P_loss.backward()
        self.optimizer_P.step()

        return float(G_loss.item())
