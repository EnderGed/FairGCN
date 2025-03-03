import dgl
import torch

from ModelGCN import GCNBaseN, GCNBaseNNet
from consts import SEED


class AdGcnN(GCNBaseN):
    """
    Adversarial Debiasing GCN class

    GCN model trained with Adversarial Debiasing, implements the adversarial network and the joined training step.
    From the paper: "Say No to the Discrimination: Learning Fair Graph Neural
        Networks with Limited Sensitive Attribute Information"
    """

    fair_prefix = 'Ad'

    def __init__(self, data: dgl.DGLGraph, base_net: GCNBaseNNet, alpha: float, beta: float,
                 lr: float = 0.001, weight_decay: float = 0., inductive: bool = False, validation: bool = True,
                 run_name: str = None, gpu: str = 'cpu', debug_mode: bool = False, seed: int = SEED):
        """
        :param data: dgl.graph,
        :param base_net: base GCN network to be made fair
        :param alpha: float, importance of covariance constraint to the loss function
        :param beta: float, importance of adversarial loss to the loss function
        :param lr: float, learning rate of the optimizers
        :param weight_decay: float, weight decay of the optimizers
        :param run_name: str, name of the run for easy reading in wandb and saving results on the device
        :param inductive: bool, whether inductive setting should be used, transductive by default.
        :param validation: bool, whether validation set should be used, if False, add it to test set
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param debug_mode: bool, if True then print instead of log to wandb
        :param seed: randomness seed
        """
        # Unsqueeze sens feature once, for faster loss calculation
        data.ndata['sens_u'] = data.ndata['sens'].unsqueeze(1).double()

        GCNBaseN.__init__(self, data=data, inductive=inductive, validation=validation, gpu=gpu, debug_mode=debug_mode,
                          seed=seed)
        self.model = base_net.to(self.device).double()

        # Create adversarial network
        self.adv = torch.nn.Linear(base_net.emb_size, 1).to(self.device).double()
        self.alpha = alpha
        self.beta = beta
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_G = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # set wandb configuration
        self.set_run_config(run_name=run_name, lr=lr, weight_decay=weight_decay)
        self.config['alpha'] = self.alpha
        self.config['beta'] = self.beta

    def train_step(self):
        self.model.train()

        # update GNN (self.model)
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        emb = self.model.get_embedding(self.train_data, self.train_data.ndata['feat'])
        logits_G = self.model.classify_emb(emb)
        logits_A = self.adv(emb)  # .detach())

        sens = self.train_data.ndata['sens_u']
        y_score = torch.sigmoid(logits_G)
        cov = torch.abs(torch.mean((sens - torch.mean(sens)) * (y_score - torch.mean(y_score))))
        cls_loss = self.criterion(logits_G[self.train_data.ndata['train_mask']],
                                  self.train_data.ndata['label'][self.train_data.ndata['train_mask']])
        adv_loss = self.criterion(logits_A, sens)

        G_loss = cls_loss + self.alpha * cov - self.beta * adv_loss
        G_loss.backward()
        self.optimizer_G.step()

        # update Adv (self.adv)
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()

        logits_A = self.adv(emb.detach())
        A_loss = self.criterion(logits_A, sens)
        A_loss.backward()
        self.optimizer_A.step()
        return float(G_loss.item())
