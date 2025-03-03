import dgl
import torch

from ModelGCN import GCNBaseN, GCNBaseNNet
from consts import SEED


class FlGcnN(GCNBaseN):
    """
    Fair learning GCN class

    GCN model trained with Fair learning.
    From the paper: "Mitigating unwanted biases with adversarial learning"
    """

    fair_prefix = 'Fl'

    def __init__(self, data: dgl.DGLGraph, base_net: GCNBaseNNet, alpha: float, fairnes_type: str = 'par',
                 lr: float = 0.001, weight_decay: float = 0., run_name: str = None, inductive: bool = False,
                 validation: bool = True, gpu: str = 'cpu', debug_mode: bool = False, seed: int = SEED):
        """
        :param data: dgl.graph,
        :param base_net: base GCN network to be made fair
        :param alpha: float, importance of covariance constraint to the loss function
        :param fairnes_type: str, 'eoo' or 'par', definition of fairnes used for calculating the loss
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

        self.fair_prefix += fairnes_type
        GCNBaseN.__init__(self, data=data, inductive=inductive, validation=validation, gpu=gpu, debug_mode=debug_mode,
                          seed=seed)
        self.model = base_net.to(self.device).double()

        self.alpha = alpha
        if fairnes_type == 'par':
            subset_mask = self.train_data.ndata['train_mask']
        elif fairnes_type == 'eoo':
            subset_mask = self.train_data.ndata['train_mask'] * (self.train_data.ndata['label'].squeeze(1) == 1)
        else:
            raise Exception('fairnes_type needs to be eoo or par, got {} instead'.format(fairnes_type))
        self.sens_0_mask = (self.train_data.ndata['sens'] == 0) * subset_mask
        self.sens_1_mask = (self.train_data.ndata['sens'] == 1) * subset_mask
        self.sens_0_sum = torch.sum(self.sens_0_mask)
        self.sens_1_sum = torch.sum(self.sens_1_mask)

        self.fairnes_type = fairnes_type
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # set wandb configuration
        self.set_run_config(run_name=run_name, lr=lr, weight_decay=weight_decay)
        self.config['alpha'] = self.alpha
        self.config['fairnes_type'] = self.fairnes_type

    def train_step(self):
        self.model.train()
        # model classification loss
        logits = self.model(self.train_data, self.train_data.ndata['feat'])
        cls_loss = self.criterion(logits[self.train_data.ndata['train_mask']],
                                  self.train_data.ndata['label'][self.train_data.ndata['train_mask']])
        # model fairness loss
        fair_loss = self.get_fair_loss(logits)

        loss = cls_loss + self.alpha * fair_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def get_fair_loss(self, logits):
        probs = torch.sigmoid(logits)
        par_0 = torch.sum(probs[self.sens_0_mask]) / self.sens_0_sum
        par_1 = torch.sum(probs[self.sens_1_mask]) / self.sens_1_sum
        return (par_0 - par_1) ** 2
