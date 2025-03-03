import dgl

from ModelGCN import EwFlSageN, CensGcnN, CensFlGcnN
from consts import SEED


class CensEwFlSageN(CensFlGcnN, EwFlSageN, CensGcnN):
    """
    Fair learning with Edge weightening and privacy censoring.
    """
    fair_prefix = CensGcnN.fair_prefix + EwFlSageN.fair_prefix

    def __init__(self, data: dgl.DGLGraph, ew_alpha: float, alpha: float,
                 priv_lambda: float, hid_feats: int, fairnes_type: str = 'par', dropout: (float, float) = (0.5, 0.),
                 lr: float = 0.001, weight_decay: float = 0., run_name: str = None, gpu: str = 'cpu',
                 debug_mode: bool = False, seed: int = SEED, *args, **kwargs):
        """
        :param data: dgl.graph,
        :param ew_alpha: float, [0, 1] edge weightening strength parameter, 0 - no weights, 1 - complete counterweights
        :param alpha: float, importance of covariance constraint to the loss function
        :param priv_lambda: float, importance of adversarial loss to the loss function
        :param hid_feats: int, size of each hidden layer
        :param fairnes_type: str, 'eoo' or 'par', definition of fairnes used for calculating the loss
        :param dropout: (float, float), probabilities of zeroeing a value after first and second Convolution layers
        :param lr: float, learning rate of the optimizers
        :param weight_decay: float, weight decay of the optimizers
        :param run_name: str, name of the run for easy reading in wandb and saving results on the device
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param debug_mode: bool, if True then print instead of log to wandb
        :param seed: randomness seed
        """
        EwFlSageN.__init__(self, data=data, ew_alpha=ew_alpha, alpha=alpha, fairnes_type=fairnes_type,
                           hid_feats=hid_feats, dropout=dropout, lr=lr, weight_decay=weight_decay, run_name=run_name,
                           gpu=gpu, debug_mode=debug_mode, seed=seed)
        CensGcnN.__init__(self, data=self.data, base_net=self.model, priv_lambda=priv_lambda, lr=lr,
                          weight_decay=weight_decay, run_name=run_name, gpu=gpu, debug_mode=debug_mode, seed=seed)

    def train_step(self):
        return CensFlGcnN.train_step(self)
