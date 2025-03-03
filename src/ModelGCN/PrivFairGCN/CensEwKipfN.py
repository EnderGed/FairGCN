import dgl

from ModelGCN import EwKipfN, CensGcnN
from consts import SEED


class CensEwKipfN(EwKipfN, CensGcnN):
    """
    Edge weightening with privacy censoring
    """

    fair_prefix = CensGcnN.fair_prefix + EwKipfN.fair_prefix

    def __init__(self, data: dgl.DGLGraph, alpha: float, priv_lambda: float, hid_feats: int,
                 dropout: (float, float) = (0.5, 0.), lr: float = 1e-3, weight_decay: float = 0.,
                 gpu: str = 'cpu', run_name: str = None, debug_mode: bool = False, seed: int = SEED, *args, **kwargs):
        """
        :param data: dgl.graph,
        :param alpha: float [0, 1] edge weightening strength parameter, 0 - no weights, 1 - complete counterweights
        :param priv_lambda: float, importance of adversarial loss to the loss function
        :param hid_feats: int, size of each hidden layer
        :param dropout: (float, float), probabilities of zeroeing a value after first and second Convolution layers
        :param lr: float, learning rate of the optimizer
        :param weight_decay: float, weight decay (L2 penalty) of the optimizer
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param run_name: str, name of the run, for easy reading in wandb and saving results on the device
        :param debug_mode: bool, if yes, then print instead of log to wandb
        :param seed: randomness seed
        """
        EwKipfN.__init__(self, data=data, alpha=alpha, hid_feats=hid_feats, dropout=dropout, lr=lr,
                         weight_decay=weight_decay, gpu=gpu, run_name=run_name, debug_mode=debug_mode,
                         seed=seed)
        CensGcnN.__init__(self, data=self.data, base_net=self.model, priv_lambda=priv_lambda, lr=lr,
                          weight_decay=weight_decay, run_name=run_name, gpu=gpu, debug_mode=debug_mode, seed=seed)

    def train_step(self):
        return CensGcnN.train_step(self)