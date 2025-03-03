import dgl

from ModelGCN import AdGcnN, EwKipfN
from consts import SEED


class EwAdKipfN(AdGcnN):
    """
    Edge weightening combined with Adversarial debiasing for GCN (Kipf)
    """

    fair_prefix = 'EwAd'

    def __init__(self, data: dgl.DGLGraph, ew_alpha: float, alpha: float, beta: float, hid_feats: int,
                 dropout: (float, float) = (0.5, 0.), lr: float = 0.001, weight_decay: float = 0., run_name: str = None,
                 inductive: bool = False, validation: bool = True, gpu: str = 'cpu', debug_mode: bool = False,
                 seed: int = SEED):
        """
        :param data: dgl.graph,
        :param ew_alpha: float, [0, 1] edge weightening strength parameter, 0 - no weights, 1 - complete counterweights
        :param alpha: float, importance of covariance constraint to the loss function
        :param beta: float, importance of adversarial loss to the loss function
        :param hid_feats: int, size of each hidden layer
        :param dropout: (float, float), probabilities of zeroeing a value after first and second Convolution layers
        :param lr: float, learning rate of the optimizers
        :param weight_decay: float, weight decay of the optimizers
        :param run_name: str, name of the run for easy reading in wandb and saving results on the device
        :param inductive: bool, whether inductive setting should be used, transductive by default.
        :param validation: bool, whether validation set should be used, if False, add it to test set
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param debug_mode: bool, if True then print instead of log to wandb
        :param seed: randomness seed
        """

        ew = EwKipfN(data=data, alpha=ew_alpha, hid_feats=hid_feats, dropout=dropout, lr=lr,
                     weight_decay=weight_decay, gpu=gpu, run_name=run_name, debug_mode=debug_mode, seed=seed)
        AdGcnN.__init__(self, data=ew.data, base_net=ew.model, alpha=alpha, beta=beta, lr=lr, weight_decay=weight_decay,
                        run_name=run_name, inductive=inductive, validation=validation, gpu=gpu, debug_mode=debug_mode,
                        seed=seed)
        self.ew_alpha = ew_alpha
        self.set_run_config(run_name=run_name, lr=lr, weight_decay=weight_decay)
        self.config['ew_alpha'] = ew_alpha
