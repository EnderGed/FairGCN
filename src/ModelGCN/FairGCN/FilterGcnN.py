import dgl
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from typing import List

import wandb

from ModelGCN import GCNBaseN, GCNBaseNNet
from consts import SEED, PROJECT_NAME


class FilterGcnN(GCNBaseN):
    """
    Filtering GCN class

    Sensitive attribute filtering base class. Implements the sensitive filters and discriminators and the joined
     training step.
    From the paper: "Compositional Fairness Constraints for Graph Embeddings", instead of using TranE or GCMC we use
     GCNs for a fair comparison.
    """

    fair_prefix = 'Fil'

    def __init__(self, data: dgl.DGLGraph, base_net: GCNBaseNNet, gamma: float, d_steps: int = 5,
                 lr: float = 0.001, weight_decay: float = 0., sens_features: List[str] = None, run_name: str = '',
                 inductive: bool = False, validation: bool = True, gpu: str = 'cpu', debug_mode: bool = False,
                 seed: int = SEED):
        """
        :param data: dgl.graph,
        :param base_net: base GCN network to be made fair
        :param gamma: float, importance of adversarial loss to the loss function
        :param d_steps: int, number of training steps of discriminators in every step of base model
        :param lr: float, learning rate of the optimizers
        :param weight_decay: float, weight decay of the optimizers
        :param sens_features: [str], list of sensitive feature names in data.ndata
        :param run_name: str, name of the run for easy reading in wandb and saving results on the device
        :param inductive: bool, whether inductive setting should be used, transductive by default.
        :param validation: bool, whether validation set should be used, if False, add it to test set
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param debug_mode: bool, if True then print instead of log to wandb
        :param seed: randomness seed
        """
        GCNBaseN.__init__(self, data=data, inductive=inductive, validation=validation, gpu=gpu, debug_mode=debug_mode,
                          seed=seed)
        if sens_features is None:
            sens_features = ['sens']
        # if 'rand' in sens_features and 'rand' not in self.data.ndata:
        #     self.data.ndata['rand'] = (torch.rand(data.ndata['label'].shape) < 0.5).int().to(self.device)
        self.model = base_net.to(self.device).double()

        # Unsqueeze sens features once, for faster loss calculation
        for data in [self.data, self.train_data, self.test_data] + ([self.val_data] if self.validation else []):
            for sens_feature in sens_features:
                data.ndata[sens_feature + '_u'] = data.ndata[sens_feature].unsqueeze(1).double()

        # Create one filter, discriminator and discriminator optimizer for each sens_feature
        filters = [SensFilter(base_net.emb_size).to(self.device).double() for _ in sens_features]
        discriminators = [SensDiscriminator(base_net.emb_size).to(self.device).double() for _ in sens_features]
        disc_optimizers = [torch.optim.Adam(disc.parameters(), lr=lr) for disc in discriminators]
        # save them all as self.sens_filters together with sens_features
        self.sens_filters = np.array(list(zip(sens_features, filters, discriminators, disc_optimizers)))

        # create base model optimizer from it's and filters parameters
        # self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model_optimizer = torch.optim.Adam(itertools.chain(
            self.model.parameters(), *[fil.parameters() for fil in filters]), lr=lr, weight_decay=weight_decay)
        self.disc_criterion = torch.nn.BCEWithLogitsLoss()
        self.model_criterion = torch.nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.d_steps = d_steps
        # alpha and beta are used for run naming reasons
        self.alpha = gamma
        self.beta = d_steps

        # set wandb cofiguration
        self.set_run_config(run_name=run_name, lr=lr, weight_decay=weight_decay)
        self.config['no_sens'] = len(sens_features)
        self.config['d_steps'] = self.d_steps
        self.config['gamma'] = self.gamma

    def init_logging(self):
        if not self.debug_mode:
            wandb.init(project=PROJECT_NAME, name=self.run_name, config=self.config,
                       tags=['Fil' + self.model.name, 'node', self.data.split_name], reinit=True)
            wandb.watch(self.model)
            for sens_feature, sens_filter, disc, disc_optimizer in self.sens_filters:
                wandb.watch(sens_filter)

    def train_step(self):
        # draw a random filter mask
        mask = np.random.choice([True, False], self.sens_filters.shape[0])
        # mask = np.array([False])
        step_filters = self.sens_filters[mask]

        # train base models with selected filters
        self.model.train()
        for sens_feature, sens_filter, disc, disc_optimizer in step_filters:
            sens_filter.train()
            disc.requires_grad_(False)

        # forward through base model and all selected filters
        emb = self.model.get_embedding(self.train_data, self.train_data.ndata['feat'])
        for sens_feature, sens_filter, disc, disc_optimizer in step_filters:
            emb = sens_filter(emb)

        # get logits and loss of target task
        target_logits = self.model.classify_emb(emb)
        target_loss = self.model_criterion(target_logits[self.train_data.ndata['train_mask']],
                                           self.train_data.ndata['label'][self.train_data.ndata['train_mask']])

        # run discriminators
        adv_loss_sum = 0
        for sens_feature, sens_filter, disc, disc_optimizer in step_filters:
            adv_loss_sum += self.disc_criterion(disc(emb), self.train_data.ndata[sens_feature + '_u'])

        # gradient descend on the base model and filters
        model_loss = target_loss - self.gamma * adv_loss_sum
        self.model_optimizer.zero_grad()
        model_loss.backward() #retain_graph=False)
        self.model_optimizer.step()

        # train adversarial discriminators
        for sens_feature, sens_filter, disc, disc_optimizer in step_filters:
            disc.requires_grad_(True)
            for _ in range(self.d_steps):
                disc_loss = self.disc_criterion(disc(emb.detach()), self.train_data.ndata[sens_feature + '_u'])
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()

        return float(model_loss.item())

    def set_eval_mode(self):
        self.model.eval()
        for sens_feature, sens_filter, disc, disc_optimizer in self.sens_filters:
            sens_filter.eval()
            disc.eval()

    def get_embeddings(self, split='all'):
        """

        :param split: can be all, train, val or test
        :return:
        """
        data = self.get_data_for_split(split=split)
        emb = self.model.get_embedding(data, data.ndata['feat'])
        for sens_feature, sens_filter, disc, disc_optimizer in self.sens_filters:
            emb = sens_filter(emb)
        return emb

    def get_predictions(self, split='all'):
        emb = self.get_embeddings(split)
        return self.model.classify_emb(emb)


class SensFilter(torch.nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.lin1 = torch.nn.Linear(emb_size, emb_size * 2, bias=True)
        self.lin2 = torch.nn.Linear(emb_size * 2, emb_size, bias=True)

    def forward(self, h):
        h = F.leaky_relu(self.lin1(h))
        h = F.leaky_relu(self.lin2(h))
        return h


class SensDiscriminator(torch.nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        #
        # self.sigmoid = torch.nn.Sigmoid()
        # self.criterion = torch.nn.BCELoss()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size * 2, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(emb_size * 2, emb_size * 4, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(emb_size * 4, emb_size * 4, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(emb_size * 4, emb_size * 2, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(emb_size * 2, emb_size * 2, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(emb_size * 2, emb_size, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(emb_size, emb_size, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(emb_size, emb_size // 2, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(emb_size // 2, 1, bias=True)
            )

    def forward(self, h):
        return self.net(h)
