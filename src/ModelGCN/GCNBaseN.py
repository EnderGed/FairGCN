from abc import abstractmethod, abstractproperty, ABC
from functools import lru_cache

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from ModelGCN import GCNBase
from consts import PROJECT_NAME, MODELS_PATH


class GCNBaseN(GCNBase, ABC):
    """
    Abstract basic model for node classification.
    It assumes the features to be in self.data.ndata['feat'].

    Children classes need only to create and instantiate the model itself.
    """
    config = None

    def train_step(self):
        self.model.train()
        data = self.train_data
        logits = self.model(data, data.ndata['feat'])
        if self.data.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(logits[data.ndata['train_mask']],
                                                      data.ndata['label'][data.ndata['train_mask']])
        else:
            loss = F.cross_entropy(logits[data.ndata['train_mask']],
                                   data.ndata['label'][data.ndata['train_mask']])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def eval(self):
        self.set_eval_mode()
        splits = ['train', 'val', 'test'] if self.validation else ['train', 'test']
        datas = {'train': self.train_data, 'test': self.test_data}
        if self.validation:
            datas['val'] = self.val_data
        metrics = {}
        if not self.inductive:
            logits = self.get_predictions('all')
        for split in splits:
            if self.inductive:
                logits = self.get_predictions(split)
            mask = datas[split].ndata[split + '_mask']
        # for mask in [self.data.ndata[mask_name].cpu().numpy() for mask_name in ['train_mask', 'val_mask', 'test_mask']]:
            pred = logits[mask]
            labels = datas[split].ndata['label'][mask]
            if self.data.num_classes == 1:
                pred = pred.squeeze()
                labels = labels.squeeze()
                pred = (pred > 0).type_as(labels)
                correct = torch.sum(pred == labels)
            else:
                _, indices = torch.max(pred, dim=1)
                correct = torch.sum(indices == labels)
            metrics[split + '_acc'] = correct.item() * 1.0 / len(labels)
            # Comment it out since it's broken, but we don't care about wandb anymore
            # if 'sens' in self.data.ndata and not self.debug_mode:
            #     # fairness eval, works only for binary classifiers
            #     for fair_type in ['par', 'eoo']:
            #         metrics['{}_{}'.format(split, fair_type)] = (
            #             torch.sum(pred[self.fair_mask(fair_type, 0, split)]) / self.fair_mask_count(fair_type, 0, split)
            #             -
            #             torch.sum(pred[self.fair_mask(fair_type, 1, split)]) / self.fair_mask_count(fair_type, 1, split)
            #         ).item()
        return metrics

    @lru_cache(maxsize=6)
    def fair_mask(self, fair_type: str, sens_group: int, split: str) -> torch.Tensor:
        """
        Calculate mask for fairness evaluations.
        Thanks to @lru_cache, we never recalculate the same call
        :param fair_type: 'par' or 'eoo'
        :param sens_group: 0 or 1
        :param split: 'train', 'val' or 'test
        :return: Tensor of Bool
        """
        if fair_type == 'par':
            return (self.data.ndata['sens'] == sens_group) * (self.data.ndata[split + '_mask'])
        else:
            return (self.data.ndata['sens'] == sens_group) * (self.data.ndata['label'].squeeze(1) == 1) *\
                   (self.data.ndata[split + '_mask'])

    @lru_cache(maxsize=6)
    def fair_mask_count(self, fair_type: str, sens_group: int, split: str) -> torch.Tensor:
        """
        Calculate number of elements in a mask.
        :param fair_type: 'par' or 'eoo'
        :param sens_group: 0 or 1
        :param split: 'train', 'val' or 'test
        :return: Tensor of a single int
        """
        return torch.sum(self.fair_mask(fair_type, sens_group, split))

    def get_model_name(self):
        return self.fair_prefix + self.model.name

    def init_logging(self):
        if not self.debug_mode:
            wandb.init(project=PROJECT_NAME, name=self.run_name, config=self.config,
                       tags=[self.get_model_name(), 'node', self.data.split_name], reinit=True)
            wandb.watch(self.model)

    def set_run_config(self, run_name, lr, weight_decay):
        """
        Has to be run in init of the children classes. Sets names and starts wandb.
        :param run_name:
        :param lr:
        :param weight_decay:
        :return:
        """
        if run_name is None:
            run_name = ''
        if not self.validation:
            run_name += 'Noval'
        if self.inductive:
            run_name += 'Ind'
        self.run_name = '{}/{}_{}_{}_{}_{}_{}'.format(self.get_model_name(), self.data.split_name, run_name,
                                                      self.alpha, self.beta, self.ew_alpha, self.priv_lambda)

        self.model_path = MODELS_PATH + self.run_name

        # wandb config
        self.config = self.model.config
        self.config["lr"] = lr
        self.config["weight_decay"] = weight_decay


class GCNBaseNNet(torch.nn.Module, ABC):
    """
    Base for all GCN Networks (Modules).
    Implements classification network, children only have to implement embeddings.
    """
    def __init__(self, hid_feats, out_feats):
        super().__init__()
        self.emb_size = hid_feats
        self.fc = torch.nn.Linear(hid_feats, out_feats)

    def forward(self, g, h):
        h = self.get_embedding(g, h)
        h = self.classify_emb(h)
        return h

    @abstractmethod
    def get_embedding(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def classify_emb(self, h):
        return self.fc(h)

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def config(self):
        raise NotImplementedError
