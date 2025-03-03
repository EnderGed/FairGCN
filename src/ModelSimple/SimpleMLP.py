import copy

import torch
import torch.nn.functional as F
import random
import numpy as np

from consts import SEED
from ModelSimple import SimpleBase


class SimpleMLP(SimpleBase):
    """
    Very simple MLP for binary classification, doesn't read any files or even connect to wandb.
    """

    def __init__(self, features, labels, num_classes, train_mask, val_mask, test_mask, layers=1, dropout=0.25,
                 gpu='cpu', seed=SEED):
        self.num_classes = num_classes
        self.device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

        data = {'feat': torch.tensor(features.values, dtype=torch.double).to(self.device),
                'train_mask': train_mask.to(self.device),
                'val_mask': val_mask.to(self.device),
                'test_mask': test_mask.to(self.device)}

        if isinstance(labels, torch.Tensor):
            labels_values = labels
        else:   # it's pandas
            labels_values = labels.values
        if self.num_classes == 1:
            data['label'] = torch.tensor(labels_values, dtype=torch.double).unsqueeze(1).to(self.device)
        else:
            data['label'] = torch.tensor(labels_values, dtype=torch.long).to(self.device)

        super().__init__(data)
        # Set random seed
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = SimpleMLPNet(data['feat'].shape[1], self.num_classes, layers, dropout).double().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, epochs=200, early_stopping=20):
        best_val_acc = test_acc = 0
        best_val_epoch = 0
        best_val_state_dict = copy.deepcopy(self.model.state_dict())
        for epoch in range(1, epochs + 1):
            if early_stopping and epoch > best_val_epoch + early_stopping:
                break
            loss = self.train_step()
            metrics = self.eval()
            if early_stopping and (metrics['val_acc'] > best_val_acc):
                best_val_acc = metrics['val_acc']
                test_acc = metrics['test_acc']
                best_val_epoch = epoch
                best_val_state_dict = copy.deepcopy(self.model.state_dict())
            if not early_stopping:
                test_acc = metrics['test_acc']
        if early_stopping:
            self.model.load_state_dict(best_val_state_dict)
        return test_acc

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        logits, metrics = self.model(self.data['feat']), {}
        for split in ['train', 'val', 'test']:
            mask = self.data[split + '_mask']
            pred = logits[mask]
            labels = self.data['label'][mask]
            if self.num_classes == 1:
                pred = pred.squeeze()
                labels = labels.squeeze()
                pred = (pred > 0).type_as(labels)
                correct = torch.sum(pred.eq(labels))
            else:
                _, indices = torch.max(pred, dim=1)
                correct = torch.sum(indices.eq(labels))
            metrics[split + '_acc'] = correct.item() * 1.0 / len(labels)
        return metrics

    def train_step(self):
        self.model.train()
        logits = self.model(self.data['feat'])
        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(logits[self.data['train_mask']],
                                                      self.data['label'][self.data['train_mask']])
        else:
            loss = F.cross_entropy(logits[self.data['train_mask']], self.data['label'][self.data['train_mask']])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


class SimpleMLPNet(torch.nn.Module):
    def __init__(self, in_feats, out_feats, layers, dropout):
        super().__init__()
        self.mlist = torch.nn.ModuleList()
        for i in range(layers - 1):
            self.mlist.append(torch.nn.Linear(in_feats, in_feats//2))
            self.mlist.append(torch.nn.ReLU())
            self.mlist.append(torch.nn.Dropout(dropout))
            in_feats = int(in_feats // 2)
        self.mlist.append(torch.nn.Linear(in_feats, out_feats))

    def forward(self, x):
        for module in self.mlist:
            x = module(x)
        return x