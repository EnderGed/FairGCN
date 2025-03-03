import os
import random
from abc import abstractmethod, ABC
import copy
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import pandas as pd

from consts import MODELS_PATH, SEED, EMBEDDINGS_PATH, PREDICTIONS_PATH


class GCNBase(ABC):
    """
    Base, abstract ModelGCN.

    You have to init wandb and set a model to watch in inheriting classes init.
    """
    fair_prefix = ''
    data = None
    debug_mode = None
    model = None
    optimizer = None
    run_name = None
    model_path = None
    # network parameters, for simplicity we call them alpha and beta,
    # but could have different names in different fairness papers
    alpha = 0
    beta = 0
    # only for ew combined with other networks
    # should be in the EwCombined class, but I'm too tired to write it now
    ew_alpha = 0
    # only for Privacy censoring
    # should be in the private base class, but I'm too tired to refactor
    priv_lambda = 0

    def __init__(self, data, gpu, inductive=False, validation=True, debug_mode=False, seed=SEED):
        """

        :param data: DGL graph read by RWHelp.DglReaderN or RWHelp.DglReaderE
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param inductive: bool, whether inductive setting should be used, transductive by default.
        :param validation: bool, whether validation set should be used, if False, add it to test set
        :param debug_mode: bool, if yes, then print instead of log to wandb
        :param seed: randomness seed
        """
        self.debug_mode = debug_mode
        self.device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
        self.data = data.to(self.device)
        if len(self.data.ndata['label'].shape) == 1:
            self.data.ndata['label'] = self.data.ndata['label'].unsqueeze(1).double()
        self.inductive = inductive
        self.validation = validation
        if not validation:
            self.data.ndata['train_mask'] = self.data.ndata['train_mask'] | self.data.ndata['val_mask']
        if inductive:
            self.train_data = self.data.subgraph(self.data.ndata['train_mask'])
            if validation:
                self.val_data = self.data.subgraph(self.data.ndata['train_mask'] | self.data.ndata['val_mask'])
            self.test_data = self.data
        else:
            self.train_data = self.test_data = self.data
            if validation:
                self.val_data = self.data
        # Set random seed
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self, epochs=200, early_stopping=50, force_retrain=False):
        """
        If the model has been saved before, load it, otherwise train.

        :param epochs: number of epochs to train
        :param early_stopping: stop training and rollback when validation accuracy hasn't improved for
         `early_stopping` epochs
        :param force_retrain: train the model even if it has been saved before
        :return:
        """
        if not force_retrain and self.is_model_saved():
            self.load_model()
            print('Model {} loaded, skipping training.'.format(self.model_path))
            return

        if early_stopping and not self.validation:
            raise Exception("Cannot use early stopping without a validation set")

        self.init_logging()
        best_val_acc = test_acc = 0
        best_val_epoch = 0
        if early_stopping:
            best_val_state_dict = copy.deepcopy(self.model.state_dict())
        for epoch in range(1, epochs + 1):
            if early_stopping and epoch > best_val_epoch + early_stopping:
                break
            loss = self.train_step()
            if epoch % 5 == 0:
                metrics = self.eval()
                if self.validation:
                    if metrics['val_acc'] > best_val_acc:
                        best_val_acc = metrics['val_acc']
                        test_acc = metrics['test_acc']
                        best_val_epoch = epoch
                        if early_stopping:
                            best_val_state_dict = copy.deepcopy(self.model.state_dict())
                    else:
                        metrics['test_acc'] = test_acc
                if self.debug_mode:
                    if self.validation:
                        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Loss: {:.4f}'
                        print(log.format(epoch, metrics['train_acc'], best_val_acc, test_acc, loss))
                    else:
                        log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}, Loss: {:.4f}'
                        print(log.format(epoch, metrics['train_acc'], metrics['test_acc'], loss))
                else:
                    metrics['loss'] = loss
                    wandb.log(metrics)
        if early_stopping:
            self.model.load_state_dict(best_val_state_dict)
        if not self.debug_mode:
            wandb.join()

    def set_eval_mode(self):
        """
        Pytorch model eval mode (not training)
        :return:
        """
        self.model.eval()

    def save_model(self):
        dir_path = os.path.dirname(self.model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        self.model.eval()

    def is_model_saved(self):
        return os.path.isfile(self.model_path)

    def get_data_for_split(self, split='all'):
        if not self.inductive and split != 'all':
            raise Exception('Only all split is available for transductive models.')
        if split == 'all':
            data = self.data
        elif split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        elif split == 'test':
            data = self.test_data
        else:
            raise Exception('split should be one of: all, train, val, test')
        return data

    def get_embeddings(self):
        return self.model.get_embedding(self.data, self.data.ndata['feat'])

    def get_predictions(self, split='all'):
        """

        :param split: can be all, train, val or test
        :return:
        """
        data = self.get_data_for_split(split=split)
        return self.model(data, data.ndata['feat'])

    def save_embeddings(self, out_path=EMBEDDINGS_PATH):
        embeddings = self.get_embeddings()
        emb_df = pd.DataFrame(embeddings.cpu().detach().numpy())
        emb_df.columns = [str(i) for i in emb_df.columns]
        dir_path = os.path.dirname(os.path.realpath(out_path + self.get_embeddings_name()))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        emb_df.to_feather(out_path + self.get_embeddings_name())

    def save_predictions(self, out_path=PREDICTIONS_PATH):
        logits = self.get_predictions()
        logits_df = pd.DataFrame(logits.cpu().detach().numpy())
        logits_df.columns = [str(i) for i in logits_df.columns]
        dir_path = os.path.dirname(os.path.realpath(out_path + self.get_embeddings_name()))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        logits_df.to_feather(out_path + self.get_embeddings_name())

    def get_embeddings_name(self):
        return '{}.ftr'.format(self.run_name)

    @abstractmethod
    def train_step(self):
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        raise NotImplementedError

    @abstractmethod
    def init_logging(self):
        raise NotImplementedError
