import pickle as pkl
import dgl
from dgl import save_graphs
import numpy as np
import pandas as pd
import torch

from RWHelp import DataWriterE
from consts import DGL_PATH, SEED


class DglWriterE(DataWriterE):
    """
    Writes edge classification / regression data set in a format accepted by DGL.
    """

    def __init__(self, name, dataset_name, rand_feats=True, out_path=DGL_PATH):
        """

        :param name:
        :param dataset_name:
        :param rand_feats: create nonexistent features as random, if False wil be zeroes
        :param out_path:
        """
        super().__init__(dataset_name + '/' + name, out_path)
        self.dataset_name = dataset_name
        self.rand_feats = rand_feats

    def dump(self, data, info, split_args):
        save_graphs(self.full_name + '.data', data)
        with open(self.full_name + '.info', 'wb') as info_file:
            pkl.dump(info, info_file)
        with open(self.full_name + '.args', 'wb') as args_file:
            pkl.dump(split_args, args_file)
        print('Saving DGL data under {}.data, .args. info.'.format(self.full_name))

    def write(self, edges, nodes, train_ids, val_ids, test_ids, split_args):
        """
        Create dgl compliant data and save it on disk.
        :param edges: pandas.DataFrame, with columns 'user', 'item', 'label'
        :param nodes: {str: pandas.DataFrame}, dictionary of nodes types and their features
        :param train_ids: [int], list of training indices
        :param val_ids: [int], list of validation indices
        :param test_ids: [int], list of testing indices
        :param split_args: list of parameters used for dataset generation
        :return:
        """
        def make_mask(ids):
            mask = torch.tensor([False] * edges.shape[0])
            mask[ids] = True
            return torch.cat((mask, mask), 0)

        # Fill nans with average
        for node_type, features in nodes.items():
            for attr in features.columns:
                features[attr] = features[attr].fillna(features[attr].mean())

        # Extend users and items with random values
        np.random.seed(SEED)
        u_cols = nodes['user'].columns
        i_cols = nodes['item'].columns
        for col in u_cols:
            if self.rand_feats:
                nodes['item'][col] = np.random.uniform(0., 1., nodes['item'].shape[0])
            else:
                nodes['item'][col] = np.zeros(nodes['item'].shape[0])
        for col in i_cols:
            if self.rand_feats:
                nodes['user'][col] = np.random.uniform(0., 1., nodes['user'].shape[0])
            else:
                nodes['user'][col] = np.zeros(nodes['user'].shape[0])
        all_nodes = pd.concat([nodes['user'], nodes['item']]).reset_index(drop=True)
        edges['item'] += nodes['user'].shape[0]
        rev_edges = edges.copy()
        rev_edges.columns = ['item', 'user', 'label']
        all_edges = pd.concat([edges, rev_edges]).reset_index(drop=True)

        g = dgl.graph((all_edges['item'].values, all_edges['user'].values))
        g.ndata['feat'] = torch.tensor(all_nodes.values)
        g.edata['label'] = torch.tensor(all_edges.label.values).reshape((all_edges.shape[0], 1))
        g.edata['train_mask'] = make_mask(train_ids)
        g.edata['val_mask'] = make_mask(val_ids)
        g.edata['test_mask'] = make_mask(test_ids)
        g.num_classes = 1
        g.split_name = self.name
        info = {'num_classes': g.num_classes, 'split_name': g.split_name}
        self.dump(g, info, split_args)
        return g
