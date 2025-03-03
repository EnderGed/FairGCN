import pickle as pkl
import dgl
from dgl import save_graphs
import torch

from RWHelp import DataWriterN
from consts import DGL_PATH


class DglWriterN(DataWriterN):
    """
    Writes data set in a format accepted by DGL.
    """

    def __init__(self, split_name, dataset_name, out_path=DGL_PATH):
        super().__init__(dataset_name + '/' + split_name, out_path)
        self.dataset_name = dataset_name

    def dump(self, data, info, split_args):
        save_graphs(self.full_name + '.data', data)
        with open(self.full_name + '.info', 'wb') as info_file:
            pkl.dump(info, info_file)
        with open(self.full_name + '.args', 'wb') as args_file:
            pkl.dump(split_args, args_file)
        print('Saving DGL data under {}.data, .args .info.'.format(self.full_name))

    def write(self, graph, features, labels, labels_classes, priv, priv_classes, sens, incoming, train_ids, val_ids,
              test_ids, split_args, task='classification'):
        """
        Create dgl compliant data and save it on disk.
        :param graph: nxgraph.graph, edge graph
        :param features: pandas.DataFrame, feature matrix, index is the node id
        :param labels: pandas.Series, label class as int, index is the node id
        :param labels_classes: int, number of target classes, 1 means a binary classifier,
                            if 0, num_classes will be inferred from labels
        :param priv: pandas.Series, private attributes as int, index is the node id
        :param priv_classes: int, number of privatea attr classes, 1 means a binary classifier,
                            if 0, priv_classes will be inferred from priv
        :param sens: pandas.Series, sensitive attributes as int, index is the node id or None
        :param incoming: numpy.array((2, len(labels)),
                incoming[0] - number of edges incoming from nodes in group 0
                incoming[1] - number of edges incoming from nodes in group 1
        :param train_ids: [int], list of training indices
        :param val_ids: [int], list of validation indices
        :param test_ids: [int], list of testing indices
        :param split_args: list of parameters used for dataset generation
        :param task: 'classification' or 'regression', will convert the labels to long or double respectively
        :return:
        """

        def make_mask(ids):
            mask = torch.tensor([False] * features.shape[0])
            mask[ids] = True
            return mask

        if task == 'classification':
            label_dtype = torch.long
        elif task == 'regression':
            label_dtype = torch.double
        else:
            raise Exception("Unsupported task {}.".format(task))
        # Fill nans with average
        for attr in features.columns:
            features[attr] = features[attr].fillna(features[attr].mean())

        nodes = list(range(len(labels)))
        u, v = zip(*list(graph.edges))
        u = list(u)
        v = list(v)
        g = dgl.graph((nodes + u + v, nodes + v + u))
        # initialize edge weights to ones
        # g.edata['w'] = torch.ones(2 * len(u) + len(nodes), dtype=torch.double)
        g.ndata['feat'] = torch.tensor(features.values, dtype=torch.double)
        g.ndata['label'] = torch.tensor(labels.values, dtype=label_dtype)
        if priv is not None:
            g.ndata['priv'] = torch.tensor(priv.values, dtype=torch.long)
        if sens is not None:
            g.ndata['sens'] = torch.tensor(sens.values, dtype=torch.long)
            g.ndata['inc_0'] = torch.tensor(incoming[0], dtype=torch.long)
            g.ndata['inc_1'] = torch.tensor(incoming[1], dtype=torch.long)

        g.ndata['train_mask'] = make_mask(train_ids)
        g.ndata['val_mask'] = make_mask(val_ids)
        g.ndata['test_mask'] = make_mask(test_ids)
        g.num_classes = labels_classes if labels_classes > 0 else int(labels.max(skipna=True) + 1)
        g.priv_classes = priv_classes if priv_classes > 0 else int(priv.max(skipna=True) + 1)
        g.split_name = self.name
        info = {'num_classes': g.num_classes, 'priv_classes': g.priv_classes, 'split_name': g.split_name}
        self.dump(g, info, split_args)
        return g
