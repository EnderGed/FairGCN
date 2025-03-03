import pickle as pkl

from dgl import load_graphs
from dgl.data.utils import generate_mask_tensor

from consts import DGL_PATH
from RWHelp import DataReaderN


class DglReaderN(DataReaderN):
    """
    Reads Dgl compliant data set split.
    """

    def __init__(self, split_name, dataset_name, cv=0, in_path=DGL_PATH):
        super().__init__('{}/{}/{}'.format(dataset_name, cv, split_name), in_path)

    def read(self):
        with open(self.full_name + '.info', 'rb') as info_file:
            info = pkl.load(info_file)
        g = load_graphs(self.full_name + '.data')[0][0]
        g.num_classes = info['num_classes']
        g.priv_classes = info['priv_classes']
        g.split_name = info['split_name']
        # because pytorch cannot save booleans ???
        g.ndata['train_mask'] = generate_mask_tensor(g.ndata['train_mask'].numpy())
        g.ndata['val_mask'] = generate_mask_tensor(g.ndata['val_mask'].numpy())
        g.ndata['test_mask'] = generate_mask_tensor(g.ndata['test_mask'].numpy())
        return g

    def read_args(self):
        return pkl.load(open(self.full_name + '.args', 'rb'))
