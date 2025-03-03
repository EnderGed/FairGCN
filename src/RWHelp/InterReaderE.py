import pandas as pd
import pickle as pkl

from consts import INTER_PATH
from RWHelp import DataReaderE


class InterReaderE(DataReaderE):
    """
    Reads data set for edge classification / regression saved as an intermediate representation.
    """

    _full_name_format = '{}{}/'
    _full_path_format = '{}{}/'

    def __init__(self, name, in_path=INTER_PATH):
        super().__init__(name, in_path)

    def read(self):
        edges = pd.read_feather(self.full_name + 'edges.ftr')
        with open(self.full_name + 'keys.pkl', 'rb') as keys_file:
            keys = pkl.load(keys_file)
        nodes = dict()
        for key in keys:
            feats = pd.read_feather('{}{}.ftr'.format(self.full_name, key))
            feats.index = feats['index']
            nodes[key] = feats.drop('index', axis=1)
        return edges, nodes
