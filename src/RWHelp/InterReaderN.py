import pandas as pd
import pickle as pkl

from consts import INTER_PATH
from RWHelp import DataReaderN
from graph_utils import read_nxgraph


class InterReaderN(DataReaderN):
    """
    Reads data set for node classification / regression saved as an intermediate representation.
    """

    _full_name_format = '{}{}/'
    _full_path_format = '{}{}/'

    def __init__(self, name, in_path=INTER_PATH):
        super().__init__(name, in_path)

    def read(self):
        features = pd.read_feather('{}feat.ftr'.format(self.full_name))
        features.index = features['index']
        features = features.drop('index', axis=1)
        graph = read_nxgraph('{}edges.csv'.format(self.full_name))
        return graph, features
