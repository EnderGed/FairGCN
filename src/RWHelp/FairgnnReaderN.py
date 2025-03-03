import networkx as nx
import numpy as np
import pandas as pd

from consts import FAIRGNN_PATH
from RWHelp import DataReaderN


class FairgnnReaderN(DataReaderN):
    """
    Reads data set processed by FairGNN paper https://github.com/EnyanDai/FairGNN
    Changes all -1s to np.nan
    """

    def __init__(self, name, in_path=FAIRGNN_PATH):
        super().__init__(name, in_path, add_to_path=False)
        if name not in ['nba', 'pokec_z', 'pokec_n']:
            raise Exception('Unknown dataset name {}.'.format(name))

    def read(self):
        info = pd.read_csv('{}.csv'.format(self.full_name))
        edges = pd.read_csv('{}_relationship.txt'.format(self.full_name), sep='\t', header=None)

        # drop nodes with 0 edges
        all_ids = [idx for idx in info.user_id if (idx in edges[0].values or idx in edges[1].values)]
        info = info.loc[info.user_id.isin(all_ids), :].reset_index(drop=True)

        id_map = {v: k for (k, v) in info.user_id.iteritems()}
        info.drop('user_id', axis=1, inplace=True)
        info.replace(to_replace=-1, value=np.nan, inplace=True)
        graph = nx.from_edgelist([[id_map[val] for val in row] for row in edges.values])
        return graph, info
