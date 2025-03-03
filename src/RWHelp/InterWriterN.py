from consts import INTER_PATH
from RWHelp import DataWriterN
from graph_utils import write_nxgraph


class InterWriterN(DataWriterN):
    """
    Writes data set in an intermediate representation.
    """

    _full_name_format = '{}{}/'
    _full_path_format = '{}{}/'

    def __init__(self, name, out_path=INTER_PATH):
        super().__init__(name, out_path)

    def write(self, features, graph):
        """
        Writes features and graph into separate files in self.out_path/name_{feat.ftr, edges.csv}.
        :param features: pandas.DataFrame
        :param graph: networkx.Graph
        :return:
        """
        features.reset_index().to_feather('{}feat.ftr'.format(self.full_name))
        write_nxgraph(graph, '{}edges.csv'.format(self.full_name))
