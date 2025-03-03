import pickle as pkl
from consts import INTER_PATH
from RWHelp import DataWriterE


class InterWriterE(DataWriterE):
    """
    Writes data set for edge classification / regression in an intermediate representation.
    """

    _full_name_format = '{}{}/'
    _full_path_format = '{}{}/'

    def __init__(self, name, out_path=INTER_PATH):
        super().__init__(name, out_path)

    def write(self, edges, nodes):
        """
        Writes edges and each value of nodes under a different file
        :param edges: pandas.DataFrame
        :param nodes: dictionary {str: pandas.DataFrame}
        :return:
        """

        edges.to_feather(self.full_name + 'edges.ftr')
        with open(self.full_name + 'keys.pkl', 'wb') as keys_file:
            pkl.dump(list(nodes.keys()), keys_file)
        for key, val in nodes.items():
            val.reset_index().to_feather('{}{}.ftr'.format(self.full_name, key))