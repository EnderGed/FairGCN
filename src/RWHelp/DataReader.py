from abc import abstractmethod, ABC


class DataReader(ABC):
    """
    Abstract class for reading data sets of different formats.
    """
    name = None
    in_path = None
    full_path = None
    full_name = None
    _full_path_format = '{}'
    _full_name_format = '{}{}'

    def __init__(self, name, in_path):
        self.name = name
        self.out_name = name
        self.in_path = in_path
        self.full_path = self._full_path_format.format(self.in_path, self.name)
        self.full_name = self._full_name_format.format(self.in_path, self.name)

    @abstractmethod
    def read(self):
        """
        Reads given dataset and returns graph and feature matrix
        :return: (networkx.Graph, pandas.DataFrame)
        """
        pass
