from abc import abstractmethod, ABC
import os


class DataWriter(ABC):
    """
    Abstract class for writing data sets to different formats.
    """
    name = None
    out_path = None
    full_path = None
    full_name = None
    _full_path_format = '{}'
    _full_name_format = '{}{}'

    def __init__(self, name, out_path):
        self.name = name
        self.out_path = out_path
        self.full_path = self._full_path_format.format(self.out_path, self.name)
        self.full_name = self._full_name_format.format(self.out_path, self.name)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

    @abstractmethod
    def write(self, *args, **kwargs):
        """
        Writes required files, whatever they may be.
        """
        pass