from abc import ABC

from RWHelp import DataReader


class DataReaderE(DataReader, ABC):
    """
    Abstract class for reading edge classification / regression data sets to different formats.
    """
    pass

    def __init__(self, name, out_path, add_to_path=True):
        """
        :param name:
        :param out_path:
        :param add_to_path: if true, will add 'edge/' to `out_path`
        """
        super().__init__(name, out_path + 'edge/' if add_to_path else out_path)
