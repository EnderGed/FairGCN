from abc import ABC

from RWHelp import DataWriter


class DataWriterN(DataWriter, ABC):
    """
    Abstract class for writing node classification / regression data sets to different formats.
    """
    pass

    def __init__(self, name, out_path, add_to_path=True):
        """
        :param name:
        :param out_path:
        :param add_to_path: if true, will add 'node/' to `out_path`
        """
        super().__init__(name, out_path + 'node/' if add_to_path else out_path)
