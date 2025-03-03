from abc import abstractmethod


class SimpleBase:
    """
    Base, abstract ModelSimple.
    These models should work with geo_data and simply ignore graph data.
    """
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def train(self):
        pass
