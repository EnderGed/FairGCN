from abc import abstractmethod, ABC

from consts import SEED


class Dataset(ABC):
    """
    Basic, abstract Dataset class for all the common functions
    """

    def train_test_val_split(self, df=None, train_frac=0.5, validation_frac=0.2, seed=SEED):
        """
        Splits a dataframe into training, testing and validation sets.
        :param df: DataFrame to be split. self.features if None
        :param train_frac: fraction of the samples to be used as training set
        :param validation_frac: fraction of the samples to be used as validation set
        :param seed: randomness seed
        :return: training ids, testing ids, validation ids
        """
        if df is None:
            df = self.get_label_df()
        train_ids = df.sample(frac=train_frac, random_state=seed).index
        other_ids = df.index.difference(train_ids)
        other_df = df.loc[other_ids, :]
        val_frac = validation_frac / (1 - train_frac)
        if val_frac > 1:
            raise Exception("Validation fraction {} must be smaller then the testing set {}.".format(
                validation_frac, 1 - train_frac))
        val_ids = other_df.sample(frac=val_frac, random_state=seed+1).index
        test_ids = other_ids.difference(val_ids)
        return train_ids.sort_values(), test_ids.sort_values(), val_ids.sort_values()

    @abstractmethod
    def get_label_df(self):
        """
        Return DataFrame that has classifier labels.
        :return: pandas.DataFrame
        """
        pass
