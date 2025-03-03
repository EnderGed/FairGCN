from sklearn import preprocessing
import pandas as pd

from Datasets import Dataset
from RWHelp import InterReaderE, DataReaderE, InterWriterE, DglWriterE
from consts import INTER_PATH, SEED


class DatasetE(Dataset):
    """
    Class that holds an edge classification / regression dataset:
        connection graph with numerical labels (pandas.Dataframe)
        dictionary of node types (str) and their features (pandas.Dataframe)
    """
    name = None
    out_name = None
    edges = None
    nodes = None

    def __init__(self, reader='movielK'):
        """
        Loads given dataset
        :param reader: DataReaderE or dataset name for the default InterReaderE
        """
        if isinstance(reader, str):
            reader = InterReaderE(reader)
        assert isinstance(reader, DataReaderE)
        self.name = reader.name
        self.out_name = reader.out_name
        self.edges, self.nodes = reader.read()
        assert all(col in self.edges.columns for col in ['user', 'item', 'label'])
        assert len(self.nodes) == 2
        assert 'user' in self.nodes.keys()
        assert 'item' in self.nodes.keys()

    def dump(self, out_path=INTER_PATH):
        """
        Dumps current dataset
        :param out_path:
        :return:
        """
        writer = InterWriterE(self.out_name, out_path)
        writer.write(self.edges, self.nodes)

    def get_label_df(self):
        return self.edges

    def prepare_classification(self, regression=True, label_bins=0, features_filter=None, rand_feats=True,
                               train_frac=0.5, validation_frac=0.2, split_name=None, seed=SEED):
        """
        Prepares and saves data for edge classification or regression in dgl format.
        :param regression: is the task regression, if False, will be classification
        :param label_bins: if `regression` == False, how many classes to have
        :param features_filter: {str: [str]}, dictionary of nodes types to list of features to be used
        :param rand_feats: Boolean, create nonexistent features as random, if False wil be zeroes
        :param train_frac: float, fraction of the edge labels used for training
        :param validation_frac: float, fraction of the edge labels used for validation
        :param split_name: string, name of the resulting file, if None will be generated automatically
        :param seed: int, randomness seed
        :return:
        """
        split_args = (regression, label_bins, features_filter, train_frac, validation_frac, seed)
        edges = self.edges.copy()
        nodes = self.nodes.copy()
        # if we're preparing for a regression task, min max scale the label
        if regression:
            edges['label'] = preprocessing.MinMaxScaler().fit_transform(edges['label'].values.reshape(-1, 1))
        # if we're preparing for a classifier, bin the labels
        elif label_bins > 0:
            edges_sorted = edges.sort_values('label').reset_index()
            max_vals = [edges_sorted.loc[int((i + 1) * edges_sorted.shape[0] / label_bins) - 1, 'label']
                        for i in range(label_bins)]
            edges['label'] = [sum([label > val for val in max_vals]) for label in edges['label']]

        # filter the features:
        if features_filter is not None:
            for node_type, feat_names in features_filter:
                nodes[node_type] = nodes[node_type][feat_names]

        # normalize the node features
        for node_type, features in nodes.items():
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(features.values)
            nodes[node_type] = pd.DataFrame(x_scaled, index=features.index, columns=features.columns)

        # generate train, test, validation split
        train_ids, test_ids, val_ids = self.train_test_val_split(edges, train_frac, validation_frac, seed)

        if split_name is None:
            split_name = ('reg' if regression else ('clf' + str(label_bins))) + \
                         ('all' if features_filter is None else 'fil')

        writer = DglWriterE(split_name, self.out_name, rand_feats)
        return writer.write(edges, nodes, train_ids, val_ids, test_ids, split_args)

    def __eq__(self, other):
        if self.__class__ != other.__class__ or self.out_name != other.out_name:
            return False
        if not self.edges.equals(other.edges):
            return False
        if self.nodes.keys() != other.nodes.keys():
            return False
        return all(self.nodes[key].equals(other.nodes[key]) for key in self.nodes.keys())
