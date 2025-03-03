import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import preprocessing

from Datasets import Dataset
from RWHelp import DataReaderN, InterWriterN, InterReaderN, DglWriterN
from deprecated.GeoWriterN import GeoWriterN
from consts import INTER_PATH, SEED
from gen_utils import deprecated


class DatasetN(Dataset):
    """
    Class that holds a node classification / regression dataset: friendship graph and feature matrix
    can read multiple formats
    """
    name = None
    out_name = None
    graph = None
    features = None

    def __init__(self, reader='ny'):
        """
        Loads given dataset
        :param reader: DataReaderN or dataset name for the default InterReaderN
        """
        if isinstance(reader, str):
            reader = InterReaderN(reader)
        assert isinstance(reader, DataReaderN)
        self.name = reader.name
        self.out_name = reader.out_name
        self.graph, self.features = reader.read()

    def dump(self, out_path=INTER_PATH):
        """
        Dumps current dataset
        :param out_path:
        :return:
        """
        writer = InterWriterN(self.out_name, out_path)
        writer.write(self.features, self.graph)

    def get_label_df(self):
        return self.features

    @deprecated
    def train_test_split(self, df=None, train_frac=0.5, split_feature=None, seed=SEED):
        """
        Splits self.features into training and testing disjoint subsets.
        Makes sure every value of `split_feature` will make up equal fraction in both resulting subsets.
        :param df: dataframe, if None then use self.features
        :param train_frac: number of training samples / number of all samples
        :param split_feature: feature on which to split the data
        :param seed: randomness seed
        :return: ([training indices], [testing indices]) as a tuple of pandas.Index
        """
        if not df:
            df = self.features
        if not split_feature:
            train_ids = df.sample(frac=train_frac, random_state=seed).index
        else:
            possible_values = df[split_feature].unique()
            if len(possible_values) > 5:
                raise Exception("Feature {} has more then 5 possible values.".format(split_feature))
            train_ids = pd.Index([])
            for val in df[split_feature].unique():
                train_ids = train_ids.append(df.loc[df[split_feature] == val].sample(
                        frac=train_frac, random_state=seed
                    ).index)
        test_ids = df.index.difference(train_ids)
        return train_ids, test_ids

    @deprecated
    def prepare_kipf_format(self, label_feature, label_bins=0, features_filter=None,
                            train_frac=0.5, validation_frac=0.2, seed=SEED):
        """
        Prepares the data for kipf GCN. Return the same tuple as kipf.utils.load_data.
        :param label_feature: feature that will be a label for the prediction task
        :param label_bins: number of bins for classifiaction task, if 0 then use as many as label_feature values
        :param features_filter: list of features to be used as features in the prediction task
        :param train_frac: fraction of samples used for training
        :param validation_frac: fraction of samples used for validation
        :param seed: randomness seed
        :return: (
                    networkx.adjacency_matrix graph,
                    scipy.sparse.lil.lil_matrix features,
                    2D np.array training labels, rest is masked,
                    2D np.array validation labels, rest is masked,
                    2D np.array testing labels, rest is masked,
                    1D np.array train_mask,
                    1D np.array val_mask,
                    1D np.array test_mask
                )
        """

        def create_mask(ids):
            mask = np.zeros(self.features.shape[0])
            mask[ids] = 1
            return mask

        # reduce the labels diversity by binning them (equal number of users in each bin)
        df = self.features.copy()
        if label_bins > 0:
            df_sorted = df.sort_values(label_feature).reset_index()
            max_vals = [df_sorted.loc[int((i + 1) * df_sorted.shape[0] / label_bins) - 1, label_feature]
                        for i in range(label_bins)]
            df[label_feature] = [sum([feat > val for val in max_vals]) for feat in df[label_feature]]
        labels = pd.get_dummies(df[label_feature])
        df = df.drop(label_feature, axis=1)
        train_ids, test_ids, val_ids = self.train_test_val_split(df, train_frac, validation_frac, seed)
        if features_filter:
            df = df.loc[:, features_filter]
        adj = nx.adjacency_matrix(self.graph)
        feats = sp.csr_matrix(df.values).tolil()
        train_mask = create_mask(train_ids)
        test_mask = create_mask(test_ids)
        val_mask = create_mask(val_ids)
        y_train = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_train[train_ids, :] = labels.loc[train_ids, :].values
        y_test[test_ids, :] = labels.loc[test_ids, :].values
        y_val[val_ids, :] = labels.loc[val_ids, :].values
        return adj, feats, y_train, y_val, y_test, train_mask, val_mask, test_mask

    def discretize_feature(self, feature, bins):
        if bins <= 0:
            return self.features[feature]
        if bins == 1:
            top_class = self.features[feature].value_counts().index[0]
            return self.features[feature].map(lambda x: 0 if x == top_class else 1, na_action='ignore')
        df_sorted = self.features.loc[self.features[feature] >= 0, :].sort_values(feature).reset_index()
        max_vals = [df_sorted.loc[int((i + 1) * df_sorted.shape[0] / bins) - 1, feature] for i in range(bins)]
        return pd.Series([sum([feat > val for val in max_vals]) for feat in self.features[feature]])

    def prepare_classification(self, label_feature, label_bins=0, priv_feature=None, priv_bins=0, sens_feature=None,
                               features_filter_out=None, train_frac=0.5, validation_frac=0.2, split_name=None,
                               seed=SEED):
        """
        Prepares and saves data for node classification in dgl format.
         Also (if sens_feature) for every node count number of incoming edges from each sensitive group and save it
          on the node
        :param label_feature: string, feature that will be a label for the prediction task,
                                records with label_feature = -1 are skipped for classification task
        :param label_bins: int, number of bins for classification task, if 0 then use as many as label_feature values,
                           if 1 then take the biggest class as 0 and all others as 1
        :param priv_feature: string, feature that will be the sensitive attribute for privacy evaluation
        :param priv_bins: int, number of bins for privacy attr inference, if 0 then use as many as label_feature values,
                           if 1 then take the biggest class as 0 and all others as 1
        :param sens_feature: string, feature that will be the sensitive attribute for fairness evaluation,
                                records with sens_feature = -1 are skipped for fairness evaluation (not implemented yet)
                                or None if we have no sensitive feature
        :param features_filter_out: [string], list of features to be omitted as features in the prediction task
        :param train_frac: float, fraction of dataset labels used for training
        :param validation_frac: float, fraction of dataset labels used for validation
        :param split_name: string, name of the resulting file, if None will be generated automatically
        # :param out_path: string, path to where the resulting file should be saved if None, default dgl will be used
        :param seed: int, randomness seed
        :return:
        """
        split_args = (label_feature, label_bins, priv_bins, features_filter_out, train_frac, validation_frac, seed)
        df = self.features.copy()
        # reduce the labels diversity by binning them (equal number of users in each bin)
        labels = self.discretize_feature(label_feature, label_bins)
        # extract labels out of features
        df = df.drop(label_feature, axis=1)
        # extract privacy attr out of feature and bin it if needed
        if priv_feature is not None:
            df = df.drop(priv_feature, axis=1)
            priv = self.discretize_feature(priv_feature, priv_bins)
        else:
            priv = None
        # extract sens out of features
        if sens_feature is not None:
            sens = df[sens_feature]
            df = df.drop(sens_feature, axis=1)
        else:
            sens = None
        # filter the features out:
        if features_filter_out is not None:
            df = df[[col for col in df.columns if col not in features_filter_out]]
        # normalize the data
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(df.values)
        df = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)
        # generate train, test, validation split
        train_ids, test_ids, val_ids = self.train_test_val_split(
            df.loc[labels >= 0, :], train_frac, validation_frac, seed)

        # for each node count number of incoming edges from same and other sensitive group as the node itself
        # incoming[0] - number of edges incoming from sens group 0
        # incoming[1] - number of edges incoming from sens group 1
        incoming = np.zeros((2, len(sens)), np.int64)
        if sens_feature is not None:
            for u, v in self.graph.edges:
                incoming[sens[u], v] += 1
                incoming[sens[v], u] += 1

        if split_name is None:
            split_name = '{}{}{}{}{}_{}'.format(label_feature, label_bins, priv_feature, priv_bins, sens_feature,
                                                'all' if features_filter_out is None else 'fil')
        # save to DGL format
        writer = DglWriterN(split_name, self.out_name)
        return writer.write(graph=self.graph, features=df, labels=labels, labels_classes=label_bins, priv=priv,
                            priv_classes=priv_bins, sens=sens, incoming=incoming, train_ids=train_ids, val_ids=val_ids,
                            test_ids=test_ids, split_args=split_args)

    @deprecated
    def gen_crossval_sets(self, writer, label_feature, label_bins=0, features=None, num_crossval=5,
                          train_frac=0.5, train_label_frac=0.2, split_feature=None, seed=SEED):
        """
        Creates `num_sets` train test splits and saves them with a chosen writer
         in subdirs corresponding to `label_feature`, `split_feature` and set number
        :param writer: RWHelp.DataWriter
        :param label_feature: feature that will be a label for the prediction task
        :param label_bins: number of bins for classification task, if 0 then use as many as label_feature values
        :param features: list of features to be used as features in the prediction task
        :param num_crossval: number of training testing pairs
        :param train_frac: fraction of samples used for training
        :param train_label_frac: fraction of training samples that will be labled
        :param split_feature: feature on which to split the data equally
        :param seed: randomness seed
        :return:
        """
        # reduce the labels diversity by binning them (equal number of users in each bin)
        df = self.features.copy()
        if label_bins > 0:
            df_sorted = df.sort_values(label_feature).reset_index()
            max_vals = [df_sorted.loc[int((i + 1) * df_sorted.shape[0] / label_bins) - 1, label_feature]
                        for i in range(label_bins)]
            df[label_feature] = [sum([feat > val for  val in max_vals]) for feat in df[label_feature]]

        for i in range(num_crossval):
            train_ids, test_ids = self.train_test_split(df, train_frac, split_feature, seed + i)
            train_label_ids = train_ids[:int(train_label_frac * len(train_ids))]
            subdir = '{}/'.format(split_feature) if split_feature else '' + '{}/'.format(i)
            if features:
                df = df.loc[:, features + [label_feature]]
            writer.write(features=df, graph=self.graph, label_feature=label_feature, train_label_ids=train_label_ids,
                         train_ids=train_ids, test_ids=test_ids, subdir=subdir)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.out_name == other.out_name and
            self.graph.edges == other.graph.edges and
            self.features.equals(other.features)
        )
