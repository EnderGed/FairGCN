import logging
import sys
import os

from research_tests.commons import skip_if_done, dirs

# myPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, myPath + '/../')

from Datasets import DatasetE
from ModelGCN import SageN, KipfN, GatN
from ModelSimple import Xgboost

import pytest

from consts import SEED
from Datasets.DatasetN import DatasetN
from RWHelp import DglReaderN, FairgnnReaderN, MovielensReaderE


# ##########--------------  PREPARE FOR TESTING  --------------###########
@pytest.mark.setup
def test_make_subdirs():
    for tmp_dir in dirs.values():
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)


@pytest.mark.parametrize('name', ['pokec_z', 'pokec_n', 'nba'])
@pytest.mark.inter
def test_transform_pokec_nba(name):
    res_file = 'node/{}/feat.ftr'.format(name)
    skip_if_done(res_file, dirs['inter'])
    ds = DatasetN(FairgnnReaderN(name))
    ds.dump()


@pytest.mark.parametrize('name', ['movielK', 'movielM'])
@pytest.mark.inter
def test_transform_movielens(name):
    res_file = 'edge/{}/edges.ftr'.format(name)
    skip_if_done(res_file, dirs['iter'])
    ds = DatasetE(MovielensReaderE(name))
    ds.dump()


@pytest.mark.parametrize('split_name,ds_name,label,label_bins,priv_feature,priv_bins,sens,filters', [
    # ('race1gen_cin', 'ny_ci', 'race', 'gender', 1, ['age']),
    # ('race1age_cin', 'ny_ci', 'race', 'age', 1, ['gender']),
    ('work1gen1reg', 'pokec_n', 'I_am_working_in_field', 1, 'gender', 1, 'region', None),
    ('work1gen1reg', 'pokec_z', 'I_am_working_in_field', 1, 'gender', 1, 'region', None),
    ('salary1age4cou', 'nba', 'SALARY', 1, 'AGE', 4, 'country', None)
])
@pytest.mark.dgl
def test_make_node_dgl(split_name, ds_name, label, label_bins, priv_feature, priv_bins, sens, filters, cv):
    split_name = '{}/{}'.format(cv, split_name)
    res_file = 'node/{}/{}'.format(ds_name, split_name)
    skip_if_done(res_file, dirs['dgl'], '.data')
    ds = DatasetN(ds_name)
    ret_vals = ds.prepare_classification(label_feature=label, label_bins=label_bins, priv_feature=priv_feature,
                                         priv_bins=priv_bins, sens_feature=sens, features_filter_out=filters,
                                         split_name=split_name, seed=SEED + cv)
    print('Prepared dgl format for split {}'.format(split_name))



@pytest.mark.parametrize('model_class,hid_feats,dropout', [
    (KipfN, 16, (0.25, 0.25)),
    (SageN, 32, (0.75, 0.75)),
    # (GinN, 32, (0.5, 0)),
])
@pytest.mark.parametrize('epochs', [300])
@pytest.mark.gcns
@pytest.mark.skip("Skip for now")
def test_train_gcns(split_ds_names, model_class, hid_feats, dropout, cv, epochs, gpu):
    split_name, ds_name = split_ds_names
    gpu = 'cuda:' + gpu
    run_name = '{}'.format(hid_feats)
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    model = model_class(reader.read(), hid_feats=hid_feats, dropout=dropout, gpu=gpu, run_name=run_name)
    model.train(epochs=epochs)


@pytest.mark.parametrize('hid_feats', [32])
@pytest.mark.parametrize('epochs', [300])
@pytest.mark.gcns
@pytest.mark.skip("Skip for now")
def test_train_gat(split_ds_names, hid_feats, cv, epochs, gpu):
    split_name, ds_name = split_ds_names
    gpu = 'cuda:' + gpu
    run_name = '{}'.format(hid_feats)
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    model = GatN(reader.read(), hid_feats, (2, 2), gpu=gpu, run_name=run_name)
    model.train(epochs=epochs)


@pytest.mark.parametrize('subsample', [0.7])  # [0.7, 1])  # (0, 1], the higher, the bigger overfitting
@pytest.mark.parametrize('lr', [0.3])  # It is common to have small values in the range of 0.1 to 0.3, as well as values less than 0.1.
@pytest.mark.parametrize('max_depth', [4])  # the higher, the bigger overfitting
@pytest.mark.model_simple
def test_train_xgboost(split_ds_names, gpu, subsample, lr, max_depth, cv):
    split_name, ds_name = split_ds_names
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    xgb = Xgboost(reader.read(), max_depth=max_depth, subsample=subsample, lr=lr, gpu=gpu)
    xgb.train()
    # xgb.save_model()
