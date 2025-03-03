import multiprocessing
import os
from datetime import date

import pytest
import torch
from networkx.tests.test_convert_pandas import pd

from Datasets import DatasetN
from ModelGCN import KipfN, SageN, GatN, GinN
from ModelSimple import SimpleMLP
from RWHelp import DglReaderN
from research_tests.commons import dirs, skip_if_emb_done
from research_tests.test_attacks import attr_bins_dict, test_attr_inference, test_mia
from scripts import get_network_file_name


@pytest.mark.parametrize('model_class', [KipfN, SageN, GinN])
@pytest.mark.parametrize('hid_feats', [64])
@pytest.mark.parametrize('dropout', [(0.5, 0.5)])
@pytest.mark.parametrize('validation', [False])
@pytest.mark.parametrize('inductive', [True, False])
@pytest.mark.ind_gnn
def test_train_gnns_variants(split_ds_names, model_class, hid_feats, dropout, validation, inductive, cv, epochs, gpu):
    split_name, ds_name = split_ds_names
    gpu = 'cuda:' + gpu
    run_name = '{}'.format(hid_feats)
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    model = model_class(reader.read(), hid_feats=hid_feats, dropout=dropout, validation=validation, inductive=inductive,
                        gpu=gpu, run_name=run_name, debug_mode=True)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=0)
    model.save_embeddings()
    model.save_predictions()


@pytest.mark.parametrize('hid_feats', [64])
@pytest.mark.parametrize('validation', [False])
@pytest.mark.parametrize('inductive', [True, False])
@pytest.mark.ind_gnn
def test_train_gat_variants(split_ds_names, hid_feats, validation, inductive, cv, epochs, gpu):
    split_name, ds_name = split_ds_names
    gpu = 'cuda:' + gpu
    run_name = '{}'.format(hid_feats)
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    model = GatN(reader.read(), hid_feats, (2, 2), validation=validation, inductive=inductive, gpu=gpu,
                 run_name=run_name, debug_mode=True)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=0)
    model.save_embeddings()
    model.save_predictions()


@pytest.mark.parametrize('attack', ['priv', 'fair'])
@pytest.mark.parametrize('layers', [2])
@pytest.mark.parametrize('hyper_name', ['64Noval', '64NovalInd'])
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat', 'Gin'])
@pytest.mark.parametrize('fair_net,alpha,beta,ew_alpha,priv_lambda', [('', 0, 0, 0, 0)])
@pytest.mark.parametrize('dropout', [0.25])
@pytest.mark.ind_attacks
def test_attr_inference_variants(split_ds_names, hyper_name, model_name, fair_net, alpha, beta, ew_alpha, priv_lambda,
                                 attack, layers, cv, epochs, dropout, gpu):
    test_attr_inference(split_ds_names, hyper_name, model_name, fair_net, alpha, beta, ew_alpha, priv_lambda,
                        attack, layers, cv, epochs, dropout, gpu)


@pytest.mark.parametrize('hyper_name', ['64Noval', '64NovalInd'])
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat', 'Gin'])
@pytest.mark.parametrize('fair_net,alpha,beta,ew_alpha,priv_lambda', [('', 0, 0, 0, 0)])
@pytest.mark.parametrize('layers', [2])
@pytest.mark.parametrize('dropout', [0.25])
@pytest.mark.ind_attacks
def test_mia_variants(split_ds_names, hyper_name, model_name, fair_net, alpha, beta, ew_alpha, priv_lambda, layers, cv,
                      epochs, dropout, gpu):
    test_mia(split_ds_names, hyper_name, model_name, fair_net, alpha, beta, ew_alpha, priv_lambda, layers, cv, epochs,
             dropout, gpu)
