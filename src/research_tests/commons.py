import os
import pickle

import pytest

from ModelGCN import KipfN, SageN, GatN, CensAdGcnN, CensEwKipfN, CensEwAdKipfN, CensEwFlKipfN, CensFlGcnN, CensGcnN, \
    CensEwSageN, CensEwAdSageN, CensEwFlSageN
from consts import RESULTS_PATH, EMBEDDINGS_PATH, INTER_PATH, DGL_PATH, MODELS_PATH, PREDICTIONS_PATH


def is_done(name, directory, appendix=''):
    return os.path.isfile('{}{}{}'.format(directory, name, appendix))


def skip_if_done(name, directory, appendix=''):
    if is_done(name, directory, appendix):
        pytest.skip('{}{}{} already exists'.format(directory, name, appendix))


dirs = {
    'inter': INTER_PATH,
    # 'pokec': POKEC_PATH,
    # 'geo': GEO_PATH,
    'dgl': DGL_PATH,
    'model': MODELS_PATH,
    'emb': EMBEDDINGS_PATH,
    'res': RESULTS_PATH,
    'preds': PREDICTIONS_PATH,
    }


# a set of completed and deleted embeddings
# Once all attacks requiring the embeddings (fair_leak, priv_leak, mia_acc) are run, we don't need the embedding anymore
# Preds will still be saved
deleted_embeds_file = '{}deleted_embeds.pkl'.format(dirs['res'])
if os.path.exists(deleted_embeds_file):
    with open(deleted_embeds_file, 'rb') as f:
        deleted_embeds = pickle.load(f)
else:
    deleted_embeds = []




def skip_if_emb_done(emb_name):
    if '{}{}'.format(EMBEDDINGS_PATH, emb_name) in deleted_embeds:
        pytest.skip("embedding {} has served it's purpose and was deleted".format(emb_name))
    else:
        skip_if_done(emb_name, EMBEDDINGS_PATH)


early_stopping = 50

# parameters for Fair algorithms
fil_params = {'gammas': [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 0.5, 1., 5., 10., 15., 20., 50., 100., 1000.], 'd_stepss': [3, 4]}
ep_params = {}  # embedding projection is not parametrizable
ad_params = {'alphas': [1., 2.], 'betas': [0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.5, 0.7]}
ew_params = {'alphas': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                        0.85, 0.9, 0.95, 1.0],
             'ew_alphas': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
fl_params = {'alphas': [0.0001, 0.001, 0.01, 0.1, 0.5, 1., 5., 10., 20., 40., 60., 80., 100., 1000.]}

AUTO_OPT = -100.
# Parameters for optimal Fair algorithms, telling functions to automatically take parameters from fairgnn_opt_params
# ad_params = {'alphas': [AUTO_OPT], 'betas': [AUTO_OPT]}
# ew_params = {'alphas': [AUTO_OPT], 'ew_alphas': [AUTO_OPT]}
# fl_params = {'alphas': [AUTO_OPT]}


# dictionary from (ds_name, model_name, validation) into parameters to instantiate specified GNN model
gnn_params = {
    ('nba',     'Kipf', True):  (KipfN, {'hid_feats': 55,  'dropout': (0.5632, 0.5592),  'lr': 0.01933,   'weight_decay': 0.000762,  'run_name': 'opt'}),  # 0.7613 0.7581 0.6667
    ('pokec_n', 'Kipf', True):  (KipfN, {'hid_feats': 49,  'dropout': (0.7487, 0.03426), 'lr': 0.00933,   'weight_decay': 0.0004119, 'run_name': 'opt'}),  # 0.7779 0.7034 0.7188
    ('pokec_z', 'Kipf', True):  (KipfN, {'hid_feats': 53,  'dropout': (0.4519, 0.4526),  'lr': 0.0008374, 'weight_decay': 0.0007648, 'run_name': 'opt'}),  # 0.7507 0.732  0.7116
    ('nba',     'Sage', True):  (SageN, {'hid_feats': 110, 'dropout': (0.6156, 0.557),   'lr': 0.0005227, 'weight_decay': 0.0007721, 'run_name': 'opt'}),  # 0.8    0.7742 0.6344
    ('pokec_n', 'Sage', True):  (SageN, {'hid_feats': 63,  'dropout': (0.7621, 0.1426),  'lr': 0.002484,  'weight_decay': 0.0009233, 'run_name': 'opt'}),  # 0.869  0.6989 0.7116
    ('pokec_z', 'Sage', True):  (SageN, {'hid_feats': 50,  'dropout': (0.7696, 0.5413),  'lr': 0.003148,  'weight_decay': 0.0004365, 'run_name': 'opt'}),  # 0.8983 0.7368 0.7142
    ('nba',     'Gat',  True):  (GatN,  {'hid_feats': 96, 'num_heads': (2, 1), 'feat_drop': (0.18, 0.5308),    'att_drop': (0.24, 0.3488),   'negative_slope': 0.4264,  'lr': 0.003051, 'weight_decay': 0.0003003, 'run_name': 'opt'}),  # 0.8645 0.7581 0.7527
    ('pokec_n', 'Gat',  True):  (GatN,  {'hid_feats': 61, 'num_heads': (2, 4), 'feat_drop': (0.2687, 0.3562),  'att_drop': (0.4344, 0.2722), 'negative_slope': 0.1513,  'lr': 0.01796,  'weight_decay': 0.0000312, 'run_name': 'opt'}),  # 0.8158 0.7011 0.7086
    ('pokec_z', 'Gat',  True):  (GatN,  {'hid_feats': 55, 'num_heads': (3, 2), 'feat_drop': (0.1139, 0.05198), 'att_drop': (0.3492, 0.2879), 'negative_slope': 0.02188, 'lr': 0.0094,   'weight_decay': 0.0004047, 'run_name': 'opt'}),  # 0.8129 0.7281 0.7135
    ('nba',     'Kipf', False): (KipfN, {'hid_feats': 64, 'dropout': (0.5, 0.5), 'lr':1e-3, 'weight_decay': 0., 'run_name': '64'}),
    ('pokec_n', 'Kipf', False): (KipfN, {'hid_feats': 64, 'dropout': (0.5, 0.5), 'lr':1e-3, 'weight_decay': 0., 'run_name': '64'}),
    ('pokec_z', 'Kipf', False): (KipfN, {'hid_feats': 64, 'dropout': (0.5, 0.5), 'lr':1e-3, 'weight_decay': 0., 'run_name': '64'}),
    ('nba',     'Sage', False): (SageN, {'hid_feats': 64, 'dropout': (0.5, 0.5), 'lr':1e-3, 'weight_decay': 0., 'run_name': '64'}),
    ('pokec_n', 'Sage', False): (SageN, {'hid_feats': 64, 'dropout': (0.5, 0.5), 'lr':1e-3, 'weight_decay': 0., 'run_name': '64'}),
    ('pokec_z', 'Sage', False): (SageN, {'hid_feats': 64, 'dropout': (0.5, 0.5), 'lr':1e-3, 'weight_decay': 0., 'run_name': '64'}),
    ('nba',     'Gat',  False): (GatN,  {'hid_feats': 64, 'num_heads': (2, 2),   'lr':1e-3, 'weight_decay': 0., 'run_name': '64'}),
    ('pokec_n', 'Gat',  False): (GatN,  {'hid_feats': 64, 'num_heads': (2, 2),   'lr':1e-3, 'weight_decay': 0., 'run_name': '64'}),
    ('pokec_z', 'Gat',  False): (GatN,  {'hid_feats': 64, 'num_heads': (2, 2),   'lr':1e-3, 'weight_decay': 0., 'run_name': '64'}),
}

# For private models
priv_models = {
    ('Kipf', 'Ad'):      CensAdGcnN,
    ('Kipf', 'Ew'):      CensEwKipfN,
    ('Kipf', 'EwAd'):    CensEwAdKipfN,
    ('Kipf', 'EwFlpar'): CensEwFlKipfN,
    ('Kipf', 'Flpar'):   CensFlGcnN,
    ('Kipf', 'base'):    CensGcnN,
    ('Sage', 'Ad'):      CensAdGcnN,
    ('Sage', 'Ew'):      CensEwSageN,
    ('Sage', 'EwAd'):    CensEwAdSageN,
    ('Sage', 'EwFlpar'): CensEwFlSageN,
    ('Sage', 'Flpar'):   CensFlGcnN,
    ('Sage', 'base'):    CensGcnN,
}
# fairgnn_opt_params = generate_fairgnn_opt_params_dict()
fairgnn_opt_params = {
    ('nba', 'Kipf', 'base'): {'alpha': 0.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('nba', 'Kipf', 'Ad'): {'alpha': 1.0, 'beta': 0.01, 'ew_alpha': 0.0},
    ('nba', 'Kipf', 'Flpar'): {'alpha': 5.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('nba', 'Kipf', 'Ew'): {'alpha': 0.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('nba', 'Kipf', 'EwAd'): {'alpha': 1.0, 'beta': 0.19, 'ew_alpha': 0.3},
    ('nba', 'Kipf', 'EwFlpar'): {'alpha': 20.0, 'beta': 0.0, 'ew_alpha': 0.6},
    ('nba', 'Sage', 'base'): {'alpha': 0.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('nba', 'Sage', 'Ad'): {'alpha': 1.0, 'beta': 0.0001, 'ew_alpha': 0.0},
    ('nba', 'Sage', 'Flpar'): {'alpha': 0.0001, 'beta': 0.0, 'ew_alpha': 0.0},
    ('nba', 'Sage', 'Ew'): {'alpha': 0.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('nba', 'Sage', 'EwAd'): {'alpha': 2.0, 'beta': 0.01, 'ew_alpha': 0.9},
    ('nba', 'Sage', 'EwFlpar'): {'alpha': 0.0001, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_n', 'Kipf', 'base'): {'alpha': 0.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_n', 'Kipf', 'Ad'): {'alpha': 1.0, 'beta': 0.03, 'ew_alpha': 0.0},
    ('pokec_n', 'Kipf', 'Flpar'): {'alpha': 100.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_n', 'Kipf', 'Ew'): {'alpha': 0.4, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_n', 'Kipf', 'EwAd'): {'alpha': 2.0, 'beta': 0.01, 'ew_alpha': 0.0},
    ('pokec_n', 'Kipf', 'EwFlpar'): {'alpha': 20.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_n', 'Sage', 'base'): {'alpha': 0.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_n', 'Sage', 'Ad'): {'alpha': 2.0, 'beta': 0.03, 'ew_alpha': 0.0},
    ('pokec_n', 'Sage', 'Flpar'): {'alpha': 80.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_n', 'Sage', 'Ew'): {'alpha': 0.35, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_n', 'Sage', 'EwAd'): {'alpha': 2.0, 'beta': 0.19, 'ew_alpha': 0.4},
    ('pokec_n', 'Sage', 'EwFlpar'): {'alpha': 10.0, 'beta': 0.0, 'ew_alpha': 0.3},
    ('pokec_z', 'Kipf', 'base'): {'alpha': 0.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_z', 'Kipf', 'Ad'): {'alpha': 1.0, 'beta': 0.09, 'ew_alpha': 0.0},
    ('pokec_z', 'Kipf', 'Flpar'): {'alpha': 5.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_z', 'Kipf', 'Ew'): {'alpha': 0.35, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_z', 'Kipf', 'EwAd'): {'alpha': 1.0, 'beta': 0.17, 'ew_alpha': 0.2},
    ('pokec_z', 'Kipf', 'EwFlpar'): {'alpha': 10.0, 'beta': 0.0, 'ew_alpha': 0.3},
    ('pokec_z', 'Sage', 'base'): {'alpha': 0.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_z', 'Sage', 'Ad'): {'alpha': 2.0, 'beta': 0.07, 'ew_alpha': 0.0},
    ('pokec_z', 'Sage', 'Flpar'): {'alpha': 80.0, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_z', 'Sage', 'Ew'): {'alpha': 0.3, 'beta': 0.0, 'ew_alpha': 0.0},
    ('pokec_z', 'Sage', 'EwAd'): {'alpha': 2.0, 'beta': 0.09, 'ew_alpha': 0.2},
    ('pokec_z', 'Sage', 'EwFlpar'): {'alpha': 100.0, 'beta': 0.0, 'ew_alpha': 0.1}
}

# privacy lambda parameters for Cens
lambdas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1., 3., 6., 10.]
