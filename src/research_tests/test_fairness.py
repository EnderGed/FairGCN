import os
import pickle
import pytest

from ModelGCN import AdGcnN, FilterGcnN, GatN, KipfN, SageN, GinN, FlGcnN, EpKipfN, EpSageN, EpGinN, EpGatN, EwKipfN, \
    EwSageN, EwGinN, EwGatN, EwAdKipfN, EwAdSageN, EwFlKipfN, EwFlSageN
from RWHelp import DglReaderN
from consts import CV_NUM
from research_tests.commons import dirs, deleted_embeds_file, gnn_params, skip_if_emb_done, early_stopping, \
    ad_params, fl_params, ew_params, AUTO_OPT, fairgnn_opt_params, fil_params


@pytest.mark.test
def test_pytest_test(split_ds_names):
    split_name, ds_name = split_ds_names
    if ds_name != 'nba':
        pytest.skip('test only on nba')
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=0)
    model = KipfN(reader.read(), 45, debug_mode=True)
    model.train(10)


@pytest.mark.setup
def test_make_emb_dirs():
    for subdir in [dirs['emb'], dirs['preds'], dirs['model']]:
        for gcn in ['Kipf', 'Sage', 'Gat', 'Gin']:
            for fairnes in ['', 'Ew', 'Ep', 'Ad', 'Fil', 'Flpar', 'Fleoo', 'EwAd', 'EwFlpar', 'Cens', 'CensAd',
                            'CensFlpar', 'CensEw', 'CensEwAd', 'CensEwFlpar']:
                for ds in ['nba', 'pokec_n', 'pokec_z']:
                    for cv in range(CV_NUM):
                        dir_path = '{}{}{}/{}/{}/'.format(subdir, fairnes, gcn, ds, cv)
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)


@pytest.mark.setup
def test_make_deleted_emb_file():
    if not os.path.exists(deleted_embeds_file):
        deleted_embeds = set()
        with open(deleted_embeds_file, 'wb') as f:
            pickle.dump(deleted_embeds, f)


# Unfair GCNs with optimal parameters
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat'])
@pytest.mark.parametrize('validation,inductive', [(False, False), (True, False), (False, True)])
@pytest.mark.gcns
def test_gcn_embeddings(split_ds_names, model_name, validation, inductive, cv, epochs, gpu):
    gpu = 'cuda:' + gpu
    split_name, ds_name = split_ds_names
    model_class, model_kwargs = gnn_params[(ds_name, model_name, validation)]
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    model = model_class(reader.read(), validation=validation, inductive=inductive, gpu=gpu, debug_mode=True, **model_kwargs)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=early_stopping if validation else 0)
    # model.save_model()
    model.save_embeddings()
    model.save_predictions()


# Adversarial debiasing
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat'])
@pytest.mark.parametrize('validation,inductive', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('alpha', ad_params['alphas'])
@pytest.mark.parametrize('beta', ad_params['betas'])
@pytest.mark.fair
def test_ad_gcn_embeddings(split_ds_names, model_name, alpha: float, beta: float, validation, inductive, cv, epochs, gpu):
    gpu = 'cuda:' + gpu
    split_name, ds_name = split_ds_names
    if alpha == beta == AUTO_OPT:
        opt_params = fairgnn_opt_params[(ds_name, model_name, 'Ad')]
        alpha = opt_params['alpha']
        beta = opt_params['beta']
    base_model_class, model_kwargs = gnn_params[(ds_name, model_name, validation)]
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    base_model = base_model_class(reader.read(), validation=validation, inductive=inductive, debug_mode=True, **model_kwargs)
    model = AdGcnN(data=base_model.data, base_net=base_model.model, alpha=alpha, beta=beta, lr=model_kwargs['lr'],
                   weight_decay=model_kwargs['weight_decay'], run_name=model_kwargs['run_name'], validation=validation,
                   inductive=inductive, gpu=gpu, debug_mode=True)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=early_stopping if validation else 0)
    model.save_embeddings()
    model.save_predictions()


# Fair learning
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat'])
@pytest.mark.parametrize('validation,inductive', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('alpha', fl_params['alphas'])
@pytest.mark.parametrize('fairnes_type', ['par'])  # ['eoo', 'par'])
@pytest.mark.fair
def test_fl_gcn_embeddings(split_ds_names, model_name, alpha, fairnes_type, validation, inductive, cv, epochs, gpu):
    gpu = 'cuda:' + gpu
    split_name, ds_name = split_ds_names
    if alpha == AUTO_OPT:
        opt_params = fairgnn_opt_params[(ds_name, model_name, 'Fl'+fairnes_type)]
        alpha = opt_params['alpha']
    base_model_class, model_kwargs = gnn_params[(ds_name, model_name, validation)]
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    base_model = base_model_class(reader.read(), validation=validation, inductive=inductive, debug_mode=True, **model_kwargs)
    model = FlGcnN(data=base_model.data, base_net=base_model.model, alpha=alpha, fairnes_type=fairnes_type,
                   lr=model_kwargs['lr'], weight_decay=model_kwargs['weight_decay'], run_name=model_kwargs['run_name'],
                   validation=validation, inductive=inductive, gpu=gpu, debug_mode=True)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=early_stopping if validation else 0)
    model.save_embeddings()
    model.save_predictions()


# Adversarial filtering (Bose, Hamilton)
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat'])
@pytest.mark.parametrize('validation,inductive', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('gamma', fil_params['gammas'])
@pytest.mark.parametrize('d_steps', fil_params['d_stepss'])
@pytest.mark.fair
def test_fil_gcn_embeddings(split_ds_names, model_name, gamma: float, d_steps: int, validation, inductive, cv, epochs, gpu):
    gpu = 'cuda:' + gpu
    split_name, ds_name = split_ds_names
    base_model_class, model_kwargs = gnn_params[(ds_name, model_name, validation)]
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    base_model = base_model_class(reader.read(), validation=validation, inductive=inductive, debug_mode=True, **model_kwargs)
    model = FilterGcnN(data=base_model.data, base_net=base_model.model, gamma=gamma, d_steps=d_steps,
                       lr=model_kwargs['lr'], weight_decay=model_kwargs['weight_decay'],
                       run_name=model_kwargs['run_name'], validation=validation, inductive=inductive, gpu=gpu,
                       debug_mode=True)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=early_stopping if validation else 0)
    model.save_embeddings()
    model.save_predictions()


# Embedding projection
ep_version = {KipfN: EpKipfN, SageN: EpSageN, GinN: EpGinN, GatN: EpGatN}


@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat'])
@pytest.mark.parametrize('validation,inductive', [(False, False), (True, False), (False, True)])
@pytest.mark.fair
def test_ep_gcn_embeddings(split_ds_names, model_name, validation, inductive, cv, epochs, gpu):
    gpu = 'cuda:' + gpu
    split_name, ds_name = split_ds_names
    base_model_class, model_kwargs = gnn_params[(ds_name, model_name, validation)]
    model_class = ep_version[base_model_class]
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    model = model_class(reader.read(), validation=validation, inductive=inductive, gpu=gpu, debug_mode=True, **model_kwargs)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=early_stopping if validation else 0)
    model.save_embeddings()
    model.save_predictions()


# Edge weightening
ew_version = {KipfN: EwKipfN, SageN: EwSageN, GinN: EwGinN, GatN: EwGatN}


@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat'])
@pytest.mark.parametrize('validation,inductive', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('alpha', ew_params['alphas'])
@pytest.mark.fair
def test_ew_gcn_embeddings(split_ds_names, model_name, alpha, validation, inductive, cv, epochs, gpu):
    gpu = 'cuda:' + gpu
    split_name, ds_name = split_ds_names
    if alpha == AUTO_OPT:
        opt_params = fairgnn_opt_params[(ds_name, model_name, 'Ew')]
        alpha = opt_params['alpha']
    base_model_class, model_kwargs = gnn_params[(ds_name, model_name, validation)]
    model_class = ew_version[base_model_class]
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    model = model_class(reader.read(), alpha=alpha, validation=validation, inductive=inductive, gpu=gpu,
                        debug_mode=True, **model_kwargs)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=early_stopping if validation else 0)
    model.save_embeddings()
    model.save_predictions()


# Edge weightening with Adversarial Training
ew_ad_version = {KipfN: EwAdKipfN, SageN: EwAdSageN}


@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat'])
@pytest.mark.parametrize('validation,inductive', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('ew_alpha', ew_params['ew_alphas'])
@pytest.mark.parametrize('alpha', ad_params['alphas'])
@pytest.mark.parametrize('beta', ad_params['betas'])
@pytest.mark.fair_skip
def test_ew_ad_gcn_embeddings(split_ds_names, model_name, ew_alpha, alpha, beta, validation, inductive, cv, epochs, gpu):
    gpu = 'cuda:' + gpu
    split_name, ds_name = split_ds_names
    if alpha == beta == ew_alpha == AUTO_OPT:
        opt_params = fairgnn_opt_params[(ds_name, model_name, 'EwAd')]
        alpha = opt_params['alpha']
        beta = opt_params['beta']
        ew_alpha = opt_params['ew_alpha']
    base_model_class, model_kwargs = gnn_params[(ds_name, model_name, validation)]
    model_class = ew_ad_version[base_model_class]
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    model = model_class(data=reader.read(), ew_alpha=ew_alpha, alpha=alpha, beta=beta, validation=validation,
                        inductive=inductive, gpu=gpu, debug_mode=True, **model_kwargs)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=early_stopping if validation else 0)
    model.save_embeddings()
    model.save_predictions()


# Edge weightening with Fair learning parity
ew_fl_version = {KipfN: EwFlKipfN, SageN: EwFlSageN}


@pytest.mark.parametrize('model_name', ['Kipf', 'Sage', 'Gat'])
@pytest.mark.parametrize('validation,inductive', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('ew_alpha', ew_params['ew_alphas'])
@pytest.mark.parametrize('alpha', fl_params['alphas'])
@pytest.mark.parametrize('fairnes_type', ['par'])
@pytest.mark.fair_skip
def test_ew_fl_gcn_embeddings(split_ds_names, model_name, ew_alpha, alpha, fairnes_type, validation, inductive, cv,
                              epochs, gpu):
    gpu = 'cuda:' + gpu
    split_name, ds_name = split_ds_names
    if alpha == ew_alpha == AUTO_OPT:
        opt_params = fairgnn_opt_params[(ds_name, model_name, 'EwFlpar')]
        alpha = opt_params['alpha']
        ew_alpha = opt_params['ew_alpha']
    base_model_class, model_kwargs = gnn_params[(ds_name, model_name, validation)]
    model_class = ew_fl_version[base_model_class]
    reader = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv)
    model = model_class(data=reader.read(), ew_alpha=ew_alpha, alpha=alpha, fairnes_type=fairnes_type,
                        validation=validation, inductive=inductive, gpu=gpu, debug_mode=True, **model_kwargs)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=early_stopping if validation else 0)
    model.save_embeddings()
    model.save_predictions()
