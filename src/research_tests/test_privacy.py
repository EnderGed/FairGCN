import pandas as pd
import pytest

from RWHelp import DglReaderN
from consts import RESULTS_PATH
from research_tests.commons import lambdas, fairgnn_opt_params, priv_models
from research_tests.test_fairness import skip_if_emb_done, early_stopping, gnn_params


@pytest.mark.parametrize('model_name', ['Kipf', 'Sage'])
@pytest.mark.parametrize('fair_net', ['base', 'Ad', 'Flpar', 'Ew', 'EwAd', 'EwFlpar'])
@pytest.mark.parametrize('priv_lambda', lambdas)
@pytest.mark.censoring
def test_cens_base_gcn(split_ds_names, model_name, fair_net, priv_lambda, cv, epochs, gpu):
    gpu = 'cuda:' + gpu
    split_name, ds_name = split_ds_names
    base_model_class, base_par = gnn_opt_params[(ds_name, model_name)]
    # prepare the base model
    dataset = DglReaderN(split_name=split_name, dataset_name=ds_name, cv=cv).read()
    base_net = base_model_class(data=dataset, debug_mode=True, gpu=gpu, **base_par).model
    # take the best fairness hyperparameters
    priv_par = {**base_par, **fairgnn_opt_params[(ds_name, model_name, fair_net)]}
    # best = pd.read_feather(RESULTS_PATH + 'best.ftr')
    # best = best.loc[best.ds_name == ds_name].loc[best.emb_net == model_name].loc[best.fair_net == fair_net]
    # alpha = best.alpha.item()
    # beta = best.beta.item()
    # ew_alpha = best.ew_alpha.item()
    priv_net = priv_models[(model_name, fair_net)]
    model = priv_net(data=dataset, base_net=base_net, priv_lambda=priv_lambda, debug_mode=True, gpu=gpu, **priv_par)
    skip_if_emb_done(model.get_embeddings_name())
    model.train(epochs=epochs, early_stopping=early_stopping)
    model.save_embeddings()
    model.save_predictions()
