# ###################### Privacy, fairness, membership inference attacks ####################### #
import itertools
import multiprocessing
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch

from Datasets import DatasetN
from ModelSimple import SimpleMLP
from RWHelp import DglReaderN
from consts import SEED, CV_NUM
from research_tests.commons import dirs, ad_params, fil_params, ew_params, fl_params, lambdas, fairgnn_opt_params, \
    deleted_embeds, deleted_embeds_file
from scripts import combine_leak_performance_data, average_leak_data, choose_best_utility_fairness, \
    get_network_file_name


res_write_lock = multiprocessing.Lock()
acc_file = '{}accuracy.csv'.format(dirs['res'])
if not os.path.isfile(acc_file):
    with open(acc_file, 'w') as f:
        f.write('date,ds_name,split_name,emb_net,hyper_name,fair_net,alpha,beta,ew_alpha,priv_lambda,attack,attr,bins,layers,cv,measure,value\n')
accs = pd.read_csv(acc_file)


def skip_attack_if_done(ds_name, split_name, emb_net, hyper_name, fair_net, alpha, beta, ew_alpha, priv_lambda, attack,
                        attr, bins, layers, cv, measure):
    """
    Check if the attack has already been performed and if so, skip calculating it again.

    :param ds_name: dataset name, for now: 'nba', 'pokec_z', 'pokec_n'
    :param split_name: prediction task name, for now: 'salary1cou' for 'nba' and 'work1reg' for 'pokec_*'
    :param emb_net: name of the network architecture, for now: 'Kipf', 'Sage', 'Git', 'Gat'
    :param hyper_name: name of the network hyperparameter set, for now: will only be 'opt'
    :param fair_net: short name of the fairness mechanism, for now: '', 'Ad', 'Ep', 'Ew', 'Fil', 'Flpar', 'Fleoo'
    :param alpha: first fairness parameter, 0 if this mechanism doesn't have parameters
    :param beta: second fairness parameter, 0 if this mechanism has only one or no parameters
    :param ew_alpha: edge weightening alpha for ew combined with others
    :param priv_lambda: lambda parameter for privacy censoring
    :param attack: name of the attack being evaluated, for now: 'priv', 'fair', 'mia'
    :param attr: on which attribute is the attack evaluated
    :param bins: in how many bins is the `attr` binned, 0 for no binning
    :param layers: how many layers does the attacking network has
    :param cv: cross validation number, for now: 0, 1, 2, 3, 4
    :param measure: what do we measure, for now: 'test_acc'
    :return:
    """
    if len(accs.loc[(accs.ds_name == ds_name) & (accs.split_name == split_name) &
                    (accs.emb_net == emb_net if emb_net else accs.emb_net.isna()) &
                    (accs.hyper_name == hyper_name) & (accs.fair_net == fair_net) & (accs.alpha == alpha) &
                    (accs.beta == beta) & (accs.ew_alpha == ew_alpha) & (accs.priv_lambda == priv_lambda) &
                    (accs.attack == attack) & (accs.attr == attr) & (accs.bins == bins) & (accs.layers == layers) &
                    (accs.cv == cv) & (accs.measure == measure), :]) > 0:
        pytest.skip("Already calculated.")


def save_attack_res(date_stamp, ds_name, split_name, emb_net, hyper_name, fair_net, alpha, beta, ew_alpha, priv_lambda,
                    attack, attr, bins, layers, cv, measure, value):
    """
    Record an attack in the database. Right now - a simple padas.DataFrame

    :param date_stamp: date and hour of performing the attack
    :param ds_name: dataset name, for now: 'nba', 'pokec_z', 'pokec_n'
    :param split_name: prediction task name, for now: 'salary1cou' for 'nba' and 'work1reg' for 'pokec_*'
    :param emb_net: name of the network architecture, for now: 'Kipf', 'Sage', 'Git', 'Gat'
    :param hyper_name: name of the network hyperparameter set, for now: will only be 'opt'
    :param fair_net: short name of the fairness mechanism, for now: '', 'Ad', 'Ep', 'Ew', 'Fil', 'Flpar', 'Fleoo'
    :param alpha: first fairness parameter, 0 if this mechanism doesn't have parameters
    :param beta: second fairness parameter, 0 if this mechanism has only one or no parameters
    :param ew_alpha: edge weightening alpha for ew combined with others
    :param priv_lambda: lambda parameter for privacy censoring
    :param attack: name of the attack being evaluated, for now: 'priv', 'fair', 'mia'
    :param attr: on which attribute is the attack evaluated
    :param bins: in how many bins is the `attr` binned, 0 for no binning
    :param layers: how many layers does the attacking network has
    :param cv: cross validation number, for now: 0, 1, 2, 3, 4
    :param measure: what do we measure, for now: 'test_acc'
    :param value: value of the attack measurement
    :return:
    """
    with res_write_lock:
        with open(acc_file, 'a+') as f:
            f.write(','.join((str(i) for i in (date_stamp, ds_name, split_name, emb_net, hyper_name, fair_net, alpha,
                                               beta, ew_alpha, priv_lambda, attack, attr, bins, layers, cv, measure,
                                               value))))
            f.write('\n')


base_nets = ['Kipf', 'Sage', 'Gat']
emb_args = 'model_name,fair_net,alpha,beta,ew_alpha,priv_lambda'
emb_params = list(itertools.chain(
    itertools.product(base_nets, ['', 'Ep'], [0], [0], [0], [0]),
    itertools.product(base_nets, ['Ad'], ad_params['alphas'], ad_params['betas'], [0], [0]),
    itertools.product(base_nets, ['Fil'], fil_params['gammas'], fil_params['d_stepss'], [0], [0]),
    itertools.product(base_nets, ['Ew'], ew_params['alphas'], [0], [0], [0]),
    itertools.product(base_nets, ['Flpar'], fl_params['alphas'], [0], [0], [0]),  # , 'Fleoo']
    # itertools.product(base_nets[:2], ['EwAd'], ad_params['alphas'], ad_params['betas'], ew_params['ew_alphas'], [0]),
    # itertools.product(base_nets[:2], ['EwFlpar'], fl_params['alphas'], [0], ew_params['ew_alphas'], [0]),
))
attr_bins_dict = {
    ('nba', 'priv'): ('AGE', 4),
    ('nba', 'fair'): ('country', 0),
    ('pokec_n', 'priv'): ('gender', 0),
    ('pokec_n', 'fair'): ('region', 0),
    ('pokec_z', 'priv'): ('gender', 0),
    ('pokec_z', 'fair'): ('region', 0),
}


@pytest.mark.parametrize('attack', ['priv', 'fair'])
@pytest.mark.parametrize('layers', [2])
@pytest.mark.parametrize('hyper_name', ['opt', '64Noval', '64NovalInd'])
@pytest.mark.parametrize(emb_args, emb_params)
@pytest.mark.parametrize('dropout', [0.25])
@pytest.mark.attack
def test_attr_inference(split_ds_names, hyper_name, model_name, fair_net, alpha, beta, ew_alpha, priv_lambda,
                        attack, layers, cv, epochs, dropout, gpu):
    split_name, ds_name = split_ds_names
    attr, bins = attr_bins_dict[(ds_name, attack)]
    skip_attack_if_done(ds_name, split_name, model_name, hyper_name, fair_net, alpha, beta, ew_alpha, priv_lambda,
                        'mino_' + attack, attr, bins, layers, cv, 'test_acc')
    gpu = 'cuda:' + gpu
    emb_file_name = get_network_file_name(dirs['emb'], fair_net, model_name, ds_name, cv, split_name, hyper_name,
                                          alpha, beta, ew_alpha, priv_lambda)
    try:
        features = pd.read_feather(emb_file_name)   # embeddings are features now
    except FileNotFoundError:
        pytest.xfail("Embedding file {} doesn't exist yet")
    inter = DatasetN(ds_name)                   # get label from old features
    label = inter.discretize_feature(attr, bins)
    dgl = DglReaderN(split_name, ds_name, cv=cv).read()      # get splits from the dgl
    mlp = SimpleMLP(features, label, max(bins, 1), dgl.ndata['train_mask'], dgl.ndata['val_mask'],
                    dgl.ndata['test_mask'], layers, dropout, gpu)
    test_acc = mlp.train(epochs)
    save_attack_res(date.today(), ds_name, split_name, model_name, hyper_name, fair_net, alpha, beta, ew_alpha,
                    priv_lambda, attack, attr, bins, layers, cv, 'test_acc', test_acc)
    # calculate accuracy for minority and majority fairness groups
    min_mask = dgl.ndata['test_mask'].logical_and(dgl.ndata['sens'])
    maj_mask = dgl.ndata['test_mask'].logical_and(dgl.ndata['sens'].logical_not())
    for mask, mask_name in [(min_mask, 'mino_'), (maj_mask, 'majo_')]:
        preds = mlp.model(mlp.data['feat'][mask])
        labels = mlp.data['label'][mask]
        if bins <= 1:
            preds = preds.squeeze()
            labels = labels.squeeze()
            pred = (preds > 0).type_as(labels)
            correct = torch.sum(pred.eq(labels))
        else:
            _, indices = torch.max(preds, dim=1)
            correct = torch.sum(indices.eq(labels))
        accuracy = correct.item() * 1.0 / len(labels)
        save_attack_res(date.today(), ds_name, split_name, model_name, hyper_name, fair_net, alpha, beta, ew_alpha,
                        priv_lambda, mask_name+attack, attr, bins, layers, cv, 'test_acc', accuracy)


@pytest.mark.parametrize('hyper_name', ['opt'])
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage'])
@pytest.mark.parametrize('fair_net', ['base', 'Ad', 'Flpar', 'Ew', 'EwAd', 'EwFlpar'])
@pytest.mark.parametrize('attack', ['priv', 'fair'])
@pytest.mark.parametrize('layers', [2])
@pytest.mark.parametrize('dropout', [0.25])
@pytest.mark.attack_skip
def test_opt_attr_inference(split_ds_names, hyper_name, model_name, fair_net, attack, layers, cv, epochs, dropout, gpu):
    fair_par = fairgnn_opt_params[(split_ds_names[1], model_name, fair_net)]
    fair_net = '' if fair_net == 'base' else fair_net
    test_attr_inference(split_ds_names=split_ds_names, hyper_name=hyper_name, model_name=model_name, fair_net=fair_net,
                        priv_lambda=0, attack=attack, layers=layers, cv=cv, epochs=epochs, dropout=dropout,
                        gpu=gpu, **fair_par)


@pytest.mark.parametrize('hyper_name', ['opt'])
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage'])
@pytest.mark.parametrize('fair_net', ['base', 'Ad', 'Flpar', 'Ew', 'EwAd', 'EwFlpar'])
@pytest.mark.parametrize('priv_lambda', lambdas)
@pytest.mark.parametrize('attack', ['priv', 'fair'])
@pytest.mark.parametrize('layers', [2])
@pytest.mark.parametrize('dropout', [0.25])
@pytest.mark.attack_priv
def test_priv_attr_inference(split_ds_names, hyper_name, model_name, fair_net, priv_lambda, attack,
                        layers, cv, epochs, dropout, gpu):
    fair_par = fairgnn_opt_params[(split_ds_names[1], model_name, fair_net)]
    fair_net = 'Cens' + (fair_net if fair_net != 'base' else '')
    test_attr_inference(split_ds_names=split_ds_names, hyper_name=hyper_name, model_name=model_name, fair_net=fair_net,
                        priv_lambda=priv_lambda, attack=attack, layers=layers, cv=cv, epochs=epochs, dropout=dropout,
                        gpu=gpu, **fair_par)

# We're turning off this experiment, because it didn't show any significant difference as compared to the standard split
# @pytest.mark.parametrize('name,ds_name,attr,bins,layers', [
#     # these are for privacy leak
#     ('work1reg', 'pokec_n', 'gender', 0, 2),
#     ('work1reg', 'pokec_z', 'gender', 0, 2),
#     ('salary1cou', 'nba', 'AGE', 4, 2),
# ])
# @pytest.mark.parametrize('hyper_name', ['opt'])
# @pytest.mark.parametrize(emb_args, emb_params)
# @pytest.mark.parametrize('epochs', [300])
# @pytest.mark.parametrize('cv', range(cv_num))
# @pytest.mark.parametrize('dropout', [0.25])
# @pytest.mark.parametrize('split_used', ['train', 'test'])
# @pytest.mark.attrinf
# def test_attr_inference_dif_splits(name, ds_name, hyper_name, alpha, beta, emb_net, attr, bins, layers, cv, epochs,
#                                    dropout, split_used, gpu):
#     if emb_net.startswith('Ad'):
#         emb_name = '{}_{}_{}'.format(hyper_name, alpha, beta)
#     else:
#         emb_name = hyper_name
#     skip_attack_if_done(ds_name, name, emb_name, emb_net, attr, bins, layers, cv, '{}_priv_leak'.format(split_used))
#     gpu = 'cuda:' + gpu
#     emb_file_name = '{}{}/{}/{}/{}_{}.ftr'.format(dirs['emb'], emb_net, ds_name, cv, name, emb_name)
#     features = pd.read_feather(emb_file_name)   # embeddings are features now
#     inter = DatasetN(ds_name)                   # get label from old features
#     label = inter.discretize_feature(attr, bins)
#     dgl = DglReaderN(name, ds_name, cv=cv).read()      # get splits from the dgl
#     # redo the splits based on the `split_used`
#     o_mask = dgl.ndata['{}_mask'.format(split_used)]
#     np.random.seed(SEED + 20)
#     attack_random = torch.tensor(np.random.rand(len(label)) < 0.8)
#     attack_train = o_mask.bitwise_and(attack_random)
#     attack_test = o_mask.bitwise_and(attack_random.bitwise_not())
#     # we pass 'val_mask', to avoid a crash, but evaluation on it is not used because early_stopping is False
#     mlp = SimpleMLP(features, label, max(bins, 1), attack_train, dgl.ndata['val_mask'], attack_test, layers, dropout, gpu)
#     test_acc = mlp.train(epochs, early_stopping=False)
#     save_attack_res(date.today(), ds_name, name, emb_name, emb_net, attr, bins, layers, cv,
#                     '{}_priv_leak'.format(split_used), test_acc)


@pytest.mark.parametrize('hyper_name', ['opt', '64Noval', '64NovalInd'])
@pytest.mark.parametrize(emb_args, emb_params)
@pytest.mark.parametrize('layers', [2])
@pytest.mark.parametrize('dropout', [0.25])
@pytest.mark.attack
def test_mia(split_ds_names, hyper_name, model_name, fair_net, alpha, beta, ew_alpha, priv_lambda, layers, cv, epochs,
             dropout, gpu):
    split_name, ds_name = split_ds_names
    attack, attr, bins = 'mia', 'None', 0
    skip_attack_if_done(ds_name, split_name, model_name, hyper_name, fair_net, alpha, beta, ew_alpha, priv_lambda,
                        attack, attr, bins, layers, cv, 'test_acc')
    gpu = 'cuda:' + gpu
    emb_file_name = get_network_file_name(dirs['emb'], fair_net, model_name, ds_name, cv, split_name, hyper_name,
                                          alpha, beta, ew_alpha, priv_lambda)
    try:
        features = pd.read_feather(emb_file_name)   # embeddings are features now
    except FileNotFoundError:
        pytest.xfail("Embedding file {} doesn't exist yet")
    dgl = DglReaderN(split_name, ds_name, cv=cv).read()   # get original splits from the dgl
    o_train = dgl.ndata['train_mask']
    o_test = dgl.ndata['test_mask']
    label = torch.zeros(dgl.ndata['label'].shape)
    label[o_train] = torch.ones(sum(o_train))
    np.random.seed(SEED + 20)
    mia_random = torch.tensor(np.random.rand(len(label)) < 0.8)
    mia_train = o_train.bitwise_or(o_test).bitwise_and(mia_random)
    mia_test = o_train.bitwise_or(o_test).bitwise_and(mia_random.bitwise_not())
    # we pass 'val_mask', to avoid a crash, but evaluation on it is not used because early_stopping is False
    mlp = SimpleMLP(features, label, 1, mia_train, dgl.ndata['val_mask'], mia_test, layers, dropout, gpu)
    mia_acc = mlp.train(epochs, early_stopping=False)
    save_attack_res(date.today(), ds_name, split_name, model_name, hyper_name, fair_net, alpha, beta, ew_alpha,
                    priv_lambda, attack, attr, bins, layers, cv, 'test_acc', mia_acc)
    # calculate accuracy for minority and majority fairness groups
    min_mask = mia_test.logical_and(dgl.ndata['sens'])
    maj_mask = mia_test.logical_and(dgl.ndata['sens'].logical_not())
    for mask, mask_name in [(min_mask, 'mino_'), (maj_mask, 'majo_')]:
        preds = mlp.model(mlp.data['feat'][mask])
        labels = mlp.data['label'][mask]
        if bins <= 1:
            preds = preds.squeeze()
            labels = labels.squeeze()
            pred = (preds > 0).type_as(labels)
            correct = torch.sum(pred.eq(labels))
        else:
            _, indices = torch.max(preds, dim=1)
            correct = torch.sum(indices.eq(labels))
        accuracy = correct.item() * 1.0 / len(labels)
        save_attack_res(date.today(), ds_name, split_name, model_name, hyper_name, fair_net, alpha, beta, ew_alpha,
                        priv_lambda, mask_name+attack, attr, bins, layers, cv, 'test_acc', accuracy)


@pytest.mark.parametrize('hyper_name', ['opt'])
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage'])
@pytest.mark.parametrize('fair_net', ['base', 'Ad', 'Flpar', 'Ew', 'EwAd', 'EwFlpar'])
@pytest.mark.parametrize('layers', [2])
@pytest.mark.parametrize('dropout', [0.25])
@pytest.mark.attack_skip
def test_opt_mia(split_ds_names, hyper_name, model_name, fair_net, layers, cv, epochs, dropout, gpu):
    fair_par = fairgnn_opt_params[(split_ds_names[1], model_name, fair_net)]
    fair_net = '' if fair_net == 'base' else fair_net
    test_mia(split_ds_names=split_ds_names, hyper_name=hyper_name, model_name=model_name, fair_net=fair_net,
             priv_lambda=0, layers=layers, cv=cv, epochs=epochs, dropout=dropout, gpu=gpu, **fair_par)


@pytest.mark.parametrize('hyper_name', ['opt'])
@pytest.mark.parametrize('model_name', ['Kipf', 'Sage'])
@pytest.mark.parametrize('fair_net', ['base', 'Ad', 'Flpar', 'Ew', 'EwAd', 'EwFlpar'])
@pytest.mark.parametrize('priv_lambda', lambdas)
@pytest.mark.parametrize('layers', [2])
@pytest.mark.parametrize('dropout', [0.25])
@pytest.mark.attack_priv
def test_priv_mia(split_ds_names, hyper_name, model_name, fair_net, priv_lambda, layers, cv, epochs, dropout, gpu):
    fair_par = fairgnn_opt_params[(split_ds_names[1], model_name, fair_net)]
    fair_net = 'Cens' + (fair_net if fair_net != 'base' else '')
    test_mia(split_ds_names=split_ds_names, hyper_name=hyper_name, model_name=model_name, fair_net=fair_net,
             priv_lambda=priv_lambda, layers=layers, cv=cv, epochs=epochs, dropout=dropout, gpu=gpu, **fair_par)


def remove_embeddings():
    num_deleted = 0
    comb = pd.read_feather(dirs['res'] + 'combined.ftr')
    comb = comb.loc[comb.num_vals == CV_NUM].loc[comb.fair_net != 'base']
    for params in comb.iterrows():
        params = params[1]
        if params['priv_leak'] == 0 or params['fair_leak'] == 0 or params['mia_acc'] == 0:
            continue
        # only Ep has no alpha parameter, so the default alpha should be 0
        params['alpha'] = 0 if params['fair_net'] == 'Ep' else params['alpha']
        params['beta'] = 0 if params['beta'] == 0 else params['beta']
        # all Ew networks have ew_alpha and it can be 0.0
        params['ew_alpha'] = 0 if params['ew_alpha'] == 0 and 'Ew' not in params['fair_net'] else params['ew_alpha']
        params['priv_lambda'] = 0 if params['priv_lambda'] == 0 else params['priv_lambda']
        if 'Fil' in params['fair_net']:
            params['beta'] = int(params['beta'])
        for cv in range(CV_NUM):
            path = get_network_file_name(dirs['emb'], params['fair_net'], params['emb_net'], params['ds_name'], cv,
                                         params['split_name'], params['hyper_name'], params['alpha'], params['beta'],
                                         params['ew_alpha'], params['priv_lambda'])
            if os.path.isfile(path):
                os.remove(path)
                num_deleted += 1
                if path not in deleted_embeds:
                    deleted_embeds.append(path)
    with open(deleted_embeds_file, 'wb') as f:
        pickle.dump(deleted_embeds, f)
    print('Deleted {} embedding files.'.format(num_deleted))


@pytest.mark.cleanup
def test_aggregate_priv_results():
    combine_leak_performance_data()
    average_leak_data()
    # choose_best_utility_fairness()
    remove_embeddings()
