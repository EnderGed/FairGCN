import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from RWHelp import DglReaderN
from consts import RESULTS_PATH, PREDICTIONS_PATH
from gen_utils import deprecated
from plot_utils import plot_dataset_measures, plot_alpha_beta_importance, plot_beta_importance_by_train_acc, \
    plot_5_privs_mia_fair, plot_utiliy_fairnes_comp, plot_utiliy_fairnes_comp_3_best, plot_fairness_privacy


# priv_attrs = ['AGE', 'gender', 'player_weight', 'player_height']


def get_network_file_name(path, fair_net, emb_net, ds_name, cv, split_name, hyper_name, alpha, beta, ew_alpha,
                          priv_lambda):
    """
    Get a file name for embedding / model / predictions, based on the arguments.
     It accepts all numerical values as floats (as saved by pandas) and converts those that needs converting to ints.
    """
    if fair_net == 'base':
        fair_net = ''
    if fair_net in ['', 'Ep', 'Cens', 'CensEp']:
        alpha = int(alpha)
    if 'Ad' not in fair_net:
        beta = int(beta)
    if fair_net in ['Ew', 'CensEw'] or 'Ew' not in fair_net:
        ew_alpha = int(ew_alpha)
    if 'Cens' not in fair_net:
        priv_lambda = int(priv_lambda)
    return '{}{}{}/{}/{}/{}_{}_{}_{}_{}_{}.ftr'.format(path, fair_net, emb_net, ds_name, cv, split_name, hyper_name,
                                                       alpha, beta, ew_alpha, priv_lambda)


def generate_fairgnn_opt_params_dict():
    """
    For faster pytest execution, generate a dictionary of optimal FairGnns parameters.
    It requires the `best.ftr` file created by `choose_best_utility_fairness`.
    If `best.ftr` doesn't exist, return an empty dict.
    :return: dict of dicts
    """
    dss = ['nba', 'pokec_n', 'pokec_z']
    nets = ['Kipf', 'Sage']
    fairs = ['base', 'Ad', 'Flpar', 'Ew', 'EwAd', 'EwFlpar']
    opt_params = {}
    try:
        best = pd.read_feather(RESULTS_PATH + 'best.ftr')
    except FileNotFoundError:
        return {}

    for ds in dss:
        for net in nets:
            for fair in fairs:
                single_best = best.loc[best.ds_name == ds].loc[best.emb_net == net].loc[best.fair_net == fair]
                opt_params[(ds, net, fair)] = {par: single_best[par].item() for par in ['alpha', 'beta', 'ew_alpha']}
    return opt_params


def get_additional_measures(ds_name, split_name, emb_net, hyper_name, fair_net, alpha, beta, ew_alpha, priv_lambda, cv):
    """
    Use dgl (for splits) and preds file to measure model performance:
     training accuracy, test accuracy, test parity, test equality of opportunity
    It actually does GCNBaseN.eval()
    :return: (float, float, float, float), or all zeros if record doesn't exist
    """
    cv = int(cv)
    data = DglReaderN(split_name, ds_name, cv).read()
    if 'sens' not in data.ndata:
        raise Exception('This data split has no sensitive attribute data')
    pred_file_name = get_network_file_name(PREDICTIONS_PATH, fair_net, emb_net, ds_name, cv, split_name, hyper_name,
                                           alpha, beta, ew_alpha, priv_lambda)
    try:
        logits = torch.tensor(pd.read_feather(pred_file_name).values, dtype=torch.float)
    except FileNotFoundError:
        print('File {} not found.'.format(pred_file_name))
        return 0., 0., 0., 0.
    metrics = {}
    for split in ['train', 'val', 'test']:
        mask = data.ndata[split + '_mask']
        pred = logits[mask]
        labels = data.ndata['label'][mask]
        if data.num_classes == 1:
            pred = pred.squeeze()
            labels = labels.squeeze()
            pred = (pred > 0).type_as(labels)
            correct = torch.sum(pred == labels)
        else:
            _, indices = torch.max(pred, dim=1)
            correct = torch.sum(indices == labels)
        metrics[split + '_acc'] = correct.item() * 1.0 / len(labels)
        # fairness eval
        sens_attr = data.ndata['sens'][mask].cpu().numpy()
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        # Statistical / Demographic Parity
        sens_0 = sens_attr == 0
        sens_1 = sens_attr == 1
        metrics[split + '_par'] = sum(pred[sens_0]) / sum(sens_0) - sum(pred[sens_1]) / sum(sens_1)
        # Equality of Opportunity
        sens_0_true = np.bitwise_and(sens_0, labels == 1.)
        sens_1_true = np.bitwise_and(sens_1, labels == 1.)
        metrics[split + '_eoo'] = sum(pred[sens_0_true]) / sum(sens_0_true) - \
                                  sum(pred[sens_1_true]) / sum(sens_1_true)
    '''
    # simple MIA doesn't show anything so we skip it now
    # calculate simple Membership Inference
    y_score = torch.sigmoid(logits).numpy()
    # flip score of those where labels are 0
    y_score[data.ndata['label'] == 0] = [1 - i for i in y_score[data.ndata['label'] == 0]]
    y_score_train = y_score[data.ndata['train_mask']]
    y_score_test = y_score[data.ndata['test_mask']]
    ordered_y_score = np.concatenate([y_score_train, y_score_test])
    members = np.concatenate([np.ones(y_score_train.shape), np.zeros(y_score_test.shape)])
    '''
    return metrics['train_acc'], metrics['test_acc'], metrics['test_par'], metrics['test_eoo']


def combine_leak_performance_data():
    """
    Create one file with privacy leak and model performance measurements.
    :return:
    """
    file = '{}accuracy.csv'.format(RESULTS_PATH)
    accs = pd.read_csv(file)

    cols_to_merge_on = ['ds_name', 'split_name', 'emb_net', 'hyper_name', 'fair_net', 'alpha', 'beta', 'ew_alpha',
                        'priv_lambda', 'cv']
    accs.sort_values('date', ascending=False, inplace=True)
    accs.drop_duplicates(subset=cols_to_merge_on + ['attack', 'attr', 'bins', 'layers', 'measure'], keep='first',
                         inplace=True)
    # attacks
    cols_to_drop = ['date', 'measure', 'attr', 'bins', 'layers', 'attack']
    attack_res = []
    for att, att_name in [('mia', 'mia_acc'), ('fair', 'fair_leak'), ('priv', 'priv_leak')]:
        for group in ['', 'mino_', 'majo_']:
            attack_res.append(
                accs.loc[accs.attack == group+att].drop(cols_to_drop, axis=1).rename(columns={'value': group+att_name}))
    # merge them together
    accs = attack_res[0]
    for res in attack_res[1:]:
        accs = accs.merge(res, how='left', on=cols_to_merge_on)
    # name basic GCNs
    accs.fair_net = accs.fair_net.fillna('base')
    # additional measures
    measures = [get_additional_measures(*params[1][['ds_name', 'split_name', 'emb_net', 'hyper_name', 'fair_net',
                                                    'alpha', 'beta', 'ew_alpha', 'priv_lambda', 'cv']])
                for params in accs.iterrows()]
    accs[['train_acc', 'test_acc', 'par', 'eoo']] = measures
    accs = accs.reset_index(drop=True)
    accs.to_feather('{}accs_measures.ftr'.format(RESULTS_PATH))


def average_leak_data():
    accs = pd.read_feather('{}accs_measures.ftr'.format(RESULTS_PATH))
    accs['num_vals'] = np.ones(accs.shape[0])
    measures = ['priv_leak', 'mino_priv_leak', 'majo_priv_leak', 'fair_leak', 'mino_fair_leak', 'majo_fair_leak',
                'train_acc', 'test_acc', 'par', 'eoo', 'mia_acc', 'mino_mia_acc', 'majo_mia_acc']
    for measure in measures:
        accs[measure + '_min'] = accs[measure].copy()
        accs[measure + '_max'] = accs[measure].copy()
    aggregators = {**{val: np.mean for val in measures},
                   **{val + '_min': np.min for val in measures},
                   **{val + '_max': np.max for val in measures}}
    aggregators['num_vals'] = np.sum
    comb = accs.groupby(['ds_name', 'split_name', 'emb_net', 'hyper_name', 'fair_net', 'alpha', 'beta', 'ew_alpha',
                         'priv_lambda']).agg(aggregators).reset_index()
    comb = comb[
        ['ds_name', 'split_name', 'emb_net', 'hyper_name', 'fair_net', 'alpha', 'beta', 'ew_alpha',
         'priv_lambda'] + list(aggregators.keys())]
    # comb['alpha'] = comb['alpha'].astype(float, copy=False)
    # comb['beta'] = comb['beta'].astype(float, copy=False)
    # comb['hyper_name'] = comb['hyper_name'].astype(str, copy=False)
    comb.to_feather('{}combined.ftr'.format(RESULTS_PATH))


def plot_all_alpha_beta_importance():
    for fair_net in ['Ad', 'Ew', 'Ep', 'Fil', 'Fl']:
        for ds_name in ['nba', 'pokec_n', 'pokec_z']:
            for emb_net in ['Kipf', 'Sage', 'Gat']:
                plot_alpha_beta_importance(ds_name=ds_name, emb_net=emb_net, fair_net=fair_net)


def plot_all_utility_fairness_comp(errorbars=False, acc_drop=0.):
    for ds_name in ['nba', 'pokec_n', 'pokec_z']:
        for emb_net in ['Kipf', 'Sage', 'Gat']:
            plot_utiliy_fairnes_comp(ds_name=ds_name, hyper_name='opt', emb_net=emb_net, errorbars=errorbars,
                                     acc_drop=acc_drop)


def plot_chosen_utility_fairness_comp(errorbars=False, acc_drop=0.):
    fair_nets = ['EwAd', 'EwFlpar', 'Ad', 'Flpar', 'Ew', 'base']
    for ds_name in ['nba', 'pokec_n', 'pokec_z']:
        for emb_net in ['Kipf', 'Sage']:
            # plot_utiliy_fairnes_comp(ds_name=ds_name, hyper_name='opt', emb_net=emb_net, errorbars=errorbars,
            #                          acc_drop=acc_drop, fair_nets=fair_nets)
            plot_utiliy_fairnes_comp_3_best(ds_name=ds_name, hyper_name='opt', emb_net=emb_net, errorbars=errorbars,
                                            acc_drop=acc_drop, fair_nets=fair_nets)


def plot_all_best_utility_fairness_comp(errorbars=False, acc_drop=0.):
    # fair_nets = ['Ad', 'Fil', 'Ew', 'Flpar', 'Fleoo', 'base', 'Ep']
    for ds_name in ['nba', 'pokec_n', 'pokec_z']:
        for emb_net in ['Kipf', 'Sage', 'Gat']:
            # plot_utiliy_fairnes_comp(ds_name=ds_name, hyper_name='opt', emb_net=emb_net, errorbars=errorbars,
            #                          acc_drop=acc_drop, fair_nets=fair_nets)
            plot_utiliy_fairnes_comp_3_best(ds_name=ds_name, hyper_name='opt', emb_net=emb_net, errorbars=errorbars,
                                            acc_drop=acc_drop)

def plot_privacy_fairness_comp():
    fair_nets = ['EwAd', 'EwFlpar', 'Ad', 'Flpar', 'Ew', 'base']
    for ds_name in ['nba', 'pokec_n', 'pokec_z']:
        for emb_net in ['Kipf', 'Sage']:
            plot_fairness_privacy(ds_name=ds_name, hyper_name='opt', emb_net=emb_net, fair_nets=fair_nets)
            plot_fairness_privacy(ds_name=ds_name, hyper_name='opt', emb_net=emb_net)


def choose_best_utility_fairness():
    """
    For each dataset, for each GNN, for each Fairness mechanism,
     rank their utility and rank their fairness (par and eoo),
     choose one fairness params that minimizes the sum of squares of ranks.
    :return:
    """
    comb = pd.read_feather('{}combined.ftr'.format(RESULTS_PATH))
    best = comb.loc[comb.emb_net.isin(['Kipf', 'Sage'])].loc[comb.fair_net == 'base']
    # comb['is_best'] = np.zeros(comb.shape[0])
    comb['abs_par'] = comb.par.abs()
    comb['abs_eoo'] = comb.eoo.abs()
    for ds in ['nba', 'pokec_n', 'pokec_z']:
        for gnn in ['Kipf', 'Sage', 'Gat']:
            for fair_net in ['Ad', 'Flpar', 'Ew', 'EwAd', 'EwFlpar']:
                if gnn == 'Gat' and fair_net in ['EwAd', 'EwFlpar']:
                    continue
                tmp = comb.loc[comb.ds_name == ds].loc[comb.emb_net == gnn].loc[comb.fair_net == fair_net]
                tmp = tmp.sort_values('test_acc', ascending=False).reset_index(drop=True)
                tmp['util_rank'] = tmp.index
                tmp = tmp.sort_values('abs_par').reset_index(drop=True)
                tmp['par_rank'] = tmp.index
                tmp = tmp.sort_values('abs_eoo').reset_index(drop=True)
                tmp['eoo_rank'] = tmp.index
                tmp['score_2'] = tmp.apply(
                    lambda row: sum((row[rank] ** 2 for rank in ['util_rank', 'par_rank', 'eoo_rank'])), axis=1)
                # tmp['score'] = tmp.apply(lambda row: sum((row[rank] for rank in ['util_rank', 'par_rank', 'eoo_rank'])), axis=1)
                tmp = tmp.sort_values('score_2').iloc[:, :-6]
                best = best.append(tmp.head(1), ignore_index=True)
    best.to_feather('{}best.ftr'.format(RESULTS_PATH))


def compute_priv_utility_correlation():
    """
    For each dataset, for each GNN, compute correlation of attr_inference to fairness
    :return:
    """
    comb = pd.read_feather('{}combined.ftr'.format(RESULTS_PATH))
    res = {}
    for ds in ['nba', 'pokec_n', 'pokec_z']:
        for gnn in ['Kipf', 'Sage']:
            tmp = comb.loc[comb.ds_name == ds].loc[comb.emb_net == gnn]
            res['{} {}'.format(ds, gnn)] = tmp.priv_leak.corr(tmp.test_acc)
    return res


# ########## obsolete after the refactor ########## #


@deprecated
def plot_leak_performance_data():
    """
    Use a file generated by `combine_leak_performance_data` to plot the results for each dataset
    :return:
    """
    for ds_name in ['nba', 'pokec_n', 'pokec_z']:
        for emb_name in ['128', 'opt']:
            plot_dataset_measures(ds_name, emb_name)


@deprecated
def plot_all_beta_importance():
    for ds_name in ['nba', 'pokec_n', 'pokec_z']:
        plot_beta_importance_by_train_acc(ds_name=ds_name, all_runs=False)
        plot_beta_importance_by_train_acc(ds_name=ds_name, all_runs=True)


@deprecated
def plot_all_priv_mia_attacks():
    for ds_name in ['nba', 'pokec_n', 'pokec_z']:
        plot_5_privs_mia_fair(ds_name=ds_name)
