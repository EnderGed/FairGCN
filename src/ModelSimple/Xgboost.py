import wandb
import xgboost as xgb
from ModelSimple import SimpleBase
from consts import PROJECT_NAME, MODELS_PATH


class Xgboost(SimpleBase):
    """
    XBoost model for label prediction using node features and ignoring graph edges.
    """

    bst = None

    def __init__(self, data, max_depth=6, subsample=1, lr=1., gpu=None):
        super().__init__(data)
        run_name = 'X{}_{}_{}_{}'.format(self.data.split_name, max_depth, subsample, lr)
        self.model_name = '{}_{}'.format(run_name, max_depth)
        wandb.init(project=PROJECT_NAME, name=run_name, config={
            "max_depth": max_depth,
            "lr": lr,
            "subsample": subsample,
        }, tags=['xgboost', 'node', self.data.split_name], reinit=True)
        self.param = {
            'max_depth': max_depth,
            'eta': lr,
            'objective': 'multi:softmax',
            'num_class': max(2, self.data.num_classes),
            'subsample': subsample,
        }
        if gpu is not None:
            self.param['tree_method'] = 'gpu_hist'
            self.param['gpu_id'] = gpu
        self.accLogger = AccLogger()

    def train(self):
        train_x = self.data.ndata['feat'][self.data.ndata['train_mask']].cpu().detach().numpy()
        train_y = self.data.ndata['label'][self.data.ndata['train_mask']].cpu().detach().numpy()
        test_x = self.data.ndata['feat'][self.data.ndata['test_mask']].cpu().detach().numpy()
        test_y = self.data.ndata['label'][self.data.ndata['test_mask']].cpu().detach().numpy()
        val_x = self.data.ndata['feat'][self.data.ndata['val_mask']].cpu().detach().numpy()
        val_y = self.data.ndata['label'][self.data.ndata['val_mask']].cpu().detach().numpy()
        dtrain = xgb.DMatrix(train_x, train_y)
        dtest = xgb.DMatrix(test_x, test_y)
        dval = xgb.DMatrix(val_x, val_y)
        self.bst = xgb.train(self.param, dtrain, 30, evals=[(dtrain, 'train'), (dtest, 'test'), (dval, 'val')],
                             callbacks=[self.accLogger.callback_log])
        wandb.join()

    def save_model(self, out_path=MODELS_PATH):
        self.bst.save_model('{}{}.model'.format(out_path, self.model_name))

    def load_model(self, in_path=MODELS_PATH):
        bst = xgb.Booster()
        bst.load_model('{}{}.model'.format(in_path, self.model_name))


class AccLogger:
    def __init__(self):
        self.best_val_acc = 0
        self.test_acc = 0

    def callback_log(self, epoch, evals_log=''):
        # log = dict(epoch.evaluation_result_list)
        acc_log = dict((key[:-7] + '_acc', 1 - val) for (key, val) in epoch.evaluation_result_list)
        if acc_log['val_acc'] > self.best_val_acc:
            self.best_val_acc = acc_log['val_acc']
            self.test_acc = acc_log['test_acc']
        acc_log['test_acc'] = self.test_acc
        wandb.log(acc_log)
        return False
