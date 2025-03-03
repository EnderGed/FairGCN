# script for hyperparameter tuning with wandb sweeps for Kipf, Sage and Gin

import argparse

from ModelGCN import KipfN, SageN, GinN
from RWHelp import DglReaderN

parser = argparse.ArgumentParser()
parser.add_argument("--hid_feats", type=int)
parser.add_argument("--dropout_0", type=float)
parser.add_argument("--dropout_1", type=float)
parser.add_argument("--lr", type=float)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--ds_name", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--hid_layers", type=int)
args = parser.parse_args()

if args.ds_name == 'nba':
    name = 'salary1cou'
else:
    name = 'work1reg'
data = DglReaderN(split_name=name, dataset_name=args.ds_name).read()
if args.model_name == 'Kipf':
    model = KipfN(data=data, hid_feats=args.hid_feats, dropout=(args.dropout_0, args.dropout_1),
                  hid_layers=args.hid_layers, lr=args.lr, weight_decay=args.weight_decay, gpu='cuda:0',
                  run_name='{}_{}_{}'.format(args.hid_feats, args.dropout_0, args.dropout_1))
elif args.model_name == 'Sage':
    model = SageN(data=data, hid_feats=args.hid_feats, dropout=(args.dropout_0, args.dropout_1),
                  lr=args.lr, weight_decay=args.weight_decay, gpu='cuda:0',
                  run_name='{}_{}_{}'.format(args.hid_feats, args.dropout_0, args.dropout_1))
elif args.model_name == 'Gin':
    model = GinN(data=data, hid_feats=args.hid_feats, dropout=(args.dropout_0, args.dropout_1),
                  lr=args.lr, weight_decay=args.weight_decay, gpu='cuda:0',
                  run_name='{}_{}_{}'.format(args.hid_feats, args.dropout_0, args.dropout_1))
else:
    raise Exception("Unknown model")


model.train(500)
