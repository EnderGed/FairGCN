# script for hyperparameter tuning with wandb sweeps for Gat

import argparse
from RWHelp import DglReaderN
from ModelGCN import GatN

parser = argparse.ArgumentParser()
parser.add_argument("--hid_feats", type=int)
parser.add_argument("--num_heads_0", type=int)
parser.add_argument("--num_heads_1", type=int)
parser.add_argument("--feat_drop_0", type=float)
parser.add_argument("--feat_drop_1", type=float)
parser.add_argument("--att_drop_0", type=float)
parser.add_argument("--att_drop_1", type=float)
parser.add_argument("--negative_slope", type=float)
parser.add_argument("--lr", type=float)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--ds_name", type=str)
args = parser.parse_args()

if args.ds_name == 'nba':
    name = 'salary1cou'
else:
    name = 'work1reg'

model = GatN(DglReaderN(split_name=name, dataset_name=args.ds_name).read(), hid_feats=args.hid_feats,
             num_heads=(args.num_heads_0, args.num_heads_1), feat_drop=(args.feat_drop_0, args.feat_drop_1),
             att_drop=(args.att_drop_0, args.att_drop_1), negative_slope=args.negative_slope, lr=args.lr,
             weight_decay=args.weight_decay, gpu='cuda:0',
             run_name='{}_{}_{}'.format(args.hid_feats, args.num_heads_0, args.num_heads_1))
model.train(500)
