#!/bin/bash
# $1 is the sweep id
CUDA_VISIBLE_DEVICES=1 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=2 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=3 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=4 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=5 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=6 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=7 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=8 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=9 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=10 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=11 wandb agent surma/gnn-fair-gcn/"$1" &
CUDA_VISIBLE_DEVICES=12 wandb agent surma/gnn-fair-gcn/"$1" &
