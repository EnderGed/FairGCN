#!/bin/bash
if [ "$1" != "" ]; then
  GPUS=$1
else
  GPUS="2,3,4,5,6,7,8,9,10,11,12,13,14,15"
fi


#nice -n 10 pytest research_tests/ -v -n 6  -m test         --gpus=$GPUS  --html=../data/results/test.html       --self-contained-html
#nice -n 10 pytest research_tests/ -v -n 1   -m setup        --gpus=$GPUS
#nice -n 10 pytest research_tests/ -v -n 6  -m inter         --gpus=$GPUS  --html=../data/results/1-inter.html       --self-contained-html
#nice -n 10 pytest research_tests/ -v -n 13 -m dgl           --gpus=$GPUS  --html=../data/results/2-dgl.html         --self-contained-html
#nice -n 10 pytest research_tests/ -v -n 14  -m gcns         --gpus=$GPUS  --html=../data/results/3-gcns.html        --self-contained-html  --full-trace
#nice -n 10 pytest research_tests/ -v -n 4  -m model_simple  --gpus=$GPUS  --html=../data/results/4-simple.html      --self-contained-html
#nice -n 10 pytest research_tests/ -v -n 6  -m estimate      --gpus=$GPUS  --html=../data/results/5-estimate.html   --self-contained-html
nice -n 10 pytest research_tests/ -v -n 14  -m fair         --gpus=$GPUS  --html=../data/results/6-fair.html        --self-contained-html
nice -n 10 pytest research_tests/ -v -n 42  -m attack      --gpus=$GPUS  --html=../data/results/7-attack.html     --self-contained-html
nice -n 10 pytest research_tests/ -v -n 1   -m cleanup      --gpus=$GPUS  --html=../data/results/10a-cleanup.html     --self-contained-html
#nice -n 10 pytest research_tests/ -v -n 7  -m censoring    --gpus=$GPUS  --html=../data/results/8-cens.html        --self-contained-html
#nice -n 10 pytest research_tests/ -v -n 8  -m attack_priv  --gpus=$GPUS  --html=../data/results/9-attack_priv.html     --self-contained-html
nice -n 10 pytest research_tests/ -v -n 14  -m fair2         --gpus=$GPUS  --html=../data/results/6b-fair.html        --self-contained-html
nice -n 10 pytest research_tests/ -v -n 42  -m attack      --gpus=$GPUS  --html=../data/results/7b-attack.html     --self-contained-html
nice -n 10 pytest research_tests/ -v -n 1   -m cleanup      --gpus=$GPUS  --html=../data/results/10b-cleanup.html     --self-contained-html
#nice -n 10 pytest research_tests/ -v -n 10  -m ind_gnn      --gpus=$GPUS  --html=../data/results/21-ind_gnn.html     --self-contained-html
#nice -n 10 pytest research_tests/ -v -n 30  -m ind_attack      --gpus=$GPUS  --html=../data/results/27-ind_attack.html     --self-contained-html
#nice -n 10 pytest research_tests/ -v -n 1   -m cleanup      --gpus=$GPUS  --html=../data/results/29-cleanup.html     --self-contained-html
