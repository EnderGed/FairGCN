# Fairness and / or Privacy on Social Graphs

You'll need NBA and pokec datasets: https://github.com/EnyanDai/FairGNN/tree/main/dataset
Although not used in the paper, you can also do Edge prediction, for which you'd need Movielens dataset

Project setup:
1. git clone git@github.com:EnderGed/FairGCN.git
1. pyenv install 3.7.5
1. Create a virtualenv:
* pip3 install virtualenv
* virtualenv -p ~/.pyenv/versions/3.7.5/bin/python venv
* source venv/bin/activate
* pip3 install -r requirements.txt
* deactivate

Working on project (in venv):
* source venv/bin/activate
* ... your work ...
* deactivate

Managing pip requirements
* pip3 freeze >requirements.txt # saving new requirements
* pip3 install -r requirements.txt # updating requirements

We use pytest to schedule running research experiments, where each experiment is a "test".
This allows us to manage failures, parametrize, track progress and parallelize over multiple GPUs.
We modify `main.sh` and run it to fire different stages of the experiments, like preprocessing the data, training models,
using embeddings to test fairness and privacy hypotheses.
