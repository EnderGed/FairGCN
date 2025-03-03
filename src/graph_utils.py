#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:04:30 2018

@author: enderged
"""

import json
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import pandas as pd
import random
import scipy.io as sio

from numpy.random import choice

from consts import DATA_PATH, INTER_PATH
    
def split_graph(path, percentage):
    """
    Reads graph from `path`.
    Splits edges of a graph into `percentage` of the graph
    and 100 - `percentage` of the graph.
    """
    nxgraph = read_nxgraph(path + '.edgelist')
    edges = list(nxgraph.edges())
    random.shuffle(edges)
    split_point = int(percentage/100.*len(edges))
    write_nxgraph(
            nx.from_edgelist(edges[:split_point]),
            path + str(percentage) + '.edgelist')
    write_nxgraph(
            nx.from_edgelist(edges[split_point:]),
            path + str(100 - percentage) + '.edgelist')
    
def split_graph_in_5(path, graph_file):
    """
    Reads graph from `path`.
    Splits edges of a graph into 20 / 80 5 different ways
    and saves each pair in different folder.
    Also creates bash script to run random walks for each split.
    """
    graph = read_nxgraph('{}/{}'.format(path, graph_file))
    edges = list(graph.edges())
    random.shuffle(edges)
    size = len(edges)
    for i in range(5):
        new_path = '{}/{}'.format(path, str(i))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        range_begin = int(size / 5. * i)
        range_end = int(size / 5. * (i + 1))
        graph20 = nx.from_edgelist(edges[range_begin: range_end])
        graph80 = nx.from_edgelist(edges[:range_begin] + edges[range_end:])
        write_nxgraph(graph20, new_path + '/known_20.edgelist')
        write_nxgraph(graph80, new_path + '/known_80.edgelist')
    with open('{}/walks.sh'.format(path), 'w') as f:
        for i in range(5):
            f.write('~/random_walk/walk --if=./{}/known_80.edgelist --of=./{}/known_80.walk --length=80 --walks=20\n'.format(i, i))            
        

def sample_false_edges(nxgraph, n, allowed=None):
    """
    For each node in `nxgraph` samples `n` edges,
    that don't exist and are not in `forbidden`.
    Returns array of arrays. Ret[0] are `n` non-conected to node 0 nodes.
    """
    if not allowed:
        allowed = list(nxgraph.nodes())
    def sample_for_node(u):
        """
        sample `n` non-connected for user `u`
        """
        friends = [nxgraph.neighbors(u)] + [u]
        sampled = []
        while len(sampled) < n:
            ran = random.choice(allowed)
            if ran not in friends and ran not in sampled:
                sampled.append(ran)
        return sampled
    return dict([(u, list(sample_for_node(u))) for u in allowed])


def sample_false_edges_5_sets(path, name='known_80', n=50, allowed=None):
    """
    For each file `path`/{0,1,2,3,4}/`name`.edgelist,
    run sample_false_edges with `n` and `allowed`.
    Save the results with pickle in:
        `path`/{0,1,2,3,4}/`name`_rand`n`.pick
    """
    for i in range(5):
        nxgraph = read_nxgraph('{}/{}/{}.edgelist'.format(path, i, name))
        sampled = sample_false_edges(nxgraph, n, allowed)
        write_pickle(sampled, '{}/{}/{}_rand{}.pick'.format(path, i, name, n))
        
        
def sample_false_edges_equal_amount(nxgraph, nxgraph_equal):
    """
    For each node in `nxgraph` sample as many non friends
    as the amount of his friends in 'nxgraph_equal'.
    only those that are not connected in either of those graphs will be sampled.
    """
    def sample_for_node(u):
        friends = set(list(nxgraph.neighbors(u)) + [u] + list(nxgraph_equal.neighbors(u)))
        sampled = []
        while len(sampled) < len(nxgraph_equal.neighbors(u)):
            ran = random.choice(nxgraph.nodes())
            if ran not in friends and ran not in sampled:
                sampled.append(ran)
        return sampled
    return dict([(u, sample_for_node(u)) for u in nxgraph.nodes() if u in nxgraph_equal.nodes()])
    

def sample_false_edges_equal_amount_5_sets(path, name='known_80', name_equal='known_20'):
    """
    For each file `path`/{0,1,2,3,4}/`name`.edgelist,
    run sample_false_edges_equal_amount.
    Save the results with pickle in:
        `path`/{0,1,2,3,4}/`name`_rand.pick
    """
    for i in range(5):
        nxgraph = read_nxgraph('{}/{}/{}.edgelist'.format(path, i, name))
        nxgraph_equal = read_nxgraph('{}/{}/{}.edgelist'.format(path, i, name_equal))
        sampled = sample_false_edges_equal_amount(nxgraph, nxgraph_equal)
        write_pickle(sampled, '{}/{}/{}_rand.pick'.format(path, i, name))


def write_json(anything, out_path):
    """
    Writes array of arrays `neighbors` into `out_path`.
    """
    open(out_path, 'w').write(json.dumps(anything))
    
    
def read_json(in_path):
    """
    Reads array of arrays from `in_path` and returns it.
    """
    return json.loads(open(in_path, 'r').read())


def write_nxgraph(nxgraph, path):
    """
    Write nxgraph to file.
    """
    with open(path, 'w') as out:
        for u, v in nxgraph.edges():
            out.write('{},{}\n'.format(u,v))
            out.write('{},{}\n'.format(v,u))
    # nx.write_edgelist(nxgraph, path + '.edgelist', delimiter = ',', data = False)
    
def read_nxgraph(path):
    """
    Read nxgraph from file.
    """
    return nx.read_edgelist(path, delimiter = ',', nodetype = int)


def write_pickle(obj, path):
    """
    Pickle an object and dump it.
    """
    pkl.dump(obj, open(path, 'wb'))
    
    
def read_pickle(path):
    """
    Load and unpickle.
    """
    return pkl.load(open(path, 'rb'))
