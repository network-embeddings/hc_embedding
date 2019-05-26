""" Hyperbolic Coaelesccent Embeddings

input: an edgelist

output: a table with the each node's coordinate

"""
import argparse
import math
from itertools import combinations

import plfit
import numpy as np
import networkx as nx
from scipy.stats import linregress
import matplotlib.pyplot as plt


def radial_coord(G, zeta, beta):
    """
        Nodes are first sorted i = 1,2, ... N according to their degree
        (descending).

        Then r_i is calculated by the formula in pg. 3

        r_i = \frac{2}{\zeta} ( \beta ln i + (1 - \beta) ln N )
    """
    sorted_node_ids = [i for i, k in sorted(G.degree, key=lambda x: x[1], reverse=True)]

    for i, n in sorted_node_ids:
        r = 2/zeta * ( beta * math.log(i) + (1 - beta) math.log(len(G)))
        yield i, r


def get_pl_exponent(g):
    degree = np.array(g.degree())[:, 1]
    myplfit = plfit.plfit(degree)
    x = plfit.plfit.discrete_best_alpha(myplfit, finite=False, verbose=0)
    return x[0]


def number_of_common_neighbors(g):
    N = g.number_of_nodes()
    cn = np.zeros((N, N))
    for (u, v), (i, j) in zip(combinations(g.nodes(), 2),
                              combinations(range(N), 2)):
        cn[i, j] = len(list(nx.common_neighbors(g, u, v)))
    return cn + cn.T


def external_degree(g):
    degree = np.array(g.degree())[:, 1]
    N = g.number_of_nodes()
    adj = nx.to_numpy_array(g)
    cn = number_of_common_neighbors(g)
    d = np.repeat(degree.reshape(1, -1), degree.shape[0], axis=0)
    ext_degree = (d - cn - 1) * adj
    return ext_degree.T

def RA1_weights(g):
    degree = np.array(g.degree())[:, 1]
    di, dj = np.meshgrid(degree, degree)
    cn = number_of_common_neighbors(g)
    return (di + dj + di * dj) / (1 + cn)


def RA2_weights(g):
    ei = external_degree(g)
    cn = number_of_common_neighbors(g)
    return (ei + ei.T + ei * ei.T) / (1 + cn)


def EBC_weights(g):
    w = nx.edge_betweenness_centrality(g)
    edges = [(u, v, w[(u,v)]) for u, v in w]
    _g = g.copy()
    _g.add_weighted_edges_from(edges)


    return nx.to_numpy_array(_g)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
