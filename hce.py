""" Hyperbolic Coaelesccent Embeddings

input: an edgelist

output: a table with the each node's coordinate

"""
import argparse
import math
import logging
from itertools import combinations

from powerlaw import Power_Law
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
        r = 2/zeta * ( beta * math.log(i) + (1 - beta) * math.log(len(G)))
        yield i, r


def get_pl_exponent(g):
    degree = np.array(g.degree())[:, 1]
    pl = Power_Law()
    pl.fit(degree)
    return pl.alpha


def number_of_common_neighbors(g):
    N = g.number_of_nodes()
    weights = []
    for e in g.edges():
        u, v = e[0], e[1]
        w = len(list(nx.common_neighbors(g, u, v)))
        weights.append((u, v, w))
        print(u, v, w)

    _g = g.copy()
    _g.add_weighted_edges_from(weights)

    return nx.to_numpy_array(_g)


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
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='coalescent embedding in the hyperbolic space.')
    parser.add_argument('infile', metavar='edgelist', type=str,
                    help='an input network file (edgelist)')
    args = parser.parse_args()
    logging.info(args)


